"""Tests that the synthetic benchmark generator produces valid, loadable
models and that its ground-truth manifest is internally consistent."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from peekaboo.benchmark.stego import extract_lsb
from peekaboo.loaders import load_model

EXPECTED_VARIANTS = {"clean", "noisy", "steganographic", "backdoored", "combined"}


class TestManifestStructure:
    def test_five_variants_present(self, manifest: list[dict]) -> None:
        assert {entry["variant"] for entry in manifest} == EXPECTED_VARIANTS

    def test_clean_is_not_flagged_tampered(self, manifest: list[dict]) -> None:
        clean = next(e for e in manifest if e["variant"] == "clean")
        assert clean["is_tampered"] is False
        assert clean["tamper_types"] == []

    def test_other_variants_flagged_tampered(self, manifest: list[dict]) -> None:
        for entry in manifest:
            if entry["variant"] != "clean":
                assert entry["is_tampered"] is True
                assert len(entry["tamper_types"]) > 0

    def test_combined_has_both_tamper_types(self, manifest: list[dict]) -> None:
        combined = next(e for e in manifest if e["variant"] == "combined")
        assert set(combined["tamper_types"]) == {"backdoor_trigger", "steganographic_payload"}

    def test_all_referenced_files_exist(self, benchmark_dir: Path, manifest: list[dict]) -> None:
        for entry in manifest:
            for filename in entry["files"].values():
                assert (benchmark_dir / filename).exists()

    def test_csv_manifest_matches_json_variants(self, benchmark_dir: Path, manifest: list[dict]) -> None:
        with open(benchmark_dir / "manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        csv_variants = {row["variant"] for row in rows}
        json_variants = {entry["variant"] for entry in manifest}
        assert csv_variants == json_variants


class TestGeneratedModelsAreLoadable:
    def test_each_variant_safetensors_loads_and_matches_manifest(
        self, benchmark_dir: Path, manifest: list[dict]
    ) -> None:
        for entry in manifest:
            lm = load_model(str(benchmark_dir / entry["files"]["safetensors"]))
            assert len(lm) == entry["num_tensors"]
            assert lm.total_params() == entry["total_params"]


class TestBackdoorGroundTruth:
    def test_backdoor_attack_success_rate_high(self, manifest: list[dict]) -> None:
        backdoored = next(e for e in manifest if e["variant"] == "backdoored")
        trigger_gt = next(
            t for t in backdoored["tampering"] if t["tamper_type"] == "backdoor_trigger"
        )
        assert trigger_gt["attack_success_rate"] > 0.9

    def test_backdoor_retains_reasonable_clean_accuracy(self, manifest: list[dict]) -> None:
        backdoored = next(e for e in manifest if e["variant"] == "backdoored")
        assert backdoored["training"]["clean_accuracy"] > 0.4  # well above 1/4 chance


class TestNoiseGroundTruth:
    def test_noisy_weights_differ_from_clean_only_on_affected_layers(
        self, benchmark_dir: Path, manifest: list[dict]
    ) -> None:
        clean = load_model(str(benchmark_dir / "clean.safetensors"))
        noisy = load_model(str(benchmark_dir / "noisy.safetensors"))
        noisy_entry = next(e for e in manifest if e["variant"] == "noisy")
        affected = set(noisy_entry["tampering"][0]["affected_layers"])

        for name in clean.tensors:
            clean_arr = clean.tensors[name].array
            noisy_arr = noisy.tensors[name].array
            if name in affected:
                assert not np.array_equal(clean_arr, noisy_arr)
            else:
                assert np.array_equal(clean_arr, noisy_arr)


class TestSteganographicGroundTruth:
    def test_payload_is_recoverable_from_affected_layers(
        self, benchmark_dir: Path, manifest: list[dict]
    ) -> None:
        stego_entry = next(e for e in manifest if e["variant"] == "steganographic")
        tamper = stego_entry["tampering"][0]
        lm = load_model(str(benchmark_dir / "steganographic.safetensors"))

        recovered = b""
        for layer in tamper["affected_layers"]:
            n_bits = tamper["bits_embedded_per_layer"][layer]
            if n_bits == 0:
                continue
            arr = lm.tensors[layer].array.astype(np.float32)
            recovered += extract_lsb(arr, n_bits, bits_per_value=tamper["bits_per_value"])

        assert len(recovered) == tamper["payload_length_bytes"]
        assert hashlib.sha256(recovered).hexdigest() == tamper["payload_sha256"]

    def test_steganographic_weights_differ_from_clean_only_on_affected_layers(
        self, benchmark_dir: Path, manifest: list[dict]
    ) -> None:
        clean = load_model(str(benchmark_dir / "clean.safetensors"))
        stego = load_model(str(benchmark_dir / "steganographic.safetensors"))
        stego_entry = next(e for e in manifest if e["variant"] == "steganographic")
        affected = set(stego_entry["tampering"][0]["affected_layers"])

        for name in clean.tensors:
            clean_arr = clean.tensors[name].array
            stego_arr = stego.tensors[name].array
            if name in affected:
                assert not np.array_equal(clean_arr, stego_arr)
                # LSB embedding should perturb values only slightly.
                assert np.allclose(clean_arr, stego_arr, atol=1e-2)
            else:
                assert np.array_equal(clean_arr, stego_arr)
