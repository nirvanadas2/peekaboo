"""Tests for Stage 1 (Metadata Integrity)."""

from __future__ import annotations

import json
import pickle
import shutil
import struct
from pathlib import Path

import pytest

from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.schema import Severity


def _finding(report, check_name):
    return next(f for f in report.findings if f.check == check_name)


class TestBenchmarkModelsPassMetadataCheck:
    """None of Phase 0's tampering (noise, stego, backdoor) touches file
    format or header structure, only weight content — so every benchmark
    variant, in every format, must pass Stage 1 cleanly."""

    @pytest.mark.parametrize(
        "variant", ["clean", "noisy", "steganographic", "backdoored", "combined"]
    )
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_variant_passes(self, benchmark_dir: Path, variant: str, suffix: str) -> None:
        report = run_metadata_check(str(benchmark_dir / f"{variant}.{suffix}"))
        assert report.passed, f"findings: {[f.to_dict() for f in report.findings if not f.passed]}"
        assert not report.hard_fail
        assert report.declared_format == report.detected_format
        assert len(report.file_hash) == 64
        assert report.file_size > 0


class TestFileHashAndSize:
    def test_hash_is_deterministic_and_matches_sha256(self, benchmark_dir: Path) -> None:
        import hashlib

        path = benchmark_dir / "clean.safetensors"
        report = run_metadata_check(str(path))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert report.file_hash == expected
        assert report.file_size == path.stat().st_size

    def test_missing_file_hard_fails(self, tmp_path: Path) -> None:
        report = run_metadata_check(str(tmp_path / "does_not_exist.safetensors"))
        assert report.hard_fail
        assert not report.passed
        assert _finding(report, "file_exists").severity == Severity.CRITICAL

    def test_empty_file_hard_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.safetensors"
        empty.write_bytes(b"")
        report = run_metadata_check(str(empty))
        assert report.hard_fail
        assert _finding(report, "file_not_empty").severity == Severity.CRITICAL

    def test_unknown_extension_hard_fails(self, tmp_path: Path) -> None:
        bogus = tmp_path / "model.xyz"
        bogus.write_bytes(b"not a real model file, just some bytes")
        report = run_metadata_check(str(bogus))
        assert report.hard_fail
        assert _finding(report, "known_extension").severity == Severity.CRITICAL


class TestTruncatedFiles:
    def test_truncated_safetensors_mid_header_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        src = benchmark_dir / "clean.safetensors"
        truncated = tmp_path / "truncated_header.safetensors"
        # Cut off in the middle of the 8-byte header-length prefix.
        truncated.write_bytes(src.read_bytes()[:4])

        report = run_metadata_check(str(truncated))
        assert report.hard_fail
        assert not report.passed

    def test_truncated_safetensors_mid_data_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        src = benchmark_dir / "clean.safetensors"
        data = src.read_bytes()
        # Keep the full header but chop off most of the tensor data.
        header_len = struct.unpack("<Q", data[:8])[0]
        cutoff = 8 + header_len + 4
        truncated = tmp_path / "truncated_data.safetensors"
        truncated.write_bytes(data[:cutoff])

        report = run_metadata_check(str(truncated))
        assert report.hard_fail
        assert not report.passed
        assert _finding(report, "safetensors_offsets_valid").severity == Severity.CRITICAL

    def test_truncated_onnx_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        src = benchmark_dir / "clean.onnx"
        data = src.read_bytes()
        truncated = tmp_path / "truncated.onnx"
        truncated.write_bytes(data[: len(data) // 2])

        report = run_metadata_check(str(truncated))
        assert report.hard_fail
        assert not report.passed

    def test_truncated_pt_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        src = benchmark_dir / "clean.pt"
        data = src.read_bytes()
        truncated = tmp_path / "truncated.pt"
        truncated.write_bytes(data[: len(data) // 2])

        report = run_metadata_check(str(truncated))
        assert report.hard_fail
        assert not report.passed


class TestMismatchedExtensions:
    def test_onnx_renamed_to_safetensors_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "sneaky.safetensors"
        shutil.copyfile(benchmark_dir / "clean.onnx", renamed)

        report = run_metadata_check(str(renamed))
        assert report.hard_fail
        assert not report.passed
        assert report.declared_format == "safetensors"
        assert report.detected_format != "safetensors"

    def test_safetensors_renamed_to_onnx_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "sneaky.onnx"
        shutil.copyfile(benchmark_dir / "clean.safetensors", renamed)

        report = run_metadata_check(str(renamed))
        assert report.hard_fail
        assert not report.passed

    def test_pt_renamed_to_safetensors_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "sneaky.safetensors"
        shutil.copyfile(benchmark_dir / "clean.pt", renamed)

        report = run_metadata_check(str(renamed))
        assert report.hard_fail
        assert not report.passed
        assert _finding(report, "extension_matches_content").severity == Severity.CRITICAL

    def test_safetensors_renamed_to_pt_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "sneaky.pt"
        shutil.copyfile(benchmark_dir / "clean.safetensors", renamed)

        report = run_metadata_check(str(renamed))
        assert report.hard_fail
        assert not report.passed

    def test_onnx_renamed_to_pt_hard_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "sneaky.pt"
        shutil.copyfile(benchmark_dir / "clean.onnx", renamed)

        report = run_metadata_check(str(renamed))
        assert report.hard_fail
        assert not report.passed


class TestCorruptedSafetensorsHeader:
    def test_non_json_header_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.safetensors"
        garbage_header = b"{not valid json!!"
        path.write_bytes(struct.pack("<Q", len(garbage_header)) + garbage_header + b"\x00" * 16)

        report = run_metadata_check(str(path))
        assert report.hard_fail
        assert _finding(report, "safetensors_header_valid").severity == Severity.CRITICAL

    def test_header_length_out_of_bounds_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.safetensors"
        # Claim a header far larger than the file actually is. (Avoid a
        # length whose little-endian first byte is 0x80 or 'P'/0x50 — that
        # would coincidentally collide with the pickle/zip magic sniff.)
        path.write_bytes(struct.pack("<Q", 3_000_000) + b"\x00" * 32)

        report = run_metadata_check(str(path))
        assert report.hard_fail
        assert _finding(report, "safetensors_header_valid").severity == Severity.CRITICAL

    def test_unexpected_extra_key_in_tensor_entry_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "extra_key.safetensors"
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
                "hidden_payload": "sneaky extra field",
            }
        }
        header_bytes = json.dumps(header).encode("utf-8")
        data = b"\x00" * 16
        path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)

        report = run_metadata_check(str(path))
        assert not report.passed
        finding = _finding(report, "safetensors_no_extra_keys")
        assert not finding.passed
        assert "hidden_payload" in str(finding.details)

    def test_data_offsets_out_of_bounds_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_offsets.safetensors"
        header = {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 999]}}
        header_bytes = json.dumps(header).encode("utf-8")
        data = b"\x00" * 16
        path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)

        report = run_metadata_check(str(path))
        assert report.hard_fail
        assert _finding(report, "safetensors_offsets_valid").severity == Severity.CRITICAL

    def test_valid_metadata_block_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "with_metadata.safetensors"
        header = {
            "__metadata__": {"format": "pt", "author": "test"},
            "weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        data = b"\x00" * 16
        path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)

        report = run_metadata_check(str(path))
        assert report.passed
        assert not report.hard_fail


class TestUnsafePickleIsHardGated:
    def test_code_exec_pickle_hard_fails(self, tmp_path: Path) -> None:
        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned') or {}",))

        evil_path = tmp_path / "evil.pt"
        with open(evil_path, "wb") as f:
            pickle.dump({"state": Evil()}, f)

        report = run_metadata_check(str(evil_path))
        assert report.hard_fail
        assert not report.passed
        finding = _finding(report, "pickle_weights_only_safe")
        assert finding.severity == Severity.CRITICAL
        assert not finding.passed


class TestOnnxOpsetSanity:
    def test_insane_opset_version_fails(self, benchmark_dir: Path, tmp_path: Path) -> None:
        import onnx

        model = onnx.load(str(benchmark_dir / "clean.onnx"))
        for imp in model.opset_import:
            if not imp.domain:
                imp.version = 99999
        path = tmp_path / "insane_opset.onnx"
        onnx.save(model, str(path))

        report = run_metadata_check(str(path))
        assert not report.passed
        finding = _finding(report, "onnx_opset_sane")
        assert not finding.passed
        assert finding.severity == Severity.HIGH


class TestOnnxCheckerFailureIsNotHardFail:
    """A model that fails onnx.checker (e.g. a dangling node reference) can
    still have perfectly extractable initializer tensors, so it should be
    flagged strongly but must not skip Stage 2+ analysis."""

    def test_checker_invalid_graph_is_high_not_critical(self, tmp_path: Path) -> None:
        import onnx
        from onnx import helper

        node = helper.make_node("Identity", ["undefined_input"], ["out"])
        graph = helper.make_graph(
            [node], "g", [], [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4, 4])]
        )
        model = helper.make_model(graph)
        path = tmp_path / "checker_invalid.onnx"
        onnx.save(model, str(path))

        # Sanity check the fixture: onnx.load succeeds, onnx.checker rejects it.
        loaded = onnx.load(str(path))
        with pytest.raises(Exception):
            onnx.checker.check_model(loaded)

        report = run_metadata_check(str(path))
        finding = _finding(report, "onnx_checker")
        assert not finding.passed
        assert finding.severity == Severity.HIGH
        assert not report.hard_fail
        assert not report.passed


class TestBroadenedExtensionAllowlist:
    """Stage 1 recognizes .bin (HuggingFace legacy pytorch_model.bin) and
    .ckpt (PyTorch Lightning) as pytorch_pickle content, independent of
    whether Phase 0's loader dispatch knows those extensions."""

    @pytest.mark.parametrize("suffix", ["bin", "ckpt"])
    def test_legacy_pickle_extension_passes_metadata_check(
        self, benchmark_dir: Path, tmp_path: Path, suffix: str
    ) -> None:
        renamed = tmp_path / f"pytorch_model.{suffix}"
        shutil.copyfile(benchmark_dir / "clean.pt", renamed)

        report = run_metadata_check(str(renamed))
        assert report.passed
        assert not report.hard_fail
        assert report.declared_format == "pytorch_pickle"
        assert report.detected_format == "pytorch_pickle"

    @pytest.mark.parametrize("suffix", ["bin", "ckpt"])
    def test_unsafe_pickle_is_still_hard_gated_under_legacy_extension(
        self, tmp_path: Path, suffix: str
    ) -> None:
        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned') or {}",))

        evil_path = tmp_path / f"evil.{suffix}"
        with open(evil_path, "wb") as f:
            pickle.dump({"state": Evil()}, f)

        report = run_metadata_check(str(evil_path))
        assert report.hard_fail
        assert not report.passed

    @pytest.mark.parametrize("suffix", ["bin", "ckpt"])
    def test_known_gap_passes_stage1_but_phase0_dispatch_cannot_load_it(
        self, benchmark_dir: Path, tmp_path: Path, suffix: str
    ) -> None:
        """Documents the known, temporary gap: Stage 1's extension mapping is
        broader than Phase 0's loader dispatch, so a .bin/.ckpt file can pass
        Stage 1 and still fail to load via `load_model()`."""
        from peekaboo.loaders import load_model

        renamed = tmp_path / f"pytorch_model.{suffix}"
        shutil.copyfile(benchmark_dir / "clean.pt", renamed)

        report = run_metadata_check(str(renamed))
        assert report.passed  # Stage 1 is satisfied...

        with pytest.raises(ValueError):
            load_model(str(renamed))  # ...but Phase 0's dispatcher still isn't.


class TestOnnxExternalDataNotFound:
    """A .onnx graph descriptor scanned without its companion external-data
    file is a common, legitimate scenario (not evidence of tampering) and
    must be distinguished from a genuinely corrupted/mismatched file."""

    def _build_external_data_model(self, path: Path, data_filename: str) -> None:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        weight = numpy_helper.from_array(np.random.rand(4, 4).astype("float32"), name="w")
        node = helper.make_node("Identity", ["w"], ["out"])
        graph = helper.make_graph(
            [node],
            "g",
            [],
            [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4, 4])],
            initializer=[weight],
        )
        model = helper.make_model(graph)
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=0,
        )

    def test_missing_companion_data_file_is_medium_not_hard_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        self._build_external_data_model(path, "model.onnx.data")

        # Simulate scanning the .onnx file in isolation from its companion.
        (tmp_path / "model.onnx.data").unlink()

        report = run_metadata_check(str(path))
        finding = _finding(report, "external_data_not_found")
        assert not finding.passed
        assert finding.severity == Severity.MEDIUM
        assert not report.hard_fail
        assert not report.passed

    def test_present_companion_data_file_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        self._build_external_data_model(path, "model.onnx.data")

        report = run_metadata_check(str(path))
        assert report.passed
        assert not report.hard_fail
