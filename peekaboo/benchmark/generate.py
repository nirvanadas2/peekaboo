"""Synthetic benchmark generator.

Produces 5 ground-truth-labeled variants of a small TinyCNN model:

  clean          - normally trained weights, no tampering
  noisy          - clean weights + Gaussian noise injected into some layers
  steganographic - clean weights + payload hidden in mantissa bits
  backdoored     - trained on trigger-poisoned data, misbehaves on trigger input
  combined       - backdoored weights + steganographic payload

Each variant is saved as .safetensors, .pt, and .onnx, and a manifest
(JSON + CSV) records exactly what was modified, for testing detectors
against known ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from peekaboo.benchmark.io_utils import save_onnx, save_pt, save_safetensors, state_dict_to_numpy
from peekaboo.benchmark.models import TinyCNN
from peekaboo.benchmark.tamper import add_gaussian_noise, embed_steganographic_payload
from peekaboo.benchmark.data import DEFAULT_TRIGGER, TriggerSpec, make_dataset
from peekaboo.benchmark.train import train_backdoored, train_clean, trigger_response

ARCH = "cnn"
DUMMY_INPUT = torch.randn(1, 1, 16, 16)


def _num_params(state: dict[str, Any]) -> int:
    return sum(v.size for v in state.values())


def _save_variant(name: str, state: dict, out_dir: Path) -> dict[str, str]:
    st_path = out_dir / f"{name}.safetensors"
    pt_path = out_dir / f"{name}.pt"
    onnx_path = out_dir / f"{name}.onnx"

    save_safetensors(state, str(st_path))
    save_pt(state, str(pt_path))
    save_onnx(TinyCNN(), state, str(onnx_path), DUMMY_INPUT)

    return {
        "safetensors": st_path.name,
        "pt": pt_path.name,
        "onnx": onnx_path.name,
    }


def generate_benchmark(
    out_dir: str, seed: int = 0, trigger: TriggerSpec = DEFAULT_TRIGGER
) -> list[dict[str, Any]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    # --- clean baseline ---
    # Seed before constructing the model, not just before training: model
    # construction (Kaiming/uniform init of conv/linear/BN parameters) is
    # itself a random draw. Seeding only inside `_train` (as this used to
    # do) left initial weights dependent on whatever the global RNG state
    # happened to be at import/process-start time, which is NOT
    # reproducible across fresh processes despite the explicit `seed`
    # argument. See tests/test_benchmark.py::TestCleanVariantReproducibility.
    torch.manual_seed(seed)
    clean_model = TinyCNN()
    clean_training_info = train_clean(clean_model, seed=seed)
    clean_state = state_dict_to_numpy(clean_model)
    clean_files = _save_variant("clean", clean_state, out_path)
    manifest.append(
        {
            "variant": "clean",
            "architecture": ARCH,
            "files": clean_files,
            "is_tampered": False,
            "tamper_types": [],
            "num_tensors": len(clean_state),
            "total_params": _num_params(clean_state),
            "training": clean_training_info,
            "tampering": [],
        }
    )

    # --- noisy: clean + Gaussian noise ---
    noisy_state, noise_gt = add_gaussian_noise(clean_state, seed=seed + 1)
    noisy_files = _save_variant("noisy", noisy_state, out_path)
    manifest.append(
        {
            "variant": "noisy",
            "architecture": ARCH,
            "files": noisy_files,
            "is_tampered": True,
            "tamper_types": ["gaussian_noise"],
            "num_tensors": len(noisy_state),
            "total_params": _num_params(noisy_state),
            "training": clean_training_info,
            "tampering": [noise_gt],
        }
    )

    # --- steganographic: clean + hidden payload ---
    stego_state, stego_gt = embed_steganographic_payload(clean_state)
    stego_files = _save_variant("steganographic", stego_state, out_path)
    manifest.append(
        {
            "variant": "steganographic",
            "architecture": ARCH,
            "files": stego_files,
            "is_tampered": True,
            "tamper_types": ["steganographic_payload"],
            "num_tensors": len(stego_state),
            "total_params": _num_params(stego_state),
            "training": clean_training_info,
            "tampering": [stego_gt],
        }
    )

    # --- backdoored: trained on trigger-poisoned data ---
    torch.manual_seed(seed + 2)  # see the note above the clean baseline
    backdoor_model = TinyCNN()
    backdoor_training_info = train_backdoored(backdoor_model, seed=seed + 2, trigger=trigger)
    backdoor_state = state_dict_to_numpy(backdoor_model)
    backdoor_files = _save_variant("backdoored", backdoor_state, out_path)
    # Same held-out triggered set the backdoor's ASR is measured on, run
    # through the CLEAN model: the task-only baseline the ASR must clear.
    test_images, test_labels = make_dataset(200, seed=seed + 2 + 1000)
    clean_trigger_baseline = trigger_response(clean_model, test_images, test_labels, trigger)
    backdoor_gt = {
        "tamper_type": "backdoor_trigger",
        "trigger_description": trigger.describe(),
        "trigger": trigger.to_dict(),
        "target_class": backdoor_training_info["trigger_target_class"],
        "poison_fraction": backdoor_training_info["poison_fraction"],
        "attack_success_rate": backdoor_training_info["attack_success_rate"],
        "attack_success_rate_non_target_labels": backdoor_training_info["attack_success_rate_non_target_labels"],
        "clean_model_trigger_to_target_rate": clean_trigger_baseline["trigger_to_target_rate"],
        "clean_model_trigger_to_target_rate_non_target_labels": clean_trigger_baseline[
            "trigger_to_target_rate_non_target_labels"
        ],
    }
    manifest.append(
        {
            "variant": "backdoored",
            "architecture": ARCH,
            "files": backdoor_files,
            "is_tampered": True,
            "tamper_types": ["backdoor_trigger"],
            "num_tensors": len(backdoor_state),
            "total_params": _num_params(backdoor_state),
            "training": backdoor_training_info,
            "tampering": [backdoor_gt],
        }
    )

    # --- combined: backdoored + steganographic payload ---
    combined_state, combined_stego_gt = embed_steganographic_payload(backdoor_state)
    combined_files = _save_variant("combined", combined_state, out_path)
    manifest.append(
        {
            "variant": "combined",
            "architecture": ARCH,
            "files": combined_files,
            "is_tampered": True,
            "tamper_types": ["backdoor_trigger", "steganographic_payload"],
            "num_tensors": len(combined_state),
            "total_params": _num_params(combined_state),
            "training": backdoor_training_info,
            "tampering": [dict(backdoor_gt), combined_stego_gt],
        }
    )

    _write_manifest_json(manifest, out_path / "manifest.json")
    _write_manifest_csv(manifest, out_path / "manifest.csv")

    return manifest


def _write_manifest_json(manifest: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_manifest_csv(manifest: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "variant",
        "architecture",
        "file_safetensors",
        "file_pt",
        "file_onnx",
        "is_tampered",
        "tamper_types",
        "affected_layers",
        "num_tensors",
        "total_params",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest:
            affected_layers = sorted(
                {
                    layer
                    for tamper in entry["tampering"]
                    for layer in tamper.get("affected_layers", [])
                }
            )
            writer.writerow(
                {
                    "variant": entry["variant"],
                    "architecture": entry["architecture"],
                    "file_safetensors": entry["files"]["safetensors"],
                    "file_pt": entry["files"]["pt"],
                    "file_onnx": entry["files"]["onnx"],
                    "is_tampered": entry["is_tampered"],
                    "tamper_types": "|".join(entry["tamper_types"]),
                    "affected_layers": "|".join(affected_layers),
                    "num_tensors": entry["num_tensors"],
                    "total_params": entry["total_params"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Peekaboo synthetic benchmark models")
    parser.add_argument("--out-dir", default="benchmark_output", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    manifest = generate_benchmark(args.out_dir, seed=args.seed)
    print(f"Generated {len(manifest)} benchmark variants in '{args.out_dir}'")
    for entry in manifest:
        print(f"  {entry['variant']:<16} tampered={entry['is_tampered']!s:<5} types={entry['tamper_types']}")


if __name__ == "__main__":
    main()
