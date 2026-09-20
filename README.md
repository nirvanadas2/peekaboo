# Peekaboo

Peekaboo is a pre-deployment scanner that inspects AI model weight files (SafeTensors, pickle-based PyTorch checkpoints, and ONNX) to detect steganographic payloads hidden in low-order mantissa bits, backdoor triggers that cause targeted misbehavior on specific inputs, and statistically abnormal parameter distributions — producing a Model Risk Score with per-layer flags and explanations before a model is trusted and deployed.

## Status

Phase 0: Setup & Ground Truth. Provides format-agnostic model loaders, a synthetic benchmark generator for ground-truth testing (clean / noisy / steganographic / backdoored / combined variants), and a stub schema for the eventual Model Risk Score.

Phase 1: Metadata Integrity & Structural Consistency pre-checks now run ahead of deep analysis — file/format integrity, safe-checkpoint gating, and self-consistency/spec-diff structural checks, wired together via `run_pre_checks`. See [PHASE1.md](PHASE1.md) for details (what each stage checks, the severity policy, and known/documented gaps). Detection/scoring logic (statistical, steganographic, behavioral pillars) is not yet implemented.

## Layout

- `peekaboo/loaders/` — load `.safetensors`, `.pt`/`.pth`, and `.onnx` files into a common internal representation (layer names, shapes, dtypes, raw tensors).
- `peekaboo/benchmark/` — generates small synthetic models with known ground-truth tampering for testing detectors.
- `peekaboo/schema/` — data classes for the Model Risk Score output, plus the Phase 1 `MetadataReport`/`StructuralReport`/`Finding` types.
- `peekaboo/pipeline/` — Phase 1 pre-checks: `metadata_check.py` (Stage 1), `structural_check.py` (Stage 2), `gate.py` (`run_pre_checks`, wiring the two together). See [PHASE1.md](PHASE1.md).
- `tests/` — unit tests for loaders, the benchmark generator, and the Phase 1 pre-checks.

## Install

```
pip install -e ".[dev]"
```

## Run tests

```
pytest
```
