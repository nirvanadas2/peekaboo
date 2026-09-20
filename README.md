# Peekaboo

Peekaboo is a pre-deployment scanner that inspects AI model weight files (SafeTensors, pickle-based PyTorch checkpoints, and ONNX) to detect steganographic payloads hidden in low-order mantissa bits, backdoor triggers that cause targeted misbehavior on specific inputs, and statistically abnormal parameter distributions — producing a Model Risk Score with per-layer flags and explanations before a model is trusted and deployed.

## Status

Phase 0: Setup & Ground Truth. Provides format-agnostic model loaders, a synthetic benchmark generator for ground-truth testing (clean / noisy / steganographic / backdoored / combined variants), and a stub schema for the eventual Model Risk Score. `TinyCNN` (the benchmark architecture) has 7 conv/linear weight-bearing layers — deepened from an original 3-layer design specifically so Phase 2's relative-outlier detection has enough population to run; see [PHASE2.md](PHASE2.md) for why. Benchmark generation is fully deterministic given a seed (a seeding-order bug that broke this was found and fixed).

Phase 1: Metadata Integrity & Structural Consistency pre-checks now run ahead of deep analysis — file/format integrity, safe-checkpoint gating, and self-consistency/spec-diff structural checks, wired together via `run_pre_checks`. See [PHASE1.md](PHASE1.md) for details (what each stage checks, the severity policy, and known/documented gaps).

Phase 2: Statistical Analysis (Stage 3) — per-layer mean/std/skewness/kurtosis/entropy with within-model relative outlier detection (robust median/MAD z-score, log-scale for std, ceiling-normalized for entropy — both corrections found empirically necessary to avoid false positives from legitimate architectural scale variation), falling back to conservative absolute ranges on small models. Not yet wired into `run_pre_checks`. See [PHASE2.md](PHASE2.md) for the full detection story: false positives are resolved, but recall on Phase 0's benchmark tampering magnitudes is honestly weak (noise: undetected; backdoor: a weak, format-inconsistent partial signal) — reported as measured, not tuned to pass. Remaining detection/scoring logic (steganographic, behavioral pillars, the Model Risk Score itself) is not yet implemented.

## Layout

- `peekaboo/loaders/` — load `.safetensors`, `.pt`/`.pth`, and `.onnx` files into a common internal representation (layer names, shapes, dtypes, raw tensors).
- `peekaboo/benchmark/` — generates small synthetic models with known ground-truth tampering for testing detectors.
- `peekaboo/schema/` — data classes for the Model Risk Score output, plus the `MetadataReport`/`StructuralReport`/`StatisticalReport`/`Finding` types (Phases 1-2).
- `peekaboo/pipeline/` — pre-check and analysis stages: `metadata_check.py` (Stage 1), `structural_check.py` (Stage 2), `statistical_check.py` (Stage 3), `gate.py` (`run_pre_checks`, currently wiring Stages 1-2). See [PHASE1.md](PHASE1.md) and [PHASE2.md](PHASE2.md).
- `tests/` — unit tests for loaders, the benchmark generator, and the pipeline stages.

## Install

```
pip install -e ".[dev]"
```

## Run tests

```
pytest
```
