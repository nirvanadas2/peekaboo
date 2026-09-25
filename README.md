# Peekaboo

Peekaboo is a pre-deployment scanner that inspects AI model weight files (SafeTensors, pickle-based PyTorch checkpoints, and ONNX) to detect steganographic payloads hidden in low-order mantissa bits, backdoor triggers that cause targeted misbehavior on specific inputs, and statistically abnormal parameter distributions — producing a Model Risk Score with per-layer flags and explanations before a model is trusted and deployed.

## Status

Phase 0: Setup & Ground Truth. Provides format-agnostic model loaders, a synthetic benchmark generator for ground-truth testing (clean / noisy / steganographic / backdoored / combined variants), and a stub schema for the eventual Model Risk Score. `TinyCNN` (the benchmark architecture) has 7 conv/linear weight-bearing layers — deepened from an original 3-layer design specifically so Phase 2's relative-outlier detection has enough population to run; see [PHASE2.md](PHASE2.md) for why. Benchmark generation is deterministic given a seed *within one environment* (a seeding-order bug that broke this was found and fixed). It is not bit-identical across torch/CPU builds, hence the committed fixtures. The backdoor trigger is configurable (`TriggerSpec`) and guarded against being confounded with the task.

Phase 1: Metadata Integrity & Structural Consistency pre-checks now run ahead of deep analysis — file/format integrity, safe-checkpoint gating, and self-consistency/spec-diff structural checks, wired together via `run_pre_checks`. See [PHASE1.md](PHASE1.md) for details (what each stage checks, the severity policy, and known/documented gaps).

Phase 2: Statistical Analysis (Stage 3). This stage computes per-layer mean, std, skewness, kurtosis and entropy, and finds outliers relative to the model's other layers, falling back to conservative absolute ranges on small models. It is wired into `run_pre_checks`. It produces no false positives, but on this benchmark it detects **none** of the tampered variants. See [PHASE2.md](PHASE2.md).

Phase 3: Steganographic Detection (Stage 4). This stage runs bit-level tests on the low mantissa bits, with a Benjamini-Hochberg false-discovery-rate correction applied within each report. It is not wired into `run_pre_checks`.
- It reliably detects the **noisy** variant.
- It has one remaining clean false positive.
- It does **not** detect the benchmark's steganographic payload.
- A density sweep shows it can only catch *structured*, dense payloads. A random-looking (e.g. encrypted) payload is undetectable at any density.

See [PHASE3.md](PHASE3.md).

Phase 4: Behavioral Probing (Stage 5). This stage runs the model on patched probe inputs. It requires a `forward_fn` supplied by the caller and is not wired into `run_pre_checks`. It uses two complementary tests, a local-inconsistency ("island") test and a class-reach asymmetry test.
- On a held-out suite of 6 backdoors that was specified before the design was built, it detected 4 of 6, with 0 false positives across 18 untampered models, and identified the correct target class in every detection.
- Both sample sizes are small.
- The asymmetry test relies on an assumption specific to this benchmark's task.

See [PHASE4.md](PHASE4.md).

Phase 5: Anomaly Fusion (Stage 6). This stage combines the evidence from Stages 3-5 into a `ModelRiskScore`. It keeps a pillar that didn't run distinct from one that ran and found nothing. `run_pre_checks` now runs Stages 1-6, with Stage 5 opt-in via `forward_fn`.

End-to-end runs on independent clean models exposed Stage 3 and Stage 4 false positives. Both stages were fixed: Stage 4 is now init-lattice aware, and Stage 3 uses a noise-aware z for mean and kurtosis. Both fixes were validated on a pre-registered fresh suite of 50 models:
- **Stage 4:** 0/10 clean false positives, 9/10 noise recall.
- **Stage 5:** 0/10 clean false positives, 8/10 backdoor recall.
- **Stage 3:** still 4/10 clean false positives (now from entropy).
- **Fused ROC AUC:** noisy 0.98, backdoored 0.91.

See [PHASE5.md](PHASE5.md). The explainable report (Stage 7) is not started.

Every benchmark-backed number is measured on committed fixtures (`tests/fixtures/`), because regenerated weights differ in their lowest mantissa bits across torch/CPU builds.

## Layout

- `peekaboo/loaders/` — load `.safetensors`, `.pt`/`.pth`, and `.onnx` files into a common internal representation (layer names, shapes, dtypes, raw tensors).
- `peekaboo/benchmark/` — generates small synthetic models with known ground-truth tampering for testing detectors.
- `peekaboo/schema/` — data classes for the Model Risk Score output, plus the `MetadataReport`/`StructuralReport`/`StatisticalReport`/`StegoReport`/`BehavioralReport`/`Finding` types.
- `peekaboo/pipeline/`: the pre-check and analysis stages. [PHASE1.md](PHASE1.md) through [PHASE5.md](PHASE5.md) document them.
  - `metadata_check.py` (Stage 1)
  - `structural_check.py` (Stage 2)
  - `statistical_check.py` (Stage 3)
  - `stego_check.py` (Stage 4)
  - `behavioral_probe.py` (Stage 5)
  - `fusion.py` (Stage 6)
  - `multiple_testing.py` (shared Benjamini-Hochberg correction)
  - `gate.py` (`run_pre_checks`, which wires Stages 1-6)
- `tests/`: unit tests for the loaders, the benchmark generator, and the pipeline stages.
  - `tests/fixtures/benchmark/` is the canonical benchmark.
  - `tests/fixtures/stage5_validation/` holds Stage 5's development and held-out backdoors.

## Install

```
pip install -e ".[dev]"
```

## Run tests

```
pytest
```
