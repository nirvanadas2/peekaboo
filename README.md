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

Phase 5: Anomaly Fusion (Stage 6). This stage combines the evidence from Stages 3-5 into a `ModelRiskScore`. A pillar that didn't run is marked "not assessed", never "clean". Stage 4 and Stage 3 false-positive fixes were validated on a pre-registered fresh suite. See [PHASE5.md](PHASE5.md).

Phase 6: Explainable Report (Stage 7), plus final validation. Stage 3 is now report-only. The explainable report comes as Markdown or JSON, and the CLI returns CI-friendly exit codes. The complete system was validated once on a final pre-registered suite:
- **Clean false positives:** 0/10.
- **Backdoors:** 9/10 with a `forward_fn` (AUC 0.95); reported as *not assessed* without one.
- **Noise:** 7/10.

Cumulative held-out results:
- **Backdoors:** 21/26 detected, 0/78 false positives.
- **Noise:** 16/20 detected, 0/80 false positives.
- **Stego:** encrypted payloads are undetectable, and the benchmark's small payload is not detected.

See [PHASE6.md](PHASE6.md) for what the project can and cannot claim.

Every benchmark-backed number is measured on committed fixtures (`tests/fixtures/`), because regenerated weights differ in their lowest mantissa bits across torch/CPU builds.

## Usage

```
python -m peekaboo scan model.safetensors --md report.md --json report.json
```

The exit code is 0 for no scored evidence, 1 for MEDIUM, 2 for HIGH, and 3 for an unsafe file. Probing for backdoors needs a runnable model:

```python
from peekaboo.pipeline import run_pre_checks
from peekaboo.report import render_markdown

result = run_pre_checks(path, forward_fn=my_batch_to_logits, input_shape=(1, 16, 16), num_classes=4)
print(render_markdown(result))
```

`--tinycnn` supplies the forward function for this repo's synthetic benchmark files only.

## Layout

- `peekaboo/loaders/` — load `.safetensors`, `.pt`/`.pth`, and `.onnx` files into a common internal representation (layer names, shapes, dtypes, raw tensors).
- `peekaboo/benchmark/` — generates small synthetic models with known ground-truth tampering for testing detectors.
- `peekaboo/report.py` (Stage 7): the explainable Markdown/JSON report and exit codes. `peekaboo/__main__.py` is the CLI.
- `peekaboo/schema/` — data classes for the Model Risk Score output, plus the `MetadataReport`/`StructuralReport`/`StatisticalReport`/`StegoReport`/`BehavioralReport`/`Finding` types.
- `peekaboo/pipeline/`: the pre-check and analysis stages. [PHASE1.md](PHASE1.md) through [PHASE6.md](PHASE6.md) document them.
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
