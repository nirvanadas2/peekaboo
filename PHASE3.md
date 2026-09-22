# Phase 3 — Stage 4: Steganographic Detection

## Status: calibrated against the real benchmark; NOT yet wired into `gate.py`

This phase implements bit-plane / mantissa-level statistical tests to
catch mantissa-LSB steganographic payloads — the threat class Stage 3
explicitly cannot detect (confirmed empirically in Phase 2: zero
findings on the steganographic variant, by design).

Since the draft below was written (outside this repo, without access to
the loaders or benchmark), `calibrate_on_clean()` has been run against
the real clean-variant benchmark and `analyze_model()` against the full
5×3 matrix, per the "suggested next steps" this doc originally laid out.
See **"Calibration & full-benchmark results"** near the end for the
honest results table — including a **known multiple-comparisons gap
that any future work on this stage (or a Model Risk Score that consumes
its findings) needs to account for.**

## Files

- `peekaboo/pipeline/stego_check.py` — core implementation
- `tests/test_stego_check.py` — unit tests for the statistical
  primitives, using synthetic bit sequences and synthetic float32/16
  tensors. These do NOT touch the project's actual `.safetensors`
  benchmark files — they validate the math in isolation (e.g. "does
  the runs test correctly flag a perfectly alternating sequence as
  non-random", "does block homogeneity catch a localized payload").

## Four tests per layer

| Test | What it catches | Has a real null distribution? |
|---|---|---|
| Bit-balance chi-square | Whole-layer 0/1 imbalance in mantissa LSBs | Yes — chi-square(df=1) |
| Block homogeneity chi-square | Payload localized to part of a tensor | Yes — chi-square(df=n_blocks-1) |
| Wald-Wolfowitz runs test | Non-random ordering (not just count) of bits | Yes — normal approx |
| Bit autocorrelation (lags 1/2/4/8) | Periodic structure (e.g. byte-boundary effects) | **No** — currently informational only, no permutation-test p-value yet |

Unlike Stage 3, these are genuine hypothesis tests with known null
distributions, not descriptive statistics needing a same-model
baseline — that's the whole point of moving to bit-level analysis.

## What is NOT done yet (be upfront about this before Phase 4/5)

1. **No calibration run against the actual benchmark.** I do not have
   access to the repo or the `.safetensors`/`.pt`/`.onnx` benchmark
   files, so `SEVERITY_THRESHOLDS` in `stego_check.py` are
   placeholders. **Before trusting any finding**, run
   `calibrate_on_clean()` (included in the module) against the clean
   variant, across all three formats, and check the false-positive
   rate at MEDIUM+ severity. If it's non-trivial, loosen the
   thresholds — this mirrors how Phase 2 was honest that Stage 3's
   absolute-heuristic fallback needed real tuning.
2. **Bit autocorrelation has no formal p-value.** It's reported as
   informational (`p_value=None`) rather than assigned a severity from
   a real null distribution. A rigorous version needs a permutation
   test: shuffle the bit sequence N times, recompute the
   autocorrelation, and see where the observed value falls in that
   null distribution. Not implemented — flagged as a TODO, same
   spirit as Phase 1's `.bin`/`.ckpt` loader-dispatch gap: a known,
   deliberate, documented gap rather than a silent one.
3. **Only float32/float16 mantissas are extracted.** Int8/int4
   quantized weight formats aren't handled — if the project needs to
   support quantized models later, this needs a different extraction
   strategy (quantized formats don't have a "mantissa" in the IEEE754
   sense).
4. **Only the lowest N mantissa bits are tested** (`n_bits` parameter,
   default 4). If probing shows payloads hiding in less-obvious bit
   positions, this is the place to extend.
5. **No integration into `gate.py` yet.** This module is standalone;
   wiring it into the existing pipeline (alongside `metadata_check.py`
   and `structural_check.py`) and adding it to `PreCheckResult` /
   `ModelRiskScore` is the next concrete step once thresholds are
   calibrated.

## Suggested next steps (in order)

1. ~~Drop `stego_check.py` into `peekaboo/pipeline/`, `test_stego_check.py`
   into `tests/`. Run the unit tests — they should all pass
   independent of any real model files, confirming the primitives are
   correct.~~ **Done.** Two of the 22 tests initially failed —
   `TestBlockHomogeneity::test_uniform_blocks_not_significant` and
   `::test_localized_payload_is_detected` — not a primitives bug but a
   missing dependency: `scipy` isn't installed (and isn't declared in
   `pyproject.toml`), and `_chi2_sf`'s scipy-free fallback only handled
   `df == 1`, returning `p=None` for the block-homogeneity test's
   `df=19`. Fixed by implementing a scipy-free regularized incomplete
   gamma function (`_regularized_gamma_p`/`_regularized_gamma_q`,
   series + continued-fraction) rather than adding scipy as a
   dependency, matching `statistical_check.py`'s stated Phase 2
   "no new dependency" principle; scipy is still used as a fast path if
   present. Validated against textbook chi-square critical values
   (df=1,5,19,50) to 4 decimal places. All 22 tests pass; full suite
   204/204.
2. ~~Check `analyze_model()` against the real `LoadedModel` interface.~~
   **Done.** `analyze_model()`/`calibrate_on_clean()` originally took a
   plain `dict[str, np.ndarray]`, but `LoadedModel.tensors` is
   `dict[str, TensorInfo]` (the ndarray lives at `TensorInfo.array`) —
   the same shape Stage 2/3 already handle by taking `LoadedModel`
   directly. Both functions now take `model: LoadedModel`, matching
   `run_structural_check`/`run_statistical_check`'s signature.
3. ~~Run `calibrate_on_clean()` against the real clean-variant benchmark
   (all 3 formats) and record the false-positive rate.~~ **Done** — see
   results below.
4. ~~Run the full `analyze_model()` against all 5 variants.~~ **Done** —
   see results below.
5. Wire the module into `gate.py`, following the same non-blocking
   (severity-label-only, no `hard_fail`) pattern Stage 3 used — **not
   yet done.** Before doing this, resolve the multiple-comparisons gap
   documented below (or at minimum decide how `PreCheckResult`/a future
   `ModelRiskScore` should represent a finding whose p-value hasn't been
   corrected for how many tests produced it).
6. Only then move to Stage 5 (behavioral probing).

## Calibration & full-benchmark results

Everything below was run against the real Phase 0 benchmark
(`generate_benchmark(seed=0)`, the same deterministic generation the test
suite uses), at the default `n_bits=4`, using `LoadedModel`s loaded
through Phase 0's actual loaders — not synthetic fixtures. Reported as
measured, in the same spirit as PHASE2.md: including the results that
complicate the story, not just the ones that support it.

### Calibration: false-positive rate on `clean`

| Format | Total findings | INFO | LOW | MEDIUM+ | Layers analyzed / skipped |
|---|---|---|---|---|---|
| safetensors | 214 | 164 | 50 | **0** | 30 / 4 |
| pt | 214 | 164 | 50 | **0** | 30 / 4 |
| onnx | 98 | 83 | 15 | **0** | 14 / 0 |

**MEDIUM+ false-positive rate is zero across all three formats** — the
placeholder `SEVERITY_THRESHOLDS` don't need loosening at MEDIUM+. (The
4 skipped layers on safetensors/pt are the `*.num_batches_tracked` int64
counters, correctly INFO-skipped. ONNX has none because BatchNorm gets
algebraically fused into the preceding conv at export — the same fact
PHASE2.md established for Stage 3.)

The 50 (safetensors/pt) / 15 (onnx) LOW findings are not evenly spread:
86-87% of them are `bit_autocorrelation`, and every one of those traces
to a tiny bias/BatchNorm-affine tensor (4–64 elements — `fc3.bias`,
`bn1.bias`, `conv1.bias`, etc.). This is the same root cause PHASE2.md
documented for Stage 3's own false positives: a small tensor's sample
statistic has much higher natural variance, so `|corr| > 0.1` is easy to
hit from pure sampling noise. It's compounded here because
`bit_autocorrelation` has no formal null distribution yet (see the known
limitations below) — its `LOW` cutoff is a hardcoded magnitude, not a
statistic corrected for sample size the way the other three tests are.
Not fixed here — flagged for whoever builds the permutation-test
p-value this stage's own docstring already calls out as missing.

### Full 5×3 matrix

| Variant | Format | INFO | LOW | MEDIUM | HIGH |
|---|---|---|---|---|---|
| clean | safetensors / pt / onnx | 164/164/83 | 50/50/15 | 0/0/0 | 0/0/0 |
| noisy | safetensors / pt / onnx | 163/163/81 | 49/49/15 | 0/0/0 | **2/2/2** |
| steganographic | safetensors / pt | 164/164 | 49/49 | **1/1** | 0/0 |
| steganographic | onnx | 83 | 15 | 0 | 0 |
| backdoored | safetensors / pt | 172/172 | 40/40 | **2/2** | 0/0 |
| backdoored | onnx | 84 | 13 | **1** | 0 |
| combined | safetensors / pt | 172/172 | 40/40 | **2/2** | 0/0 |
| combined | onnx | 84 | 13 | **1** | 0 |

Per-variant detail, each traced back to `manifest.json`'s ground truth
rather than taken at face value:

- **`noisy` — a real signal Stage 3 completely missed.**
  `fc1.weight` (one of the two ground-truth `NOISE_TARGET_LAYERS`) fires
  **HIGH** on `bit_balance_chi_square` (chi2=510.0, p≈6×10⁻¹¹³) and
  `wald_wolfowitz_runs` (z=-6.94, p≈4×10⁻¹²), identically across all
  three formats. PHASE2.md recorded Stage 3 as "zero findings, all
  formats" on this exact variant — this is the first real detection of
  it anywhere in the pipeline. `conv1.weight`, the *other* ground-truth
  noise target, shows nothing (all three tests land INFO, p>0.1) —
  plausibly a statistical-power/tensor-size effect (`conv1.weight` has
  72 elements / 288 bit positions at n_bits=4 vs. `fc1.weight`'s 8192 /
  32,768), not a detection-logic gap. Worth stating plainly: the
  tampering here is genuine Gaussian noise added directly to float32
  weights (`arr + noise`, `std=0.05`) — not a mantissa-bit trick — so a
  mantissa-LSB test picking it up this strongly is itself a new,
  previously-undocumented empirical result: floating-point addition
  rounding leaves a detectable statistical signature distinct from a
  directly-trained weight's mantissa distribution.
- **`steganographic` — the make-or-break case, and it's a partial win.**
  `conv3.weight` (the actual, sole embed target — 536 payload bits per
  the manifest) fires **MEDIUM** `bit_balance_chi_square` (chi2=7.88,
  p≈0.005) on **safetensors and pt, but not ONNX**. So: yes, this stage
  now produces a signal on the exact variant Stage 3 is designed to
  miss by construction — but it's MEDIUM, not HIGH, and it's
  format-inconsistent in the same shape as Stage 3's own backdoor
  finding (PHASE2.md's ONNX-only asymmetry, mirrored to the opposite
  two formats here). Why ONNX drops it hasn't been investigated — left
  for follow-up before this is called reliable.
- **`backdoored` / `combined` — an unplanned, more format-consistent
  signal than Stage 3 had, but it's one observation, not two.**
  `fc2.weight` fires **MEDIUM** `block_homogeneity_chi_square` (chi2=
  33.38, p≈0.004) identically across **all three formats**, in both
  variants. Important: `fc2.weight` is untouched by the steganographic
  payload (manifest confirms `affected_layers: [conv3.weight]` only for
  the stego tamper), and `backdoored`/`combined` share the exact same
  backdoor-trained `fc2.weight` tensor byte-for-byte — so this is really
  **one** underlying observation (backdoor training measurably disturbs
  `fc2.weight`'s mantissa-LSB block homogeneity) surfacing in two
  manifest rows, not independent replication. Still, it's more
  consistent across formats than Stage 3's own backdoor signal
  (PHASE2.md: ONNX-only, z≈3.6, barely past threshold). `conv3.weight`
  also fires MEDIUM on `backdoored` (chi2=7.04, p=0.008) and more
  strongly on `combined` (chi2=9.38, p=0.002, safetensors/pt only, not
  ONNX) — plausibly backdoor retraining shifting `conv3.weight`
  generally, compounded in `combined` by the stego overwrite on top.

## KNOWN LIMITATION: no multiple-comparisons correction

**This is not a minor caveat — it materially affects how every MEDIUM+
finding above (and any future finding from this stage) should be read,
and it must be accounted for before Stage 4 is wired into `gate.py` or
before any downstream Model Risk Score treats its p-values as
calibrated evidence.**

Each model report runs **84 hypothesis tests with a real p-value**
(`bit_balance_chi_square` + `block_homogeneity_chi_square` +
`wald_wolfowitz_runs`, one of each per analyzable float layer — roughly
28 layers on this benchmark's architecture; `bit_autocorrelation` has no
p-value so isn't counted). `SEVERITY_THRESHOLDS[MEDIUM]` is `p < 0.01`.
**No Bonferroni, Benjamini-Hochberg/FDR, or any other multiple-testing
correction is applied anywhere in `stego_check.py`** — every layer's
every test is compared to the raw, uncorrected threshold independently.

Applying elementary multiple-comparisons arithmetic: with 84 independent
tests at alpha=0.01, the expected number of spurious MEDIUM+ findings on
a **genuinely clean, untampered model** is `84 × 0.01 ≈ 0.84` per
report — i.e., seeing one MEDIUM+ finding purely by chance on any given
clean model is not a rare event under this design. The clean-variant
calibration above happened to show zero across all three formats, which
is *consistent with* (not proof against) that expected rate given only
three samples — it should not be read as evidence the thresholds are
conservative enough at scale.

This directly changes how some of the findings above should be
weighted:

- `noisy`'s `fc1.weight` result (p≈6×10⁻¹¹³) and the `fc2.weight`
  backdoor result (p≈0.004, though — see above — effectively one
  observation, not two) are many orders of magnitude past what 84
  uncorrected comparisons at alpha=0.01 would produce by chance. These
  are not plausibly multiple-comparisons artifacts.
- `backdoored`'s `conv3.weight` finding (p=0.008, barely under the 0.01
  cutoff) is exactly the kind of borderline result this gap makes
  ambiguous — it is fully consistent with either a real, weak backdoor
  signature on that layer, or one of the ~0.84-per-report spurious hits
  this design is expected to produce regardless of tampering. Nothing
  in the current implementation lets you tell those apart.

**Why this matters for whoever picks up Stage 5+ next:** if this
stage's findings are folded into a combined ablation or a Model Risk
Score (the eventual goal per the README/PHASE2.md), every MEDIUM+
finding needs to first survive a real correction — Benjamini-Hochberg
FDR across each report's own 84 tests is the natural fit (controls the
false-discovery rate within one model's report, which is the right unit
here, rather than a single global alpha across unrelated models). Until
that's implemented, this stage's `SEVERITY_THRESHOLDS` should be read as
**per-test, uncorrected p-value cutoffs** — a label conveying "this one
test looked unusual in isolation," not "this layer is tampered." No
threshold in this stage was adjusted to manufacture a better-looking
result in the table above, and none should be adjusted later purely to
make Stage 4 look more decisive without first fixing this — the same
standing rule PHASE2.md set for Stage 3.
