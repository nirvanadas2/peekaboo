# Phase 2: Statistical Analysis (Stage 3)

Stage 3 is the first stage that looks at actual weight *content* rather
than file structure. It lives in `peekaboo/pipeline/statistical_check.py`
and reuses Phase 1's `Finding`/`Severity`/`compute_passed` machinery
(`peekaboo/schema/reports.py`) rather than reinventing them. Output type
is `StatisticalReport` (same file), with the same no-`hard_fail` principle
as `StructuralReport` — see below.

This document went through two rounds of empirical correction after the
initial build (see "Second benchmark redesign" below) — both discovered by
actually running the numbers against Phase 0's benchmark rather than
trusting the design on paper, and both recorded here rather than silently
folded in, since they explain why the code looks the way it does.

## What it computes

For every floating-point tensor in an already-loaded model: **mean, std,
skewness, excess kurtosis** (computed directly via raw moments, not
`scipy` — no new dependency), and **histogram-based Shannon entropy**
(adaptive bin count, `clip(sqrt(n), 4, 64)`, so a tiny tensor doesn't get
an artificially inflated entropy from having as many bins as data points).

## Detection approach: within-model relative outliers

The core method is a **robust modified z-score** (median/MAD-based,
Iglewicz & Hoaglin's standard `0.6745` constant and `3.5`/`5.0`
MEDIUM/HIGH thresholds) computed across the population of layers *within
the model being scanned* — never an absolute threshold, never a paired
clean-vs-tampered comparison. In deployment there's no trusted baseline
for an unknown model pulled from a public hub; the only thing to compare
a layer against is the rest of that same model. Median/MAD is used
instead of mean/stdev specifically because a single tampered layer can
otherwise skew the very baseline it's being measured against — median/MAD
has a much higher breakdown point.

### The population: weight matrices only, and why

The cross-layer comparison population is restricted to **real learned
weight matrices** (conv/linear/embedding kernels, `ndim >= 2`) —
explicitly excluding biases, normalization-layer affine parameters
(BatchNorm/LayerNorm weight+bias), and running statistics. Per-layer
stats are still computed and reported for *every* floating tensor
(`metadata["per_layer_stats"]`); only the cross-layer comparison itself is
restricted.

This was **not** the first design tried, and the two rejected designs are
worth recording because they failed for structural, not incidental,
reasons:

1. **All floating tensors together** (weights + biases + BN params).
   Failed immediately on Phase 0's clean baseline: BatchNorm's scale
   parameter is initialized/trained to center near 1.0, its shift near
   0.0, and `running_var` is strictly positive — none of that reflects
   tampering, it's just what that *kind* of parameter looks like. Mixing
   roles produced multiple HIGH-severity "outliers" on an untampered
   model purely from comparing different parameter roles against each
   other.
2. **Weight matrices + their true paired biases** (`X.weight`+`X.bias`,
   identified structurally). This fixed the BatchNorm problem but
   introduced a subtler one: a bias vector is typically tiny (a handful to
   a few dozen elements) next to a weight matrix (hundreds to thousands).
   A sample mean's standard error shrinks with `1/sqrt(n)`, so a small
   bias's sample mean has much higher natural sampling variance than a
   weight matrix's — comparing raw mean/std across tensors of very
   different sizes conflates "this layer is unusual" with "this tensor is
   just small." It also diluted sensitivity: a real, substantial std
   shift in one weight matrix (from noise injection) stopped registering
   as an outlier once mixed into a population whose range was already
   wide because of small-bias sampling noise.

Restricting to weight matrices only fixes both problems, at the cost of a
smaller population — which directly motivated the architecture change
below.

## FALLBACK for small models

Below `_MIN_LAYERS_FOR_RELATIVE = 6` weight matrices, "relative to the
rest of this model" doesn't have enough data to mean anything, so
detection falls back to conservative absolute heuristic ranges (loose
bounds on mean/std/kurtosis/entropy for typically-initialized/trained
float32 weights). This is marked explicitly in the finding's
`details["mode"] == "absolute_fallback"` and the message is prefixed
`[fallback: too few layers for relative comparison]`. Fallback severity is
**capped at MEDIUM** — never HIGH or CRITICAL — by construction, so a
fallback finding can never be presented with the confidence of a real
relative-outlier detection.

The fallback decision is made **per-statistic**, on that statistic's own
reliable population (e.g. kurtosis excludes tensors with `n < 4` or
zero variance), not just on the model's total layer count — a stat can
independently fall back even when the model overall has enough layers.

## `StatisticalReport` has no `hard_fail`

Same principle as `StructuralReport` (Stage 2): the models Peekaboo most
needs to catch are, by construction, perfectly loadable and — for
backdoored/noisy models specifically — this is exactly the stage meant to
notice something's off in their weight distributions. But severity here
is still a triage label, never a control-flow signal; this stage never
gates anything downstream.

## Second benchmark redesign: deepening TinyCNN, and what that surfaced

The original TinyCNN (2 conv + 1 fc = 3 weight matrices) never gave Stage
3 enough population to run in relative mode at all — every benchmark
variant, in every format, fell back to the absolute-heuristic path
regardless of tampering. Since the submission's core evidence claims
(precision/recall/F1/ROC AUC, per-layer localization, the "combined beats
any single signal" ablation) all depend on the benchmark actually
demonstrating detection rather than perpetually falling back, TinyCNN was
deepened (see `peekaboo/benchmark/models.py`) to **4 conv blocks + 3 fc
layers = 7 weight-bearing layers**, keeping the same synthetic 4-class
quadrant task. `NOISE_TARGET_LAYERS`/`STEGO_TARGET_LAYERS` in
`peekaboo/benchmark/tamper.py` were updated to the new layer names
(`conv1.weight`/`fc1.weight` for noise, `conv3.weight`/`fc2.weight` for
stego). All of Phase 0's original ground-truth invariants were
re-verified on the new architecture: backdoor attack success rate 1.0,
clean accuracy 0.67-0.75 (both far above the 1/4 chance floor), stego
payload round-trips exactly (fully embedded in `conv3.weight`, SHA-256
match), noise injection isolated exactly to `conv1.weight`/`fc1.weight`
with nothing else touched.

**A real reproducibility bug was also fixed while touching this code**:
benchmark values weren't fully reproducible across fresh runs of
`generate_benchmark(seed=0)` despite the explicit seed argument, because
`TinyCNN()` was constructed *before* `train_clean`/`train_backdoored`
called `torch.manual_seed(seed)` (the seeding happened inside `_train`, in
`train.py`) — so initial weights depended on ambient global RNG state, not
the seed. Confirmed by observing different `conv1.bias` values across two
fresh Python processes with identical `seed=0` before the fix. Fixed by
seeding immediately before each `TinyCNN()` construction in
`generate.py`. Regression test:
`tests/test_benchmark.py::TestCleanVariantReproducibility` spawns two
genuinely separate Python processes and confirms byte-identical
`clean.safetensors` output. A useful side effect: the entire benchmark
(and therefore this whole test suite) is now fully deterministic — a full
`pytest` run was repeated multiple times with identical results.

### This surfaced a new false positive, fixed by two structural corrections (not threshold tuning)

Population=7 does unlock relative mode — but the first thing it did was
fail the **clean** baseline. The cause, found by inspecting per-layer
numbers directly:

```
conv1.weight  n=72    std=0.2052  entropy=2.93   fan-in = 1×3×3 = 9   (smallest)
conv2.weight  n=1152  std=0.0923  entropy=4.61   fan-in = 8×3×3 = 72
conv3.weight  n=3456  std=0.0761  entropy=5.12   fan-in = 16×3×3 = 144
conv4.weight  n=6912  std=0.0637  entropy=5.11   fan-in = 24×3×3 = 216
fc1.weight    n=8192  std=0.0846  entropy=5.46   fan-in = 128
fc2.weight    n=2048  std=0.1015  entropy=5.06   fan-in = 64
fc3.weight    n=128   std=0.1500  entropy=3.44   fan-in = 32  (small n too)
```

Two distinct, legitimate (non-tampering) sources of scale variation, both
concentrated on `conv1.weight`:

1. **std scales multiplicatively with fan-in** under Kaiming/Xavier init
   (`std ~ 1/sqrt(fan_in)`). `conv1` has the smallest fan-in by far, so it
   legitimately has the largest std. A modified z-score on *raw* std
   conflates "architecturally different scale" with "tampered." **Fix:**
   compare `log(std)` instead — turns the multiplicative relationship into
   an additive one, which the same z-score math is suited to
   (`_LOG_SCALE_STATS` in `statistical_check.py`).
2. **Entropy's adaptive bin count scales with tensor size**
   (`_adaptive_bin_count`), so the *maximum possible* entropy
   (`log2(bin_count)`) differs by tensor size too — `conv1` (72 elements,
   8 bins, ceiling 3 bits) can never reach the ~5.5-bit entropy a
   1000+-element tensor with 64 bins can, even if equally well-spread
   relative to its own ceiling. **Fix:** compare `normalized_entropy`
   (`entropy / log2(bin_count)`, a bounded ~[0,1] ratio) instead of raw
   bits (`_LayerStats.normalized_entropy`).

Both fixes target a confirmed, understood root cause in the comparison
method — not a threshold moved until a test went green. Raw values are
still what's reported in each finding (`details["outliers"][name]["value"]`);
only the z-score comparison itself uses the transform. After both fixes,
**the clean baseline passes with zero findings on all three formats**.

### The honest result on noise and backdoor detection, after the fix

False positives are solved. Detection is not strong:

| Variant | Result |
|---|---|
| clean | Zero findings, all formats. |
| steganographic | Zero findings, all formats — correct and expected. |
| noisy | **Zero findings, all formats.** Traced directly: `fc1.weight`'s std genuinely shifts (0.0846→0.0979 from the injected noise), but the population's own median/MAD shift with it, so its z-score lands *near 0*, not further out — the real shift is smaller than this 7-layer architecture's own natural fan-in-driven spread. This is the measured result, not a design goal — the "noise = Stage 3's strongest case" expectation did not hold on this benchmark. |
| backdoored / combined | Zero findings on safetensors/pt. A single weak MEDIUM `kurtosis_outliers` flag on `conv4.weight` (`onnx::Conv_66` post-BN-fusion) — ONNX only. Backdoor training has no single "target layer" (the whole model is retrained on poisoned data), so this is at least plausibly a genuine partial signal rather than an artifact, but it's weak (z≈3.6, barely past the 3.5 MEDIUM cutoff) and inconsistent across formats. |

This was deliberately **not** pushed further (e.g. by lowering the
z-score threshold to catch noise, which would also have caught `conv1`'s
z≈2.8-3.0 on the clean baseline again) — the explicit instruction was to
report what's observed rather than tune until tests are green. Stage 3's
recall on this benchmark's specific tampering magnitudes is genuinely
weak; that's recorded here rather than hidden. Revisiting detection
sensitivity (e.g. a larger tampering magnitude, or a different
statistic/transform) is left for a later, separate pass.

## Implications for Phases 3-5

Stated plainly, because it shapes how the rest of the submission's
evidence has to be built:

1. **Stage 3 alone does not reliably detect any of the four threat
   variants in the current benchmark.** Clean and steganographic
   correctly show zero findings (that's the intended, designed behavior
   for both). Noisy and backdoored/combined are, in practice,
   undetected: noisy shows zero signal on any format, and
   backdoored/combined show exactly one borderline signal (ONNX-only
   `kurtosis_outliers` on `conv4.weight`, z≈3.6 against a 3.5 cutoff) —
   not a result that would survive being called a reliable detection.
2. **The project's "combined multi-signal detector outperforms any
   single signal alone" claim currently depends entirely on Stage 4
   (bit-level steganalysis) and Stage 5 (behavioral probing) actually
   working.** Stage 3 cannot be assumed to contribute meaningfully to
   detection in its current form — any ablation or precision/recall
   result that credits Stage 3 with catching noise or backdoor tampering
   on this benchmark would not be supported by what's actually been
   measured here.
3. **This is carried forward as a known, accepted risk, not a defect
   being silently patched over.** No threshold or design choice in Stage
   3 was adjusted to manufacture a better-looking result, and none should
   be adjusted later purely to make an ablation table look more balanced.
   If Stage 3's contribution needs to improve, that's a deliberate,
   separately-scoped piece of future work (see the "later, separate pass"
   note above), not something to quietly retrofit once Phases 3-5 are
   built and the combined numbers are being assembled.

## Testing

`tests/test_statistical_check.py` (43 tests) plus one new reproducibility
regression test in `tests/test_benchmark.py`:

- **Synthetic robust-outlier logic** (`TestRobustOutlierLogicOnSyntheticArrays`):
  a single extreme outlier among many normal values is caught with a huge
  margin (z≈660) without dragging the normal cluster's own z-scores out of
  range; realistic depth-wise-decreasing std and varied layer
  types/sizes don't produce a false HIGH (MEDIUM is tolerated — it's a
  "worth a second look" label, not a tampering claim); an all-identical
  population doesn't divide by zero.
- **Relative mode on synthetic multi-layer models** (`TestRelativeModeOnSyntheticModel`):
  an 8-layer synthetic model actually reaches `relative_outlier` mode; one
  substantially-wider layer among eight normal ones is flagged; norm-affine-like
  1-D tensors are excluded from the population even when there are plenty
  of them.
- **Explicit small-model fallback fixtures** (`TestSmallModelFallback`):
  a 2-weight-matrix model uses fallback mode; passes when within
  conservative ranges; a degenerate (near-zero-entropy) layer is flagged
  MEDIUM (never HIGH); fewer-than-two-layer and zero-weight-tensor edge
  cases report cleanly.
- **`StatisticalReport` never hard-fails** (`TestNeverHardFails`).
- **Full 5×3 benchmark matrix** (`TestFullBenchmarkMatrix`), with
  expectations exactly as measured in the table above — including the
  weak/absent detection results, not adjusted to look better.
- **`TestCleanVariantReproducibility`** (`tests/test_benchmark.py`):
  regression test for the seeding-order bug, using two genuinely separate
  Python processes.

182 tests pass across the whole suite (was 183 before this round — the
old benchmark-matrix tests, written for the 3-layer architecture and the
weaker false-positive-only claim, were replaced with tests reflecting the
7-layer reality one-for-one, net one fewer test after consolidating the
backdoored/combined per-format checks).
