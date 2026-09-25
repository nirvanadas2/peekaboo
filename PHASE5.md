# Phase 5 — Stage 6: Anomaly Fusion → Model Risk Score

## Status

Implemented and wired into `run_pre_checks`, together with Stage 4 (on
by default) and Stage 5 (opt-in via `forward_fn`). The first end-to-end
run exposed large Stage 3 and Stage 4 false-positive rates on
independent clean models. **Both were then fixed and validated on a
pre-registered fresh suite of 50 models** (see "Fixing Stages 3 and 4"
at the end):

| Stage | Result on the fresh suite |
|---|---|
| Stage 4 | **fixed**: 0/10 clean false positives |
| Stage 5 | 0/10 clean false positives |
| Stage 3 | **still 4/10 clean false positives**, now from entropy |
| Fused | every false positive comes from Stage 3, capped at MEDIUM/0.4 |

Whether Stage 3 should keep any weight in the score is flagged for the
team.

The sections below "Design decisions" record the *first* run as it was
measured, before the fixes.

## Design decisions (agreed at the step-7 check-in)

### 1. Evidence fusion, and no Isolation Forest

The original brief asked for an Isolation Forest ensemble. It was
deliberately **not** built, for four reasons:

- **Too few layers to fit on.** Within one model the population is its
  layers: 7 weight matrices in TinyCNN, about 30 tensors in total. That is
  too few for isolation depth to separate anything. Stage 3 already does
  within-model outlier detection on those same statistics.
- **No trusted reference set.** Fitting across models would need a
  trusted reference set of clean models of the same architecture, which
  is exactly the baseline Peekaboo is designed not to need (PHASE1/2).
  On this benchmark it would also be circular.
- **It would discard calibrated evidence.** Stages 4 and 5 emit
  BH-corrected q-values. An Isolation Forest would replace them with an
  uncalibrated depth.
- **It would need scikit-learn or a from-scratch implementation.**

### 2. Scoring

**Pillar scores:**
- **Stages 4 and 5:** a pillar's score comes from its strongest MEDIUM+
  finding:
  `evidence_score(q) = 0.4 + 0.6·clip(log10(0.05/q) / log10(0.05/1e-10), 0, 1)`.
  That gives 0.4 at the q=0.05 MEDIUM cutoff, rising to 1.0 at q ≤ 1e-10.
- **Stage 3:** its z-scores aren't p-values, so it contributes through
  its severity labels only, **capped**. Any MEDIUM+ Stage 3 finding
  scores exactly 0.4, and its severity is capped at MEDIUM. Stage 3 alone
  can never raise a model above MEDIUM.

**Overall score = the max over pillars that ran.** On this benchmark the
pillars cover different threats (Stage 4 catches noise, Stage 5 catches
backdoors), so their union is the right combination. There are **no
fitted parameters**, so evaluating on the committed fixtures is not
circular.

### 3. "No findings" versus "not run"

This required an approved Phase 0 schema change. `PillarScore` gained a
`status` field (`not_run` / `ran_clean` / `flagged`) and an `Optional`
score. A `not_run` pillar has score `None`, is excluded from the overall
max, and is named in the explanation ("absence of evidence, not evidence
of absence"). Stage 5's `mode="not_runnable"` report maps to `not_run`,
never to "clean".

### 4. Other rules

- **Layer-level versus model-level flags.** Stage 3 and 4 flags attach
  to layers and feed `metadata["layer_risk"]`. Stage 5 flags are
  model-level, recorded as `layer_name="(model behavior)"`: they describe
  input positions and classes, not layers.
- **Explanation.** It is exact, not SHAP: because the overall score is a
  max, each pillar's attribution is simply its own score.
- **Uncorrected Stage 4 reports are rejected.**

### 5. Gate wiring (the "option (b)" decision)

`run_pre_checks` now runs Stages 1-4, then Stage 5, then fusion:

| Stage | When it runs | Cost on the benchmark |
|---|---|---|
| Stage 4 | always | about 10 ms |
| Stage 5 | probes only when the caller passes `forward_fn`, `input_shape` and `num_classes`; otherwise it returns the explicit `not_runnable` report | about 0.2 s through torch when probing; zero forward passes otherwise |
| Fusion | always, into `PreCheckResult.risk_score` | — |

Stage 1's short-circuit is unchanged: on a hard-fail, nothing runs and
there is no risk score.

## End-to-end results

These are measured through `run_pre_checks` on the committed fixtures,
and locked into `tests/test_fusion.py`.

### Seed-0 benchmark (with `forward_fn`)

The results are identical across safetensors, pt, and ONNX, except that
ONNX has no `bn4` tensor.

| Variant | Flagged pillars | Overall |
|---|---|---|
| clean | steganographic (`bn4.running_var` false positive), except ONNX: none | 0.45 (ONNX 0.00) |
| noisy | steganographic (`fc1.weight`) | 1.00 |
| steganographic | steganographic — **the same `bn4` false positive, not the payload** | 0.45 (ONNX 0.00) |
| backdoored / combined | behavioral | 0.98 |

### Independent models (Stage 5 validation seeds 1-7, safetensors, with `forward_fn`)

| Seed | clean | backdoored |
|---|---|---|
| 1 | **stego false positive** (`fc2.weight`, HIGH) | stat, behavioral |
| 2 | — | stat, behavioral |
| 3 | **stat false positive** (`conv1` mean) | stat only (Stage 5 miss) |
| 4 | **stat false positive** (`conv1` mean, z=21.6) | stat, behavioral |
| 5 | **stat false positive** (`conv1` kurtosis) | behavioral |
| 6 | **stego false positive** (`fc2.weight`, HIGH) | — (Stage 5 miss) |
| 7 | — | stat, **stego false positive** (`fc2.weight`), behavioral |

None of these 14 models carries noise or a payload, so **every Stage 4
flag here is a false positive**. Stage 3 never detected a backdoor on
seed 0, so its flags on these clean models are false positives too.

### Ablation: which pillars flag each variant (MEDIUM+, i.e. score ≥ 0.4)

Counts are pooled across the seed-0 files (all formats) and the
validation seeds.

| Variant (n) | Statistical | Steganographic | Behavioral | **Fused** |
|---|---|---|---|---|
| clean (10) | 3 | 4 | 0 | **7 (false positives)** |
| noisy (3) | 0 | 3 | 0 | 3 |
| steganographic (3) | 0 | 2 (both the `bn4` false positive) | 0 | 2 |
| backdoored (10) | 5 | 1 | 8 | 9 |
| combined (3) | 0 | 0 | 3 | 3 |

ROC AUC of the overall score, using all files:

| Comparison | With `forward_fn` | Without `forward_fn` |
|---|---|---|
| noisy vs clean | 1.000 | 1.000 |
| **backdoored vs clean** | **0.830** | **0.355** |

Without `forward_fn`, the backdoor AUC falls *below chance*. The
behavioral pillar is `not_run`, and the remaining Stage 3 flags on
backdoored models are no more informative than its flags on clean
models. The fused backdoor "detections" without Stage 5 are Stage 3
false positives, **not detection**.

**What the ablation does and does not show:**

- **Combining pillars adds coverage, not synergy.** Stage 4 alone covers
  noise, Stage 5 alone covers backdoors, and the union covers both. No
  case was found where combining weak signals beat each alone, and none
  should be claimed.
- **Max-fusion passes every pillar's false positives straight through.**
  So the fused clean false-positive rate is at least the worst single
  pillar's: 7 of 10 clean files here. The fusion is doing what it was
  designed to do. The pillars' calibration is the problem.
- **Stage 5 alone is 0/10 on clean and 8/10 on backdoored** in this
  pooled view. The pooled view overlaps with Stage 5's own development
  seeds (0 and 1), so the held-out number to quote remains 4/6
  (PHASE4.md).

## New findings about Stages 3 and 4

These models are independent of those stages' designs, so they give an
honest first estimate of their false-positive rates.

### Stage 4: `fc2.weight` breaks the uniform-bits null

On seeds 1 and 6 (clean) and seed 7 (backdoored), `fc2.weight` fails
bit-balance and/or block homogeneity at q between 3×10⁻⁵ and 5×10⁻⁷. This
is not noise. Some trained `fc2.weight` tensors genuinely have
non-uniform, non-homogeneous low mantissa bits, so the test's null
hypothesis (independent uniform bits) is **wrong** for them. BH controls
false discoveries only when the null is right. This is very likely also
what PHASE3.md originally reported as a "backdoor signal" on
`fc2.weight`.

- **Clean false-positive rate on independent models:** 2 of 7 at HIGH.
- On seed 0, Stage 4 happened to show only the `bn4` false positive.

### Stage 3: small-tensor sampling noise, again

Most flags are `conv1.weight` **mean** outliers, with z up to 21.6.
`conv1.weight` has 72 elements, so its sample mean has far higher
variance than the other layers'. This is the root cause PHASE2.md fixed
for std (log scale) and entropy (normalization), but not for the mean.

- **Clean false-positive rate on independent models:** 3 of 7.
- Seed 0 passed by chance.

### Calibration on one model was not enough

Both stages were "calibrated" on a single clean model (seed 0). This
phase is the first time either saw a second one.

## Fixing Stages 3 and 4

### Protocol

This follows the same protocol Stage 5 used (PHASE4.md):

1. **A fresh suite was specified and generated first.** It has seeds
   10-19, each with all 5 variants, and 10 new trigger positions. See
   `tests/fixtures/fresh_validation/README.md`. Only ground-truth
   validity was printed: backdoor ASR ≥ 0.968, and clean
   trigger→target rate 0.000.
2. **The fixes were diagnosed and designed on development data only:**
   seed 0 and seeds 1-7, all already spent.
3. **The fixes were frozen.** They were implemented and verified against
   the prototypes, and the files were hashed.
4. **The fresh suite was run once,** through `run_pre_checks` with a
   `forward_fn`.

### Stage 4: the default-init lattice

**Diagnosis.** On the models where `fc2.weight` fired, 5-11% of its
values are **bit-identical to their random initialization**. They got
zero gradient throughout training.

- **Removing just those values** restores the null: the smallest p-value
  goes from 5×10⁻⁷, 4×10⁻⁸ and 6×10⁻⁹ to 0.6, 0.6 and 0.1.
- **Mechanism.** PyTorch's default Conv/Linear init draws uniform(−b, b)
  with b = 1/√fan_in, from **24 random bits**. That puts every init
  value on a lattice of spacing 2b·2⁻²⁴. Below b, the lattice is coarser
  than float32's own spacing, so an init value's lowest bits are pinned
  rather than random. The uniform-bits null reads that as a payload.
- **The lattice can be reproduced exactly.** It is 100% bit-exact on
  every TinyCNN init tensor.
- **It can't be used to pick out individual init values.** About 19% of
  ordinary floats also land on the lattice by chance, so *excluding*
  lattice values would bias the test the other way.

**Fix (option C).** For each conv/linear tensor, b is computed from the
paired weight's fan-in. Each value is tested on its `n_bits` starting
**just above its own lattice floor**, rather than at bit 0. Those bits
are uniform for trained values *and* for never-updated init values, so
the null holds either way. The fix is `init_lattice_aware=True` in
`analyze_model` (the default). `False` reproduces the old behavior.

**Two alternatives were tried on development data:**

| Variant | Dev false positives (16 models) | Noisy recall | Text payload at 25 / 50 / 100% |
|---|---|---|---|
| bit-0 (old) | 4 | ✓ | ✓ ✓ ✓ |
| A: only values with \|v\| ≥ b | 5 | ✓ | ✗ ✗ ✓ |
| **C: bits above the lattice floor** | **1** (`bn4.running_var`, kept on purpose) | ✓ | ✗ ✓ ✓ |

**Cost.** Option C no longer looks at the true lowest bits of small
values, so it lost the 25%-density text payload.

**Scope limit.** The fix is specific to **PyTorch's default init**. A
model that used a different initializer (TensorFlow/JAX defaults, or
custom schemes) gets no protection, and would need its own lattice.

### Stage 3: noise-aware robust z for mean and kurtosis

**Diagnosis.** The flags came from `conv1.weight` (n=72). Its sample mean
and kurtosis carry far more sampling noise than those of the n=8192
layers.

**First attempt, on development data, which was wrong.** Dividing each
value by its own standard error flagged **21 of 29** development models.
That approach assumes every layer's true mean and kurtosis are
identical, so it inflated legitimate systematic differences in the large
layers.

**Fix.** z_i = (x_i − median) / √(σ²_between + SE_i²), where σ_between =
MAD/0.6745 and SE_i is std/√n for the mean or √(24/n) for kurtosis. It
reduces exactly to the old modified z when every SE is 0. A noisy layer
gets a wider tolerance, and a large layer is judged as before. On
development data it flagged 1 of 29 models, and **0 of 8 clean ones**.

### Fresh-suite result (single run; `tests/test_fusion.py::TestFreshSuite`)

| Variant (10 each) | Stage 3 | Stage 4 | Stage 5 | **Fused** |
|---|---|---|---|---|
| clean | **4** (false positives) | **0** | **0** | 4 (all Stage 3, score 0.40) |
| noisy | 2 | **9** | 0 | 10 |
| steganographic | 4 (same as clean) | 0 | 0 | 4 |
| backdoored / combined | 5 | 0 | **8** | 9 |

ROC AUC of the overall score against clean:

| Variant | AUC |
|---|---|
| noisy | 0.98 |
| backdoored / combined | 0.91 |
| steganographic | 0.50 |

What this shows, stage by stage:

- **Stage 4:** the fix generalized. Clean false positives went from 2/7
  before the fix to **0/10** on unseen models, and noise recall held at
  9/10 (seed 11 missed). The steganographic payload is still undetected,
  as PHASE3.md §5 predicts for a 4%-density payload.
- **Stage 5:** this was its second held-out run. Combined with PHASE4.md,
  it has detected **12 of 16** held-out backdoors with **0 of 28** clean
  false positives, and the asymmetry finding named the true target class
  every time.
- **Stage 3: not fixed.** The mean/kurtosis fix removed the `conv1` mean
  false positives, but 4 of 10 fresh clean models are still flagged,
  mostly by **entropy** outliers on the small layers (`conv1`, `fc3`).
  That is the same small-sample root cause in a statistic the
  development data didn't expose. There were also one `fc1` mean and one
  `conv4` kurtosis flag. The fresh suite is now spent too, so no further
  iteration was done. Iterating here would repeat the development-set
  trap.
- **Fused:** every fused clean false positive comes from Stage 3, at
  exactly the capped floor of 0.40 (risk MEDIUM). Every Stage 4 and
  Stage 5 finding on the fresh suite scores 0.42 or more.

## Still open (for the team)

1. **Stage 3's weight in the score.** It detects nothing on any suite and
   still has a 4/10 clean false-positive rate. Options are to keep the
   MEDIUM cap, make it report-only (zero weight), or fix entropy (a
   noise-aware or size-aware null, like mean and kurtosis) and validate
   on a *new* suite. This decision is being made *after* seeing the
   fresh-suite results, so it's yours, not mine.
2. **Stage 4 covers PyTorch-default init only.** Other initializers are
   unprotected.
3. **Stage 4 misses random-looking stego payloads at any density**
   (PHASE3.md §5).
4. **Stage 7** (the explainable report) has not started.
