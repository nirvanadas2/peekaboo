# Phase 6 — Stage 3 report-only, Stage 7 (explainable report), final validation

## Status: the full pipeline (Stages 1-7) is complete and validated end to end on a final pre-registered suite

## 1. Protocol

**The final suite was registered first**, before any change in this
phase: seeds 20-29, all 5 variants, with 10 new trigger positions. See
`tests/fixtures/final_validation/README.md`. Only ground-truth validity
was printed: every backdoor has ASR ≥ 0.98, and every clean model's
trigger→target rate is ≤ 0.005.

The reason for a third suite: making Stage 3 report-only is a scoring
change decided **after** seeing the fresh-suite results (PHASE5.md). On
that spent suite it trivially removes all 4 fused clean false positives,
which is not evidence. Only a suite nothing was tuned on can test the
complete system honestly.

**The system was then frozen and hashed, and run once.** The run went
through `run_pre_checks`, both with and without a `forward_fn`, and
rendered a Markdown and a JSON report for every one of the 50 models.

## 2. Stage 3 is report-only

`fusion.PILLAR_WEIGHTS = {"statistical": 0.0, "steganographic": 1.0, "behavioral": 1.0}`.

**Stage 3 findings are still computed and shown:**
- in the pillar, capped at MEDIUM
- in `layer_flags`
- in the explanation text
- in the report, under "Report-only observations"

**They contribute nothing to:**
- `overall_score`
- `risk_level`
- `layer_risk`
- the exit code

**Why:** on all four suites (seed 0, seeds 1-7, 10-19, 20-29), Stage 3
detected no tampering, and on the fresh suite it flagged 4 of 10 clean
models.

## 3. Stage 7: the explainable report

This is `peekaboo/report.py`. It has no detection logic, only rendering.

| Output | What it contains |
|---|---|
| `render_markdown(result)` | Verdict, risk level and score, the fusion explanation, a "What was checked" table covering all 5 checks with status and score, the findings that drive the score (each with a plain-language explanation and its evidence: q, raw p, class reach, hit rate, position), report-only observations, per-layer risk, integrity/structure issues, **"Not assessed"** (with how to enable it), and a fixed **scope and limitations** section in every report |
| `render_json(result)` | The same content as a machine-readable document |
| Stage 1 hard-fail | Rendered as **UNSAFE / NOT ANALYZED**, stating that the file was not loaded |

**`exit_code`, for CI gating:**

| Code | Meaning |
|---|---|
| 0 | no scored evidence |
| 1 | MEDIUM |
| 2 | HIGH |
| 3 | unsafe file |

**CLI:** `python -m peekaboo scan MODEL [--md OUT] [--json OUT] [--tinycnn]`.
`--tinycnn` supplies the benchmark architecture's `forward_fn`. It is for
the synthetic benchmark **only**. Without it, backdoors are reported as
"not assessed".

**Design principle:** the report states what was *not* checked as
prominently as what was found. Every report carries the measured blind
spots, such as encrypted stego payloads, non-patch triggers, and
PyTorch-default-init only.

## 4. Final-suite result (single run; `tests/test_fusion.py::TestFinalSuite`)

| Variant (10 each) | Stage 3 (report-only) | Stage 4 | Stage 5 | **Fused, with `forward_fn`** | **Fused, static only** |
|---|---|---|---|---|---|
| clean | 1 | 0 | 0 | **0** | **0** |
| noisy | 0 | 7 | 0 | 7 | 7 |
| steganographic | 1 | 0 | 0 | 0 | 0 |
| backdoored / combined | 3 | 0 | 9 | **9** | 0 (*not assessed*) |

ROC AUC of the overall score against clean:

| Variant | With `forward_fn` | Static only |
|---|---|---|
| noisy | 0.85 | 0.85 |
| backdoored | **0.95** | 0.50 |
| steganographic | 0.50 | 0.50 |

- **Target class:** the asymmetry finding named the true target class in
  all 9 backdoor detections.
- **Misses:** the noise injection on seeds 22, 25 and 28, and the
  backdoor on seed 22.
- **Report rendering:** all 100 reports (50 models × 2 configurations)
  rendered in both formats without error, and every exit code matched
  its risk level.

## 5. Cumulative held-out record

Each suite was pre-registered and run once:

| Detector | Suites | Detected | False positives |
|---|---|---|---|
| Stage 5, backdoors (with `forward_fn`) | seeds 2-7, 10-19, 20-29 | **21 / 26** | **0 / 78** backdoor-free models |
| Stage 4, noise injection | seeds 10-19, 20-29 | **16 / 20** | **0 / 80** noise-free models (clean, steganographic, backdoored, combined) |
| Stage 4, 67-byte stego payload | seeds 10-19, 20-29 | 0 / 20 | — |
| Fused score, clean models | seeds 20-29 (complete system) | — | **0 / 10** |

**How to read these numbers:**

- **The sample sizes are small.** At 95% confidence, 0/78 bounds the
  false-positive rate below about 4%, 0/10 below about 26%, and 21/26 is
  compatible with recall from roughly 61% to 93%.
- **One architecture and one synthetic task.** Everything here is TinyCNN
  on a single task. These are evidence that the mechanisms work, not
  real-world rates.
- **Weak noise evidence.** Noise detection after the Stage 4 lattice fix
  is often only MEDIUM. For example, the seed-0 noisy `fc1.weight` went
  from q=3.5×10⁻¹¹³ before the fix to q=2.4×10⁻⁴ after it, because the
  test no longer reads the true lowest bits of small values.
- **Combining stages buys coverage, not synergy.** Stage 4 covers noise
  and Stage 5 covers backdoors. No combined-signal gain beyond their
  union was found or is claimed.

## 6. What the project can and cannot claim

**Can claim** (measured, held-out):

- On this benchmark, **patch-trigger backdoors are detected behaviorally**
  when the model can be run, with no false positives observed, and the
  target class is identified.
- **Float-level noise injection is detected statically** from low-bit
  statistics.
- **Unsafe pickles are hard-gated** before loading.
- **Every scan explains itself** and says what it did not check.

**Cannot claim:**

- **Stego detection:** encrypted payloads are undetectable at any density,
  and small structured ones are not detected either (PHASE3.md §5).
- **Backdoor detection without a runnable model:** no static stage
  detects backdoors.
- **Non-patch trigger families.**
- **Architectures, tasks, or initializers beyond** TinyCNN, the quadrant
  task, and PyTorch's default init.
- **Any benefit from Stage 3.**
