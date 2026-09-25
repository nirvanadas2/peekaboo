# Phase 4 — Stage 5: Behavioral Probing

## Status: second method implemented and validated on a pre-registered held-out suite: 4/6 backdoors detected, 0/18 false positives. NOT wired into `gate.py`.

Sections 1-7 below are the original design doc, kept as written. The
record after that runs in chronological order:

1. **The first implementation did not discriminate.** Clean got 39 of 40
   candidates at MEDIUM, the same as every tampered variant.
2. **The benchmark trigger was confounded with its task.** This is
   fixed.
3. **Intensity-matched controls gave zero recall.**
4. **A local-inconsistency ("island") design passed the development set
   but failed on held-out data.**
5. **The current method (island plus class-reach asymmetry) was
   validated on a pre-registered held-out suite.** See the last section,
   "Pre-registered held-out suite and the current method". Read that
   section before trusting any Stage 5 output.

---

*Original design doc follows.*

This document proposes a design for Stage 5 (behavioral / trigger
probing). **No code had been written at the time.** Per Stage 4's own honest
accounting (PHASE3.md), the pipeline currently has: a weak, ONNX-only
partial signal on `backdoored`/`combined` from Stage 3, and Stage 4
(mantissa-bit steganalysis) is structurally incapable of catching
weight-*value* tampering like a backdoor — it looks at bit-plane
statistics, not what the model actually does when run. Stage 5 exists to
close that gap the other way: **run the model and see if it behaves
abnormally**, rather than inspecting its weights statically.

This turned out to be a much bigger departure from Stages 1-4 than the
purpose statement alone suggests, for one structural reason that shapes
everything else in this doc: **Stages 1-4 only ever need a `LoadedModel`
— named tensors, nothing else. Stage 5 needs a runnable model — something
you can call with an input and get an output from.** Those are not the
same requirement, and closing that gap is most of this design.

---

## 1. Runnability: what it actually takes to call the model

### The blocker, found while grounding this design in the actual code: even our own `LoadedModel` can't be run

All three of Phase 0's loaders (`safetensors_loader.py`,
`pytorch_pickle_loader.py`, `onnx_loader.py`) reduce their input down to
`LoadedModel.tensors: dict[str, TensorInfo]` — named tensors and their
shape/dtype, nothing else. This was the right call for Stages 1-4, which
only ever need to look *at* tensors. But it means even `load_onnx()` —
despite ONNX being the one format that actually embeds a full
computation graph — **currently discards that graph**: it reads
`model.graph.initializer` (the weights) into `tensors` and only keeps
summary metadata (`num_nodes`, `ir_version`, etc.) from the rest;
`model.graph.node` (the actual list of ops and how they're wired
together) is never retained. So "just run the `LoadedModel` we already
have" isn't available for *any* format today, ONNX included — this has
to be built, not merely wired up.

### Per format, what's actually recoverable

- **Safetensors** — structurally has no notion of a computation graph,
  by design; that lack of arbitrary structure is exactly why the format
  is considered safe. There is no way to recover a forward pass from a
  safetensors file alone, full stop, regardless of tooling. The only way
  to run one is if the caller supplies the architecture from outside
  Peekaboo.
- **PyTorch pickle (`.pt`/`.pth`/`.bin`/`.ckpt`)** — Stage 1's loader
  (`load_pytorch_pickle`) deliberately uses a restricted weights-only
  unpickler that refuses to reconstruct arbitrary Python objects/classes
  — that refusal *is* Stage 1's hard safety gate (PHASE1.md's
  `UnsafeCheckpointError`). Even if the original file happened to
  pickle a full `nn.Module`, generically reconstructing and running it
  would mean either (a) executing arbitrary pickled code — reopening
  exactly the hole Stage 1 exists to close — or (b) a human independently
  writing a correct reimplementation of an architecture they'd have to
  already recognize. Neither is something Peekaboo can do generically
  for an arbitrary, untrusted hub upload. Same conclusion as safetensors.
- **ONNX** — the one format that *does* carry both weights and a real
  computation graph, designed specifically to be executable without
  code-exec risk (that's ONNX's whole appeal for hub distribution). Two
  things are still missing to actually use that: the graph needs to be
  retained somewhere (see above), and something needs to execute it — the
  `onnx` package (already a dependency, used today for `checker.check_model`
  in Stage 1) only parses and validates the graph; it does not run it.
  Actually executing ops requires a runtime — `onnxruntime` is the
  standard one, and it is **not currently a dependency of this project**.

### Proposed approach: two-tier runnability, mirroring Stage 2's spec/self-consistency split

**Tier A — "native" runnability, ONNX only.** Extend ONNX loading to
retain the graph (either add a field to `LoadedModel`/`load_onnx`, or
have Stage 5 re-open the `.onnx` file directly via the `onnx` package
rather than going through `LoadedModel` at all — see open question below),
and execute it via `onnxruntime`. **This is a new-dependency decision
that needs your sign-off, not mine** — every prior phase deliberately
avoided adding a dependency (Stage 3's raw-moment statistics instead of
scipy; Stage 4's from-scratch incomplete-gamma function instead of
scipy) specifically because that was established as this project's
convention. A hand-written ONNX-op interpreter narrow enough to cover
just what `TinyCNN`'s own export uses (Conv, BatchNorm-fusion, ReLU,
MaxPool, Flatten, Gemm/Linear, Softmax) is technically possible and
would keep the no-new-dependency streak going, but it would silently be
wrong or unsupported the moment it met any real-world ONNX graph using a
different op — which is most of them. My recommendation is
`onnxruntime`, but flagging this explicitly rather than deciding it,
since it breaks an established pattern.

**Tier B — caller-supplied runnability (safetensors/pt, and the fallback
for ONNX if `onnxruntime` isn't adopted).** `run_behavioral_check` takes
an **optional** `forward_fn: Callable[[np.ndarray], np.ndarray] | None`
— a batch-in/logits-out function the *caller* builds however they trust
(their own `nn.Module` instantiation with a known architecture, their
own `onnxruntime` session, anything). If no `forward_fn` is supplied and
the model isn't ONNX-with-a-runtime, Stage 5 **does not run** — it
returns exactly one `mode="not_runnable"` `INFO` finding stating why,
never a silent skip and never a fabricated/estimated result. This
mirrors Stage 1's "flagged, not silently ignored" principle and Stage
2's optional-`spec` design (self-consistency when absent, exact-diff
when present) — except here there is no self-consistency fallback,
because *running* a model requires actual computational semantics
(activation functions, exact wiring, skip connections) that cannot be
inferred from tensor shapes the way Stage 2's dimension-chain check
infers structural plausibility. That's a hard ceiling, not a gap to
close later.

**For internal validation only — never the shipped general path.**
`peekaboo.benchmark.models.TinyCNN` is a real, known `nn.Module` this
project controls. A test/calibration harness can legitimately construct
one and `load_state_dict()` a benchmark file's tensors into it to get a
genuinely runnable model — this is how Stage 5 gets validated end-to-end
against `backdoored`/`combined` (see §3). This must never be confused
with, or silently substituted for, how Stage 5 would behave against a
real, unknown public-hub upload with no known class — for those, Tier B
applies and a `forward_fn` is required from the caller.

**Net effect:** Stage 5's real-world applicability is honestly narrower
than Stages 1-4's. Stages 1-4 work on *any* loadable model, by design —
that's the whole premise of a static, weights-only scanner (PHASE1.md:
"Peekaboo's actual job is scanning unknown models from public hubs with
no access to the original training pipeline"). Stage 5 cannot keep that
promise unmodified: for safetensors and pickle checkpoints — very likely
the majority of real hub uploads — it can only run if the caller already
somehow has a trusted way to instantiate the architecture, which is
exactly the kind of out-of-band knowledge Peekaboo was designed not to
require. This should be stated plainly in whatever docs/README updates
eventually cover Stage 5, not left implicit.

---

## 2. What "trigger probing" means, concretely

Two methods were on the table — pure fuzzing vs. replaying our own
benchmark's known trigger. The right answer is **both, but kept strictly
separate**, because they serve different, non-interchangeable purposes
(this separation is what makes §3's circularity problem tractable at
all):

### The general-purpose method (what actually ships)

**Paired carrier probing against a parameterized patch family**, not
literal noise-only fuzzing and not the literal known trigger:

1. **Carriers.** Draw N base inputs matching the model's inferred input
   shape (recoverable the same way Stage 2's dimension-chain check
   already infers shapes — from the first weight tensor's expected input
   dimension). Use both pure random noise *and*, where feasible, a small
   family of structured carriers (solid colors, gradients, checkerboards)
   — noise-only carriers risk conflating "abnormal" with the well-known
   generic phenomenon that classifiers often collapse toward one class on
   pure out-of-distribution noise, backdoor or not (more on this in §4).
2. **Candidate patches.** A parameterized *family* of localized
   patches — varying position (grid over the input), size (a few
   relative sizes), and color/intensity (spanning the value range,
   including unusually extreme values, since real trigger literature
   tends to use out-of-training-distribution intensities) — deliberately
   broader than, and not centered on, our own benchmark's specific 3×3
   top-left/value-6.0 patch.
3. **Paired measurement.** For each carrier, run it clean and run it
   again with each candidate patch stamped on. The signature of a real
   trigger isn't just "one class gets more hits than others" — it's that
   the patch's predicted class becomes **invariant to the carrier**,
   overriding whatever the carrier's own (unpatched) content would have
   produced. So the statistic that matters is the *shift*: across many
   different carriers, does stamping this one candidate patch collapse
   the prediction onto a single class far more than the carriers'
   unpatched predictions, or a size/position-matched **control** patch
   filled with random (non-trigger-like) pixel content, would predict?
   The control-patch comparison (same footprint, no structured color) is
   what isolates "trigger-like forced classification" from the mundane
   fact that any sufficiently large occlusion perturbs some predictions
   on any classifier, backdoored or not.
4. **A cheaper, secondary signal.** Plain random-noise-only fuzzing
   (no patch) — does one class dominate across totally unrelated random
   inputs, with no patch involved at all — is worth computing too, as a
   fast, low-cost, low-confidence `INFO`/`LOW`-level signal. It's
   explicitly weaker and more prone to the OOD-collapse false-positive
   risk in §4, so it should never on its own justify MEDIUM+.

### The validation-only method (never the shipped detector)

Using the benchmark's own known ground truth (`add_trigger`/
`TRIGGER_PATCH_SIZE=3`/`TRIGGER_VALUE=6.0`/top-left/`TRIGGER_TARGET_CLASS=0`
in `peekaboo/benchmark/data.py`), directly measure: does the general
method in §2.1-2.4 — run *blind*, without being told where or what the
real trigger is — actually flag a candidate patch at or near that
specific position/size/intensity? This is diagnostic tooling for proving
the mechanism works at all (the same role `calibrate_on_clean()` plays
for Stage 4), reported the same honest way PHASE2.md/PHASE3.md reported
their results — including if it *doesn't* find it cleanly. It is
**never** the thing that ships as "the detector," because doing that
would be circular by construction (see §3).

---

## 3. The circularity problem

Our `backdoored`/`combined` benchmark variants use exactly one
hand-picked trigger: a 3×3, value-6.0, top-left-corner patch, forcing
class 0 (`peekaboo/benchmark/data.py`). If Stage 5's actual detection
method searched specifically for *that* patch, "detecting" it would
prove nothing beyond "we can find the exact thing we ourselves planted"
— a worse, more literal version of the threshold-tuning-to-pass-tests
trap PHASE2.md explicitly refused to fall into for Stage 3. An apparent
100% recall in that setup is fully uninformative about generalization to
a real attacker's trigger, which could be:

- a different **patch** — different size, position, color, shape (not
  necessarily square/solid);
- a **blended/watermark** trigger (a faint pattern added across the
  whole image rather than localized);
- a **frequency-domain** trigger (a pattern only visible/effective in
  frequency space, invisible in the raw pixel patch our method probes
  for);
- a **warping-based** trigger (a spatial transform rather than an
  additive pattern);
- a **semantic/natural** trigger (e.g. "a stop sign" or "a red car" for
  a real-world classifier — not a synthetic artifact at all, and not
  something any pixel-patch search would ever find).

This is a real, citable line of prior work in the backdoor-detection
literature (BadNets-style patch triggers, blended backdoors,
frequency-domain triggers, WaNet-style warping triggers), and it's worth
being explicit that Peekaboo's synthetic benchmark only ever instantiates
the first of these — a narrow, high-contrast localized patch. **This is
documented here as an explicit, accepted limitation, the same way
PHASE2.md documented Stage 3's weak recall and PHASE3.md documented
Stage 4's multiple-comparisons gap**: nothing about Stage 5's design
should be read as claiming general backdoor-trigger detection, only
detection of the localized-patch trigger family this benchmark
represents, validated (§2, validation-only method) against one concrete
instance of that family.

### What a more general version would need

1. **Diverse trigger types in the benchmark itself.** Extending
   `peekaboo/benchmark/tamper.py`/`data.py` with blended, frequency, or
   warping-based triggers, so Stage 5 (or a future revision of it) can
   be validated against more than one trigger family — out of scope for
   this design doc, flagged as the concrete next step if trigger-type
   generalization becomes a priority.
2. **External validation.** Peekaboo's own tiny synthetic benchmark can,
   by construction, only ever prove detection of the trigger types it
   itself was built to contain. Real generalization evidence would need
   testing against published backdoor-attack implementations/datasets
   from the academic literature, not just this repo's own generator.
3. **A trigger-shape-agnostic detection principle**, which the
   parameterized patch-family search in §2 already is a first step
   toward (it doesn't require knowing the trigger's exact form up
   front, only that it's patch-shaped) — but the principled, state-of-
   the-art general approach is closer to **Neural Cleanse**-style
   reverse-engineering: for each output class, optimize for the
   *minimal* input perturbation that reliably forces that class, then
   check whether any one class's minimal-perturbation norm is a
   statistical outlier relative to the others. That's a substantially
   bigger undertaking (an optimization loop per class per scan, not a
   fixed probe grid) and is flagged here as a stretch goal for a later,
   separate pass — not promised as part of this one, the same way
   PHASE2.md flagged revisiting Stage 3's detection sensitivity as
   future, separately-scoped work rather than something to fold in now.

---

## 4. What counts as "abnormal" — the statistical baseline

Same root problem Stage 3 had to solve: **there is no trusted external
reference model for an arbitrary unknown model**, so "abnormal" has to
mean "abnormal relative to this model's own behavior," never an absolute
threshold and never a clean-vs-tampered pairing.

Stage 3's specific solution — a robust median/MAD z-score across a
population of comparable layers within the same model — **does not
transfer directly here**, for a concrete, checkable reason: the natural
"population" analogue would be the model's own output classes, and this
benchmark has only 4 of them. That's below even Stage 3's
`_MIN_LAYERS_FOR_RELATIVE = 6` floor — and that floor was arrived at
empirically, after the original 3-layer `TinyCNN` proved too small to
ever leave absolute-fallback mode (PHASE2.md's "second benchmark
redesign" story). A 4-class population would hit the exact same failure
mode Stage 3 already had to fix once, and there's no equivalent "deepen
the architecture" escape hatch for number-of-classes the way there was
for number-of-layers — the benchmark's task is fixed at 4 quadrants.

**Proposed alternative: a bootstrap/permutation baseline, built from
repeated *control* probes rather than a within-model population of
comparable units.** This reuses an idea Stage 4's own docstring already
flagged as the rigorous way to give `bit_autocorrelation` a real
p-value (a permutation test) but never implemented — Stage 5 adopts it
as its primary mechanism from the start rather than deferring it:

1. Run many size/position-matched **control** patches (§2.3's
   random-pixel-content control, not the structured candidate patches)
   across the carrier population, and compute the `max_class_hit_rate`
   statistic (the fraction of carriers whose patched prediction converges
   on the single most common resulting class) for each control trial.
   This builds an empirical null distribution for "how much class
   concentration does an occlusion of this size/position normally cause
   on *this specific model*, with no structured trigger-like content" —
   computed fresh per model, never borrowed from elsewhere.
2. Compare each **candidate** (structured) patch's `max_class_hit_rate`
   against that empirical null distribution — a percentile or z-score
   position within it — rather than against a fixed absolute cutoff.
   Severity scales with how far out in that empirical distribution the
   candidate lands.

This keeps the "relative to this model, not an external baseline"
principle Stage 3 established, while sidestepping the small-population
problem that would come from naively reusing Stage 3's exact mechanism.

### A mistake this design must not repeat: Stage 5 is a *worse* multiple-comparisons problem than Stage 4, by construction

PHASE3.md documented, prominently, that Stage 4 runs 84 uncorrected
hypothesis tests per model report with no Bonferroni/FDR correction
anywhere, and flagged that as a real gap for whoever builds on it next.
**Stage 5's candidate-patch grid (multiple positions × multiple sizes ×
multiple colors, each compared against the bootstrap null) is easily a
larger number of comparisons than Stage 4's 84** — and unlike Stage 4,
which at least tests each layer once, Stage 5 would be testing many
patch configurations *by design*, specifically to search for one that
works. A correction (Benjamini-Hochberg FDR across all candidate-patch
comparisons in one report, the same fix PHASE3.md recommended for Stage
4) needs to be built into Stage 5 **from the first implementation**, not
retrofitted after the fact the way Stage 4's gap is currently sitting
undocumented-until-now. Any implementation of this design that skips
that correction should be considered incomplete, not a later nice-to-have.

---

## 5. Cost, and where this fits in the pipeline

Stages 1-4 are all effectively O(single pass over the model's tensors)
— cheap, and safe to run synchronously inside `run_pre_checks` the way
Stage 3 now is. **Stage 5 is qualitatively different**: it requires
actually *executing* the model many times — carriers × (candidate
patches + control patches), easily hundreds to thousands of forward
passes per scan, not a single pass over static data. This has real
consequences worth deciding explicitly rather than discovering later:

- It likely should **not** be wired into `run_pre_checks` unconditionally
  the way Stage 3 was — a caller who only wants Stage 1-4's cheap static
  checks shouldn't be forced to pay for hundreds of forward passes (and,
  per §1, may not even be able to supply a `forward_fn` at all). Proposal:
  a separate entry point (e.g. `run_behavioral_check` called explicitly,
  not folded into `PreCheckResult`), with `PreCheckResult` gaining an
  optional slot for it only when the caller opts in.
- A configurable **probe budget** (number of carriers, patch-grid
  resolution) is needed so cost is a caller-tunable knob, not a fixed
  constant baked into the module — unlike Stages 1-4, where "run every
  check" is always affordable.
- Same non-blocking, severity-label-only, no-`hard_fail` pattern as
  Stages 2-4 — this doc isn't proposing any change to that principle,
  just flagging that *when* and *whether* Stage 5 runs at all is a new
  kind of decision the earlier stages never had to make.

---

## 6. Proposed report shape (schema-level only — no code)

Stages 1-3 share `peekaboo.schema.reports.Finding`/`Severity`/
`compute_passed`. Stage 4, as delivered, does **not** — it defines its
own local `Severity` enum and `LayerStegoFinding`/`StegoReport`
dataclasses instead of reusing the shared ones, which is a pre-existing
inconsistency (introduced in the outside-the-repo draft, not addressed
here) that whoever wires Stage 4 into `gate.py` will need to resolve
too. **Stage 5 should reuse the shared `Finding`/`Severity`/
`compute_passed` machinery**, the way Stage 3's own docstring states as
the deliberate convention ("reuses Phase 1's Finding/Severity/
compute_passed machinery... rather than reinventing them").

Sketch, at the field level only:

- `BehavioralReport`: `model_path`, `mode` (one of `"not_runnable"` |
  `"probed"`), `runnable_source` (`"caller_supplied"` | `"onnx_runtime"`
  | `None`), `passed`, `findings: list[Finding]`, `metadata` (probe
  budget used, inferred input shape, number of carriers/candidates/
  controls, per-candidate-patch `max_class_hit_rate` and its bootstrap
  percentile).
- Findings per notable candidate patch (position/size/color, its
  hit-rate, its bootstrap percentile, FDR-corrected significance),
  plus one always-present `behavioral_coverage` `INFO` finding
  recording what was actually probed (mirrors Stage 3's
  `statistical_coverage` finding) — and, critically, the
  `mode="not_runnable"` case, which must produce exactly one finding
  explaining why, never zero findings (a silently-empty report would be
  indistinguishable from "probed and found nothing").

---

## 7. Open questions before any code gets written

1. **`onnxruntime` as a new dependency** (§1, Tier A) — adopt it, or
   restrict Stage 5 to caller-supplied `forward_fn` only (Tier B) for
   every format including ONNX, at least initially?
2. **Where the ONNX graph lives** — extend `LoadedModel`/`load_onnx` to
   retain `model.graph.node` (a Phase 0 public-interface change,
   same category PHASE1.md flagged needing approval for the `.bin`/
   `.ckpt` allowlist gap), or have Stage 5 re-open the `.onnx` file
   itself, independently of `LoadedModel`, for the ONNX case only?
3. **Pipeline placement** (§5) — a separate opt-in entry point, or
   folded into `PreCheckResult` behind an optional flag? And does
   `run_pre_checks`'s caller get to pass a `forward_fn` through, the
   same way it currently passes an optional `spec`?
4. **Probe budget defaults** — how many carriers / how fine a
   position-size-color grid is an acceptable default runtime cost, given
   this is the first stage where "run every check" isn't automatically
   affordable?
5. **Multiple-comparisons correction method** (§4) — Benjamini-Hochberg
   FDR is proposed as the default; confirm before it's load-bearing in
   the first implementation, since (per §4) this one isn't optional the
   way it regrettably was left for Stage 4.

---

## Calibration & full-benchmark results

Everything below was run against the real Phase 0 benchmark
(`generate_benchmark(seed=0)`, fully deterministic). The run used the
**default** probe budget (`n_carriers=64`, `n_bootstrap=64`, patch sizes
0.25/0.5 → 4×4 and 8×8, colors ±3.0 → 40 candidates, 128 control trials).
Results are reported as measured, in the same spirit as PHASE2.md and
PHASE3.md. No threshold, alpha, or budget default was changed.

**How the model was run.** `peekaboo/benchmark/runnable.py` builds the
caller-side `forward_fn`. It is for validation only. For safetensors/pt it
loads the tensors into the known `TinyCNN` class. For `.onnx` it runs the
file's own graph through `onnx.reference.ReferenceEvaluator`. That
evaluator ships inside the `onnx` package, which is already a dependency,
so this needed **no `onnxruntime`**. Its logits match the torch path to
within 1e-4 (`tests/test_behavioral_benchmark.py::TestOnnxReferencePath`).
It is slow: about 55s per default-budget probe on TinyCNN, compared with
under 1s through torch. The calibration entry point is
`behavioral_probe.calibrate_on_clean(model, forward_fn, **probe_kwargs)`,
which mirrors Stage 4's.

### Finding 1: the benchmark's backdoor is confounded with its task

The benchmark task is "which quadrant has the highest mean"
(`data.make_dataset`), and quadrant 0 is the top-left one. The trigger is
a 3×3 patch of value 6.0 **in the top-left corner**, and its target class
is **0**. Stamping the trigger raises the top-left quadrant's mean by
about 54/64 ≈ 0.84. The per-quadrant mean of the N(0,1) images has a
standard deviation of only 1/8, so the trigger makes top-left the
brightest quadrant, and class 0 becomes the *correct* answer under the
task's own labeling rule.

Measured on the same 200 triggered test images the manifest's ASR is
computed on:

| Model | Triggered → class 0 | Same, restricted to true label ≠ 0 |
|---|---|---|
| **clean** (never saw a trigger) | **0.985** | **0.980** |
| backdoored | 1.000 | 1.000 |

The manifest's `attack_success_rate: 1.0` is therefore almost entirely
**task behavior, not backdoor behavior**. An untampered model already
"obeys" this trigger 98.5% of the time. The backdoor adds 1.5 percentage
points on its own trigger. **The expectation that backdoored/combined
would be Stage 5's strongest case cannot hold on this benchmark as
built.** No behavioral detector can separate a backdoor from clean
behavior when the clean model responds to the trigger in the same way.
This is a Phase 0 benchmark-design problem, not a Stage 5 problem, and it
has not been fixed here. Changing it means regenerating the
backdoored/combined weights, which in turn invalidates PHASE2.md's and
PHASE3.md's measured numbers for those variants. That decision is flagged
for the team.
(`tests/test_behavioral_benchmark.py::TestBenchmarkTriggerConfound`.)

### Finding 2: at default settings Stage 5 flags nearly every candidate, on every variant

| Variant | Format | Trigger-patch HIGH | MEDIUM | LOW | Candidates whose majority is class 0 |
|---|---|---|---|---|---|
| clean | safetensors / pt / onnx | 0/0/0 | **39/39/39** | 0 | 13/39 |
| noisy | safetensors / pt / onnx | 0/0/0 | 40/40/40 | 0 | 14/40 |
| steganographic | safetensors / pt / onnx | 0/0/0 | 39/39/39 | 0 | 13/39 |
| backdoored | safetensors / pt / onnx | 0/0/0 | 39/39/39 | 0 | 20/39 |
| combined | safetensors / pt / onnx | 0/0/0 | 39/39/39 | 0 | 20/39 |

All three formats give identical results, including ONNX run natively
from its graph with BN fused. The `behavioral_baseline_concentration`
secondary signal is INFO everywhere: 0.38 → class 1 for the clean-trained
variants and 0.36 → class 0 for the backdoor-trained ones.

**The clean false-positive rate at MEDIUM+ is 39/40 = 97.5% of
candidates.** The tampered variants cannot be told apart from clean by
severity.

**Root cause, traced rather than assumed:** the candidates and the
controls differ in *intensity*, not only in whether they look like a
trigger:

- A candidate is a *constant* ±3.0 patch.
- A control is a patch of the same footprint filled with *N(0,1)*
  content, which has mean about 0, the same distribution as the carriers.

On a brightness-driven task, a constant +3 patch in any quadrant
legitimately makes that quadrant the brightest, and a −3 patch makes it
the darkest. In both cases predictions converge on one class, and the
clean model does this 90-100% of the time: every +3 candidate on clean
reaches 87.5-100% hit rate. The null distribution never sees that kind of
intensity shift, so almost every candidate lands at the empirical p-value
floor.

The control design separates "structured high-intensity content" from
"any occlusion", but on this benchmark the intensity itself carries the
class. The design treats that as a trigger signal. It is actually the
task's legitimate decision feature. This limitation may matter beyond
this benchmark: saturated patches are out-of-distribution for most real
classifiers too, so the same mismatch could generate false positives on
real-world models.

### Finding 3: BH-FDR is correct on real data, but at the default budget it is saturated and can't do its job

- **It computes correctly.** On every variant each q ≥ its raw p, and the
  number of discoveries at q<0.05 and q<0.01 matches an independent BH
  step-up implementation
  (`TestFullBenchmarkMatrix::test_fdr_q_values_match_an_independent_bh`).
- **At the default budget it discriminates nothing.** With
  `n_bootstrap=64`, the smallest possible empirical p is 1/65 ≈ 0.0154.
  Nearly every candidate ties at that floor. When most p-values are tied
  at the minimum, BH's step-up barely moves them (0.0154 → 0.0162), so
  essentially every raw-significant candidate survives correction.
  - **HIGH (q<0.01) is mathematically unreachable at the default
    budget**: the minimum possible q is at least 1/(n_bootstrap+1) > 0.01
    whenever n_bootstrap < 99.
  - MEDIUM needs at least 13 of 40 candidates tied at the floor. That's
    common here only because of Finding 2.
- **More budget makes the false positives worse, not better.** A
  sensitivity run at `n_bootstrap=999` (safetensors, all 5 variants)
  moves **clean to 39 HIGH**, and every other variant looks the same
  (noisy: 39 HIGH + 1 MEDIUM). The floor effect was hiding the
  intensity-mismatch false positives; it wasn't causing them. So this is
  not a probe-budget problem. The fix, if there is one, belongs in the
  control/statistic design (Finding 2).

### Checked and working

- **`mode="not_runnable"` is visible, never silent.** On all 15 benchmark
  files, calling with `forward_fn=None` returns `mode="not_runnable"`
  with exactly one `behavioral_runnable` INFO finding
  (`TestNotRunnableOnRealFiles`). As §1 says, a safetensors/pickle file
  from an unknown source always takes this path.

### A possible, unvalidated signal (NOT acted on)

The one visible difference between the backdoor-trained variants and the
clean-trained ones is class asymmetry across the grid. Candidates whose
majority is class 0: backdoored/combined 20/39, clean 13/39. Most of that
increase comes from −3 (dark) patches, which push the backdoored model to
class 0 in positions where the clean model spreads across classes. That
fits "the backdoor gave class 0 a lower activation threshold", and it is
roughly the class-level-outlier idea §3 attributes to Neural Cleanse.
**It was noticed by looking at the backdoored model's own outputs.**
Building a detector around it now, and validating that detector on the
same benchmark, would be the circular trap that §3 and PHASE2.md warn
against. If the team pursues it, it needs a fresh benchmark: a different
seed, and, per Finding 1, a trigger whose target class is not the
trigger quadrant's natural class.

### Implications

1. **Stage 5 as implemented does not contribute a usable signal on this
   benchmark.** Any fusion or ablation result crediting Stage 5 with
   detecting backdoored/combined would not be supported by these
   measurements. This is the same standing rule PHASE2.md set for
   Stage 3.
2. **Two independent problems, and fixing one alone is not enough:**
   - **(a) Benchmark.** The trigger/target confound (Finding 1) must be
     fixed before *any* behavioral detector can be validated on this
     benchmark.
   - **(b) Stage 5 design.** The candidate/control intensity mismatch
     (Finding 2) must be fixed before Stage 5's false-positive rate means
     anything.

   Both are decisions for the team. Neither was changed here.
3. The status of Stages 3-5 as fusion inputs, as measured:

   | Stage | Status |
   |---|---|
   | Stage 3 | Weak (PHASE2.md). |
   | Stage 4 | Real signals on noisy/stego/backdoor, but still no FDR correction (PHASE3.md). |
   | Stage 5 | Non-discriminating. |

   The precondition for starting Stage 6 ("3, 4, and 5 all producing
   validated, calibrated signals") is **not met**.

---

## Update: trigger retargeted; intensity-matched controls prototyped

### Finding 1 is resolved in the benchmark

`TRIGGER_TARGET_CLASS` is now **3** (bottom-right), diagonally opposite
the top-left trigger. The trigger itself is unchanged: 3×3, value 6.0,
top-left. Measured on the committed fixture:

| Model | Triggered → class 3 | Same, true label ≠ 3 |
|---|---|---|
| clean | **0.000** | 0.000 |
| backdoored | **1.000** | 1.000 |

The backdoor is now purely backdoor behavior, and the backdoored model's
clean accuracy is 0.65 (chance is 0.25). The manifest now records the
clean model's trigger→target rate alongside the ASR
(`clean_model_trigger_to_target_rate`,
`attack_success_rate_non_target_labels`), so this confound stays
measurable in future. A regression test guards it:
`tests/test_behavioral_benchmark.py::TestBenchmarkTriggerNotConfounded`.

The retarget has consequences for the other stages. On the new backdoor,
**Stage 3 and Stage 4 both show nothing** (PHASE2.md, PHASE3.md). Stage 5
is now the only stage that could possibly detect it.

### Stage 5 as implemented, on the retargeted benchmark

There is still no discrimination at the default settings. Backdoored and
combined get 38 of 40 candidates at MEDIUM, against 39 of 40 for clean.
The ±3.0 color grid also **cannot reach this trigger**. The closest
candidate, a 4×4 +3 patch at the top-left, sends 59% of carriers to
class 0 on the backdoored model, not to class 3.

### A trigger-independent color rule

Colors should be set at **±3σ and ±6σ of the carrier noise** (σ=1 by
construction here, since carriers are N(0,1)). The 6σ level is chosen
because it is extreme and out of distribution, which §2 already calls
for. It is not chosen because the benchmark trigger happens to be 6.0. It
is still a choice made *after* seeing this benchmark's trigger value, so
any result that depends on it has to be validated on a fresh seed.

### Prototype result: intensity-matched controls remove the false positives, and remove all detection with them

The prototype is scratch code, **not implemented in the repo**. It used
the ±3σ/±6σ colors, 80 candidates, and 200 null samples per
footprint/color. Candidates reaching q<0.05:

| Control design | clean | backdoored |
|---|---|---|
| Current: N(0,1)-content controls | 79 / 80 | 75 / 80 |
| Mean-matched: N(c,1) content | **0** | **0** |
| Intensity-matched: same color c at a random position | **0** | **0** |

The blind grid **does now reach the trigger**. The 4×4 +6 patch at the
top-left sends **100%** of carriers to **class 3** on the backdoored
model, and 100% to **class 0** on the clean model. The backdoor is
clearly visible in *which class* a patch forces. It is invisible to
Stage 5's statistic, which measures *how concentrated* the predictions
are (`max_class_hit_rate`). On a brightness task, every bright or dark
patch concentrates predictions at about 100%, whatever its position, so
the candidate and every intensity-matched control score the same.

The chosen fix, intensity-matched controls, therefore turns Stage 5 from
"flags everything" into "flags nothing". It has **zero recall** on a
backdoor that is behaviorally blatant (100% vs 0%). This fix was not
built into the repo, because it would look better on paper (no false
positives) while detecting nothing. The real limitation is the
concentration statistic. Replacing it means a statistic about *which
class* each patch forces, which is closer to the class-asymmetry option
that was not chosen. That is a design decision for the team.

A caution for whoever picks this up: every design idea from here on was
formed *after* looking at this backdoored model's outputs. Whatever is
chosen must be validated on a benchmark the design has not seen, such as
a fresh seed and ideally a different trigger position. The same-benchmark
result alone doesn't count, per §3 and PHASE2.md.

---

## Experiment: local-inconsistency ("island") statistic, with held-out validation

This is scratch prototype code, **not implemented in the repo**. The
experiment protocol:
- The **development set** is the committed seed-0 fixture, which was
  already seen.
- The design was **frozen** before any held-out data was generated.
- The **held-out set** uses seed 1 and a different trigger position, and
  was run exactly once.

### The idea

A legitimate decision feature should be spatially smooth, while a small
trigger should form an *island*. The test for a candidate patch p is:
does p force a class that **none** of its neighbors force?
- **Per neighbor:** an exact one-sided McNemar test on p's majority class,
  paired over the same carriers.
- **Across neighbors:** the candidate's p-value is the max over all
  neighbors (an intersection-union test), so it only counts if p differs
  from all of them.
- **Across candidates:** BH, with q<0.01 for HIGH and q<0.05 for MEDIUM.

No bootstrap controls are needed.

### Development-set iterations (seed 0), all on the same data

1. **4×4 and 8×8 patches, neighbors one patch-width away.** Clean got 53
   HIGH. An 8×8 patch is the size of the model's own feature (a quadrant),
   so every quadrant is a legitimate "island".
2. **4×4 only.** Clean still got 24 HIGH, from cells whose patch-width
   neighbors straddle quadrant boundaries. Backdoored's **#1 finding was
   the true trigger**, (0,0) +6 → class 3.
3. **4×4 only, with overlapping (half-stride) cells included as
   neighbors.** Clean got 2 HIGH, both dark −6 patches. Backdoored and
   combined got **exactly 1 finding: the true trigger**, q=3.6×10⁻¹⁰.

**The frozen design was variant 3:**

| Parameter | Frozen value |
|---|---|
| Patch size | 0.25 of the input extent |
| Stride | size/2 |
| Neighbors | all grid cells within one patch-width (Chebyshev distance) |
| Colors | ±3σ and ±6σ |
| Carriers | 64 |
| Per-neighbor test | exact McNemar |
| Combining neighbors | IUT (max over neighbors) |
| Correction and severity | BH; q<0.01 HIGH, q<0.05 MEDIUM |

### Held-out result (seed 1; trigger in the bottom-left corner, target class 1)

The trigger is parameterized via `TriggerSpec`. The held-out backdoor is
real: ASR is 0.995, and the clean model's trigger→target rate is 0.0.

| Variant | MEDIUM+ | Trigger found? |
|---|---|---|
| clean / noisy / steganographic | **0** | n/a (no false positives) |
| backdoored / combined | 1 MEDIUM | **No.** The flag is a dark patch at (6,12) → class 0, far from the trigger. |

**Held-out recall is 0.** The design's one finding on the backdoored model
is a false positive.

### Why it missed

This diagnostic was run after the held-out result and did not change the
design. It looked at the forced class of +6 4×4 patches around the
trigger cell:

- **Dev (seed 0):** the backdoor is **point-specific**. Only cell (0,0)
  forces class 3. Every neighbor forces the task-correct class 0 at
  61–100%. That is an island, so it was detected.
- **Held-out (seed 1):** the backdoor **generalized to a whole region**.
  *Every* bright patch in the bottom-left forces class 1 at 80–100%,
  where the clean model says class 2. There is no island to detect, even
  though the model's behavior is badly hijacked.

**The same attack with a different seed produced a backdoor of a different
shape.** Local inconsistency sees only point-like backdoors. A region-wide
one looks like legitimate spatial semantics to it. Only knowledge of the
task, or a statistic about which *classes* get forced across the whole
grid, would reveal it. Here class 1 is forced by bright patches in two
quadrants and class 2 by none, which is the class-asymmetry statistic.
The two statistics look **complementary**, but that is a hypothesis, not a
measurement.

### Status and what this rules out

- **Not implemented in `behavioral_probe.py`.** The one design validated
  on held-out data fails to detect its backdoor, and shipping it would
  claim recall that was not measured.
- **The seed-1 held-out set is now contaminated**, because its outputs
  were inspected during the diagnosis. Any next design, whether
  asymmetry, island plus asymmetry, or something else, needs a **new**
  unseen benchmark (a new seed and trigger position). It should be
  generated only after the design is frozen, and ideally by someone who
  doesn't look at its outputs first.
- **What was learned generally:** there is large variance *across
  backdoors*, not just across models. A single held-out backdoor per
  design gives little statistical evidence. A credible recall claim needs
  several held-out backdoors (seeds × trigger positions), which is cheap
  here because the generator now accepts a `TriggerSpec`.

---

## Pre-registered held-out suite and the current method

### Protocol

1. **The held-out suite was written down first**, before any new design
   work:

   | Seed | Trigger (3×3, value 6.0) | Target class |
   |---|---|---|
   | 2 | top-right corner | 2 |
   | 3 | bottom-right corner | 0 |
   | 4 | rows 3-5, cols 3-5 (interior) | 3 |
   | 5 | rows 2-4, cols 10-12 (interior) | 2 |
   | 6 | bottom-left corner | 3 (adjacent quadrant, not opposite) |
   | 7 | rows 10-12, cols 4-6 (interior) | 1 |

   The suite was generated, and only ground-truth validity was printed:
   every backdoor has ASR ≥ 0.993 on inputs whose true label isn't the
   target, and every clean model's trigger→target rate is 0.000. No
   detector was run on it.
2. **The design was developed on seed 0 (point backdoor) and seed 1
   (region backdoor) only.** Seed 1 had already been inspected, so it
   counts as development data.
3. **The design was frozen.** A snapshot of the scratch code was hashed
   (sha256 `6f2208f5…`).
4. **It was run once on seeds 2-7.**
5. **It was implemented in `behavioral_probe.py`,** and verified to
   reproduce the frozen code exactly on all 32 models (every p-value,
   q-value, and severity).

### What the development data showed (seeds 0 and 1)

Per color, this counts how many 4×4 grid positions force each class:

- **Bright patches on clean models reach every class about equally**:
  [10,12,13,13] and [10,13,11,15] on the half-stride grid, and
  [4,4,4,4] on the non-overlapping grid.
- **The region backdoor is extremely lopsided under +6σ**: [6, **33**,
  **0**, 10] on the half-stride grid, and [2,11,0,3] non-overlapping.
- **The point backdoor is barely visible this way**: [12,10,11,15].
  That's the case the island test catches.
- **Dark patches are skewed even on clean models**: [7,16,3,22] and
  [0,18,2,4]. All 3 island false positives seen in development were
  dark patches.

That led to two changes on the development set, each recorded here:

- **v2:** island plus asymmetry, bright colors only, one BH across all
  tests. Result: 0 false positives, but the seed-1 asymmetry (raw
  p≈6×10⁻⁴) was diluted by pooling it with 98 island tests.
- **v3 (frozen):** the same, with **BH per hypothesis family**. Result:
  both development backdoors flagged HIGH, and 0 false positives on the
  6 untampered development models.

### The method as implemented

The full description is in the `behavioral_probe.py` docstring.

| Component | Setting |
|---|---|
| Patches | 4×4 (0.25 of the input extent), on a half-stride grid of 49 positions |
| Colors | +3σ and +6σ |
| Carriers | 64 |
| **Island test** (per position) | exact one-sided McNemar per neighbor within one patch-width; the position's p-value is the max over neighbors (IUT) |
| **Class-reach asymmetry** (per color) | chi-square goodness-of-fit of per-class reach over the 16 non-overlapping positions, against uniform |
| Correction | BH **within each family**; q<0.01 HIGH, q<0.05 MEDIUM |

Findings are reported under the checks `behavioral_trigger_island` and
`behavioral_class_asymmetry`. Cost is 99 forward batches of 64: about
0.2s through torch, about 14s through `onnx.reference`.

### Held-out result (seeds 2-7, run once)

| | Result |
|---|---|
| False positives (6 clean, 6 noisy, 6 steganographic) | **0 / 18** |
| Backdoors detected | **4 / 6**, all HIGH (seeds 2, 4, 5, 7) |
| Missed | seed 3 (target 0; +6σ reach [8,4,4,0], raw p≈0.05) and seed 6 (target 3; [4,4,1,7]) |
| Target class identified | the asymmetry finding named the **correct target class in all 4** detections |
| Trigger location | **not reliable**. On seeds 4 and 5 the island findings flag *legitimate* cells surrounded by the hijacked region, not the trigger |

All of this is locked into `tests/test_behavioral_benchmark.py`, with
the misses included, using `tests/fixtures/stage5_validation/`.

**Sample sizes are small, and the claims must stay within them.** At 95%
confidence, 0 false positives out of 18 means a rate below about 17%.
4 detections out of 6 means recall somewhere between about 22% and 96%.
This is evidence the method works on **this benchmark's backdoor
family**. It is not a precise recall number.

### Limits (all measured or structural, none hidden)

1. **The asymmetry null is specific to this task.** It assumes that on a
   clean model, bright patches reach every class uniformly. That is true
   by construction for this spatially symmetric quadrant task, and not
   for real classifiers. Real-world use would need a per-model baseline,
   and there is currently no principled source for one.
2. **Only bright triggers are in scope by default.** Dark patches were
   dropped on the development set.
3. **The report-wide FDR is bounded by the sum of the per-family levels**
   (up to 2×0.05 at MEDIUM), not by a single alpha.
4. **Island findings are hints, not trigger locations.**
5. **The suite is now spent.** Seeds 2-7 are no longer unseen, so any
   redesign needs a new pre-registered suite (see
   `tests/fixtures/stage5_validation/README.md`).
6. **The earlier limits still apply:** a single trigger family (localized
   high-contrast patches), a caller-supplied `forward_fn`, and a single
   synthetic task (§1, §3).
