"""
peekaboo/pipeline/behavioral_probe.py

Stage 5: Behavioral Probing
============================

Design doc: PHASE4.md. This is the first implementation pass; read
PHASE4.md before extending it -- several scope decisions made there are
repeated here only in summary.

Purpose
-------
Stages 1-4 are all static: they inspect a model's weights (structure,
distributions, mantissa bits) without ever running it. That makes them
blind to a backdoor's actual defining behavior -- a trigger input
forcing a specific, wrong classification. Stage 5 closes that gap by
actually executing the model against many probe inputs and looking for
that forced-classification signature.

NOT wired into gate.py / run_pre_checks
----------------------------------------
Unlike Stages 1-3 (wired) and Stage 4 (implemented but deliberately not
yet wired), this stage is not folded into the default pipeline at all,
and that isn't a "not done yet" placeholder -- it's a deliberate first-
pass decision. Two reasons, both structural, not just "still needs
calibration":

1. Cost. Stages 1-4 are a single pass over static tensor data. Stage 5
   requires actually running the model many times (carriers x candidate
   patches x bootstrap control trials -- see below), which is a
   qualitatively different, caller-tunable cost, not something every
   `run_pre_checks` call should be forced to pay.
2. Runnability. Per PHASE4.md Sec 1, most real-world models (safetensors,
   pytorch pickle) cannot be run from a `LoadedModel` alone -- there is
   no computation graph in those formats, and the pickle loader's
   restricted unpickler deliberately refuses to reconstruct one (that
   refusal is Stage 1's safety gate, not a limitation to route around).
   This stage requires a caller-supplied `forward_fn`. Native ONNX
   execution (which does carry a graph) was flagged in PHASE4.md as
   needing a new `onnxruntime` dependency -- a decision requiring
   approval, matching the project's established no-new-dependency-
   without-discussion pattern (see Stage 3/4's scipy avoidance). That
   approval hasn't been given, so this first pass supports ONLY the
   caller-supplied `forward_fn` path, for every format including ONNX.
   `run_behavioral_check` is a standalone entry point; call it
   explicitly, not through `run_pre_checks`.

The not-runnable case is never a silent skip
----------------------------------------------
When `forward_fn` is `None`, this returns a `BehavioralReport` with
`mode="not_runnable"` and exactly one real, visible `Finding`
(`check="behavioral_runnable"`) explaining why -- never zero findings,
never a fabricated result. `mode` is the field a downstream Model Risk
Score should key off of to tell "probed, found nothing" apart from
"could not probe this model at all."

Method
------
Paired carrier probing against a parameterized patch family (PHASE4.md
Sec 2) -- NOT a replay of any specific known trigger (that would be
circular, see PHASE4.md Sec 3):

1. Draw carrier inputs: mostly random noise, plus a handful of
   structured (constant-value) carriers, at the caller-supplied
   `input_shape`.
2. Build a grid of candidate patches -- varying position, size (as
   fractions of the input's spatial/feature extent), and color/
   intensity -- and stamp each onto every carrier.
3. For each patch footprint (size), build an empirical null
   distribution of "how much does an occlusion this size normally
   concentrate predictions onto one class" from many random-CONTENT
   control patches of the same footprint, on the same carriers. This
   isolates "trigger-like forced classification" from the mundane fact
   that any big enough occlusion perturbs some predictions on any
   classifier, backdoored or not (PHASE4.md Sec 4).
4. Compare each candidate's class-concentration statistic against that
   footprint's own null distribution -> an empirical p-value.
5. Apply Benjamini-Hochberg FDR correction across every candidate patch
   compared in the same report -- built into this first implementation,
   not deferred. PHASE4.md Sec 4 is explicit that Stage 5's patch grid
   is a *larger* multiple-comparisons problem than Stage 4's already-
   documented 84-tests-uncorrected gap (PHASE3.md), and that mistake
   must not be repeated here.

A cheap, deliberately weaker secondary signal (plain random-noise-only
class concentration, no patch involved) is also reported, capped at LOW
severity -- see `_max_class_hit_rate`'s use in the baseline finding.

Circularity (read before trusting any output)
------------------------------------------------
This module's candidate-patch grid is a generic, parameterized search
over position/size/color -- it does not know about, and is not centered
on, this project's own benchmark trigger (`peekaboo/benchmark/data.py`'s
3x3/value=6.0/top-left patch). That is intentional (PHASE4.md Sec 3):
searching specifically for the known trigger would prove nothing beyond
"we can find what we ourselves planted." Any validation against the real
benchmark must run this generic search *blind* and check whether it
happens to land near the known trigger's position/size -- never
special-cased to look there. This module only instantiates one trigger
*family* (localized high-contrast patches); blended, frequency-domain,
warping, and semantic triggers are out of scope for this pass, per
PHASE4.md Sec 3.

Shared schema
-------------
Uses `peekaboo.schema.reports.Finding`/`BehavioralReport`/
`compute_passed` and `peekaboo.schema.model_risk_score.Severity` --
the same convention Stage 3 established and Stage 4 has now been fixed
to follow, rather than adding a third inconsistent shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from peekaboo.loaders.common import LoadedModel
from peekaboo.pipeline.multiple_testing import benjamini_hochberg
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import BehavioralReport, Finding, compute_passed

# Structured (non-random) carrier values, mixed in alongside random-noise
# carriers -- broadens coverage beyond pure noise, since noise-only
# carriers risk conflating "abnormal" with the well-known generic
# phenomenon that classifiers often collapse toward one class on pure
# out-of-distribution input, backdoor or not (PHASE4.md Sec 2/4).
_STRUCTURED_CARRIER_VALUES: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _passed_for_severity(severity: Severity) -> bool:
    """Same convention as Stages 1-4: INFO/LOW findings pass; MEDIUM+
    findings don't. Severity is still a triage label, not control flow --
    this module has no hard_fail concept."""
    return severity in (Severity.INFO, Severity.LOW)


# ---------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------

def _build_carriers(rng: np.random.Generator, n_carriers: int, input_shape: tuple[int, ...]) -> np.ndarray:
    """n_carriers samples at the given per-sample shape: mostly random
    noise, plus up to len(_STRUCTURED_CARRIER_VALUES) constant-value
    carriers (fewer if n_carriers is too small to fit them)."""
    n_structured = min(len(_STRUCTURED_CARRIER_VALUES), max(0, n_carriers - 1))
    n_random = n_carriers - n_structured

    parts: list[np.ndarray] = []
    if n_random > 0:
        parts.append(rng.standard_normal((n_random,) + tuple(input_shape)).astype(np.float32))
    if n_structured > 0:
        structured = np.stack(
            [
                np.full(input_shape, v, dtype=np.float32)
                for v in _STRUCTURED_CARRIER_VALUES[:n_structured]
            ]
        )
        parts.append(structured)
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------
# Patch footprints: where/how big a candidate patch can be
# ---------------------------------------------------------------------

@dataclass
class _Footprint:
    """One patch size, and every position it can be placed at, for a
    given input_shape. kind="box" for >=2D (image-like) inputs -- a
    square region over the trailing two dims, across all leading
    (channel) dims; kind="segment" for 1D (flat feature vector) inputs
    -- a contiguous slice of the trailing dim. Positions are (row, col)
    tuples for "box", plain ints for "segment"."""

    kind: str
    size: int
    positions: list = field(default_factory=list)


def _iter_footprints(
    input_shape: tuple[int, ...], patch_size_fractions: tuple[float, ...]
) -> list[_Footprint]:
    footprints: list[_Footprint] = []
    if len(input_shape) >= 2:
        h, w = input_shape[-2], input_shape[-1]
        extent = min(h, w)
        sizes = sorted({min(max(1, round(frac * extent)), h, w) for frac in patch_size_fractions})
        for s in sizes:
            positions = [(r, c) for r in range(0, h - s + 1, s) for c in range(0, w - s + 1, s)]
            if not positions:
                positions = [(0, 0)]
            footprints.append(_Footprint(kind="box", size=s, positions=positions))
    else:
        d = input_shape[-1] if input_shape else 1
        sizes = sorted({min(max(1, round(frac * d)), d) for frac in patch_size_fractions})
        for s in sizes:
            positions = list(range(0, d - s + 1, s)) or [0]
            footprints.append(_Footprint(kind="segment", size=s, positions=positions))
    return footprints


def _region_shape(carriers: np.ndarray, footprint: _Footprint, position) -> tuple[int, ...]:
    if footprint.kind == "box":
        r, c = position
        s = footprint.size
        return carriers[..., r : r + s, c : c + s].shape
    start = position
    s = footprint.size
    return carriers[..., start : start + s].shape


def _apply_patch(carriers: np.ndarray, footprint: _Footprint, position, fill) -> np.ndarray:
    """Returns a COPY of carriers with the given patch region set to
    `fill` (a scalar color, or an ndarray of per-carrier random content
    matching the region's shape)."""
    out = carriers.copy()
    if footprint.kind == "box":
        r, c = position
        s = footprint.size
        out[..., r : r + s, c : c + s] = fill
    else:
        start = position
        s = footprint.size
        out[..., start : start + s] = fill
    return out


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _max_class_hit_rate(preds: np.ndarray, num_classes: int) -> tuple[float, int]:
    """Fraction of predictions landing on the single most common class,
    and which class that is."""
    counts = np.bincount(preds, minlength=num_classes)
    majority = int(np.argmax(counts))
    return float(counts[majority]) / float(preds.size), majority


def _empirical_p_value(null_samples: np.ndarray, observed: float) -> float:
    """One-sided empirical p-value: P(null >= observed), with the
    standard +1 continuity correction so a p-value is never exactly
    zero (matches the permutation-test convention PHASE4.md Sec 4
    proposes, and Stage 4's own docstring flagged as the rigorous
    approach it never got to build)."""
    return float((1 + int(np.sum(null_samples >= observed))) / (null_samples.size + 1))


_benjamini_hochberg = benjamini_hochberg  # shared with Stage 4; see multiple_testing.py


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_behavioral_check(
    model: LoadedModel,
    forward_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    *,
    input_shape: Optional[tuple[int, ...]] = None,
    num_classes: Optional[int] = None,
    n_carriers: int = 64,
    n_bootstrap: int = 64,
    patch_size_fractions: tuple[float, ...] = (0.25, 0.5),
    patch_colors: tuple[float, ...] = (-3.0, 3.0),
    fdr_alpha_high: float = 0.01,
    fdr_alpha_medium: float = 0.05,
    raw_p_report_threshold: float = 0.05,
    seed: int = 0,
) -> BehavioralReport:
    """Run Stage 5 (Behavioral Probing).

    `forward_fn`: batch-in/logits-out callable the CALLER builds and
    trusts (see module docstring -- this stage never constructs a
    runnable model itself). `None` means "not runnable": returns
    immediately with `mode="not_runnable"` and exactly one Finding, no
    probing attempted.

    `input_shape`/`num_classes` are required whenever `forward_fn` is
    given -- this stage does not attempt to infer them (recovering the
    full spatial input shape from weight tensors alone isn't reliable
    in general; only the caller, who built `forward_fn`, actually knows
    it).
    """
    if forward_fn is None:
        finding = Finding(
            check="behavioral_runnable",
            severity=Severity.INFO,
            passed=True,
            message=(
                "No forward_fn supplied -- Stage 5 requires a caller-provided "
                "runnable model and cannot reconstruct a forward pass from "
                "tensors alone (see PHASE4.md Sec 1). This model was not probed."
            ),
            details={"reason": "no_forward_fn_supplied"},
        )
        return BehavioralReport(
            model_path=model.source_path,
            mode="not_runnable",
            passed=compute_passed([finding]),
            findings=[finding],
            metadata={"n_carriers": 0, "n_candidates_tested": 0},
        )

    if input_shape is None or num_classes is None:
        raise ValueError(
            "forward_fn was supplied but input_shape/num_classes was not -- "
            "both are required to build probe inputs and interpret outputs."
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 for behavioral probing, got {num_classes}")

    rng = np.random.default_rng(seed)
    carriers = _build_carriers(rng, n_carriers, input_shape)
    n_carriers_actual = carriers.shape[0]

    def predict(batch: np.ndarray) -> np.ndarray:
        out = np.asarray(forward_fn(batch))
        if out.ndim != 2 or out.shape[0] != batch.shape[0] or out.shape[1] != num_classes:
            raise ValueError(
                f"forward_fn must return shape ({batch.shape[0]}, {num_classes}); got {out.shape}"
            )
        return out.argmax(axis=1)

    findings: list[Finding] = []

    # --- cheap secondary signal: plain random-noise-only concentration,
    # never elevated past LOW (PHASE4.md Sec 2, "a cheaper, secondary signal") ---
    baseline_preds = predict(carriers)
    baseline_hit_rate, baseline_majority = _max_class_hit_rate(baseline_preds, num_classes)
    uniform = 1.0 / num_classes
    baseline_severity = Severity.LOW if baseline_hit_rate > min(1.0, 2 * uniform) else Severity.INFO
    findings.append(
        Finding(
            check="behavioral_baseline_concentration",
            severity=baseline_severity,
            passed=_passed_for_severity(baseline_severity),
            message=(
                f"Unpatched carriers: class {baseline_majority} got "
                f"{baseline_hit_rate:.2%} of predictions (uniform would be "
                f"{uniform:.2%}). Cheap, low-confidence signal -- OOD noise "
                "commonly collapses toward one class on any classifier, "
                "backdoored or not; never elevated past LOW on its own."
            ),
            details={
                "baseline_hit_rate": baseline_hit_rate,
                "majority_class": baseline_majority,
                "n_carriers": n_carriers_actual,
            },
        )
    )

    # --- main method: candidate patches vs. per-footprint bootstrap null ---
    footprints = _iter_footprints(input_shape, patch_size_fractions)
    candidate_results: list[dict] = []
    n_control_trials = 0

    for footprint in footprints:
        null_samples = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            position = footprint.positions[i % len(footprint.positions)]
            region_shape = _region_shape(carriers, footprint, position)
            fill = rng.standard_normal(region_shape).astype(np.float32)
            patched = _apply_patch(carriers, footprint, position, fill)
            preds = predict(patched)
            stat, _ = _max_class_hit_rate(preds, num_classes)
            null_samples[i] = stat
            n_control_trials += 1

        for color in patch_colors:
            for position in footprint.positions:
                patched = _apply_patch(carriers, footprint, position, np.float32(color))
                preds = predict(patched)
                stat, majority = _max_class_hit_rate(preds, num_classes)
                p = _empirical_p_value(null_samples, stat)
                candidate_results.append(
                    {
                        "kind": footprint.kind,
                        "size": footprint.size,
                        "position": list(position) if isinstance(position, tuple) else position,
                        "color": float(color),
                        "hit_rate": stat,
                        "majority_class": majority,
                        "p_value": p,
                    }
                )

    q_values = _benjamini_hochberg([r["p_value"] for r in candidate_results])
    for r, q in zip(candidate_results, q_values):
        r["fdr_p_value"] = q

    n_raw_significant = 0
    n_fdr_high = 0
    n_fdr_medium = 0
    for r in candidate_results:
        if r["p_value"] >= raw_p_report_threshold:
            continue
        n_raw_significant += 1
        if r["fdr_p_value"] < fdr_alpha_high:
            severity = Severity.HIGH
            n_fdr_high += 1
        elif r["fdr_p_value"] < fdr_alpha_medium:
            severity = Severity.MEDIUM
            n_fdr_medium += 1
        else:
            severity = Severity.LOW
        findings.append(
            Finding(
                check="behavioral_trigger_patch",
                severity=severity,
                passed=_passed_for_severity(severity),
                message=(
                    f"{r['kind']} patch size={r['size']} pos={r['position']} "
                    f"color={r['color']}: {r['hit_rate']:.2%} of carriers -> "
                    f"class {r['majority_class']} (raw p={r['p_value']:.4g}, "
                    f"FDR-adjusted p={r['fdr_p_value']:.4g} across "
                    f"{len(candidate_results)} candidates tested this report)"
                ),
                details=dict(r),
            )
        )

    findings.append(
        Finding(
            check="behavioral_coverage",
            severity=Severity.INFO,
            passed=True,
            message=(
                f"Probed {n_carriers_actual} carriers against "
                f"{len(candidate_results)} candidate patches across "
                f"{len(footprints)} size(s) x {len(patch_colors)} color(s), "
                f"plus {n_control_trials} control trials for the bootstrap "
                f"null. {n_raw_significant} candidate(s) had raw p < "
                f"{raw_p_report_threshold}; after Benjamini-Hochberg FDR "
                f"correction across all {len(candidate_results)} candidates, "
                f"{n_fdr_high} reached HIGH (q<{fdr_alpha_high}) and "
                f"{n_fdr_medium} reached MEDIUM (q<{fdr_alpha_medium})."
            ),
            details={
                "n_carriers": n_carriers_actual,
                "n_footprints": len(footprints),
                "n_candidates_tested": len(candidate_results),
                "n_control_trials": n_control_trials,
                "n_raw_significant": n_raw_significant,
                "n_fdr_high": n_fdr_high,
                "n_fdr_medium": n_fdr_medium,
            },
        )
    )

    return BehavioralReport(
        model_path=model.source_path,
        mode="probed",
        passed=compute_passed(findings),
        findings=findings,
        metadata={
            "input_shape": list(input_shape),
            "num_classes": num_classes,
            "n_carriers": n_carriers_actual,
            "n_candidates_tested": len(candidate_results),
            "n_control_trials": n_control_trials,
            "seed": seed,
        },
    )


def calibrate_on_clean(
    model: LoadedModel,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    **probe_kwargs,
) -> dict:
    """Run Stage 5 on a known-clean model and summarize the false-positive
    rate, mirroring Stage 4's `stego_check.calibrate_on_clean`. Use this on
    the 'clean' benchmark variant BEFORE trusting any finding on the
    backdoored/combined variants. `probe_kwargs` are forwarded unchanged to
    `run_behavioral_check` (input_shape, num_classes, probe budget, ...).

    Every candidate patch on a clean model is a potential false positive,
    so `candidate_fp_rate_medium_plus` (MEDIUM+ trigger-patch findings /
    candidates tested) is the number to read. Results on the real
    benchmark are recorded in PHASE4.md -- do not loosen/tighten the FDR
    alphas to move this number without first reading that section.
    """
    report = run_behavioral_check(model, forward_fn, **probe_kwargs)
    counts = {s.value: 0 for s in Severity}
    for f in report.findings:
        counts[f.severity.value] += 1
    n_candidates = report.metadata.get("n_candidates_tested", 0)
    n_fp = sum(
        1
        for f in report.findings
        if f.check == "behavioral_trigger_patch" and f.severity not in (Severity.INFO, Severity.LOW)
    )
    return {
        "mode": report.mode,
        "total_findings": len(report.findings),
        "by_severity": counts,
        "n_candidates_tested": n_candidates,
        "n_trigger_patch_medium_plus": n_fp,
        "candidate_fp_rate_medium_plus": (n_fp / n_candidates) if n_candidates else 0.0,
    }
