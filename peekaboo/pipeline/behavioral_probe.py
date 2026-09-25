"""
peekaboo/pipeline/behavioral_probe.py

Stage 5: Behavioral Probing
============================

Design doc: PHASE4.md. Read its "Held-out validation" sections before
extending this -- the method below replaced a first implementation that
was shown NOT to discriminate on the real benchmark, and the current
method's measured recall/false-positive numbers (and their limits) are
recorded there.

Purpose
-------
Stages 1-4 are all static: they inspect a model's weights (structure,
distributions, mantissa bits) without ever running it. That makes them
blind to a backdoor's actual defining behavior -- a trigger input
forcing a specific, wrong classification. Stage 5 closes that gap by
actually executing the model against many probe inputs and looking for
that forced-classification signature. On this project's benchmark, no
static stage detects the backdoor at all (PHASE2.md, PHASE3.md).

NOT wired into gate.py / run_pre_checks
----------------------------------------
Two structural reasons:

1. Cost. Stages 1-4 are a single pass over static tensor data. Stage 5
   runs the model many times (carriers x candidate patches), a
   qualitatively different, caller-tunable cost.
2. Runnability. Per PHASE4.md Sec 1, most real-world models (safetensors,
   pytorch pickle) cannot be run from a `LoadedModel` alone -- there is
   no computation graph in those formats, and the pickle loader's
   restricted unpickler deliberately refuses to reconstruct one (that
   refusal is Stage 1's safety gate, not a limitation to route around).
   This stage requires a caller-supplied `forward_fn`, for every format
   including ONNX (no `onnxruntime` dependency; `onnx.reference` works
   but is slow -- see peekaboo/benchmark/runnable.py).

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
Carriers (mostly N(0,1) noise, plus a few constant carriers) are stamped
with constant-color square patches over a half-stride grid of positions
(patch side = `patch_size_fractions` x input extent; colors in units of
the carrier noise's sigma, which is 1 by construction). Each patched
batch's predictions give, per position, the class the patch *forces*
(majority) and how reliably. Two complementary hypothesis families are
tested on that, because backdoors of the same attack come out in two
different shapes (PHASE4.md "Why it missed"):

1. ISLAND (local inconsistency) -- catches POINT-like backdoors. A
   legitimate decision feature should be spatially smooth; a small
   trigger forms an island. For each patch p with forced class k, and
   each grid neighbour q within one patch-width: an exact one-sided
   McNemar test, paired over carriers, of "p sends carriers to k more
   often than q does". p's p-value is the MAX over neighbours (an
   intersection-union test: an island must differ from ALL of them).
2. CLASS-REACH ASYMMETRY -- catches REGION-like backdoors, where a whole
   area has been hijacked to the target class (so there is no island).
   Per color, over NON-overlapping positions: how many positions force
   each class (majority hit >= 0.5). Chi-square goodness-of-fit vs.
   uniform. ASSUMPTION: under clean behavior, bright patches reach every
   class about equally often. That holds for this benchmark's
   spatially-symmetric quadrant task by construction; it is NOT a
   general property of real classifiers, and would need a per-model
   baseline there.

Benjamini-Hochberg FDR is applied PER FAMILY (island candidates;
asymmetry tests) -- pooling ~100 dependent island tests with 2
asymmetry tests drowned the latter. Consequence: the report-wide FDR is
bounded by the SUM of the per-family levels (<= 2 x fdr_alpha_medium at
MEDIUM), not by one alpha.

Default colors are bright only (+3 sigma, +6 sigma). Dark patches were
excluded on development data: clean models' responses to them are
legitimately skewed, and every island false positive observed in
development came from a dark patch. Consequence: dark-colored triggers
are out of scope by default.

Known limits (measured -- PHASE4.md):
  - Held-out: 4/6 backdoors detected (all HIGH), 0/18 false positives on
    clean/noisy/steganographic models. Small samples -- wide intervals.
  - The ASYMMETRY finding names the target class correctly; ISLAND
    findings do NOT reliably localize the trigger (on region backdoors
    they can flag the legitimate cells surrounded by the hijacked area).
    Treat island positions as hints, not trigger locations.
  - A cheap secondary signal (unpatched-carrier class concentration) is
    also reported, never above LOW.

Circularity (read before trusting any output)
------------------------------------------------
The candidate grid is a generic search over position/size/color; it
does not know about, and is not centered on, the benchmark trigger.
Validation must run blind, on backdoors the design never saw (PHASE4.md
Sec 3 and "Pre-registered held-out suite"). Only localized
high-contrast patch triggers are in scope; blended, frequency-domain,
warping, and semantic triggers are not.

Shared schema
-------------
Uses `peekaboo.schema.reports.Finding`/`BehavioralReport`/
`compute_passed` and `peekaboo.schema.model_risk_score.Severity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from peekaboo.loaders.common import LoadedModel
from peekaboo.pipeline.multiple_testing import benjamini_hochberg
from peekaboo.pipeline.stego_check import _chi2_sf
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import BehavioralReport, Finding, compute_passed

# Structured (non-random) carrier values, mixed in alongside random-noise
# carriers -- broadens coverage beyond pure noise, since noise-only
# carriers risk conflating "abnormal" with the well-known generic
# phenomenon that classifiers often collapse toward one class on pure
# out-of-distribution input, backdoor or not (PHASE4.md Sec 2/4).
_STRUCTURED_CARRIER_VALUES: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

# A position "reaches" a class for the asymmetry test when its patch
# forces that class on at least this fraction of carriers.
_REACH_MIN_HIT_RATE = 0.5


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
    input_shape: tuple[int, ...],
    patch_size_fractions: tuple[float, ...],
    stride_divisor: int = 1,
) -> list[_Footprint]:
    """Positions step by `size // stride_divisor` (at least 1):
    stride_divisor=1 gives non-overlapping positions, 2 a half-stride
    (overlapping) grid."""
    footprints: list[_Footprint] = []
    if len(input_shape) >= 2:
        h, w = input_shape[-2], input_shape[-1]
        extent = min(h, w)
        sizes = sorted({min(max(1, round(frac * extent)), h, w) for frac in patch_size_fractions})
        for s in sizes:
            step = max(1, s // stride_divisor)
            positions = [(r, c) for r in range(0, h - s + 1, step) for c in range(0, w - s + 1, step)]
            if not positions:
                positions = [(0, 0)]
            footprints.append(_Footprint(kind="box", size=s, positions=positions))
    else:
        d = input_shape[-1] if input_shape else 1
        sizes = sorted({min(max(1, round(frac * d)), d) for frac in patch_size_fractions})
        for s in sizes:
            step = max(1, s // stride_divisor)
            positions = list(range(0, d - s + 1, step)) or [0]
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
    `fill` (a scalar color, or an ndarray of per-carrier content
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


def _neighbors(footprint: _Footprint, position, grid: set) -> list:
    """Grid positions within one patch-width (Chebyshev distance for
    boxes), excluding `position` itself."""
    s = footprint.size
    if footprint.kind == "box":
        r, c = position
        return [q for q in grid if q != position and abs(q[0] - r) <= s and abs(q[1] - c) <= s]
    return [q for q in grid if q != position and abs(q - position) <= s]


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _max_class_hit_rate(preds: np.ndarray, num_classes: int) -> tuple[float, int]:
    """Fraction of predictions landing on the single most common class,
    and which class that is."""
    counts = np.bincount(preds, minlength=num_classes)
    majority = int(np.argmax(counts))
    return float(counts[majority]) / float(preds.size), majority


def _binom_sf_half(k: int, n: int) -> float:
    """Exact P(Binomial(n, 0.5) >= k) -- the one-sided exact McNemar
    p-value for k of n discordant pairs favouring one side."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def _island_p_value(p_preds: np.ndarray, neighbor_preds: list[np.ndarray], forced_class: int) -> float:
    """Intersection-union test: max over neighbours of the exact McNemar
    p-value for "this patch sends carriers to forced_class more often
    than that neighbour does". 1.0 when there are no neighbours."""
    worst = 0.0
    for q_preds in neighbor_preds:
        b = int(np.sum((p_preds == forced_class) & (q_preds != forced_class)))
        c = int(np.sum((q_preds == forced_class) & (p_preds != forced_class)))
        worst = max(worst, _binom_sf_half(b, b + c))
    return worst if neighbor_preds else 1.0


def _reach_asymmetry(reach: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Chi-square goodness-of-fit of per-class reach counts vs. uniform.
    (None, None) when fewer than 2 decisive positions per class exist --
    too few to test."""
    k = reach.size
    n = int(reach.sum())
    if n < 2 * k:
        return None, None
    expected = n / k
    chi2 = float(((reach - expected) ** 2 / expected).sum())
    return chi2, _chi2_sf(chi2, k - 1)


_benjamini_hochberg = benjamini_hochberg  # shared with Stage 4; see multiple_testing.py


def _severity_from_q(q: float, alpha_high: float, alpha_medium: float) -> Severity:
    if q < alpha_high:
        return Severity.HIGH
    if q < alpha_medium:
        return Severity.MEDIUM
    return Severity.LOW


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
    patch_size_fractions: tuple[float, ...] = (0.25,),
    patch_colors: tuple[float, ...] = (3.0, 6.0),
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
    given -- this stage does not attempt to infer them.

    `patch_colors` are in units of the carrier noise's sigma (1 here).
    The defaults are the design frozen before held-out validation
    (PHASE4.md); changing them means the validation numbers no longer
    apply.
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
    n_forward_batches = 0

    def predict(batch: np.ndarray) -> np.ndarray:
        nonlocal n_forward_batches
        n_forward_batches += 1
        out = np.asarray(forward_fn(batch))
        if out.ndim != 2 or out.shape[0] != batch.shape[0] or out.shape[1] != num_classes:
            raise ValueError(
                f"forward_fn must return shape ({batch.shape[0]}, {num_classes}); got {out.shape}"
            )
        return out.argmax(axis=1)

    findings: list[Finding] = []

    # --- cheap secondary signal: plain carrier class concentration,
    # never elevated past LOW (PHASE4.md Sec 2) ---
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

    island_results: list[dict] = []
    asymmetry_results: list[dict] = []
    grids = _iter_footprints(input_shape, patch_size_fractions, stride_divisor=2)
    disjoint = {fp.size: fp.positions for fp in _iter_footprints(input_shape, patch_size_fractions)}

    for footprint in grids:
        grid = set(footprint.positions)
        for color in patch_colors:
            preds: dict = {}

            def preds_at(position):
                if position not in preds:
                    preds[position] = predict(_apply_patch(carriers, footprint, position, np.float32(color)))
                return preds[position]

            for position in footprint.positions:
                p_preds = preds_at(position)
                hit, forced = _max_class_hit_rate(p_preds, num_classes)
                nbrs = _neighbors(footprint, position, grid)
                island_results.append(
                    {
                        "kind": footprint.kind,
                        "size": footprint.size,
                        "position": list(position) if isinstance(position, tuple) else position,
                        "color": float(color),
                        "forced_class": forced,
                        "hit_rate": hit,
                        "n_neighbors": len(nbrs),
                        "p_value": _island_p_value(p_preds, [preds_at(q) for q in nbrs], forced),
                    }
                )

            reach = np.zeros(num_classes, dtype=int)
            for position in disjoint[footprint.size]:
                hit, forced = _max_class_hit_rate(preds_at(position), num_classes)
                if hit >= _REACH_MIN_HIT_RATE:
                    reach[forced] += 1
            chi2, p = _reach_asymmetry(reach)
            asymmetry_results.append(
                {
                    "kind": footprint.kind,
                    "size": footprint.size,
                    "color": float(color),
                    "reach": reach.tolist(),
                    "n_positions": len(disjoint[footprint.size]),
                    "chi2": chi2,
                    "p_value": p,
                    "over_represented_class": int(np.argmax(reach)),
                    "under_represented_class": int(np.argmin(reach)),
                }
            )

    # --- BH per hypothesis family ---
    for r, q in zip(island_results, benjamini_hochberg([r["p_value"] for r in island_results])):
        r["fdr_p_value"] = q
    testable = [r for r in asymmetry_results if r["p_value"] is not None]
    for r, q in zip(testable, benjamini_hochberg([r["p_value"] for r in testable])):
        r["fdr_p_value"] = q

    counts = {"island": {"high": 0, "medium": 0}, "asymmetry": {"high": 0, "medium": 0}}
    n_island_reported = 0
    for r in island_results:
        if r["p_value"] >= raw_p_report_threshold:
            continue
        n_island_reported += 1
        severity = _severity_from_q(r["fdr_p_value"], fdr_alpha_high, fdr_alpha_medium)
        if severity.value in counts["island"]:
            counts["island"][severity.value] += 1
        findings.append(
            Finding(
                check="behavioral_trigger_island",
                severity=severity,
                passed=_passed_for_severity(severity),
                message=(
                    f"{r['kind']} patch size={r['size']} pos={r['position']} color=+{r['color']:g}sigma "
                    f"forces class {r['forced_class']} on {r['hit_rate']:.0%} of carriers, more than "
                    f"all {r['n_neighbors']} neighbouring positions (worst-neighbour exact McNemar "
                    f"p={r['p_value']:.3g}, BH q={r['fdr_p_value']:.3g} over {len(island_results)} "
                    "island candidates). Position is a hint, not a trigger location."
                ),
                details=dict(r),
            )
        )

    for r in asymmetry_results:
        if r["p_value"] is None:
            severity = Severity.INFO
            message = (
                f"color=+{r['color']:g}sigma: reach {r['reach']} -- too few decisive positions "
                "to test class-reach asymmetry."
            )
        else:
            severity = _severity_from_q(r["fdr_p_value"], fdr_alpha_high, fdr_alpha_medium)
            if severity == Severity.LOW:
                severity = Severity.INFO if r["p_value"] >= raw_p_report_threshold else Severity.LOW
            message = (
                f"color=+{r['color']:g}sigma: class reach {r['reach']} over {r['n_positions']} "
                f"non-overlapping positions (chi2={r['chi2']:.2f}, p={r['p_value']:.3g}, BH "
                f"q={r['fdr_p_value']:.3g}); most-reached class {r['over_represented_class']}. "
                "Assumes bright patches reach classes uniformly on a clean model -- true for "
                "the benchmark task, not in general."
            )
        if severity.value in counts["asymmetry"]:
            counts["asymmetry"][severity.value] += 1
        findings.append(
            Finding(
                check="behavioral_class_asymmetry",
                severity=severity,
                passed=_passed_for_severity(severity),
                message=message,
                details=dict(r),
            )
        )

    findings.append(
        Finding(
            check="behavioral_coverage",
            severity=Severity.INFO,
            passed=True,
            message=(
                f"Probed {n_carriers_actual} carriers with {len(island_results)} island "
                f"candidates and {len(asymmetry_results)} class-asymmetry tests across "
                f"{len(grids)} patch size(s) x {len(patch_colors)} color(s) "
                f"({n_forward_batches} forward batches). BH per family: island "
                f"{counts['island']['high']} HIGH / {counts['island']['medium']} MEDIUM; "
                f"asymmetry {counts['asymmetry']['high']} HIGH / "
                f"{counts['asymmetry']['medium']} MEDIUM. Report-wide FDR is bounded by the "
                "sum of the per-family levels."
            ),
            details={
                "n_carriers": n_carriers_actual,
                "n_candidates_tested": len(island_results),
                "n_island_reported": n_island_reported,
                "n_asymmetry_tests": len(asymmetry_results),
                "n_forward_batches": n_forward_batches,
                "severity_counts": counts,
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
            "n_candidates_tested": len(island_results),
            "n_asymmetry_tests": len(asymmetry_results),
            "n_forward_batches": n_forward_batches,
            "patch_colors_sigma": list(patch_colors),
            "fdr": "benjamini_hochberg_per_family",
            "seed": seed,
        },
    )


def calibrate_on_clean(
    model: LoadedModel,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    **probe_kwargs,
) -> dict:
    """Run Stage 5 on a known-clean model and summarize the false-positive
    rate, mirroring Stage 4's `stego_check.calibrate_on_clean`. Any
    MEDIUM+ island or asymmetry finding on a clean model is a false
    positive. `probe_kwargs` are forwarded unchanged to
    `run_behavioral_check`. Measured results are in PHASE4.md -- do not
    move the FDR alphas to change this number without reading them."""
    report = run_behavioral_check(model, forward_fn, **probe_kwargs)
    counts = {s.value: 0 for s in Severity}
    for f in report.findings:
        counts[f.severity.value] += 1
    detection_checks = ("behavioral_trigger_island", "behavioral_class_asymmetry")
    n_fp = sum(
        1
        for f in report.findings
        if f.check in detection_checks and f.severity not in (Severity.INFO, Severity.LOW)
    )
    n_tests = report.metadata.get("n_candidates_tested", 0) + report.metadata.get("n_asymmetry_tests", 0)
    return {
        "mode": report.mode,
        "total_findings": len(report.findings),
        "by_severity": counts,
        "n_tests": n_tests,
        "n_detection_medium_plus": n_fp,
        "fp_rate_medium_plus": (n_fp / n_tests) if n_tests else 0.0,
    }
