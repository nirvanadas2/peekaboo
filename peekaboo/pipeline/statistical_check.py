"""Stage 3: Statistical Analysis.

Per-layer statistics (mean, std, skewness, excess kurtosis, and
histogram-based Shannon entropy) computed on every floating-point tensor
in an already-loaded model, then screened for **within-model relative
outliers** — never absolute thresholds, and never a paired clean-vs-
tampered comparison. In deployment there is no trusted baseline for an
unknown model pulled from a public hub; the only thing Peekaboo can
compare a layer against is the rest of that same model's layers.

Detection method: a robust (median/MAD-based) modified z-score per
statistic, computed across the population of layers in the model being
scanned. Median/MAD is used instead of mean/stdev specifically because a
single tampered layer can otherwise skew the very baseline it's being
compared against — a classic breakdown-point problem for mean/stdev-based
outlier detection that median/MAD is far more resistant to.

FALLBACK: when a model has too few layers for "relative to the rest of
this model" to mean anything (fewer than `_MIN_LAYERS_FOR_RELATIVE`
layers with a reliable value for that statistic), detection falls back to
conservative absolute heuristic ranges. This is clearly marked in the
finding's `details["mode"]` and always capped at MEDIUM severity — a
fallback finding is never presented with the same confidence as a real
relative-outlier detection.

Same principle as Stage 2: `StatisticalReport` has no `hard_fail`. This
stage never gates anything downstream — severity here is a triage label,
not a control-flow signal.

KNOWN LIMITATION, found empirically against Phase 0's benchmark before
settling on the population definition below: Phase 0's TinyCNN has only 3
conv/linear weight matrices. No matter how the tensor population for
comparison is stratified, any role-homogeneous stratum of TinyCNN sits
well under `_MIN_LAYERS_FOR_RELATIVE`, so every Phase 0 benchmark variant
in every format falls back to the absolute-heuristic path — the relative-
outlier mode this stage is actually built around never gets to run on
that specific benchmark. That's a property of the benchmark being
deliberately tiny (see peekaboo/benchmark/models.py's docstring), not a
flaw in the detection logic — see `tests/test_statistical_check.py` for
where the relative-outlier logic is validated directly (synthetic
multi-layer fixtures with enough population to exercise it) versus what's
realistically observable on the real, tiny benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from peekaboo.loaders.common import LoadedModel
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import Finding, StatisticalReport, compute_passed

# A model needs at least this many layers with a reliable value for a given
# statistic before "outlier relative to the rest of this model" has enough
# data to mean anything. Below this, that statistic falls back to
# conservative absolute heuristic ranges instead.
_MIN_LAYERS_FOR_RELATIVE = 6

# Iglewicz & Hoaglin's standard modified z-score outlier thresholds
# (the 0.6745 constant makes MAD comparable to stdev under normality).
_MODIFIED_Z_CONSTANT = 0.6745
_MODIFIED_Z_HIGH = 5.0
_MODIFIED_Z_MEDIUM = 3.5

# Conservative absolute fallback ranges for typically-initialized/trained
# float32 neural network weights. Deliberately loose — these exist only to
# catch gross anomalies on models too small for relative comparison, not
# to make fine-grained judgments.
_FALLBACK_MEAN_ABS_LIMIT = 2.0
_FALLBACK_STD_MIN = 1e-4
_FALLBACK_STD_MAX = 10.0
_FALLBACK_KURTOSIS_MIN = -1.5
_FALLBACK_KURTOSIS_MAX = 20.0
_FALLBACK_ENTROPY_MIN_BITS = 1.0

_STAT_NAMES = ("mean", "std", "kurtosis", "entropy")


@dataclass
class _LayerStats:
    name: str
    n: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    entropy: float
    normalized_entropy: float  # entropy / log2(bin_count) — see _shannon_entropy
    reliable_kurtosis: bool


def _adaptive_bin_count(n: int) -> int:
    """Histogram bin count that scales gently with tensor size, so a tiny
    tensor doesn't get an artificially inflated entropy from having as
    many bins as data points."""
    return int(np.clip(np.sqrt(max(n, 1)), 4, 64))


def _shannon_entropy(flat: np.ndarray) -> tuple[float, int]:
    """Returns (entropy_bits, bin_count). Bin count scales with tensor
    size (see `_adaptive_bin_count`), so it's returned alongside the raw
    entropy — the *maximum possible* entropy for `k` bins is `log2(k)`,
    which differs by tensor size. Comparing raw entropy across tensors
    with different bin counts is exactly as unfair as comparing std
    across layers with different fan-in (see `_LOG_SCALE_STATS`'s
    docstring) — `normalized_entropy` below corrects for it."""
    if flat.size < 2:
        return 0.0, 0
    bins = _adaptive_bin_count(flat.size)
    counts, _ = np.histogram(flat, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0, bins
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs))), bins


def _compute_layer_stats(name: str, array: np.ndarray) -> _LayerStats:
    flat = array.astype(np.float64).ravel()
    n = flat.size
    mean = float(np.mean(flat)) if n >= 1 else 0.0
    std = float(np.std(flat)) if n >= 2 else 0.0

    reliable_kurtosis = n >= 4 and std > 1e-12
    if reliable_kurtosis:
        diffs = flat - mean
        m2 = float(np.mean(diffs**2))
        m3 = float(np.mean(diffs**3))
        m4 = float(np.mean(diffs**4))
        skewness = m3 / (m2**1.5)
        kurtosis = m4 / (m2**2) - 3.0  # excess kurtosis (0 for a normal distribution)
    else:
        skewness = 0.0
        kurtosis = 0.0

    entropy, entropy_bins = _shannon_entropy(flat)
    normalized_entropy = entropy / np.log2(entropy_bins) if entropy_bins > 1 else 0.0

    return _LayerStats(
        name=name, n=n, mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
        entropy=entropy, normalized_entropy=normalized_entropy, reliable_kurtosis=reliable_kurtosis,
    )


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    """Robust (median/MAD-based) z-scores. Public because the outlier
    logic is tested directly against synthetic per-layer stat arrays,
    independent of any model loading."""
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev)
    if mad < 1e-12:
        # Median absolute deviation collapses to ~0 when the majority of
        # values are identical/near-identical (common with small
        # populations) — mean absolute deviation is a reasonable fallback
        # rather than dividing by (near-)zero.
        mad = np.mean(abs_dev)
    if mad < 1e-12:
        return np.zeros_like(values)
    return _MODIFIED_Z_CONSTANT * (values - median) / mad


def _severity_for_z(z: float) -> Severity | None:
    az = abs(z)
    if az >= _MODIFIED_Z_HIGH:
        return Severity.HIGH
    if az >= _MODIFIED_Z_MEDIUM:
        return Severity.MEDIUM
    return None


# Two statistics need a transform before cross-layer comparison in
# relative mode, both found empirically: deepening Phase 0's benchmark
# from 3 to 7 conv/linear layers (to give this stage enough population to
# run in relative mode at all) surfaced conv1.weight — the smallest layer
# in every relevant sense — as a spurious outlier on the *clean* baseline,
# for two different structural reasons:
#
# - std scales multiplicatively with a layer's fan-in under standard
#   Kaiming/Xavier init (std ~ 1/sqrt(fan_in)), so two architecturally-
#   normal layers of very different sizes can have raw stds differing
#   several-fold with zero tampering involved. A modified z-score on raw
#   values conflates "this layer's scale is architecturally different"
#   with "this layer was tampered with" — comparing log(std) instead
#   turns that multiplicative relationship into an additive one, which
#   the same z-score math is actually suited to.
# - entropy's adaptive bin count (`_adaptive_bin_count`) scales with
#   tensor size, so the *maximum possible* entropy (log2(bin_count))
#   differs by tensor size too — a small tensor's raw entropy is
#   naturally lower than a large tensor's even when both are equally
#   well-spread relative to their own achievable ceiling. Comparing
#   `normalized_entropy` (entropy / log2(bin_count), a bounded ~[0,1]
#   "how uniformly spread is this" ratio) instead of raw bits corrects
#   for it the same way log(std) corrects for fan-in scaling.
#
# In both cases the *raw* value is still what's reported in the finding
# (`value` below) — only the comparison itself uses the transform.
_LOG_SCALE_STATS = frozenset({"std"})


def _check_statistic_relative(
    stat_name: str,
    entries: list[tuple[str, float]],
    layer_stats: dict[str, _LayerStats],
    findings: list[Finding],
) -> None:
    names = [n for n, _ in entries]
    values = np.array([v for _, v in entries], dtype=np.float64)
    if stat_name in _LOG_SCALE_STATS:
        compare_values = np.log(np.clip(values, 1e-12, None))
    elif stat_name == "entropy":
        compare_values = np.array([layer_stats[n].normalized_entropy for n in names], dtype=np.float64)
    else:
        compare_values = values
    z_scores = modified_z_scores(compare_values)

    flagged: dict[str, dict[str, Any]] = {}
    for name, value, z in zip(names, values, z_scores):
        severity = _severity_for_z(z)
        if severity is not None:
            flagged[name] = {"value": float(value), "modified_z_score": float(z), "severity": severity.value}

    if flagged:
        worst = Severity.HIGH if any(v["severity"] == "high" for v in flagged.values()) else Severity.MEDIUM
        findings.append(
            Finding(
                check=f"{stat_name}_outliers",
                severity=worst,
                passed=False,
                message=(
                    f"{len(flagged)}/{len(names)} layer(s) are relative outliers in {stat_name} "
                    f"(robust median/MAD z-score) vs. the rest of this model: {sorted(flagged)}"
                ),
                details={"mode": "relative_outlier", "population_size": len(names), "outliers": flagged},
            )
        )
    else:
        findings.append(
            Finding(
                check=f"{stat_name}_outliers",
                severity=Severity.INFO,
                passed=True,
                message=f"No relative outliers detected in {stat_name} across {len(names)} layer(s)",
                details={"mode": "relative_outlier", "population_size": len(names)},
            )
        )


def _fallback_reason(stat_name: str, value: float) -> str | None:
    if stat_name == "mean" and abs(value) > _FALLBACK_MEAN_ABS_LIMIT:
        return f"|mean|={abs(value):.4g} exceeds conservative fallback limit {_FALLBACK_MEAN_ABS_LIMIT}"
    if stat_name == "std" and not (_FALLBACK_STD_MIN <= value <= _FALLBACK_STD_MAX):
        return f"std={value:.4g} outside conservative fallback range [{_FALLBACK_STD_MIN}, {_FALLBACK_STD_MAX}]"
    if stat_name == "kurtosis" and not (_FALLBACK_KURTOSIS_MIN <= value <= _FALLBACK_KURTOSIS_MAX):
        return (
            f"excess kurtosis={value:.4g} outside conservative fallback range "
            f"[{_FALLBACK_KURTOSIS_MIN}, {_FALLBACK_KURTOSIS_MAX}]"
        )
    if stat_name == "entropy" and value < _FALLBACK_ENTROPY_MIN_BITS:
        return f"entropy={value:.4g} bits is below conservative fallback minimum {_FALLBACK_ENTROPY_MIN_BITS}"
    return None


def _check_statistic_fallback(stat_name: str, entries: list[tuple[str, float]], findings: list[Finding]) -> None:
    flagged: dict[str, dict[str, Any]] = {}
    for name, value in entries:
        reason = _fallback_reason(stat_name, value)
        if reason is not None:
            flagged[name] = {"value": float(value), "reason": reason}

    if flagged:
        # Capped at MEDIUM regardless of how far outside range: a fallback
        # finding is deliberately never presented with the confidence of a
        # real relative-outlier detection.
        findings.append(
            Finding(
                check=f"{stat_name}_outliers",
                severity=Severity.MEDIUM,
                passed=False,
                message=(
                    f"[fallback: too few layers for relative comparison] {len(flagged)}/{len(entries)} "
                    f"layer(s) fall outside conservative absolute {stat_name} ranges: {sorted(flagged)}"
                ),
                details={"mode": "absolute_fallback", "population_size": len(entries), "outliers": flagged},
            )
        )
    else:
        findings.append(
            Finding(
                check=f"{stat_name}_outliers",
                severity=Severity.INFO,
                passed=True,
                message=(
                    f"[fallback: too few layers for relative comparison] All {len(entries)} layer(s) within "
                    f"conservative absolute {stat_name} ranges"
                ),
                details={"mode": "absolute_fallback", "population_size": len(entries)},
            )
        )


def _weight_bearing_population(model: LoadedModel) -> set[str]:
    """The set of tensor names eligible for *cross-layer* comparison: real
    learned weight matrices (conv/linear/embedding kernels, ndim >= 2)
    only — explicitly excluding biases, normalization-layer affine
    parameters (BatchNorm/LayerNorm weight+bias), and running statistics.

    Two exclusions are load-bearing, not cosmetic, both caught empirically
    against Phase 0's clean baseline before being added:

    1. Normalization affine/running-stat tensors have a systematically
       different role than a weight matrix (a BatchNorm scale is
       initialized and trained to center near 1.0, its shift near 0.0,
       running_var is strictly positive) — none of that reflects
       tampering, it's just what that *kind* of parameter looks like.
    2. Biases, even genuine ones, are typically tiny vectors (a handful to
       a few dozen elements) compared to weight matrices (hundreds to
       thousands). A sample mean's standard error shrinks with
       1/sqrt(n): a small bias's sample mean has much higher natural
       sampling variance than a weight matrix's, so comparing raw
       mean/std across tensors of very different sizes conflates "this
       layer is unusual" with "this tensor is just small" — a real
       robustness concern in any model with a mix of tensor sizes, not
       specific to this benchmark.

    Restricting to weight matrices keeps the population reasonably
    homogeneous in both role and (typically) scale. The tradeoff is a
    smaller population — a small model with few conv/linear layers (like
    Phase 0's benchmark) will legitimately fall back to the absolute
    heuristic path rather than force an unreliable relative comparison;
    see the module docstring's FALLBACK section.
    """
    return {name for name, t in model.tensors.items() if t.dtype.startswith("float") and len(t.shape) >= 2}


def _entries_for_stat(
    stat_name: str, layer_stats: dict[str, _LayerStats], population: set[str]
) -> list[tuple[str, float]]:
    eligible = {name: s for name, s in layer_stats.items() if name in population}
    if stat_name == "mean":
        return [(n, s.mean) for n, s in eligible.items() if s.n >= 1]
    if stat_name == "std":
        return [(n, s.std) for n, s in eligible.items() if s.n >= 2]
    if stat_name == "kurtosis":
        return [(n, s.kurtosis) for n, s in eligible.items() if s.reliable_kurtosis]
    if stat_name == "entropy":
        return [(n, s.entropy) for n, s in eligible.items() if s.n >= 2]
    raise ValueError(f"unknown statistic '{stat_name}'")


def run_statistical_check(model: LoadedModel) -> StatisticalReport:
    """Run Stage 3 (Statistical Analysis) on an already-loaded model."""
    layer_stats: dict[str, _LayerStats] = {
        name: _compute_layer_stats(name, tensor.array)
        for name, tensor in model.tensors.items()
        if tensor.dtype.startswith("float")
    }

    # Per-layer stats (mean/std/skewness/kurtosis/entropy) are computed and
    # reported for every floating-point tensor, for full transparency. But
    # only the weight-bearing population (real weight matrices + their
    # paired biases) participates in cross-layer comparison — see
    # `_weight_bearing_population`'s docstring for why.
    population = _weight_bearing_population(model)
    n_layers = len(population)
    overall_mode = "relative_outlier" if n_layers >= _MIN_LAYERS_FOR_RELATIVE else "absolute_fallback"

    findings: list[Finding] = []

    if n_layers == 0:
        findings.append(
            Finding(
                check="statistical_coverage",
                severity=Severity.INFO,
                passed=True,
                message="No weight-bearing tensors found to analyze",
            )
        )
    else:
        for stat_name in _STAT_NAMES:
            entries = _entries_for_stat(stat_name, layer_stats, population)
            if len(entries) < 2:
                findings.append(
                    Finding(
                        check=f"{stat_name}_outliers",
                        severity=Severity.INFO,
                        passed=True,
                        message=(
                            f"Fewer than two layers have a reliable {stat_name} value; nothing to compare"
                        ),
                        details={"mode": overall_mode, "population_size": len(entries)},
                    )
                )
                continue

            # A stat's own *reliable* population can be smaller than the
            # model's total layer count (e.g. kurtosis excludes tiny
            # tensors) — the fallback decision is made per-statistic on
            # that actual population, not just the model-wide count.
            if len(entries) >= _MIN_LAYERS_FOR_RELATIVE:
                _check_statistic_relative(stat_name, entries, layer_stats, findings)
            else:
                _check_statistic_fallback(stat_name, entries, findings)

    per_layer_summary = {
        name: {
            "n": s.n,
            "mean": s.mean,
            "std": s.std,
            "skewness": s.skewness,
            "kurtosis": s.kurtosis,
            "entropy": s.entropy,
            "normalized_entropy": s.normalized_entropy,
            "reliable_kurtosis": s.reliable_kurtosis,
        }
        for name, s in layer_stats.items()
    }

    return StatisticalReport(
        model_path=model.source_path,
        mode=overall_mode,
        passed=compute_passed(findings),
        findings=findings,
        metadata={
            "num_layers_analyzed": n_layers,
            "weight_bearing_population": sorted(population),
            "per_layer_stats": per_layer_summary,
        },
    )
