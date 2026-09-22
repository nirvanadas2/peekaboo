"""
peekaboo/pipeline/stego_check.py

Stage 4: Steganographic Detection (bit-plane / mantissa-level tests)
======================================================================

Purpose
-------
Stage 3 (statistical analysis) explicitly cannot detect mantissa-LSB
steganographic payloads — by design, LSB-level embedding is invisible
to layer-level descriptive statistics (mean/std/kurtosis/entropy).
That's what this stage exists to catch.

Approach
--------
Unlike Stage 3, which had to invent a "relative to other layers in the
same model" baseline (because there's no trusted reference model in
real deployment), the tests here are *classical statistical hypothesis
tests* with known null distributions — they don't need a baseline
model to compare against. Each test asks: "does this bit sequence look
like it came from the assumed null process (independent, unbiased
bits)?" and returns a p-value.

Four tests per layer, on the N lowest mantissa bits of that layer's
weights:

1. Bit-balance chi-square test
   Tests whether P(bit=0) == P(bit=1) == 0.5 across the whole layer.
   Rationale: compressed/encrypted payload bytes are close to uniform
   random at the bit level, so embedding tends to push LSBs toward a
   50/50 split. CAVEAT: natural trained-weight mantissas are NOT
   guaranteed to be 50/50 themselves — floating point rounding during
   training can leave real, non-malicious bias. This test alone WILL
   false-positive on some clean models until calibrated (see bottom).

2. Block homogeneity chi-square test
   Splits the bit sequence into contiguous blocks and tests whether
   the 1-proportion is homogeneous across blocks (classic
   goodness-of-fit across blocks, not a single global count).
   Rationale: catches payloads embedded in only part of a tensor,
   which test #1 would dilute/average away across the whole layer.

3. Wald-Wolfowitz runs test
   Tests whether the *ordering* of bits (not just their count) looks
   random, via the number of runs (maximal same-value streaks).
   Rationale: structured payload bytes tend to produce a different run
   count than natural per-value rounding noise, even when the 0/1
   balance looks fine.

4. Bit autocorrelation at lags {1, 2, 4, 8}
   Pearson correlation between bit[i] and bit[i+lag], mapped to ±1.
   Rationale: per-value or per-byte payload embedding often introduces
   periodic structure (e.g. byte-boundary effects at lag 8) that
   pure independent noise will not show.

Honesty / known limitations (read before trusting any output)
---------------------------------------------------------------
- These are PLACEHOLDER severity thresholds. They have NOT been
  calibrated against this project's actual synthetic benchmark
  (clean / noisy / steganographic / backdoored / combined variants).
  Before relying on this stage's findings for anything, run
  `calibrate_on_clean()` against the clean variant across all three
  formats (safetensors/pt/onnx) and check the false-positive rate.
  If clean models trigger findings at more than a token rate, the
  thresholds in `SEVERITY_THRESHOLDS` need loosening.
- Only float32 and float16 tensors are supported for mantissa
  extraction. Integer tensors (rare in weight files, but possible for
  buffers/counters) are skipped with an INFO finding, not silently
  ignored.
- Like Stage 3's `StructuralReport`, severity here is a label only,
  never control flow — this module has no `hard_fail` concept and
  must not short-circuit `gate.py`.
- This stage tests the LOWEST mantissa bits only. If Stage 5's
  behavioral probing or future work suggests payloads live in higher
  mantissa bits or in exponent bits, `n_bits` and the extraction
  function will need extending — flagged as a TODO, not fixed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from peekaboo.loaders.common import LoadedModel

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Placeholder p-value cutoffs. NOT yet calibrated against the real
# benchmark — see module docstring. Smaller p-value = stronger
# rejection of "this looks random" = more suspicious.
SEVERITY_THRESHOLDS = {
    Severity.HIGH: 0.001,
    Severity.MEDIUM: 0.01,
    Severity.LOW: 0.05,
}


def _severity_from_pvalue(p: Optional[float]) -> Severity:
    if p is None:
        return Severity.INFO
    if p < SEVERITY_THRESHOLDS[Severity.HIGH]:
        return Severity.HIGH
    if p < SEVERITY_THRESHOLDS[Severity.MEDIUM]:
        return Severity.MEDIUM
    if p < SEVERITY_THRESHOLDS[Severity.LOW]:
        return Severity.LOW
    return Severity.INFO


@dataclass
class LayerStegoFinding:
    layer_name: str
    test_name: str
    severity: Severity
    statistic: float
    p_value: Optional[float]
    message: str

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "test_name": self.test_name,
            "severity": self.severity.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "message": self.message,
        }


@dataclass
class StegoReport:
    findings: list[LayerStegoFinding] = field(default_factory=list)
    layers_analyzed: int = 0
    layers_skipped: int = 0

    @property
    def max_severity(self) -> Severity:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM,
                 Severity.HIGH, Severity.CRITICAL]
        if not self.findings:
            return Severity.INFO
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "layers_analyzed": self.layers_analyzed,
            "layers_skipped": self.layers_skipped,
            "max_severity": self.max_severity.value,
        }


# ---------------------------------------------------------------------
# Bit extraction
# ---------------------------------------------------------------------

def extract_mantissa_lsbs(tensor: np.ndarray, n_bits: int = 1) -> Optional[np.ndarray]:
    """Extract the lowest n_bits of the mantissa from each float value.

    Returns a flat uint8 array of shape (numel * n_bits,), bit i of
    value v at position [v * n_bits + i] (i=0 is the true LSB).
    Returns None for unsupported dtypes (caller should emit an INFO
    finding and skip, not silently drop the layer).
    """
    flat = np.ascontiguousarray(tensor).ravel()

    if flat.dtype == np.float32:
        as_int = flat.view(np.uint32)
        max_bits = 23  # float32 mantissa width
    elif flat.dtype == np.float16:
        as_int = flat.view(np.uint16)
        max_bits = 10  # float16 mantissa width
    else:
        return None

    n_bits = min(n_bits, max_bits)
    bits = np.empty((flat.size, n_bits), dtype=np.uint8)
    for i in range(n_bits):
        bits[:, i] = ((as_int >> i) & 1).astype(np.uint8)
    return bits.reshape(-1)


# ---------------------------------------------------------------------
# Test 1: bit-balance chi-square
# ---------------------------------------------------------------------

def bit_balance_chi_square_test(bits: np.ndarray) -> tuple[float, Optional[float]]:
    """Chi-square goodness-of-fit of bit sequence against 50/50 null."""
    n = bits.size
    if n == 0:
        return 0.0, None
    ones = int(bits.sum())
    zeros = n - ones
    expected = n / 2.0
    statistic = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
    p_value = _chi2_sf(statistic, df=1)
    return float(statistic), p_value


# ---------------------------------------------------------------------
# Test 2: block homogeneity chi-square
# ---------------------------------------------------------------------

def block_chi_square_homogeneity_test(
    bits: np.ndarray, n_blocks: int = 16
) -> tuple[float, Optional[float]]:
    """Chi-square test for homogeneity of 1-proportion across blocks.

    Detects localized payloads that a whole-layer test would average
    away. Requires enough bits per block to be meaningful; falls back
    to fewer blocks for small layers rather than erroring.
    """
    n = bits.size
    if n < n_blocks * 20:
        n_blocks = max(1, n // 20)
    if n_blocks <= 1 or n == 0:
        return 0.0, None

    block_len = n // n_blocks
    trimmed = bits[: block_len * n_blocks].reshape(n_blocks, block_len)
    block_ones = trimmed.sum(axis=1)
    total_ones = block_ones.sum()
    overall_p = total_ones / (n_blocks * block_len)

    if overall_p in (0.0, 1.0):
        # Degenerate: every bit identical, nothing to test for
        # homogeneity (already maximally suspicious — caller should
        # weigh the balance test's result for this case).
        return 0.0, None

    expected_ones = block_len * overall_p
    expected_zeros = block_len * (1 - overall_p)
    block_zeros = block_len - block_ones

    statistic = float(
        (((block_ones - expected_ones) ** 2) / expected_ones).sum()
        + (((block_zeros - expected_zeros) ** 2) / expected_zeros).sum()
    )
    p_value = _chi2_sf(statistic, df=n_blocks - 1)
    return statistic, p_value


# ---------------------------------------------------------------------
# Test 3: Wald-Wolfowitz runs test
# ---------------------------------------------------------------------

def runs_test(bits: np.ndarray) -> tuple[float, Optional[float], int, float]:
    """Wald-Wolfowitz runs test for randomness of bit ordering.

    Returns (z_statistic, p_value, observed_runs, expected_runs).
    """
    n = bits.size
    if n < 2:
        return 0.0, None, 0, 0.0

    n1 = int(bits.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        # All-same sequence: maximally non-random, but the standard
        # runs-test formula divides by zero here. Report as a single
        # run with an explicit sentinel rather than crashing.
        return float("inf"), 0.0, 1, 0.0

    runs = 1 + int(np.sum(bits[1:] != bits[:-1]))

    expected_runs = (2.0 * n1 * n0) / n + 1
    variance = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n ** 2 * (n - 1))
    if variance <= 0:
        return 0.0, None, runs, expected_runs

    z = (runs - expected_runs) / (variance ** 0.5)
    p_value = _normal_sf_two_sided(z)
    return float(z), p_value, runs, expected_runs


# ---------------------------------------------------------------------
# Test 4: bit autocorrelation at multiple lags
# ---------------------------------------------------------------------

def bit_autocorrelation(
    bits: np.ndarray, lags: tuple[int, ...] = (1, 2, 4, 8)
) -> dict[int, float]:
    """Pearson correlation of the bit sequence (mapped to +-1) with
    itself shifted by each lag. Returns {} for sequences too short for
    any requested lag.
    """
    n = bits.size
    signed = bits.astype(np.float64) * 2.0 - 1.0  # 0/1 -> -1/+1
    result: dict[int, float] = {}
    for lag in lags:
        if lag >= n:
            continue
        a = signed[: n - lag]
        b = signed[lag:]
        if a.std() == 0 or b.std() == 0:
            result[lag] = 0.0
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        result[lag] = corr
    return result


# ---------------------------------------------------------------------
# p-value helpers (scipy if available, else lightweight fallback)
# ---------------------------------------------------------------------

def _chi2_sf(statistic: float, df: int) -> Optional[float]:
    if df <= 0:
        return None
    if _HAVE_SCIPY:
        return float(_scipy_stats.chi2.sf(statistic, df))
    if df == 1:
        # Chi-square(1) sf == 2 * normal sf(sqrt(statistic))
        return _normal_sf_two_sided(statistic ** 0.5)
    if statistic <= 0:
        return 1.0
    # General df > 1, no scipy: chi-square(df).sf(x) == Q(df/2, x/2), the
    # upper regularized incomplete gamma function. Implemented directly
    # (series for the lower branch, continued fraction for the upper —
    # the standard Numerical-Recipes-style split) rather than adding
    # scipy as a dependency, matching statistical_check.py's existing
    # "no new dependency" stance from Phase 2.
    return _regularized_gamma_q(df / 2.0, statistic / 2.0)


def _normal_sf_two_sided(z: float) -> float:
    """Two-sided survival function of |Z| for standard normal, via
    the erf-based closed form (no scipy dependency)."""
    import math
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _regularized_gamma_p(a: float, x: float) -> float:
    """Lower regularized incomplete gamma function P(a, x) = gamma(a, x) /
    Gamma(a), via its series expansion. Valid (rapidly convergent) for
    x < a + 1; callers route larger x to the continued-fraction form
    for Q instead."""
    import math

    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x),
    used as the chi-square survival function's scipy-free fallback for
    df > 1 (chi2(df).sf(x) == Q(df/2, x/2))."""
    import math

    if x < 0 or a <= 0:
        raise ValueError("_regularized_gamma_q requires x >= 0 and a > 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _regularized_gamma_p(a, x)
    # Continued fraction for Q(a, x), valid for x >= a + 1 (Numerical
    # Recipes' Lentz's-method form).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 501):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    gln = math.lgamma(a)
    return math.exp(-x + a * math.log(x) - gln) * h


# ---------------------------------------------------------------------
# Per-layer / per-model orchestration
# ---------------------------------------------------------------------

def analyze_layer(
    layer_name: str, tensor: np.ndarray, n_bits: int = 4, n_blocks: int = 16
) -> list[LayerStegoFinding]:
    bits = extract_mantissa_lsbs(tensor, n_bits=n_bits)
    if bits is None:
        return [
            LayerStegoFinding(
                layer_name=layer_name,
                test_name="dtype_support",
                severity=Severity.INFO,
                statistic=0.0,
                p_value=None,
                message=(
                    f"dtype {tensor.dtype} not supported for mantissa "
                    "extraction (only float32/float16); layer skipped."
                ),
            )
        ]

    findings: list[LayerStegoFinding] = []

    stat, p = bit_balance_chi_square_test(bits)
    findings.append(
        LayerStegoFinding(
            layer_name=layer_name,
            test_name="bit_balance_chi_square",
            severity=_severity_from_pvalue(p),
            statistic=stat,
            p_value=p,
            message=f"chi2={stat:.3f}, p={p}" if p is not None else "insufficient data",
        )
    )

    stat, p = block_chi_square_homogeneity_test(bits, n_blocks=n_blocks)
    findings.append(
        LayerStegoFinding(
            layer_name=layer_name,
            test_name="block_homogeneity_chi_square",
            severity=_severity_from_pvalue(p),
            statistic=stat,
            p_value=p,
            message=f"chi2={stat:.3f}, p={p}" if p is not None else "insufficient data",
        )
    )

    z, p, runs, expected = runs_test(bits)
    findings.append(
        LayerStegoFinding(
            layer_name=layer_name,
            test_name="wald_wolfowitz_runs",
            severity=_severity_from_pvalue(p),
            statistic=z,
            p_value=p,
            message=f"runs={runs}, expected={expected:.1f}, z={z:.3f}",
        )
    )

    autocorr = bit_autocorrelation(bits)
    for lag, corr in autocorr.items():
        # No formal null-distribution p-value here (would need a
        # permutation test against this layer's own bit count to be
        # rigorous) — reported as informational until Phase 3 adds a
        # permutation-based significance test. Flagged, not silently
        # treated as a real p-value.
        severity = Severity.LOW if abs(corr) > 0.1 else Severity.INFO
        findings.append(
            LayerStegoFinding(
                layer_name=layer_name,
                test_name=f"bit_autocorrelation_lag{lag}",
                severity=severity,
                statistic=corr,
                p_value=None,
                message=(
                    f"corr={corr:.4f} at lag {lag} "
                    "(no permutation-test p-value yet — informational only)"
                ),
            )
        )

    return findings


def analyze_model(model: LoadedModel, n_bits: int = 4) -> StegoReport:
    """model: an already-loaded model, as produced by Phase 0's loaders
    (peekaboo.loaders.load_model) -- the same LoadedModel type Stage 2
    (run_structural_check) and Stage 3 (run_statistical_check) take, for
    consistency across the pipeline."""
    report = StegoReport()
    for name, tensor_info in model.tensors.items():
        layer_findings = analyze_layer(name, tensor_info.array, n_bits=n_bits)
        report.findings.extend(layer_findings)
        if len(layer_findings) == 1 and layer_findings[0].test_name == "dtype_support":
            report.layers_skipped += 1
        else:
            report.layers_analyzed += 1
    return report


# ---------------------------------------------------------------------
# Calibration harness (run this before trusting SEVERITY_THRESHOLDS)
# ---------------------------------------------------------------------

def calibrate_on_clean(model: LoadedModel, n_bits: int = 4) -> dict:
    """Run the full test suite on a known-clean model and summarize
    the false-positive rate at each severity level. Use this on your
    'clean' benchmark variant (all three formats) BEFORE trusting any
    finding from analyze_model() on the stego/backdoored variants.

    If MEDIUM+ findings appear at more than a token rate on clean
    models, loosen SEVERITY_THRESHOLDS accordingly and re-run.
    """
    report = analyze_model(model, n_bits=n_bits)
    counts = {s.value: 0 for s in Severity}
    for f in report.findings:
        counts[f.severity.value] += 1
    return {
        "total_findings": len(report.findings),
        "by_severity": counts,
        "layers_analyzed": report.layers_analyzed,
        "layers_skipped": report.layers_skipped,
    }
