"""Multiple-comparisons correction shared across pipeline stages.

Stage 4 (many per-layer bit-level tests per report) and Stage 5 (many
candidate patches per report) both run far more hypothesis tests per
model than a raw per-test p-value cutoff can support -- see PHASE3.md's
"no multiple-comparisons correction" section and PHASE4.md Sec 4. Both
apply Benjamini-Hochberg FDR within one model's report, which is the
right unit here (rather than a single global alpha across unrelated
models).
"""

from __future__ import annotations


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values), in the same
    order as the input. Standard step-up procedure: sort ascending,
    q_(i) = p_(i) * m / rank, then enforce monotonicity via a running
    minimum from the largest rank down to the smallest."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        q = p_values[idx] * m / rank
        running_min = min(running_min, q)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted
