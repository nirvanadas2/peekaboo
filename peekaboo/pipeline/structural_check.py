"""Stage 2: Structural Consistency.

Two modes, selected by whether the caller supplies an `ArchitectureSpec`:

  - self_consistency (default, no spec): the primary mode. Peekaboo's real
    use case is scanning unknown models pulled from public hubs with no
    access to the original training pipeline, so this mode infers what a
    "normal" version of this model's structure should look like purely
    from the tensors themselves, and flags what doesn't fit:
      - matching dimensions between consecutive layers (conv/linear chain)
      - dtype uniformity, with outliers flagged
      - degenerate (zero-sized) tensor shapes
      - weight/bias shape cross-consistency
      - informational: parameter-count outliers, naming-pattern outliers
        (a best-effort stand-in for "no orphaned/unreferenced tensors" on
        formats with no real computation graph — see the docstring on
        `_check_naming_pattern_outliers` for why this is coarse by design)

  - spec_diff (caller supplies an ArchitectureSpec): a straightforward
    exact-match diff of actual vs. expected layer names/shapes/dtypes. No
    partial-spec handling, no spec versioning, no fuzzy/approximate
    matching — the caller is assumed to know the exact expected structure.

IMPORTANT: `StructuralReport` has no `hard_fail` concept. Severity here is
a triage label only, never a control-flow signal — the models Peekaboo
most needs to catch (backdoored, steganographic) are, by construction,
perfectly loadable and structurally normal. A structural anomaly must
never short-circuit the pipeline before the statistical/steganographic/
behavioral pillars (built in later phases) get to inspect the same
already-loaded tensors. Only Stage 1's `MetadataReport.hard_fail` gates
`run_pre_checks`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from peekaboo.loaders.common import LoadedModel
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import Finding, StructuralReport, compute_passed

_INTEGER_COUNTER_NAME_HINTS = ("num_batches_tracked",)

_LAYER_SUFFIX_RE = re.compile(r"\.(weight|bias|running_mean|running_var|num_batches_tracked)$")
_DIGIT_RE = re.compile(r"\d+")

# A tensor's num_params outlier ratio: an "outlier" tensor is flagged
# (informationally only) when it holds more than this many times the
# median tensor's parameter count. This is deliberately generous — real
# architectures very commonly have one dominant embedding/classifier layer.
_PARAM_COUNT_OUTLIER_RATIO = 1000


@dataclass
class LayerSpec:
    """Expected shape/dtype for one named tensor, for spec_diff mode."""

    shape: tuple[int, ...]
    dtype: str


@dataclass
class ArchitectureSpec:
    """An exact, complete expected architecture: tensor name -> LayerSpec.

    Deliberately simple by design (see module docstring): no partial-spec
    support, no versioning, no fuzzy/approximate matching. A layer missing
    from either side, or any shape/dtype difference, is a finding.
    """

    layers: dict[str, LayerSpec] = field(default_factory=dict)


def _shape_str(shape: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape) if shape else "scalar"


def _dtype_pair_severity(dtypes: set[str]) -> Severity:
    """Shared severity rule for a dtype disagreement, used by both modes:
    a float64 on either side of the disagreement is HIGH (a much wider
    float type than anything else in a typical fp16/fp32 model is a
    stronger anomaly signal), anything else is MEDIUM."""
    return Severity.HIGH if "float64" in dtypes else Severity.MEDIUM


def _classify_weight_kind(shape: tuple[int, ...]) -> str | None:
    """Classify a weight tensor's shape as a layer kind this module knows
    how to reason about dimensionally. Norm-layer affine params, biases,
    embeddings' 1-D buffers, etc. are ndim 1 (or 0) and return None — they
    aren't part of the dimension chain, though biases are still
    cross-checked against their weight's output dimension separately."""
    if len(shape) == 2:
        return "linear"
    if len(shape) == 4:
        return "conv2d"
    return None


def _dims_for(kind: str, shape: tuple[int, ...]) -> tuple[int, int]:
    """Return (in_dim, out_dim) for a classified weight shape."""
    if kind == "linear":
        out_dim, in_dim = shape
    else:  # conv2d: (out_channels, in_channels, kh, kw)
        out_dim, in_dim = shape[0], shape[1]
    return in_dim, out_dim


def _output_satisfies_input(producer_kind: str, producer_out: int, consumer_kind: str, consumer_in: int) -> bool:
    """Could `producer`'s output plausibly feed `consumer`'s input?

    - same kind (conv-conv or linear-linear): dimensions must match
      exactly — there's no legitimate reshape between two layers of the
      same kind in a direct chain.
    - conv2d -> linear: `consumer_in` must be an integer multiple of
      `producer_out` — this is the real invariant behind a flatten
      operation (in_features = out_channels * spatial_h * spatial_w for
      some positive integer spatial size), not blanket leniency.
    - linear -> conv2d: rare (e.g. project-then-reshape-into-conv) and not
      generically verifiable without knowing the target spatial shape;
      treated as inconclusive (satisfied) rather than risking a false
      positive on a legitimate but unusual pattern.
    """
    if producer_kind == consumer_kind:
        return producer_out == consumer_in
    if producer_kind == "conv2d" and consumer_kind == "linear":
        return producer_out > 0 and consumer_in % producer_out == 0
    if producer_kind == "linear" and consumer_kind == "conv2d":
        return True
    return False


def _check_degenerate_shapes(model: LoadedModel, findings: list[Finding]) -> None:
    degenerate = sorted(name for name, t in model.tensors.items() if 0 in t.shape)
    if degenerate:
        # CRITICAL is a severity/triage label only here — see module
        # docstring. It does not stop anything or skip later analysis.
        findings.append(
            Finding(
                check="no_degenerate_shapes",
                severity=Severity.CRITICAL,
                passed=False,
                message=f"{len(degenerate)} tensor(s) have a zero-sized dimension: {degenerate}",
                details={"tensors": degenerate},
            )
        )
    else:
        findings.append(
            Finding(
                check="no_degenerate_shapes",
                severity=Severity.INFO,
                passed=True,
                message="No tensors with a zero-sized dimension",
            )
        )


def _check_dtype_uniformity(model: LoadedModel, findings: list[Finding]) -> None:
    relevant = {
        name: t
        for name, t in model.tensors.items()
        if not any(hint in name for hint in _INTEGER_COUNTER_NAME_HINTS)
    }
    if not relevant:
        findings.append(
            Finding(
                check="dtype_uniformity",
                severity=Severity.INFO,
                passed=True,
                message="No non-counter tensors to check",
            )
        )
        return

    dtype_counts = Counter(t.dtype for t in relevant.values())
    majority_dtype, _ = dtype_counts.most_common(1)[0]
    outliers = {name: t.dtype for name, t in relevant.items() if t.dtype != majority_dtype}

    if outliers:
        severity = _dtype_pair_severity(set(outliers.values()) | {majority_dtype})
        findings.append(
            Finding(
                check="dtype_uniformity",
                severity=severity,
                passed=False,
                message=(
                    f"{len(outliers)} tensor(s) deviate from the majority dtype "
                    f"'{majority_dtype}': {outliers}"
                ),
                details={"majority_dtype": majority_dtype, "outliers": outliers},
            )
        )
    else:
        findings.append(
            Finding(
                check="dtype_uniformity",
                severity=Severity.INFO,
                passed=True,
                message=f"All tensors share dtype '{majority_dtype}'",
            )
        )


def _check_consecutive_dimension_chain(model: LoadedModel, findings: list[Finding]) -> None:
    layers: list[tuple[str, str, int, int]] = []  # (name, kind, in_dim, out_dim)
    for name, tensor in model.tensors.items():
        if not name.endswith(".weight"):
            continue
        kind = _classify_weight_kind(tensor.shape)
        if kind is None:
            continue
        in_dim, out_dim = _dims_for(kind, tensor.shape)
        layers.append((name, kind, in_dim, out_dim))

    if len(layers) < 2:
        findings.append(
            Finding(
                check="consecutive_dimension_chain",
                severity=Severity.INFO,
                passed=True,
                message="Fewer than two linear/conv2d weight layers found; nothing to chain-check",
            )
        )
        return

    mismatches: dict[str, str] = {}
    for i in range(1, len(layers)):
        name, kind, in_dim, out_dim = layers[i]
        # Checked against every prior layer's output, not just the
        # immediately preceding one: this is what lets a branching/skip
        # architecture (two layers both consuming an earlier layer's
        # output) pass without a false positive, while a layer whose
        # input matches nothing anyone has produced is still caught.
        satisfied = any(
            _output_satisfies_input(prev_kind, prev_out, kind, in_dim) for _, prev_kind, _, prev_out in layers[:i]
        )
        if not satisfied:
            prior_desc = ", ".join(f"{n} ({k}, out={o})" for n, k, _, o in layers[:i])
            mismatches[name] = (
                f"'{name}' ({kind}, expects input dim {in_dim}) does not match any prior "
                f"layer's output dimension. Prior layers: {prior_desc}"
            )

    if mismatches:
        findings.append(
            Finding(
                check="consecutive_dimension_chain",
                severity=Severity.HIGH,
                passed=False,
                message="; ".join(mismatches.values()),
                details={"mismatches": mismatches},
            )
        )
    else:
        findings.append(
            Finding(
                check="consecutive_dimension_chain",
                severity=Severity.INFO,
                passed=True,
                message=f"All {len(layers)} linear/conv2d layers chain consistently",
            )
        )


def _check_weight_bias_shape_consistency(model: LoadedModel, findings: list[Finding]) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    checked = 0
    for name, tensor in model.tensors.items():
        if not name.endswith(".weight"):
            continue
        kind = _classify_weight_kind(tensor.shape)
        if kind is None:
            continue
        bias_name = name[: -len(".weight")] + ".bias"
        bias = model.tensors.get(bias_name)
        if bias is None:
            continue
        checked += 1
        _, out_dim = _dims_for(kind, tensor.shape)
        if tuple(bias.shape) != (out_dim,):
            mismatches[name] = {"expected_bias_shape": [out_dim], "actual_bias_shape": list(bias.shape)}

    if checked == 0:
        findings.append(
            Finding(
                check="weight_bias_shape_consistency",
                severity=Severity.INFO,
                passed=True,
                message="No weight/bias pairs found to cross-check",
            )
        )
    elif mismatches:
        findings.append(
            Finding(
                check="weight_bias_shape_consistency",
                severity=Severity.HIGH,
                passed=False,
                message=f"{len(mismatches)} weight/bias pair(s) have inconsistent shapes: {mismatches}",
                details={"mismatches": mismatches},
            )
        )
    else:
        findings.append(
            Finding(
                check="weight_bias_shape_consistency",
                severity=Severity.INFO,
                passed=True,
                message=f"All {checked} weight/bias pairs are shape-consistent",
            )
        )


def _check_param_count_profile(model: LoadedModel, findings: list[Finding]) -> None:
    """Purely informational (always passed=True): reports the total/median
    parameter footprint and calls out any tensor that dominates the model
    by a wide margin. Never fails the check — real architectures very
    commonly have one legitimately huge layer (an embedding table, a
    vocab-projection head), so treating size skew as a failure would be a
    reliable source of false positives on valid models."""
    if not model.tensors:
        return

    counts = sorted(t.num_params for t in model.tensors.values())
    mid = len(counts) // 2
    median = counts[mid] if len(counts) % 2 == 1 else (counts[mid - 1] + counts[mid]) / 2

    outliers = {}
    if median > 0:
        outliers = {
            name: t.num_params
            for name, t in model.tensors.items()
            if t.num_params > median * _PARAM_COUNT_OUTLIER_RATIO
        }

    message = f"Total params: {model.total_params()}; median tensor size: {median:.0f}"
    if outliers:
        message += (
            f"; {len(outliers)} tensor(s) are >{_PARAM_COUNT_OUTLIER_RATIO}x the median "
            f"(commonly legitimate for embeddings/classifier heads): {list(outliers)}"
        )

    findings.append(
        Finding(
            check="param_count_profile",
            severity=Severity.INFO,
            passed=True,
            message=message,
            details={"total_params": model.total_params(), "median_tensor_params": median, "large_outliers": outliers},
        )
    )


def _layer_base_name(tensor_name: str) -> str:
    return _LAYER_SUFFIX_RE.sub("", tensor_name)


def _naming_template(base_name: str) -> str:
    return _DIGIT_RE.sub("#", base_name)


def _check_naming_pattern_outliers(model: LoadedModel, findings: list[Finding]) -> None:
    """Best-effort, name-only stand-in for "no orphaned/unreferenced
    tensors" on formats with no real computation graph (safetensors,
    pickle state dicts). Clusters layer base names by a digit-normalized
    template and flags names that don't fit the dominant template.

    This is deliberately coarse and over-inclusive, and always INFO /
    passed=True: a model's one-off head/classifier layer will typically
    show up here alongside any genuinely stray/injected tensor, since both
    are "singletons" relative to a repeating block pattern. It's report-
    only groundwork for a human or a later pillar to triage — better to
    surface too much than silently miss a real smuggled tensor. (ONNX's
    actual computation graph would support a more precise, ground-truth
    orphan check; out of scope here since this check does not gate
    anything regardless.)
    """
    base_names = sorted({_layer_base_name(name) for name in model.tensors})
    if len(base_names) < 3:
        findings.append(
            Finding(
                check="naming_pattern_outliers",
                severity=Severity.INFO,
                passed=True,
                message="Fewer than three distinct layers; not enough to profile a naming pattern",
            )
        )
        return

    template_counts = Counter(_naming_template(b) for b in base_names)
    dominant_template, dominant_count = template_counts.most_common(1)[0]

    if dominant_count < 2:
        findings.append(
            Finding(
                check="naming_pattern_outliers",
                severity=Severity.INFO,
                passed=True,
                message="No repeating layer-naming pattern detected; nothing to compare against",
            )
        )
        return

    outliers = sorted(b for b in base_names if _naming_template(b) != dominant_template)
    if outliers:
        findings.append(
            Finding(
                check="naming_pattern_outliers",
                severity=Severity.INFO,
                passed=True,
                message=(
                    f"{len(outliers)} layer name(s) don't fit the dominant naming pattern "
                    f"'{dominant_template}' (informational only — commonly includes a legitimate "
                    f"one-off layer like a final classifier head, alongside any genuinely stray "
                    f"tensor): {outliers}"
                ),
                details={"outliers": outliers, "dominant_template": dominant_template},
            )
        )
    else:
        findings.append(
            Finding(
                check="naming_pattern_outliers",
                severity=Severity.INFO,
                passed=True,
                message=f"All layer names fit the dominant naming pattern '{dominant_template}'",
            )
        )


def _run_self_consistency_checks(model: LoadedModel, findings: list[Finding]) -> None:
    _check_degenerate_shapes(model, findings)
    _check_dtype_uniformity(model, findings)
    _check_consecutive_dimension_chain(model, findings)
    _check_weight_bias_shape_consistency(model, findings)
    _check_param_count_profile(model, findings)
    _check_naming_pattern_outliers(model, findings)


def _run_spec_diff_checks(model: LoadedModel, spec: ArchitectureSpec, findings: list[Finding]) -> None:
    spec_names = set(spec.layers.keys())
    model_names = set(model.tensors.keys())

    missing = sorted(spec_names - model_names)
    extra = sorted(model_names - spec_names)

    if missing:
        findings.append(
            Finding(
                check="spec_missing_layers",
                severity=Severity.HIGH,
                passed=False,
                message=f"{len(missing)} layer(s) declared in the spec are missing from the model: {missing}",
                details={"missing_layers": missing},
            )
        )
    else:
        findings.append(
            Finding(
                check="spec_missing_layers",
                severity=Severity.INFO,
                passed=True,
                message="All spec-declared layers are present in the model",
            )
        )

    if extra:
        findings.append(
            Finding(
                check="spec_extra_layers",
                severity=Severity.MEDIUM,
                passed=False,
                message=f"{len(extra)} layer(s) in the model are not declared in the spec: {extra}",
                details={"extra_layers": extra},
            )
        )
    else:
        findings.append(
            Finding(
                check="spec_extra_layers",
                severity=Severity.INFO,
                passed=True,
                message="No undeclared extra layers in the model",
            )
        )

    shape_mismatches: dict[str, dict[str, str]] = {}
    dtype_mismatches: dict[str, dict[str, str]] = {}
    for name in sorted(spec_names & model_names):
        expected = spec.layers[name]
        actual = model.tensors[name]
        if tuple(actual.shape) != tuple(expected.shape):
            shape_mismatches[name] = {
                "expected": _shape_str(expected.shape),
                "actual": _shape_str(actual.shape),
            }
        if actual.dtype != expected.dtype:
            dtype_mismatches[name] = {"expected": expected.dtype, "actual": actual.dtype}

    if shape_mismatches:
        findings.append(
            Finding(
                check="spec_shape_match",
                severity=Severity.HIGH,
                passed=False,
                message=f"{len(shape_mismatches)} layer(s) have shapes differing from the spec: {shape_mismatches}",
                details={"mismatches": shape_mismatches},
            )
        )
    else:
        findings.append(
            Finding(
                check="spec_shape_match",
                severity=Severity.INFO,
                passed=True,
                message="All shared layers match the spec's declared shapes",
            )
        )

    if dtype_mismatches:
        all_dtypes = {v for pair in dtype_mismatches.values() for v in pair.values()}
        severity = _dtype_pair_severity(all_dtypes)
        findings.append(
            Finding(
                check="spec_dtype_match",
                severity=severity,
                passed=False,
                message=f"{len(dtype_mismatches)} layer(s) have dtypes differing from the spec: {dtype_mismatches}",
                details={"mismatches": dtype_mismatches},
            )
        )
    else:
        findings.append(
            Finding(
                check="spec_dtype_match",
                severity=Severity.INFO,
                passed=True,
                message="All shared layers match the spec's declared dtypes",
            )
        )


def run_structural_check(model: LoadedModel, spec: ArchitectureSpec | None = None) -> StructuralReport:
    """Run Stage 2 (Structural Consistency) checks on an already-loaded
    model. Defaults to self-consistency inference; pass `spec` to run an
    exact-match diff against a known architecture instead."""
    findings: list[Finding] = []
    metadata: dict[str, Any] = {
        "num_tensors": len(model),
        "total_params": model.total_params(),
    }

    if spec is not None:
        mode = "spec_diff"
        _run_spec_diff_checks(model, spec, findings)
    else:
        mode = "self_consistency"
        _run_self_consistency_checks(model, findings)

    return StructuralReport(
        model_path=model.source_path,
        mode=mode,
        passed=compute_passed(findings),
        findings=findings,
        metadata=metadata,
    )
