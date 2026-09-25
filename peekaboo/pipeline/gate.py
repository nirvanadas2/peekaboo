"""The pipeline gate: wires Stages 1-6 into a single entry point.

Control flow is intentionally simple and linear:

    metadata_report = run_metadata_check(model_path)
    if metadata_report.hard_fail:
        return <metadata only>          # nothing loaded, nothing else runs, no risk score
    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec)       # Stage 2
    statistical_report = run_statistical_check(loaded_model)           # Stage 3
    stego_report = analyze_model(loaded_model)                         # Stage 4 (BH-corrected)
    behavioral_report = run_behavioral_check(loaded_model, forward_fn) # Stage 5
    risk_score = fuse(statistical, stego, behavioral)                  # Stage 6
    return <all of it>

Only `MetadataReport.hard_fail` gates the short-circuit. No later stage
has an equivalent concept -- once Stage 1 doesn't hard-fail, every stage
runs to completion and reports findings as triage labels only.

Stage 4 runs by default: it is a single cheap pass (~10 ms on the
benchmark), like Stages 1-3, and now carries per-report FDR correction
(PHASE3.md). Stage 5 follows the agreed "option (b)" (PHASE4.md/
PHASE5.md): it probes only when the caller supplies a `forward_fn` (plus
`input_shape`/`num_classes`); otherwise it returns its explicit
`mode="not_runnable"` report -- zero forward passes -- so Stage 6 always
sees "not assessed" rather than silence. Probing costs many forward
passes; the caller opts in by supplying the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from peekaboo.loaders import LoadedModel, load_model
from peekaboo.pipeline.behavioral_probe import run_behavioral_check
from peekaboo.pipeline.fusion import fuse
from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.pipeline.statistical_check import run_statistical_check
from peekaboo.pipeline.stego_check import analyze_model
from peekaboo.pipeline.structural_check import ArchitectureSpec, run_structural_check
from peekaboo.schema.model_risk_score import ModelRiskScore
from peekaboo.schema.reports import (
    BehavioralReport,
    MetadataReport,
    StatisticalReport,
    StegoReport,
    StructuralReport,
)


@dataclass
class PreCheckResult:
    """Everything a caller needs from one scan, without having to know how
    it's split into stages.

    Every field but `metadata` is `None` exactly when Stage 1 hard-failed
    and the pipeline stopped before loading any tensors. Otherwise
    `loaded_model` is handed back so a caller can reuse the tensors
    already loaded from disk.
    """

    metadata: MetadataReport
    structural: StructuralReport | None
    statistical: StatisticalReport | None
    loaded_model: LoadedModel | None
    stego: StegoReport | None = None
    behavioral: BehavioralReport | None = None
    risk_score: ModelRiskScore | None = None

    @property
    def stopped_at_metadata(self) -> bool:
        """True if Stage 1 hard-failed and the pipeline short-circuited
        before any later stage (equivalently: before `loaded_model`
        exists)."""
        return self.structural is None

    def to_dict(self) -> dict[str, Any]:
        def maybe(report):
            return report.to_dict() if report is not None else None

        return {
            "stopped_at_metadata": self.stopped_at_metadata,
            "metadata": self.metadata.to_dict(),
            "structural": maybe(self.structural),
            "statistical": maybe(self.statistical),
            "stego": maybe(self.stego),
            "behavioral": maybe(self.behavioral),
            "risk_score": maybe(self.risk_score),
        }


def run_pre_checks(
    model_path: str,
    spec: ArchitectureSpec | None = None,
    *,
    forward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    input_shape: Optional[tuple[int, ...]] = None,
    num_classes: Optional[int] = None,
    behavioral_kwargs: Optional[dict[str, Any]] = None,
) -> PreCheckResult:
    """Run Stages 1-6 in sequence.

    Short-circuits after Stage 1 -- no `load_model()` call, nothing else
    run, no risk score -- if Stage 1 hard-fails (unsafe pickle,
    corrupt/mismatched file). Defaults Stage 2 to self-consistency mode;
    pass `spec` for an exact-match diff against a known architecture.

    Stage 5 probes only if `forward_fn` is given (then `input_shape` and
    `num_classes` are required; `behavioral_kwargs` are forwarded to
    `run_behavioral_check`, e.g. a probe budget). Without it, Stage 5
    reports `not_runnable` and the risk score marks the behavioral
    pillar NOT_RUN.
    """
    metadata_report = run_metadata_check(model_path)

    if metadata_report.hard_fail:
        return PreCheckResult(
            metadata=metadata_report, structural=None, statistical=None, loaded_model=None
        )

    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec=spec)
    statistical_report = run_statistical_check(loaded_model)
    stego_report = analyze_model(loaded_model)
    behavioral_report = run_behavioral_check(
        loaded_model,
        forward_fn,
        input_shape=input_shape,
        num_classes=num_classes,
        **(behavioral_kwargs or {}),
    )
    risk_score = fuse(
        model_path,
        statistical=statistical_report,
        stego=stego_report,
        behavioral=behavioral_report,
    )

    return PreCheckResult(
        metadata=metadata_report,
        structural=structural_report,
        statistical=statistical_report,
        loaded_model=loaded_model,
        stego=stego_report,
        behavioral=behavioral_report,
        risk_score=risk_score,
    )
