"""The Phase 1-2 pipeline gate: wires Stage 1 (Metadata Integrity), Stage 2
(Structural Consistency), and Stage 3 (Statistical Analysis) into a single
entry point.

Control flow is intentionally simple and linear:

    metadata_report = run_metadata_check(model_path)
    if metadata_report.hard_fail:
        return <metadata only>          # no load_model(), no structural/statistical checks
    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec)
    statistical_report = run_statistical_check(loaded_model)
    return <all three>

Only `MetadataReport.hard_fail` gates the short-circuit. Neither Stage 2
nor Stage 3 has an equivalent concept (see structural_check.py's and
statistical_check.py's module docstrings) — once Stage 1 doesn't
hard-fail, Stages 2 and 3 always run to completion and report their
findings as triage labels only, and neither would ever itself block a
future Stage 4+ from running on the same already-loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peekaboo.loaders import LoadedModel, load_model
from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.pipeline.statistical_check import run_statistical_check
from peekaboo.pipeline.structural_check import ArchitectureSpec, run_structural_check
from peekaboo.schema.reports import MetadataReport, StatisticalReport, StructuralReport


@dataclass
class PreCheckResult:
    """Everything a caller — including later Peekaboo phases — needs from
    Phases 1-2, without having to know that it's internally split into
    three stages.

    `structural`, `statistical`, and `loaded_model` are all `None` exactly
    when Stage 1 hard-failed and the pipeline stopped before loading any
    tensors. Otherwise `loaded_model` is handed back so a caller (e.g. a
    later phase) can reuse the tensors Stage 2/3 already loaded from disk,
    instead of loading the file a second time.
    """

    metadata: MetadataReport
    structural: StructuralReport | None
    statistical: StatisticalReport | None
    loaded_model: LoadedModel | None

    @property
    def stopped_at_metadata(self) -> bool:
        """True if Stage 1 hard-failed and the pipeline short-circuited
        before Stage 2/3 (equivalently: before `loaded_model`/`structural`/
        `statistical` exist)."""
        return self.structural is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_at_metadata": self.stopped_at_metadata,
            "metadata": self.metadata.to_dict(),
            "structural": self.structural.to_dict() if self.structural is not None else None,
            "statistical": self.statistical.to_dict() if self.statistical is not None else None,
        }


def run_pre_checks(model_path: str, spec: ArchitectureSpec | None = None) -> PreCheckResult:
    """Run Stage 1, then Stage 2, then Stage 3 in sequence.

    Short-circuits before Stage 2/3 — no `load_model()` call, no
    structural or statistical check — if Stage 1 hard-fails (unsafe
    pickle, corrupt/mismatched file). Defaults Stage 2 to self-consistency
    mode; pass `spec` to run an exact-match diff against a known
    architecture instead. Stage 3 takes no spec — it only ever looks at
    the loaded tensors' values, not the architecture.
    """
    metadata_report = run_metadata_check(model_path)

    if metadata_report.hard_fail:
        return PreCheckResult(
            metadata=metadata_report, structural=None, statistical=None, loaded_model=None
        )

    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec=spec)
    statistical_report = run_statistical_check(loaded_model)

    return PreCheckResult(
        metadata=metadata_report,
        structural=structural_report,
        statistical=statistical_report,
        loaded_model=loaded_model,
    )
