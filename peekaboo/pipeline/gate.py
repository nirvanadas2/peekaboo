"""The Phase 1 pipeline gate: wires Stage 1 (Metadata Integrity) and Stage 2
(Structural Consistency) into a single entry point.

Control flow is intentionally simple and linear:

    metadata_report = run_metadata_check(model_path)
    if metadata_report.hard_fail:
        return <metadata only>          # no load_model(), no structural check
    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec)
    return <both>

Only `MetadataReport.hard_fail` gates the short-circuit. Stage 2 has no
equivalent concept (see structural_check.py's module docstring) — once
Stage 1 doesn't hard-fail, Stage 2 always runs to completion and reports
its findings, and would never itself block a future Stage 3+ from running
on the same already-loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peekaboo.loaders import LoadedModel, load_model
from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.pipeline.structural_check import ArchitectureSpec, run_structural_check
from peekaboo.schema.reports import MetadataReport, StructuralReport


@dataclass
class PreCheckResult:
    """Everything a caller — including later Peekaboo phases — needs from
    Phase 1, without having to know that it's internally split into two
    stages.

    `structural` and `loaded_model` are `None` exactly when Stage 1
    hard-failed and the pipeline stopped before loading any tensors.
    Otherwise `loaded_model` is handed back so a caller (e.g. Phase 2's
    statistical analysis) can reuse the tensors Stage 2 already loaded
    from disk, instead of loading the file a second time.
    """

    metadata: MetadataReport
    structural: StructuralReport | None
    loaded_model: LoadedModel | None

    @property
    def stopped_at_metadata(self) -> bool:
        """True if Stage 1 hard-failed and the pipeline short-circuited
        before Stage 2 (equivalently: before `loaded_model`/`structural`
        exist)."""
        return self.structural is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_at_metadata": self.stopped_at_metadata,
            "metadata": self.metadata.to_dict(),
            "structural": self.structural.to_dict() if self.structural is not None else None,
        }


def run_pre_checks(model_path: str, spec: ArchitectureSpec | None = None) -> PreCheckResult:
    """Run Stage 1 then Stage 2 in sequence.

    Short-circuits before Stage 2 — no `load_model()` call, no structural
    check — if Stage 1 hard-fails (unsafe pickle, corrupt/mismatched file).
    Defaults Stage 2 to self-consistency mode; pass `spec` to run an
    exact-match diff against a known architecture instead.
    """
    metadata_report = run_metadata_check(model_path)

    if metadata_report.hard_fail:
        return PreCheckResult(metadata=metadata_report, structural=None, loaded_model=None)

    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec=spec)

    return PreCheckResult(metadata=metadata_report, structural=structural_report, loaded_model=loaded_model)
