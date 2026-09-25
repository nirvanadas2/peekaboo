"""Data classes for the Model Risk Score output.

Phase 0 defined the shape; Stage 6 (`peekaboo.pipeline.fusion`) populates
it. One deliberate Phase-0 interface change (approved): `PillarScore`
carries a `status` and an Optional `score`, so "this pillar could not
run" (score None) is never numerically identical to "ran and found
nothing" (score 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LayerFlag:
    """A single finding attached to one layer/tensor within a model."""

    layer_name: str
    flag_type: str
    severity: Severity
    score: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class PillarStatus(str, Enum):
    """Whether a pillar contributed evidence at all.

    NOT_RUN: the stage didn't run or couldn't (e.g. Stage 5 without a
    forward_fn). Its score is None and it is excluded from the overall
    score -- absence of evidence, not evidence of absence.
    RAN_CLEAN: ran; no finding survived to MEDIUM+ (score 0.0).
    FLAGGED: ran; at least one MEDIUM+ finding (score > 0).
    """

    NOT_RUN = "not_run"
    RAN_CLEAN = "ran_clean"
    FLAGGED = "flagged"


@dataclass
class PillarScore:
    """Sub-score for one detection pillar (statistical, steganographic, behavioral)."""

    name: str
    score: Optional[float]
    weight: float = 1.0
    summary: str = ""
    flags: list[LayerFlag] = field(default_factory=list)
    status: PillarStatus = PillarStatus.RAN_CLEAN

    def __post_init__(self) -> None:
        if (self.status == PillarStatus.NOT_RUN) != (self.score is None):
            raise ValueError("score must be None exactly when status is NOT_RUN")


@dataclass
class Explanation:
    """Placeholder for a SHAP-style human-readable explanation of the score."""

    text: str = ""
    feature_attributions: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelRiskScore:
    """Top-level output of a Peekaboo scan for a single model."""

    model_path: str
    overall_score: float
    statistical: PillarScore
    steganographic: PillarScore
    behavioral: PillarScore
    layer_flags: list[LayerFlag] = field(default_factory=list)
    explanation: Explanation = field(default_factory=Explanation)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def pillar_dict(p: PillarScore) -> dict[str, Any]:
            return {
                "name": p.name,
                "status": p.status.value,
                "score": p.score,
                "weight": p.weight,
                "summary": p.summary,
                "flags": [flag_dict(f) for f in p.flags],
            }

        def flag_dict(f: LayerFlag) -> dict[str, Any]:
            return {
                "layer_name": f.layer_name,
                "flag_type": f.flag_type,
                "severity": f.severity.value,
                "score": f.score,
                "message": f.message,
                "details": f.details,
            }

        return {
            "model_path": self.model_path,
            "overall_score": self.overall_score,
            "pillars": {
                "statistical": pillar_dict(self.statistical),
                "steganographic": pillar_dict(self.steganographic),
                "behavioral": pillar_dict(self.behavioral),
            },
            "layer_flags": [flag_dict(f) for f in self.layer_flags],
            "explanation": {
                "text": self.explanation.text,
                "feature_attributions": self.explanation.feature_attributions,
            },
            "metadata": self.metadata,
        }
