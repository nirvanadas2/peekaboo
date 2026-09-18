"""Data classes for the eventual Model Risk Score output.

These are stubs: Phase 0 only defines the shape of the output. No scoring
logic exists yet — detectors in later phases will populate these classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass
class PillarScore:
    """Sub-score for one detection pillar (statistical, steganographic, behavioral)."""

    name: str
    score: float
    weight: float = 1.0
    summary: str = ""
    flags: list[LayerFlag] = field(default_factory=list)


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
