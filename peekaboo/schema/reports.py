"""Data classes for the Phase 1/2 pipeline reports: Metadata Integrity
(Stage 1), Structural Consistency (Stage 2), and Statistical Analysis
(Stage 3).

All stages share the same pass/fail + findings pattern via `Finding`,
so downstream code (and the pipeline gate) can treat any of these
reports uniformly when deciding what to surface to a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peekaboo.schema.model_risk_score import Severity


@dataclass
class Finding:
    """One check's outcome, whether it passed or failed.

    Every check performed by a stage appends a Finding, including passing
    ones — this keeps the report a complete audit trail of what was
    checked, not just a list of problems.
    """

    check: str
    severity: Severity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


def compute_passed(findings: list[Finding]) -> bool:
    """A report passes only if every finding it recorded passed."""
    return all(f.passed for f in findings)


def compute_hard_fail(findings: list[Finding]) -> bool:
    """A finding is a hard-gate violation when it failed at CRITICAL
    severity — i.e. the file is unsafe or too malformed to trust for any
    further processing (unsafe pickle, corrupt header, mismatched
    extension, unparseable model)."""
    return any((not f.passed) and f.severity == Severity.CRITICAL for f in findings)


@dataclass
class MetadataReport:
    """Output of Stage 1 (Metadata Integrity) for a single model file."""

    model_path: str
    declared_format: str
    detected_format: str
    file_size: int
    file_hash: str
    passed: bool
    hard_fail: bool
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "declared_format": self.declared_format,
            "detected_format": self.detected_format,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "passed": self.passed,
            "hard_fail": self.hard_fail,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class StructuralReport:
    """Output of Stage 2 (Structural Consistency) for a single model file."""

    model_path: str
    mode: str  # "self_consistency" | "spec_diff"
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "mode": self.mode,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class StatisticalReport:
    """Output of Stage 3 (Statistical Analysis) for a single model file.

    Same principle as StructuralReport: no `hard_fail` field. This stage
    never gates later stages — severity here is a triage label only.
    """

    model_path: str
    mode: str  # "relative_outlier" | "absolute_fallback"
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "mode": self.mode,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }
