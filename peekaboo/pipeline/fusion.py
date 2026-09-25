"""
peekaboo/pipeline/fusion.py

Stage 6: Anomaly Fusion -> ModelRiskScore
===========================================

Design agreed at the PHASE5.md design check-in (read it for the
rationale, including why the originally-briefed Isolation Forest was
NOT built):

- EVIDENCE FUSION, no learned model. Stages 4 and 5 already emit
  Benjamini-Hochberg q-values, i.e. multiplicity-corrected evidence; an
  unsupervised anomaly model on top would discard that calibration,
  and the per-model layer population (7-30 tensors) is too small for it.
  There are no fitted parameters here, so evaluating on the committed
  benchmark fixtures is not circular.

- Pillar score, from the pillar's strongest MEDIUM+ finding:
      evidence_score(q) = 0.4 + 0.6 * clip(log10(0.05/q) / log10(0.05/1e-10), 0, 1)
  i.e. 0.4 at q = 0.05 (the MEDIUM cutoff), rising monotonically to 1.0
  at q <= 1e-10. A pillar that ran with no MEDIUM+ finding scores 0.0.

- Stage 3 (statistical) has no q-values -- its robust z-scores aren't
  p-values, and converting them would manufacture false precision -- and
  it detects nothing on the benchmark (PHASE2.md). It contributes via its
  own severity labels only, CAPPED: any MEDIUM+ Stage 3 finding scores
  exactly the floor of the flagged range (0.4) and its severity is
  capped at MEDIUM, so Stage 3 alone can never raise a model above
  MEDIUM. Not retrofitted in any other way (PHASE3.md's standing rule).

- Overall score = MAX over pillars that ran. The pillars cover disjoint
  threats on this benchmark (Stage 4: noise; Stage 5: backdoor), so the
  union is the right combination -- averaging would dilute each. Any
  "combined beats single" ablation result is therefore COVERAGE (the
  union of what each pillar catches), not synergy; report it as such.

- "No findings" vs "not run": PillarStatus.NOT_RUN has score None and is
  excluded from the max; RAN_CLEAN is 0.0. `metadata["pillars_run"]`
  lists what actually contributed. A behavioral report with
  mode="not_runnable" is NOT_RUN, never "clean".

- Explanation: exact, not SHAP -- overall = max, so the attribution is
  each pillar's own score, with the argmax pillar and its top findings
  named in the text.

- Uncorrected Stage 4 reports (analyze_model(..., fdr_correction=False))
  are rejected: their severities are per-test labels, not evidence.
"""

from __future__ import annotations

import math
from typing import Optional

from peekaboo.schema.model_risk_score import (
    Explanation,
    LayerFlag,
    ModelRiskScore,
    PillarScore,
    PillarStatus,
    Severity,
)
from peekaboo.schema.reports import BehavioralReport, Finding, StatisticalReport, StegoReport

_Q_FLAGGED = 0.05  # MEDIUM cutoff used by Stages 4 and 5
_Q_SATURATE = 1e-10
_FLAGGED_FLOOR = 0.4
_STAGE3_SEVERITY_CAP = Severity.MEDIUM
_SEVERITY_ORDER = list(Severity)
_MEDIUM_PLUS = (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
_BEHAVIORAL_CHECKS = ("behavioral_trigger_island", "behavioral_class_asymmetry")
MODEL_LEVEL = "(model behavior)"


def evidence_score(q: float) -> float:
    """0.4 at q=0.05, monotone up to 1.0 at q<=1e-10."""
    q = max(q, 1e-300)
    span = math.log10(_Q_FLAGGED / _Q_SATURATE)
    frac = min(1.0, max(0.0, math.log10(_Q_FLAGGED / q) / span))
    return _FLAGGED_FLOOR + (1.0 - _FLAGGED_FLOOR) * frac


def _max_severity(severities) -> Severity:
    return max(severities, key=_SEVERITY_ORDER.index, default=Severity.INFO)


def _cap(severity: Severity, cap: Severity) -> Severity:
    return severity if _SEVERITY_ORDER.index(severity) <= _SEVERITY_ORDER.index(cap) else cap


def _not_run(name: str, reason: str) -> PillarScore:
    return PillarScore(name=name, score=None, status=PillarStatus.NOT_RUN, summary=reason)


def _pillar(name: str, flags: list[LayerFlag], clean_summary: str) -> PillarScore:
    if not flags:
        return PillarScore(name=name, score=0.0, status=PillarStatus.RAN_CLEAN, summary=clean_summary)
    top = max(flags, key=lambda f: f.score)
    return PillarScore(
        name=name,
        score=top.score,
        status=PillarStatus.FLAGGED,
        summary=f"{len(flags)} MEDIUM+ flag(s); strongest: {top.layer_name} {top.flag_type} ({top.severity.value})",
        flags=sorted(flags, key=lambda f: -f.score),
    )


def _statistical_pillar(report: Optional[StatisticalReport]) -> PillarScore:
    if report is None:
        return _not_run("statistical", "Stage 3 did not run (Stage 1 hard-failed or not requested).")
    flags: list[LayerFlag] = []
    for f in report.findings:
        if f.severity not in _MEDIUM_PLUS:
            continue
        for layer, info in f.details.get("outliers", {}).items():
            original = Severity(info.get("severity", f.severity.value))
            flags.append(
                LayerFlag(
                    layer_name=layer,
                    flag_type=f.check,
                    severity=_cap(original, _STAGE3_SEVERITY_CAP),
                    score=_FLAGGED_FLOOR,
                    message=f"{f.check} ({f.details.get('mode')}); capped contribution (PHASE5.md)",
                    details={**info, "uncapped_severity": original.value, "mode": f.details.get("mode")},
                )
            )
    return _pillar("statistical", flags, "No MEDIUM+ statistical outliers.")


def _stego_pillar(report: Optional[StegoReport]) -> PillarScore:
    if report is None:
        return _not_run("steganographic", "Stage 4 did not run.")
    if report.metadata.get("fdr_correction") is None:
        raise ValueError(
            "Stage 4 report is not FDR-corrected (analyze_model(..., fdr_correction=False)); "
            "uncorrected per-test labels are not fusable evidence (PHASE3.md)."
        )
    flags = [
        LayerFlag(
            layer_name=f.details["layer_name"],
            flag_type=f.check,
            severity=f.severity,
            score=evidence_score(f.details["fdr_p_value"]),
            message=f.message,
            details={k: f.details[k] for k in ("statistic", "p_value", "fdr_p_value")},
        )
        for f in report.findings
        if f.severity in _MEDIUM_PLUS and f.details.get("fdr_p_value") is not None
    ]
    return _pillar("steganographic", flags, "No bit-level test survived FDR correction at MEDIUM+.")


def _behavioral_flag(f: Finding) -> LayerFlag:
    d = f.details
    if f.check == "behavioral_class_asymmetry":
        label = f"class {d['over_represented_class']} over-reached at +{d['color']:g}sigma"
    else:
        label = f"island at {d['position']} +{d['color']:g}sigma -> class {d['forced_class']}"
    return LayerFlag(
        layer_name=MODEL_LEVEL,
        flag_type=f.check,
        severity=f.severity,
        score=evidence_score(d["fdr_p_value"]),
        message=label,
        details={k: v for k, v in d.items() if k != "kind"},
    )


def _behavioral_pillar(report: Optional[BehavioralReport]) -> PillarScore:
    if report is None:
        return _not_run("behavioral", "Stage 5 was not requested.")
    if report.mode == "not_runnable":
        reason = report.findings[0].message if report.findings else "not runnable"
        return _not_run("behavioral", f"Stage 5 could not run: {reason}")
    flags = [
        _behavioral_flag(f)
        for f in report.findings
        if f.check in _BEHAVIORAL_CHECKS and f.severity in _MEDIUM_PLUS
    ]
    return _pillar("behavioral", flags, "Probed; no island or class-asymmetry finding at MEDIUM+.")


def fuse(
    model_path: str,
    statistical: Optional[StatisticalReport] = None,
    stego: Optional[StegoReport] = None,
    behavioral: Optional[BehavioralReport] = None,
) -> ModelRiskScore:
    """Combine Stage 3/4/5 reports into a ModelRiskScore. See module
    docstring for the scoring rules."""
    pillars = {
        "statistical": _statistical_pillar(statistical),
        "steganographic": _stego_pillar(stego),
        "behavioral": _behavioral_pillar(behavioral),
    }
    ran = {n: p for n, p in pillars.items() if p.status != PillarStatus.NOT_RUN}
    overall = max((p.score for p in ran.values()), default=0.0)
    all_flags = [flag for p in pillars.values() for flag in p.flags]
    risk_level = _max_severity(flag.severity for flag in all_flags)

    layer_flags = [f for f in all_flags if f.layer_name != MODEL_LEVEL]
    layer_risk: dict[str, float] = {}
    for flag in layer_flags:
        layer_risk[flag.layer_name] = max(layer_risk.get(flag.layer_name, 0.0), flag.score)

    if not ran:
        text = "No pillar ran; no risk assessment is possible."
    elif overall == 0.0:
        text = f"No MEDIUM+ evidence from the pillars that ran ({', '.join(ran)})."
    else:
        driver = max(ran.values(), key=lambda p: p.score)
        text = (
            f"Risk {risk_level.value.upper()} (score {overall:.2f}) driven by the {driver.name} pillar: "
            f"{driver.summary}."
        )
    not_run = [n for n, p in pillars.items() if p.status == PillarStatus.NOT_RUN]
    if not_run:
        text += f" NOT assessed: {', '.join(not_run)} -- absence of evidence, not evidence of absence."

    return ModelRiskScore(
        model_path=model_path,
        overall_score=overall,
        statistical=pillars["statistical"],
        steganographic=pillars["steganographic"],
        behavioral=pillars["behavioral"],
        layer_flags=sorted(layer_flags, key=lambda f: -f.score),
        explanation=Explanation(
            text=text,
            feature_attributions={n: p.score for n, p in pillars.items() if p.score is not None},
        ),
        metadata={
            "fusion": "max_over_pillars_run",
            "risk_level": risk_level.value,
            "pillars_run": list(ran),
            "pillars_not_run": not_run,
            "layer_risk": dict(sorted(layer_risk.items(), key=lambda kv: -kv[1])),
        },
    )
