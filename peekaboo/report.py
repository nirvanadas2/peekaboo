"""
peekaboo/report.py

Stage 7: Explainable Report
=============================

Renders a `PreCheckResult` (Stages 1-6) into a human-readable Markdown
report or a JSON document. No detection logic lives here -- every number
comes from the stages, and every flag is explained in plain language
with the evidence that produced it. Design principles (PHASE6.md):

- Say what was NOT checked as loudly as what was found. A pillar that
  didn't run is reported as "not assessed", never as "clean".
- Report-only signals (Stage 3) are shown, clearly marked as not scored.
- Every report carries the scope/limitations section, because each
  detector's measured blind spots matter as much as its hits.
"""

from __future__ import annotations

import json
from typing import Any

from peekaboo.pipeline.fusion import MODEL_LEVEL
from peekaboo.pipeline.gate import PreCheckResult
from peekaboo.schema.model_risk_score import LayerFlag, ModelRiskScore, PillarScore, PillarStatus, Severity

REPORT_VERSION = "1"

_VERDICT = {
    "info": "No evidence of tampering from the checks that ran",
    "low": "No evidence of tampering from the checks that ran",
    "medium": "SUSPICIOUS -- review before deployment",
    "high": "HIGH RISK -- do not deploy without investigation",
    "critical": "HIGH RISK -- do not deploy without investigation",
}

# Measured scope and blind spots (PHASE3-6.md). Kept in every report.
LIMITATIONS = [
    "Behavioral probing (backdoors) only runs when a caller-supplied forward_fn is given; "
    "safetensors/pickle files carry no computation graph. Without it, backdoors are NOT assessed -- "
    "no static check detects them on the benchmark.",
    "Backdoor detection covers localized, bright, high-contrast patch triggers only (not blended, "
    "frequency-domain, warping, dark or semantic triggers). Held-out (3 pre-registered suites): 21/26 "
    "backdoors detected, 0/78 false positives on backdoor-free models. The class-asymmetry test assumes classes are spatially symmetric under bright patches, "
    "true for the benchmark task, not in general. Trigger positions reported are hints, not locations.",
    "Steganography detection only catches STRUCTURED (e.g. unencrypted text), DENSE payloads (roughly "
    ">=50% of a layer's low-bit capacity). An encrypted/random-looking payload is undetectable at any "
    "density. The benchmark's own 67-byte payload is not detected.",
    "The bit-level tests correct for PyTorch's default initialization only; models initialized "
    "differently may show false positives from never-updated weights.",
    "Statistical (Stage 3) findings are report-only: they detected no tampering on any suite and "
    "flagged 1-4 of 10 clean models per suite, so they do not affect the score.",
    "Noise injection is detected via its low-bit signature: held-out 16/20, 0/80 false positives on "
    "noise-free models; detection evidence is sometimes weak (MEDIUM).",
    "All accuracy figures come from one small synthetic architecture (TinyCNN) and task; they are "
    "evidence the mechanisms work, not calibrated real-world rates.",
]

_CHECK_EXPLANATIONS = {
    "bit_balance_chi_square": (
        "The low mantissa bits of this tensor are unbalanced between 0s and 1s beyond what chance allows. "
        "Seen with noise added to float weights and with structured (unencrypted) hidden payloads."
    ),
    "block_homogeneity_chi_square": (
        "The share of 1s in the low mantissa bits differs between parts of this tensor beyond chance -- "
        "consistent with data hidden in only part of the layer."
    ),
    "wald_wolfowitz_runs": (
        "The ordering of the low mantissa bits is non-random (too few/many runs of equal bits) -- "
        "consistent with structured data written into the weights."
    ),
    "behavioral_class_asymmetry": (
        "Bright patches force one class from far more input positions than the others. A clean model of "
        "this task reaches every class about equally; a hijacked region pointing at one class looks like "
        "this. Consistent with a backdoor whose target is the most-reached class."
    ),
    "behavioral_trigger_island": (
        "A bright patch at this input position forces one class on most inputs while no neighbouring "
        "position does -- consistent with a small localized trigger. The position is a hint: on "
        "region-shaped backdoors this can instead flag a legitimate cell surrounded by the hijacked area."
    ),
}


def _explain(flag: LayerFlag) -> str:
    if flag.flag_type.endswith("_outliers"):
        stat = flag.flag_type[: -len("_outliers")]
        return (
            f"This layer's {stat} is an outlier relative to the model's other layers. Report-only: "
            "this check has not shown detection value and does not affect the score."
        )
    return _CHECK_EXPLANATIONS.get(flag.flag_type, "See the finding details.")


def _evidence(flag: LayerFlag) -> str:
    d = flag.details
    parts = []
    if "fdr_p_value" in d:
        parts.append(f"FDR q={d['fdr_p_value']:.2g}")
    if "p_value" in d and d["p_value"] is not None:
        parts.append(f"raw p={d['p_value']:.2g}")
    if "reach" in d:
        parts.append(f"class reach {d['reach']}")
    if "hit_rate" in d:
        parts.append(f"{d['hit_rate']:.0%} of inputs -> class {d['forced_class']} at position {d['position']}")
    if "modified_z_score" in d:
        parts.append(f"robust z={d['modified_z_score']:.1f}")
    return ", ".join(parts)


def _pillar_line(label: str, p: PillarScore) -> str:
    if p.status == PillarStatus.NOT_RUN:
        result = f"**not assessed** -- {p.summary}"
    elif p.status == PillarStatus.RAN_CLEAN:
        result = f"clean -- {p.summary}"
    else:
        result = f"**flagged** -- {p.summary}"
    scored = "report-only" if p.weight == 0 else ("—" if p.score is None else f"{p.score:.2f}")
    return f"| {label} | {p.status.value} | {scored} | {result} |"


def _finding_rows(flags: list[LayerFlag]) -> list[str]:
    lines = []
    for f in flags:
        where = "whole model (behavior)" if f.layer_name == MODEL_LEVEL else f"`{f.layer_name}`"
        lines.append(f"- **{f.severity.value.upper()}** · {where} · `{f.flag_type}` · score {f.score:.2f}")
        lines.append(f"  - {_explain(f)}")
        ev = _evidence(f)
        if ev:
            lines.append(f"  - Evidence: {ev}")
    return lines


def render_markdown(result: PreCheckResult) -> str:
    md = result.metadata
    lines = [
        "# Peekaboo scan report",
        "",
        f"- **File:** `{md.model_path}`",
        f"- **SHA-256:** `{md.file_hash}`",
        f"- **Format:** {md.detected_format} (declared: {md.declared_format}), {md.file_size:,} bytes",
        f"- **Report version:** {REPORT_VERSION}",
        "",
    ]

    if result.stopped_at_metadata:
        critical = [f for f in md.findings if not f.passed and f.severity == Severity.CRITICAL]
        lines += [
            "## Verdict: UNSAFE / NOT ANALYZED",
            "",
            "Stage 1 (metadata integrity) hard-failed, so the file was **not loaded** and no further checks ran. "
            "Do not load this file with a general-purpose deserializer.",
            "",
        ]
        lines += [f"- **{f.check}**: {f.message}" for f in critical]
        lines += ["", "## Scope and limitations", ""] + [f"- {x}" for x in LIMITATIONS]
        return "\n".join(lines) + "\n"

    rs: ModelRiskScore = result.risk_score
    level = rs.metadata["risk_level"]
    lines += [
        f"## Verdict: {_VERDICT[level]}",
        "",
        f"**Risk level:** {level.upper()} · **Overall score:** {rs.overall_score:.2f} (0 = no evidence, 1 = overwhelming)",
        "",
        rs.explanation.text,
        "",
        "## What was checked",
        "",
        "| Check | Status | Score | Result |",
        "|---|---|---|---|",
        f"| Stage 1 · metadata integrity | {'passed' if md.passed else 'issues'} | — | "
        f"{sum(not f.passed for f in md.findings)} failed of {len(md.findings)} checks |",
        f"| Stage 2 · structural consistency | {'passed' if result.structural.passed else 'issues'} | — | "
        f"{sum(not f.passed for f in result.structural.findings)} failed of {len(result.structural.findings)} checks "
        f"({result.structural.mode}) |",
        _pillar_line("Stage 3 · weight statistics", rs.statistical),
        _pillar_line("Stage 4 · bit-level steganalysis", rs.steganographic),
        _pillar_line("Stage 5 · behavioral probing", rs.behavioral),
        "",
    ]

    scored = rs.steganographic.flags + rs.behavioral.flags
    lines += ["## Findings that drive the score", ""]
    lines += _finding_rows(sorted(scored, key=lambda f: -f.score)) or ["- None."]
    lines.append("")
    if rs.statistical.flags:
        lines += ["## Report-only observations (not scored)", ""]
        lines += _finding_rows(rs.statistical.flags)
        lines.append("")

    if rs.metadata["layer_risk"]:
        lines += ["## Per-layer risk", "", "| Layer | Risk contribution |", "|---|---|"]
        lines += [f"| `{name}` | {score:.2f} |" for name, score in rs.metadata["layer_risk"].items()]
        lines.append("")

    structural_issues = [f for f in result.structural.findings if not f.passed]
    if structural_issues or not md.passed:
        lines += ["## Integrity and structure issues", ""]
        lines += [f"- **{f.severity.value.upper()}** `{f.check}`: {f.message}" for f in md.findings if not f.passed]
        lines += [f"- **{f.severity.value.upper()}** `{f.check}`: {f.message}" for f in structural_issues]
        lines.append("")

    not_run = rs.metadata["pillars_not_run"]
    if not_run:
        lines += [
            "## Not assessed",
            "",
            f"The following were **not checked**: {', '.join(not_run)}. Absence of findings there is not "
            "evidence of safety.",
            "",
        ]
        if "behavioral" in not_run:
            lines += [
                "To assess backdoors, rerun with a forward function for this model, e.g. "
                "`run_pre_checks(path, forward_fn=..., input_shape=..., num_classes=...)`.",
                "",
            ]

    lines += ["## Scope and limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"


def render_json(result: PreCheckResult) -> str:
    doc: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "verdict": "unsafe_not_analyzed"
        if result.stopped_at_metadata
        else result.risk_score.metadata["risk_level"],
        "limitations": LIMITATIONS,
        "scan": result.to_dict(),
    }
    return json.dumps(doc, indent=2, default=str)


def exit_code(result: PreCheckResult) -> int:
    """CI-friendly: 0 = no scored evidence (info/low), 1 = medium,
    2 = high/critical, 3 = unsafe file (Stage 1 hard-fail)."""
    if result.stopped_at_metadata:
        return 3
    level = result.risk_score.metadata["risk_level"]
    return {"info": 0, "low": 0, "medium": 1, "high": 2, "critical": 2}[level]
