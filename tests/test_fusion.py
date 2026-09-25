"""
tests/test_fusion.py

Stage 6 (evidence fusion -> ModelRiskScore). Unit tests on synthetic
reports, then end-to-end through run_pre_checks on the committed
fixtures, with results locked in exactly as measured -- including the
Stage 3/4 false positives on independent clean models (PHASE5.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from peekaboo.benchmark.runnable import TINYCNN_INPUT_SHAPE, TINYCNN_NUM_CLASSES, tinycnn_forward_fn
from peekaboo.loaders import load_model
from peekaboo.pipeline.fusion import MODEL_LEVEL, evidence_score, fuse
from peekaboo.pipeline.gate import run_pre_checks
from peekaboo.schema.model_risk_score import PillarScore, PillarStatus, Severity
from peekaboo.schema.reports import BehavioralReport, Finding, StatisticalReport, StegoReport

VALIDATION_DIR = Path(__file__).parent / "fixtures" / "stage5_validation"


def _stat(outliers=None, severity=Severity.MEDIUM):
    findings = []
    if outliers:
        findings.append(
            Finding(
                check="mean_outliers",
                severity=severity,
                passed=False,
                message="m",
                details={"mode": "relative_outlier", "outliers": outliers},
            )
        )
    return StatisticalReport(model_path="m", mode="relative_outlier", passed=not findings, findings=findings)


def _stego(q=None, fdr=True):
    findings = []
    if q is not None:
        sev = Severity.HIGH if q < 0.001 else Severity.MEDIUM
        findings.append(
            Finding(
                check="bit_balance_chi_square",
                severity=sev,
                passed=False,
                message="m",
                details={"layer_name": "fc1.weight", "statistic": 1.0, "p_value": q / 10, "fdr_p_value": q},
            )
        )
    return StegoReport(
        model_path="m",
        passed=not findings,
        findings=findings,
        metadata={"fdr_correction": "benjamini_hochberg" if fdr else None},
    )


def _behavioral(q=None, mode="probed"):
    if mode == "not_runnable":
        f = Finding(check="behavioral_runnable", severity=Severity.INFO, passed=True, message="no forward_fn")
        return BehavioralReport(model_path="m", mode="not_runnable", passed=True, findings=[f])
    findings = []
    if q is not None:
        findings.append(
            Finding(
                check="behavioral_class_asymmetry",
                severity=Severity.HIGH if q < 0.01 else Severity.MEDIUM,
                passed=False,
                message="m",
                details={"kind": "box", "color": 6.0, "over_represented_class": 3, "p_value": q / 2, "fdr_p_value": q},
            )
        )
    return BehavioralReport(model_path="m", mode="probed", passed=not findings, findings=findings)


class TestEvidenceScore:
    def test_anchor_points_and_monotonicity(self):
        assert evidence_score(0.05) == pytest.approx(0.4)
        assert evidence_score(1e-10) == pytest.approx(1.0)
        assert evidence_score(1e-300) == pytest.approx(1.0)
        qs = [0.05, 0.01, 1e-3, 1e-6, 1e-9]
        scores = [evidence_score(q) for q in qs]
        assert scores == sorted(scores)


class TestPillarStatus:
    def test_not_run_vs_ran_clean_are_distinct(self):
        r = fuse("m", statistical=_stat(), stego=_stego(), behavioral=_behavioral(mode="not_runnable"))
        assert r.behavioral.status == PillarStatus.NOT_RUN and r.behavioral.score is None
        assert r.steganographic.status == PillarStatus.RAN_CLEAN and r.steganographic.score == 0.0
        assert r.metadata["pillars_not_run"] == ["behavioral"]
        assert "NOT assessed: behavioral" in r.explanation.text
        assert "behavioral" not in r.explanation.feature_attributions
        assert r.to_dict()["pillars"]["behavioral"]["status"] == "not_run"

    def test_missing_report_is_not_run(self):
        r = fuse("m", statistical=_stat())
        assert r.steganographic.status == PillarStatus.NOT_RUN
        assert r.metadata["pillars_run"] == ["statistical"]

    def test_schema_rejects_inconsistent_status(self):
        with pytest.raises(ValueError):
            PillarScore(name="x", score=None, status=PillarStatus.RAN_CLEAN)
        with pytest.raises(ValueError):
            PillarScore(name="x", score=0.0, status=PillarStatus.NOT_RUN)


class TestFusionRules:
    def test_overall_is_max_of_pillars_that_ran(self):
        r = fuse("m", statistical=_stat(), stego=_stego(q=1e-4), behavioral=_behavioral(q=0.02))
        assert r.overall_score == pytest.approx(max(evidence_score(1e-4), evidence_score(0.02)))
        assert r.steganographic.status == PillarStatus.FLAGGED
        assert r.explanation.feature_attributions["steganographic"] == pytest.approx(r.overall_score)
        assert "steganographic pillar" in r.explanation.text

    def test_stage3_is_report_only(self):
        """Stage 3 flags are shown (capped at MEDIUM) but never scored
        (PHASE6.md)."""
        r = fuse("m", statistical=_stat({"conv1.weight": {"value": 1.0, "modified_z_score": 50.0, "severity": "high"}}, Severity.HIGH))
        assert r.statistical.status == PillarStatus.FLAGGED
        assert r.statistical.weight == 0.0
        assert r.overall_score == 0.0
        assert r.metadata["risk_level"] == "info"
        assert r.metadata["report_only_pillars"] == ["statistical"]
        assert r.metadata["layer_risk"] == {}
        assert r.explanation.feature_attributions["statistical"] == 0.0
        assert "report-only" in r.explanation.text
        flag = r.layer_flags[0]  # still visible
        assert flag.severity == Severity.MEDIUM and flag.details["uncapped_severity"] == "high"

    def test_stage3_does_not_change_a_scored_result(self):
        with_s3 = fuse("m", statistical=_stat({"conv1.weight": {"value": 1.0, "severity": "high"}}, Severity.HIGH),
                       stego=_stego(q=1e-4))
        without = fuse("m", statistical=_stat(), stego=_stego(q=1e-4))
        assert with_s3.overall_score == without.overall_score
        assert with_s3.metadata["risk_level"] == without.metadata["risk_level"]

    def test_behavioral_flags_are_model_level_not_layer_flags(self):
        r = fuse("m", stego=_stego(q=1e-3), behavioral=_behavioral(q=1e-4))
        assert all(f.layer_name != MODEL_LEVEL for f in r.layer_flags)
        assert r.behavioral.flags[0].layer_name == MODEL_LEVEL
        assert r.metadata["layer_risk"] == {"fc1.weight": pytest.approx(evidence_score(1e-3))}

    def test_uncorrected_stage4_is_rejected(self):
        with pytest.raises(ValueError, match="not FDR-corrected"):
            fuse("m", stego=_stego(q=1e-4, fdr=False))

    def test_nothing_ran(self):
        r = fuse("m")
        assert r.overall_score == 0.0
        assert r.metadata["pillars_run"] == []


def _scan(path: Path, probe: bool):
    fwd = tinycnn_forward_fn(load_model(str(path))) if probe else None
    kwargs = dict(input_shape=TINYCNN_INPUT_SHAPE, num_classes=TINYCNN_NUM_CLASSES) if probe else {}
    return run_pre_checks(str(path), forward_fn=fwd, **kwargs).risk_score


def _flagged(score) -> set[str]:
    return {
        n for n in ("statistical", "steganographic", "behavioral") if getattr(score, n).status == PillarStatus.FLAGGED
    }


class TestEndToEndSeed0:
    @pytest.mark.parametrize(
        "variant,expected",
        [
            ("clean", {"steganographic"}),  # bn4.running_var FP (PHASE3.md)
            ("noisy", {"steganographic"}),
            ("steganographic", {"steganographic"}),  # the SAME bn4 FP, not the payload
            ("backdoored", {"behavioral"}),
            ("combined", {"behavioral"}),
        ],
    )
    def test_flagged_pillars_as_measured(self, benchmark_dir, variant, expected):
        assert _flagged(_scan(benchmark_dir / f"{variant}.safetensors", probe=True)) == expected

    def test_without_forward_fn_backdoor_is_not_assessed(self, benchmark_dir):
        r = _scan(benchmark_dir / "backdoored.safetensors", probe=False)
        assert r.behavioral.status == PillarStatus.NOT_RUN
        assert r.overall_score == 0.0
        assert "NOT assessed: behavioral" in r.explanation.text


# Flagged pillars on the DEVELOPMENT seeds 1-7, with forward_fn, AFTER the
# Stage 3 (noise-aware z) and Stage 4 (init-lattice) fixes. These seeds
# were used to design those fixes, so they are not evidence of
# generalization -- TestFreshSuite is. (Before the fixes: 5 of 7 clean
# models flagged; see PHASE5.md.)
_EXPECTED_VALIDATION = {
    (1, "clean"): set(),
    (1, "backdoored"): {"statistical", "behavioral"},  # conv1/conv4 entropy
    (2, "clean"): set(),
    (2, "backdoored"): {"behavioral"},
    (3, "clean"): set(),
    (3, "backdoored"): set(),  # Stage 5 miss
    (4, "clean"): set(),
    (4, "backdoored"): {"behavioral"},
    (5, "clean"): set(),
    (5, "backdoored"): {"behavioral"},
    (6, "clean"): set(),
    (6, "backdoored"): set(),  # Stage 5 miss
    (7, "clean"): set(),
    (7, "backdoored"): {"behavioral"},
}


class TestEndToEndDevelopmentSeeds:
    @pytest.mark.parametrize("seed,variant", sorted(_EXPECTED_VALIDATION))
    def test_flagged_pillars_as_measured(self, seed, variant):
        path = VALIDATION_DIR / f"seed{seed}" / f"{variant}.safetensors"
        assert _flagged(_scan(path, probe=True)) == _EXPECTED_VALIDATION[(seed, variant)]


FRESH_DIR = Path(__file__).parent / "fixtures" / "fresh_validation"

# Pre-registered fresh suite (seeds 10-19), run ONCE after the Stage 3/4
# fixes were frozen. Locked in exactly as measured, misses and false
# positives included (PHASE5.md "Fresh-suite result").
_S3, _S4, _S5 = "statistical", "steganographic", "behavioral"
_EXPECTED_FRESH = {
    10: {"clean": {_S3}, "noisy": {_S4}, "backdoored": set()},
    11: {"clean": {_S3}, "noisy": {_S3}, "backdoored": {_S5}},
    12: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
    13: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
    14: {"clean": {_S3}, "noisy": {_S3, _S4}, "backdoored": {_S3}},
    15: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    16: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
    17: {"clean": {_S3}, "noisy": {_S4}, "backdoored": {_S5}},
    18: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    19: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
}


class TestFreshSuite:
    @pytest.mark.parametrize(
        "seed,variant", [(s, v) for s in sorted(_EXPECTED_FRESH) for v in ("clean", "noisy", "backdoored")]
    )
    def test_flagged_pillars_as_measured(self, seed, variant):
        path = FRESH_DIR / f"seed{seed}" / f"{variant}.safetensors"
        assert _flagged(_scan(path, probe=True)) == _EXPECTED_FRESH[seed][variant]

    def test_headline_counts(self):
        def count(variant, pillar):
            return sum(pillar in e[variant] for e in _EXPECTED_FRESH.values())

        assert (count("clean", _S4), count("clean", _S5), count("clean", _S3)) == (0, 0, 4)
        assert count("noisy", _S4) == 9
        assert count("backdoored", _S5) == 8
        assert sum(bool(e["clean"]) for e in _EXPECTED_FRESH.values()) == 4  # all Stage 3

    @pytest.mark.parametrize("seed", [s for s, e in _EXPECTED_FRESH.items() if e["clean"] == {_S3}])
    def test_stage3_only_clean_models_now_score_zero(self, seed):
        """Post-hoc on this (spent) suite -- the held-out check of the
        report-only change is TestFinalSuite."""
        rs = _scan(FRESH_DIR / f"seed{seed}" / "clean.safetensors", probe=True)
        assert rs.overall_score == 0.0 and rs.metadata["risk_level"] == "info"

    @pytest.mark.parametrize("seed", [s for s, e in _EXPECTED_FRESH.items() if _S5 in e["backdoored"]])
    def test_asymmetry_names_true_target(self, seed):
        import json

        manifest = json.loads((FRESH_DIR / f"seed{seed}" / "manifest.json").read_text(encoding="utf-8"))
        target = next(v for v in manifest if v["variant"] == "backdoored")["tampering"][0]["target_class"]
        rs = _scan(FRESH_DIR / f"seed{seed}" / "backdoored.safetensors", probe=True)
        asym = [f for f in rs.behavioral.flags if f.flag_type == "behavioral_class_asymmetry"]
        assert all(f.details["over_represented_class"] == target for f in asym)


FINAL_DIR = Path(__file__).parent / "fixtures" / "final_validation"

# Final pre-registered suite (seeds 20-29), registered BEFORE Stage 3 was
# made report-only and Stage 7 was built; run ONCE on the complete frozen
# system (PHASE6.md). Flagged pillars (Stage 3 is flagged-but-unscored).
_EXPECTED_FINAL = {
    20: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    21: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    22: {"clean": set(), "noisy": set(), "backdoored": set()},  # noise + backdoor both missed
    23: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
    24: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    25: {"clean": set(), "noisy": set(), "backdoored": {_S5}},  # noise missed
    26: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    27: {"clean": set(), "noisy": {_S4}, "backdoored": {_S5}},
    28: {"clean": {_S3}, "noisy": set(), "backdoored": {_S3, _S5}},  # noise missed
    29: {"clean": set(), "noisy": {_S4}, "backdoored": {_S3, _S5}},
}


class TestFinalSuite:
    @pytest.mark.parametrize(
        "seed,variant", [(s, v) for s in sorted(_EXPECTED_FINAL) for v in ("clean", "noisy", "backdoored")]
    )
    def test_flagged_pillars_as_measured(self, seed, variant):
        rs = _scan(FINAL_DIR / f"seed{seed}" / f"{variant}.safetensors", probe=True)
        expected = _EXPECTED_FINAL[seed][variant]
        assert _flagged(rs) == expected
        # the score is driven only by scored pillars (Stage 3 is report-only)
        assert (rs.overall_score > 0) == bool(expected - {_S3})

    def test_headline_counts(self):
        def count(variant, pillar):
            return sum(pillar in e[variant] for e in _EXPECTED_FINAL.values())

        assert sum(bool(e["clean"] - {_S3}) for e in _EXPECTED_FINAL.values()) == 0  # fused clean FPs
        assert count("noisy", _S4) == 7
        assert count("backdoored", _S5) == 9

    def test_without_forward_fn_backdoors_are_not_assessed_not_clean(self):
        rs = _scan(FINAL_DIR / "seed20" / "backdoored.safetensors", probe=False)
        assert rs.behavioral.status == PillarStatus.NOT_RUN
        assert rs.overall_score == 0.0 and "NOT assessed: behavioral" in rs.explanation.text
