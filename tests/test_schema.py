"""Tests for the Model Risk Score schema stub."""

from __future__ import annotations

from peekaboo.schema import Explanation, LayerFlag, ModelRiskScore, PillarScore, Severity


def _make_score() -> ModelRiskScore:
    flag = LayerFlag(
        layer_name="conv2.weight",
        flag_type="mantissa_lsb_anomaly",
        severity=Severity.HIGH,
        score=0.87,
        message="Low-order mantissa bits show non-random structure",
        details={"bits_per_value": 4},
    )
    return ModelRiskScore(
        model_path="benchmark_output/steganographic.safetensors",
        overall_score=0.72,
        statistical=PillarScore(name="statistical", score=0.3, weight=1.0, summary="ok"),
        steganographic=PillarScore(
            name="steganographic", score=0.9, weight=1.0, summary="payload detected", flags=[flag]
        ),
        behavioral=PillarScore(name="behavioral", score=0.1, weight=1.0, summary="no trigger found"),
        layer_flags=[flag],
        explanation=Explanation(text="High steganographic risk driven by conv2.weight anomaly."),
    )


def test_model_risk_score_constructs() -> None:
    score = _make_score()
    assert score.overall_score == 0.72
    assert score.steganographic.flags[0].severity == Severity.HIGH


def test_to_dict_round_trips_fields() -> None:
    score = _make_score()
    d = score.to_dict()

    assert d["model_path"] == score.model_path
    assert d["overall_score"] == 0.72
    assert d["pillars"]["steganographic"]["score"] == 0.9
    assert d["pillars"]["steganographic"]["flags"][0]["severity"] == "high"
    assert d["layer_flags"][0]["layer_name"] == "conv2.weight"
    assert "explanation" in d and d["explanation"]["text"]


def test_defaults_are_empty_not_shared_mutable_state() -> None:
    a = PillarScore(name="statistical", score=0.0)
    b = PillarScore(name="behavioral", score=0.0)
    a.flags.append(
        LayerFlag(
            layer_name="x",
            flag_type="t",
            severity=Severity.LOW,
            score=0.1,
            message="m",
        )
    )
    assert b.flags == []
