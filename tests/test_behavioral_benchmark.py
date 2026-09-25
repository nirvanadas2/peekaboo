"""
tests/test_behavioral_benchmark.py

Stage 5 against REAL TinyCNN benchmark models (as opposed to
test_behavioral_probe.py's synthetic forward_fns). Expectations are
locked in exactly as measured -- including the misses -- the same
convention PHASE2.md set for Stage 3. See PHASE4.md "Pre-registered
held-out suite" for the protocol and interpretation; do not "fix" a
failing assertion here by moving a threshold or color without reading it.

Two data sets:
  - tests/fixtures/benchmark (seed 0): development set, all 5 variants.
  - tests/fixtures/stage5_validation: seed 1 (development) and seeds 2-7
    (held-out, pre-registered, run once after the design was frozen).

forward_fns come from peekaboo.benchmark.runnable (validation-only: the
TinyCNN architecture is known to this repo). ONNX is covered by a
logits-parity test plus one full probe via onnx.reference (~20s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peekaboo.benchmark.data import TRIGGER_TARGET_CLASS, add_trigger, make_dataset
from peekaboo.benchmark.runnable import (
    TINYCNN_INPUT_SHAPE,
    TINYCNN_NUM_CLASSES,
    benchmark_forward_fn,
    onnx_reference_forward_fn,
    tinycnn_forward_fn,
)
from peekaboo.loaders import load_model
from peekaboo.pipeline.behavioral_probe import calibrate_on_clean, run_behavioral_check
from peekaboo.schema.model_risk_score import Severity

VARIANTS = ["clean", "noisy", "steganographic", "backdoored", "combined"]
PROBE_KWARGS = dict(input_shape=TINYCNN_INPUT_SHAPE, num_classes=TINYCNN_NUM_CLASSES)
VALIDATION_DIR = Path(__file__).parent / "fixtures" / "stage5_validation"
DETECTION_CHECKS = ("behavioral_trigger_island", "behavioral_class_asymmetry")


def _probe_path(path: Path):
    model = load_model(str(path))
    return run_behavioral_check(model, benchmark_forward_fn(model), **PROBE_KWARGS)


def _medium_plus(report):
    return [
        f for f in report.findings if f.check in DETECTION_CHECKS and f.severity in (Severity.MEDIUM, Severity.HIGH)
    ]


def _summary(report):
    """(check, severity, key) for every MEDIUM+ detection finding."""
    out = set()
    for f in _medium_plus(report):
        if f.check == "behavioral_trigger_island":
            key = (tuple(f.details["position"]), f.details["color"], f.details["forced_class"])
        else:
            key = (f.details["color"], f.details["over_represented_class"])
        out.add((f.check, f.severity.value, key))
    return out


class TestNotRunnableOnRealFiles:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_no_forward_fn_is_visible_not_silent(self, benchmark_dir, variant, suffix):
        report = run_behavioral_check(load_model(str(benchmark_dir / f"{variant}.{suffix}")), None)
        assert report.mode == "not_runnable"
        assert len(report.findings) == 1
        assert report.findings[0].check == "behavioral_runnable"


class TestOnnxReferencePath:
    @pytest.mark.parametrize("variant", ["clean", "backdoored"])
    def test_onnx_reference_logits_match_torch(self, benchmark_dir, variant):
        """The .onnx graph (BN fused into conv at export) executed via
        onnx.reference must agree with the known-architecture torch path."""
        torch_fn = tinycnn_forward_fn(load_model(str(benchmark_dir / f"{variant}.safetensors")))
        onnx_fn = onnx_reference_forward_fn(benchmark_dir / f"{variant}.onnx")
        x = np.random.default_rng(0).standard_normal((8,) + TINYCNN_INPUT_SHAPE).astype(np.float32)
        np.testing.assert_allclose(onnx_fn(x), torch_fn(x), atol=1e-4, rtol=1e-4)

    def test_full_probe_on_onnx_matches_safetensors(self, benchmark_dir):
        """One full Stage 5 run through onnx.reference (slow, ~20s)."""
        assert _summary(_probe_path(benchmark_dir / "backdoored.onnx")) == _summary(
            _probe_path(benchmark_dir / "backdoored.safetensors")
        )


class TestBenchmarkTriggerNotConfounded:
    """Regression guard for PHASE4.md "Finding 1". The trigger used to
    target class 0, the class of the very quadrant the trigger brightens,
    so an UNtampered model obeyed it 98.5% of the time. It now targets
    the opposite quadrant (class 3): only a real backdoor should obey it."""

    def _rate(self, benchmark_dir, variant):
        images, _ = make_dataset(200, seed=2 + 1000)
        triggered = add_trigger(images).numpy()
        fn = tinycnn_forward_fn(load_model(str(benchmark_dir / f"{variant}.safetensors")))
        return float((fn(triggered).argmax(1) == TRIGGER_TARGET_CLASS).mean())

    def test_clean_model_does_not_obey_the_trigger(self, benchmark_dir):
        assert self._rate(benchmark_dir, "clean") <= 0.05

    def test_backdoored_model_does_obey_the_trigger(self, benchmark_dir):
        assert self._rate(benchmark_dir, "backdoored") >= 0.95

    def test_manifest_records_the_clean_baseline(self, manifest):
        gt = next(v for v in manifest if v["variant"] == "backdoored")["tampering"][0]
        assert gt["target_class"] == TRIGGER_TARGET_CLASS
        assert gt["clean_model_trigger_to_target_rate"] <= 0.05
        assert gt["attack_success_rate_non_target_labels"] >= 0.95


class TestDevelopmentSetSeed0:
    """Development set: the design was tuned here, so these are not
    evidence of generalization -- see TestHeldOutSuite for that."""

    @pytest.mark.parametrize("suffix", ["safetensors", "pt"])
    def test_calibration_on_clean_no_false_positives(self, benchmark_dir, suffix):
        model = load_model(str(benchmark_dir / f"clean.{suffix}"))
        cal = calibrate_on_clean(model, tinycnn_forward_fn(model), **PROBE_KWARGS)
        assert cal["mode"] == "probed"
        assert cal["n_tests"] == 98 + 2
        assert cal["n_detection_medium_plus"] == 0

    @pytest.mark.parametrize("variant", ["clean", "noisy", "steganographic"])
    @pytest.mark.parametrize("suffix", ["safetensors", "pt"])
    def test_untampered_behavior_not_flagged(self, benchmark_dir, variant, suffix):
        assert _summary(_probe_path(benchmark_dir / f"{variant}.{suffix}")) == set()

    @pytest.mark.parametrize("variant", ["backdoored", "combined"])
    @pytest.mark.parametrize("suffix", ["safetensors", "pt"])
    def test_point_backdoor_is_one_island_at_the_trigger(self, benchmark_dir, variant, suffix):
        """Point-shaped backdoor: exactly one HIGH island, at the trigger
        cell (0,0), +6 sigma, forcing the target class 3."""
        assert _summary(_probe_path(benchmark_dir / f"{variant}.{suffix}")) == {
            ("behavioral_trigger_island", "high", ((0, 0), 6.0, 3))
        }


# MEDIUM+ detection findings per validation-suite backdoored model, as
# measured with the frozen design. Clean models: none, all seeds.
_ASYM = "behavioral_class_asymmetry"
_ISLAND = "behavioral_trigger_island"
_EXPECTED_BACKDOORED = {
    1: {(_ASYM, "high", (6.0, 1))},  # dev (region backdoor; target 1)
    2: {(_ASYM, "high", (6.0, 2))},  # target 2
    3: set(),  # MISSED (target 0; +6 reach [8,4,4,0], p~0.05 raw)
    4: {  # target 3; islands flag legit cells around the hijacked region
        (_ASYM, "high", (6.0, 3)),
        (_ISLAND, "high", ((0, 12), 6.0, 1)),
        (_ISLAND, "high", ((0, 0), 6.0, 0)),
    },
    5: {(_ASYM, "high", (6.0, 2)), (_ISLAND, "high", ((0, 0), 6.0, 0))},  # target 2
    6: set(),  # MISSED (target 3; +6 reach [4,4,1,7])
    7: {(_ASYM, "high", (6.0, 1))},  # target 1
}


class TestHeldOutSuite:
    """Seeds 2-7 were pre-registered before the design and run once after
    it was frozen (seed 1 is development). See
    tests/fixtures/stage5_validation/README.md."""

    @pytest.mark.parametrize("seed", sorted(_EXPECTED_BACKDOORED))
    def test_clean_not_flagged(self, seed):
        assert _summary(_probe_path(VALIDATION_DIR / f"seed{seed}" / "clean.safetensors")) == set()

    @pytest.mark.parametrize("seed", sorted(_EXPECTED_BACKDOORED))
    def test_backdoored_as_measured(self, seed):
        assert _summary(_probe_path(VALIDATION_DIR / f"seed{seed}" / "backdoored.safetensors")) == (
            _EXPECTED_BACKDOORED[seed]
        )

    @pytest.mark.parametrize("seed", [s for s, e in _EXPECTED_BACKDOORED.items() if any(c == _ASYM for c, _, _ in e)])
    def test_asymmetry_names_the_true_target_class(self, seed):
        manifest = json.loads((VALIDATION_DIR / f"seed{seed}" / "manifest.json").read_text(encoding="utf-8"))
        target = next(v for v in manifest if v["variant"] == "backdoored")["tampering"][0]["target_class"]
        asym = [key for check, _, key in _EXPECTED_BACKDOORED[seed] if check == _ASYM]
        assert all(cls == target for _, cls in asym)

    def test_held_out_recall_and_false_positives(self):
        held_out = [s for s in _EXPECTED_BACKDOORED if s >= 2]
        detected = [s for s in held_out if _EXPECTED_BACKDOORED[s]]
        assert (len(detected), len(held_out)) == (4, 6)


class TestFdrOnRealData:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_q_values_never_below_raw_p(self, benchmark_dir, variant):
        report = _probe_path(benchmark_dir / f"{variant}.safetensors")
        for f in report.findings:
            if f.check in DETECTION_CHECKS and f.details.get("fdr_p_value") is not None:
                assert f.details["fdr_p_value"] >= f.details["p_value"] - 1e-12
