"""
tests/test_behavioral_benchmark.py

Stage 5 against the REAL Phase 0 benchmark (as opposed to
test_behavioral_probe.py's synthetic forward_fns). Expectations are
locked in exactly as measured -- including results that are bad for
Stage 5 -- the same convention PHASE2.md set for Stage 3's
TestFullBenchmarkMatrix. See PHASE4.md "Calibration & full-benchmark
results" for the interpretation; do not "fix" a failing assertion here by
moving a threshold without reading that section first.

forward_fns come from peekaboo.benchmark.runnable (validation-only: the
TinyCNN architecture is known to this repo). The full probe is run on
safetensors/pt only; ONNX's reference-evaluator path takes ~55s per
default-budget probe, so ONNX is covered by a logits-parity test against
the torch path instead (the full ONNX matrix is recorded in PHASE4.md
and was identical to safetensors/pt).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from peekaboo.benchmark.data import TRIGGER_TARGET_CLASS, add_trigger, make_dataset
from peekaboo.benchmark.runnable import (
    TINYCNN_INPUT_SHAPE,
    TINYCNN_NUM_CLASSES,
    onnx_reference_forward_fn,
    tinycnn_forward_fn,
)
from peekaboo.loaders import load_model
from peekaboo.pipeline.behavioral_probe import calibrate_on_clean, run_behavioral_check
from peekaboo.schema.model_risk_score import Severity

VARIANTS = ["clean", "noisy", "steganographic", "backdoored", "combined"]
PROBE_KWARGS = dict(input_shape=TINYCNN_INPUT_SHAPE, num_classes=TINYCNN_NUM_CLASSES)


def _probe(benchmark_dir, variant, suffix):
    model = load_model(str(benchmark_dir / f"{variant}.{suffix}"))
    return run_behavioral_check(model, tinycnn_forward_fn(model), **PROBE_KWARGS)


def _severity_counts(report):
    counts = {s: 0 for s in Severity}
    for f in report.findings:
        if f.check == "behavioral_trigger_patch":
            counts[f.severity] += 1
    return counts


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


class TestCalibrationOnClean:
    @pytest.mark.parametrize("suffix", ["safetensors", "pt"])
    def test_clean_false_positive_rate_as_measured(self, benchmark_dir, suffix):
        """MEASURED, NOT A TARGET: 39 of 40 candidate patches reach MEDIUM
        on the untampered model at the default budget. Root cause (PHASE4.md):
        constant +-3 candidates vs. N(0,1)-content controls -- intensity,
        not trigger-likeness, is what separates them on this task."""
        model = load_model(str(benchmark_dir / f"clean.{suffix}"))
        cal = calibrate_on_clean(model, tinycnn_forward_fn(model), **PROBE_KWARGS)
        assert cal["mode"] == "probed"
        assert cal["n_candidates_tested"] == 40
        assert cal["n_trigger_patch_medium_plus"] == 39


class TestFullBenchmarkMatrix:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("suffix", ["safetensors", "pt"])
    def test_every_variant_looks_the_same_at_default_budget(self, benchmark_dir, variant, suffix):
        """No discrimination: every variant, tampered or not, gets 38-40
        of 40 candidates at MEDIUM and 0 HIGH -- including after the
        trigger retarget (backdoored/combined: 38). HIGH is unreachable
        at n_bootstrap=64 (min empirical p = 1/65 > 0.01; see PHASE4.md)."""
        counts = _severity_counts(_probe(benchmark_dir, variant, suffix))
        assert counts[Severity.HIGH] == 0
        assert counts[Severity.MEDIUM] in (38, 39, 40)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_fdr_q_values_match_an_independent_bh(self, benchmark_dir, variant):
        """BH on real data: every q >= its raw p, and the number of
        discoveries at the MEDIUM/HIGH alphas matches an independent
        step-up implementation."""
        report = _probe(benchmark_dir, variant, "safetensors")
        tp = [f.details for f in report.findings if f.check == "behavioral_trigger_patch"]
        assert all(d["fdr_p_value"] >= d["p_value"] for d in tp)
        m = report.metadata["n_candidates_tested"]
        # unreported candidates have raw p >= 0.05, so they can't change
        # discoveries at alpha <= 0.05 -- pad with 1.0
        p = np.sort(np.array([d["p_value"] for d in tp] + [1.0] * (m - len(tp))))
        for alpha, sev in ((0.05, (Severity.MEDIUM, Severity.HIGH)), (0.01, (Severity.HIGH,))):
            ok = np.nonzero(p <= alpha * np.arange(1, m + 1) / m)[0]
            expected = 0 if ok.size == 0 else int(ok[-1] + 1)
            got = sum(1 for f in report.findings if f.check == "behavioral_trigger_patch" and f.severity in sev)
            assert got == expected
