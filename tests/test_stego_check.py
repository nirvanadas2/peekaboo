"""
tests/test_stego_check.py

Unit tests for Stage 4's statistical primitives. These test the math
in isolation using synthetic bit sequences and synthetic float32
tensors — they do NOT require the actual peekaboo synthetic benchmark
(.safetensors files). Run these first to confirm the primitives behave
correctly on known cases; then run calibrate_on_clean() separately
against the real benchmark to set final thresholds.
"""

import numpy as np
import pytest

from peekaboo.loaders.common import LoadedModel, TensorInfo
from peekaboo.pipeline.stego_check import (
    Severity,
    bit_autocorrelation,
    bit_balance_chi_square_test,
    block_chi_square_homogeneity_test,
    calibrate_on_clean,
    extract_mantissa_lsbs,
    runs_test,
    analyze_layer,
    analyze_model,
)


def _rng():
    return np.random.default_rng(42)


def _model(named_arrays: dict[str, np.ndarray]) -> LoadedModel:
    """Build a LoadedModel from a plain name -> ndarray dict, same
    convention as tests/test_statistical_check.py -- analyze_model() and
    calibrate_on_clean() take the same LoadedModel type Stage 2/3 do."""
    tensors = {
        name: TensorInfo(name=name, shape=tuple(arr.shape), dtype=str(arr.dtype), array=arr)
        for name, arr in named_arrays.items()
    }
    return LoadedModel(source_path="synthetic", source_format="safetensors", tensors=tensors)


class TestExtractMantissaLsbs:
    def test_float32_shape(self):
        arr = _rng().standard_normal(1000).astype(np.float32)
        bits = extract_mantissa_lsbs(arr, n_bits=4)
        assert bits.shape == (4000,)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_float16_shape(self):
        arr = _rng().standard_normal(500).astype(np.float16)
        bits = extract_mantissa_lsbs(arr, n_bits=2)
        assert bits.shape == (1000,)

    def test_unsupported_dtype_returns_none(self):
        arr = np.arange(100, dtype=np.int64)
        assert extract_mantissa_lsbs(arr, n_bits=1) is None

    def test_n_bits_clamped_to_mantissa_width(self):
        arr = _rng().standard_normal(10).astype(np.float32)
        # requesting more bits than float32 mantissa has (23)
        bits = extract_mantissa_lsbs(arr, n_bits=999)
        assert bits.shape == (10 * 23,)


class TestBitBalanceChiSquare:
    def test_perfectly_balanced_gives_low_statistic(self):
        bits = np.array([0, 1] * 5000, dtype=np.uint8)
        stat, p = bit_balance_chi_square_test(bits)
        assert stat == pytest.approx(0.0, abs=1e-6)
        assert p > 0.9

    def test_heavily_biased_is_significant(self):
        bits = np.zeros(10000, dtype=np.uint8)
        bits[:9900] = 0
        bits[9900:] = 1  # 99% zeros
        stat, p = bit_balance_chi_square_test(bits)
        assert p < 0.001

    def test_random_bits_usually_not_significant(self):
        bits = _rng().integers(0, 2, size=10000).astype(np.uint8)
        stat, p = bit_balance_chi_square_test(bits)
        assert p > 0.01  # true random should rarely trip a strict cutoff

    def test_empty_input(self):
        stat, p = bit_balance_chi_square_test(np.array([], dtype=np.uint8))
        assert stat == 0.0
        assert p is None


class TestBlockHomogeneity:
    def test_uniform_blocks_not_significant(self):
        bits = _rng().integers(0, 2, size=20000).astype(np.uint8)
        stat, p = block_chi_square_homogeneity_test(bits, n_blocks=20)
        assert p is not None and p > 0.01

    def test_localized_payload_is_detected(self):
        # First half natural-ish (biased 55/45), second half a fully
        # biased "payload" block sequence.
        rng = _rng()
        natural = rng.integers(0, 2, size=10000).astype(np.uint8)
        payload_block = np.ones(10000, dtype=np.uint8)
        bits = np.concatenate([natural, payload_block])
        stat, p = block_chi_square_homogeneity_test(bits, n_blocks=20)
        assert p is not None and p < 0.01

    def test_too_few_bits_returns_none(self):
        bits = np.array([0, 1, 0], dtype=np.uint8)
        stat, p = block_chi_square_homogeneity_test(bits, n_blocks=16)
        assert p is None


class TestRunsTest:
    def test_alternating_bits_flagged_nonrandom(self):
        # Perfect alternation has far more runs than random chance
        # predicts -> should be flagged non-random (very small p).
        bits = np.array([0, 1] * 5000, dtype=np.uint8)
        z, p, runs, expected = runs_test(bits)
        assert runs == 10000
        assert p is not None and p < 1e-6

    def test_single_long_run_flagged_nonrandom(self):
        bits = np.array([0] * 5000 + [1] * 5000, dtype=np.uint8)
        z, p, runs, expected = runs_test(bits)
        assert runs == 2
        assert p is not None and p < 1e-6

    def test_random_bits_usually_not_flagged(self):
        bits = _rng().integers(0, 2, size=10000).astype(np.uint8)
        z, p, runs, expected = runs_test(bits)
        assert p is not None and p > 0.01

    def test_all_same_value_handled_without_crash(self):
        bits = np.zeros(1000, dtype=np.uint8)
        z, p, runs, expected = runs_test(bits)
        assert runs == 1
        assert p == 0.0


class TestBitAutocorrelation:
    def test_random_bits_low_correlation(self):
        bits = _rng().integers(0, 2, size=20000).astype(np.uint8)
        corr = bit_autocorrelation(bits, lags=(1, 2, 4, 8))
        for lag, c in corr.items():
            assert abs(c) < 0.05

    def test_periodic_pattern_detected_at_matching_lag(self):
        # Repeat an 8-bit "byte" pattern -> strong correlation at lag 8
        pattern = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        bits = np.tile(pattern, 2000)
        corr = bit_autocorrelation(bits, lags=(1, 8))
        assert abs(corr[8]) > 0.9

    def test_too_short_sequence_skips_lag(self):
        bits = np.array([0, 1, 0], dtype=np.uint8)
        corr = bit_autocorrelation(bits, lags=(1, 8))
        assert 8 not in corr
        assert 1 in corr


class TestAnalyzeLayerAndModel:
    def test_clean_layer_mostly_info_severity(self):
        arr = _rng().standard_normal(5000).astype(np.float32)
        findings = analyze_layer("layer.weight", arr, n_bits=4)
        # Not asserting zero findings above INFO (natural rounding
        # noise can trip these before calibration) — just confirming
        # it runs end-to-end and returns the expected test set. Each
        # Finding carries its layer name in `details`, not as a
        # top-level field (shared Finding schema -- see module docstring).
        checks = {f.check for f in findings}
        assert "bit_balance_chi_square" in checks
        assert "block_homogeneity_chi_square" in checks
        assert "wald_wolfowitz_runs" in checks
        assert any("bit_autocorrelation" in c for c in checks)
        assert all(f.details.get("layer_name") == "layer.weight" for f in findings)

    def test_unsupported_dtype_layer_gives_single_info_finding(self):
        arr = np.arange(100, dtype=np.int32)
        findings = analyze_layer("counter.buffer", arr, n_bits=4)
        assert len(findings) == 1
        assert findings[0].check == "dtype_support"
        assert findings[0].severity == Severity.INFO
        assert findings[0].passed is True

    def test_analyze_model_aggregates_layers(self):
        model = _model(
            {
                "layer1.weight": _rng().standard_normal(2000).astype(np.float32),
                "layer2.weight": _rng().standard_normal(2000).astype(np.float32),
                "layer1.num_batches_tracked": np.array([5], dtype=np.int64),
            }
        )
        report = analyze_model(model, n_bits=4)
        assert report.model_path == "synthetic"
        assert report.metadata["layers_analyzed"] == 2
        assert report.metadata["layers_skipped"] == 1
        assert report.max_severity in list(Severity)
        assert report.passed == all(f.passed for f in report.findings)

    def test_calibrate_on_clean_returns_summary(self):
        model = _model(
            {
                "layer1.weight": _rng().standard_normal(3000).astype(np.float32),
            }
        )
        summary = calibrate_on_clean(model, n_bits=4)
        assert "by_severity" in summary
        assert summary["layers_analyzed"] == 1


class TestFdrCorrection:
    """Benjamini-Hochberg across each report's p-valued tests (PHASE3.md
    "Multiple-comparisons correction"). Synthetic, platform-independent:
    the real benchmark's mantissa LSBs differ at the last-bit level across
    torch/CPU builds, so exact per-model Stage 4 counts aren't asserted."""

    def _payload_model(self, n_clean_layers=10):
        rng = _rng()
        layers = {
            f"clean{i}.weight": rng.standard_normal(4000).astype(np.float32)
            for i in range(n_clean_layers)
        }
        # Payload layer: force the lowest mantissa bit to 1 everywhere --
        # an unmistakable whole-layer bias.
        payload = rng.standard_normal(4000).astype(np.float32)
        payload = (payload.view(np.uint32) | np.uint32(1)).view(np.float32)
        layers["payload.weight"] = payload
        return _model(layers)

    def test_q_values_recorded_and_never_below_raw_p(self):
        report = analyze_model(self._payload_model())
        tested = [f for f in report.findings if f.details.get("p_value") is not None]
        assert report.metadata["fdr_correction"] == "benjamini_hochberg"
        assert report.metadata["n_tests_corrected"] == len(tested) == 11 * 3
        assert all(f.details["fdr_p_value"] >= f.details["p_value"] for f in tested)

    def test_strong_payload_survives_correction(self):
        report = analyze_model(self._payload_model())
        balance = next(
            f
            for f in report.findings
            if f.check == "bit_balance_chi_square" and f.details["layer_name"] == "payload.weight"
        )
        assert balance.severity == Severity.HIGH
        assert balance.passed is False

    def test_correction_never_raises_severity(self):
        model = self._payload_model()
        order = list(Severity)
        raw = analyze_model(model, fdr_correction=False).findings
        corrected = analyze_model(model).findings
        for r, c in zip(raw, corrected):
            assert order.index(c.severity) <= order.index(r.severity)

    def test_uncorrected_mode_is_still_available(self):
        report = analyze_model(self._payload_model(), fdr_correction=False)
        assert report.metadata["fdr_correction"] is None
        assert report.metadata["n_tests_corrected"] == 0
        assert not any("fdr_p_value" in f.details for f in report.findings)


# MEDIUM+ findings after BH, exactly as measured on the committed fixture
# (tests/fixtures/benchmark) -- see PHASE3.md "Re-measurement". Includes
# the unfavorable results: one surviving false positive on clean
# (a BatchNorm running statistic), the stego payload NOT detected, and
# no signal on the (retargeted) backdoor.
_BN4_FP = ("bn4.running_var", "wald_wolfowitz_runs", "medium")
_NOISE_HIT = ("fc1.weight", "bit_balance_chi_square", "high")
_EXPECTED_MEDIUM_PLUS = {
    ("clean", "safetensors"): {_BN4_FP},
    ("clean", "pt"): {_BN4_FP},
    ("clean", "onnx"): set(),
    ("noisy", "safetensors"): {_BN4_FP, _NOISE_HIT},
    ("noisy", "pt"): {_BN4_FP, _NOISE_HIT},
    ("noisy", "onnx"): {_NOISE_HIT},
    ("steganographic", "safetensors"): {_BN4_FP},
    ("steganographic", "pt"): {_BN4_FP},
    ("steganographic", "onnx"): set(),
    ("backdoored", "safetensors"): set(),
    ("backdoored", "pt"): set(),
    ("backdoored", "onnx"): set(),
    ("combined", "safetensors"): set(),
    ("combined", "pt"): set(),
    ("combined", "onnx"): set(),
}


class TestFullBenchmarkMatrix:
    @pytest.mark.parametrize("variant,suffix", sorted(_EXPECTED_MEDIUM_PLUS))
    def test_medium_plus_after_fdr_as_measured(self, benchmark_dir, variant, suffix):
        from peekaboo.loaders import load_model

        report = analyze_model(load_model(str(benchmark_dir / f"{variant}.{suffix}")))
        got = {
            (f.details["layer_name"], f.check, f.severity.value)
            for f in report.findings
            if f.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
        }
        assert got == _EXPECTED_MEDIUM_PLUS[(variant, suffix)]


class TestInitLatticeAwareness:
    """PHASE5.md: weights that never moved from PyTorch's default init keep
    lattice-pinned low bits, which bit-0 extraction reads as a payload."""

    def _init_like(self, n, bound, seed=0):
        import torch

        g = torch.Generator().manual_seed(seed)
        return torch.empty(n).uniform_(-bound, bound, generator=g).numpy()

    def test_bound_from_paired_weight_shape(self):
        from peekaboo.pipeline.stego_check import pytorch_default_init_bound

        model = _model({"fc.weight": np.zeros((32, 64), np.float32), "fc.bias": np.zeros(32, np.float32),
                        "bn.weight": np.ones(8, np.float32)})
        assert pytorch_default_init_bound("fc.weight", model.tensors) == pytest.approx(1 / 8)
        assert pytorch_default_init_bound("fc.bias", model.tensors) == pytest.approx(1 / 8)
        assert pytorch_default_init_bound("bn.weight", model.tensors) is None

    def test_init_values_fail_bit0_null_but_pass_lattice_aware(self):
        from peekaboo.pipeline.stego_check import bit_balance_chi_square_test

        arr = self._init_like(20000, 0.125)
        _, p_bit0 = bit_balance_chi_square_test(extract_mantissa_lsbs(arr, n_bits=4))
        _, p_aware = bit_balance_chi_square_test(extract_mantissa_lsbs(arr, n_bits=4, init_bound=0.125))
        assert p_bit0 < 1e-10
        assert p_aware > 0.01

    def test_values_above_bound_use_true_lsbs(self):
        from peekaboo.pipeline.stego_check import init_lattice_floor

        arr = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        assert list(init_lattice_floor(arr, 0.125)) == [0, 0, 0]
        assert np.array_equal(extract_mantissa_lsbs(arr, 4, init_bound=0.125), extract_mantissa_lsbs(arr, 4))

    def test_analyze_model_records_lattice_mode(self):
        model = _model({"fc.weight": self._init_like(4096, 0.125).reshape(64, 64)})
        report = analyze_model(model)
        assert report.metadata["init_lattice_aware"] is True
        assert report.metadata["n_layers_lattice_aware"] == 1
        # a layer made ENTIRELY of untouched init values is not a finding
        assert not any(f.severity in (Severity.MEDIUM, Severity.HIGH) for f in report.findings)
        legacy = analyze_model(model, init_lattice_aware=False)
        assert any(f.severity == Severity.HIGH for f in legacy.findings)
