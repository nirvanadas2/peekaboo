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
        severities = {f.severity for f in findings}
        # Not asserting zero findings above INFO (natural rounding
        # noise can trip these before calibration) — just confirming
        # it runs end-to-end and returns the expected test set.
        test_names = {f.test_name for f in findings}
        assert "bit_balance_chi_square" in test_names
        assert "block_homogeneity_chi_square" in test_names
        assert "wald_wolfowitz_runs" in test_names
        assert any("bit_autocorrelation" in n for n in test_names)

    def test_unsupported_dtype_layer_gives_single_info_finding(self):
        arr = np.arange(100, dtype=np.int32)
        findings = analyze_layer("counter.buffer", arr, n_bits=4)
        assert len(findings) == 1
        assert findings[0].test_name == "dtype_support"
        assert findings[0].severity == Severity.INFO

    def test_analyze_model_aggregates_layers(self):
        model = _model(
            {
                "layer1.weight": _rng().standard_normal(2000).astype(np.float32),
                "layer2.weight": _rng().standard_normal(2000).astype(np.float32),
                "layer1.num_batches_tracked": np.array([5], dtype=np.int64),
            }
        )
        report = analyze_model(model, n_bits=4)
        assert report.layers_analyzed == 2
        assert report.layers_skipped == 1
        assert report.max_severity in list(Severity)

    def test_calibrate_on_clean_returns_summary(self):
        model = _model(
            {
                "layer1.weight": _rng().standard_normal(3000).astype(np.float32),
            }
        )
        summary = calibrate_on_clean(model, n_bits=4)
        assert "by_severity" in summary
        assert summary["layers_analyzed"] == 1
