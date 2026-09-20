"""Tests for Stage 3 (Statistical Analysis).

Test-expectation scoping (see also PHASE2.md, "Second benchmark redesign"
section, for the full story of how these numbers were arrived at):

TinyCNN was deepened from 3 to 7 conv/linear weight-bearing layers
specifically so Stage 3's relative-outlier mode (population >= 6) would
actually run on the real benchmark instead of always falling back. It
does now, on all 5 variants in all 3 formats. Two structural corrections
were needed to get there without false positives (log-scale comparison
for std, since it scales multiplicatively with a layer's fan-in; entropy
normalized by its own adaptive bin count's ceiling) — both target
confirmed root causes in the comparison method, not tuned thresholds.

The honest result on this specific benchmark, reported as observed
rather than engineered to a target:

  - CLEAN: zero findings, all formats — false-positive control holds.
  - STEGANOGRAPHIC: zero findings, all formats — correct and expected;
    mantissa-LSB steganography is built to be invisible at this level.
    That's Stage 4's job (bit-level tests), not Stage 3's.
  - NOISY: zero findings, all formats. The hoped-for "noise = Stage 3's
    strongest case" did NOT materialize here: the affected layers' std
    genuinely shifts, but by less than the population's own natural
    layer-to-layer spread (driven by differing fan-in across a 7-layer
    architecture) — so it doesn't register as a robust outlier. Verified
    directly: the actual z-scores were inspected layer-by-layer and stay
    under the MEDIUM threshold, not just barely miss a lenient check.
  - BACKDOORED / COMBINED: zero findings on safetensors/pt; a single weak
    MEDIUM `kurtosis_outliers` flag on `conv4.weight` (`onnx::Conv_66`
    post-BN-fusion) on ONNX only. Backdoor training has no single
    "target layer" (the whole model is retrained on poisoned data), so
    this is at least plausibly a genuine partial signal rather than an
    artifact — but it's weak and format-inconsistent, not a reliable
    detection.

This is intentionally not "fixed" further (e.g. by lowering thresholds)
per an explicit instruction not to tune until tests are green — Stage 3's
recall on this benchmark's specific tampering magnitudes is genuinely
weak, and that's recorded here rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

from peekaboo.loaders import load_model
from peekaboo.loaders.common import LoadedModel, TensorInfo
from peekaboo.pipeline.statistical_check import modified_z_scores, run_statistical_check
from peekaboo.schema import Severity


def _tensor(shape: tuple[int, ...], values: np.ndarray | None = None, dtype: str = "float32") -> TensorInfo:
    if values is None:
        arr = np.zeros(shape, dtype=dtype)
    else:
        arr = values.astype(dtype).reshape(shape)
    return TensorInfo(name="", shape=tuple(shape), dtype=str(arr.dtype), array=arr)


def _model(tensors: dict[str, TensorInfo]) -> LoadedModel:
    for name, t in tensors.items():
        t.name = name
    return LoadedModel(source_path="synthetic", source_format="safetensors", tensors=tensors)


def _finding(report, check_name):
    return next(f for f in report.findings if f.check == check_name)


def _random_weight(rng: np.random.Generator, n: int, std: float, mean: float = 0.0) -> np.ndarray:
    return rng.normal(loc=mean, scale=std, size=n).astype(np.float32)


class TestRobustOutlierLogicOnSyntheticArrays:
    """The core detection logic (modified_z_scores), tested directly
    against hand-built stat arrays — independent of any model loading."""

    def test_single_extreme_outlier_among_many_normal_is_caught(self) -> None:
        normal = np.array([0.10, 0.11, 0.09, 0.105, 0.095, 0.102, 0.098, 0.107])
        with_outlier = np.append(normal, 5.0)

        z = modified_z_scores(with_outlier)

        assert abs(z[-1]) >= 5.0  # HIGH threshold
        # And the outlier doesn't drag the "normal" cluster's own z-scores
        # out of a sane range — this is exactly the breakdown-point
        # property median/MAD is chosen for over mean/stdev.
        assert np.all(np.abs(z[:-1]) < 3.5)

    def test_does_not_false_positive_on_realistic_depth_wise_variation(self) -> None:
        """A real (untampered) network's per-layer std often decreases
        smoothly with depth due to differing fan-in — this must not read
        as a HIGH-severity outlier just because the values aren't
        identical. MEDIUM is tolerated (it's a "worth a second look"
        label, not a tampering claim); HIGH is not."""
        realistic_stds = np.array([0.28, 0.19, 0.15, 0.10, 0.09, 0.07, 0.06, 0.05, 0.045, 0.04])

        z = modified_z_scores(realistic_stds)

        assert np.all(np.abs(z) < 5.0)  # no HIGH-severity outlier

    def test_does_not_false_positive_on_varied_layer_types_and_sizes(self) -> None:
        """Different layer kinds/sizes naturally produce different (but
        legitimate) statistics; smooth, explicable variation should not
        trigger a HIGH flag."""
        # e.g. a small early conv, a big middle conv, a huge fc, a tiny head
        stds = np.array([0.22, 0.10, 0.18, 0.055, 0.09, 0.15, 0.07])

        z = modified_z_scores(stds)

        assert np.all(np.abs(z) < 5.0)

    def test_all_identical_values_produce_zero_z_scores(self) -> None:
        """MAD collapsing to zero (e.g. a population of near-identical
        values) must not raise or divide by zero."""
        values = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        z = modified_z_scores(values)
        assert np.all(z == 0.0)


class TestRelativeModeOnSyntheticModel:
    """Exercises the full run_statistical_check relative-outlier path
    (mode == "relative_outlier") on a model with enough weight matrices to
    actually reach it — TinyCNN's 3 layers never do (see module docstring
    above), so this is where the primary detection mode gets proven out."""

    def _many_layer_model(self, rng: np.random.Generator, n_layers: int = 8) -> LoadedModel:
        tensors = {}
        for i in range(n_layers):
            values = _random_weight(rng, n=200, std=0.1)
            tensors[f"layer{i}.weight"] = _tensor((20, 10), values)
        return _model(tensors)

    def test_clean_many_layer_model_has_no_high_severity_findings(self) -> None:
        rng = np.random.default_rng(0)
        model = self._many_layer_model(rng)

        report = run_statistical_check(model)

        assert report.mode == "relative_outlier"
        assert not any(f.severity == Severity.HIGH for f in report.findings)

    def test_one_noised_layer_among_many_is_flagged(self) -> None:
        rng = np.random.default_rng(0)
        tensors = {}
        for i in range(8):
            std = 0.1 if i != 4 else 0.6  # layer4 gets a much wider distribution
            values = _random_weight(rng, n=300, std=std)
            tensors[f"layer{i}.weight"] = _tensor((30, 10), values)
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.mode == "relative_outlier"
        finding = _finding(report, "std_outliers")
        assert not finding.passed
        assert "layer4.weight" in finding.details["outliers"]

    def test_biases_and_norm_params_excluded_from_population_even_when_plentiful(self) -> None:
        """Even with enough 1-D tensors to nominally hit the population
        threshold, they must not be pulled into the weight-matrix
        comparison — norm-layer affine params have a systematically
        different role (see _weight_bearing_population docstring)."""
        rng = np.random.default_rng(1)
        tensors = {}
        for i in range(8):
            tensors[f"layer{i}.weight"] = _tensor((20, 10), _random_weight(rng, 200, std=0.1))
        # Add a batch of norm-affine-like 1-D tensors centered near 1.0 —
        # these must not appear in the weight-bearing population at all.
        for i in range(8):
            tensors[f"bn{i}.weight"] = _tensor((16,), np.full(16, 1.0, dtype=np.float32))

        model = _model(tensors)
        report = run_statistical_check(model)

        assert report.metadata["num_layers_analyzed"] == 8
        assert all(name.startswith("layer") for name in report.metadata["weight_bearing_population"])


class TestSmallModelFallback:
    """Explicit fallback-path fixture: fewer than _MIN_LAYERS_FOR_RELATIVE
    weight matrices, independent of the real (also-small) benchmark."""

    def test_small_model_uses_fallback_mode(self) -> None:
        tensors = {
            "layer0.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100)),
            "layer1.weight": _tensor((10, 10), np.random.default_rng(1).normal(0, 0.1, 100)),
        }
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.mode == "absolute_fallback"
        for finding in report.findings:
            assert finding.details.get("mode") == "absolute_fallback"

    def test_small_model_within_conservative_ranges_passes(self) -> None:
        tensors = {
            "layer0.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100)),
            "layer1.weight": _tensor((10, 10), np.random.default_rng(1).normal(0, 0.1, 100)),
        }
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.mode == "absolute_fallback"
        assert report.passed

    def test_small_model_outside_conservative_range_is_flagged_medium_not_high(self) -> None:
        """A fallback flag is deliberately capped at MEDIUM — it must
        never be presented with relative-mode's HIGH confidence."""
        tensors = {
            "layer0.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100)),
            # Degenerate: every value identical -> near-zero entropy, well
            # below the conservative fallback minimum.
            "layer1.weight": _tensor((10, 10), np.full(100, 0.5, dtype=np.float32)),
        }
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.mode == "absolute_fallback"
        assert not report.passed
        finding = _finding(report, "entropy_outliers")
        assert not finding.passed
        assert finding.severity == Severity.MEDIUM  # never HIGH/CRITICAL in fallback mode
        assert "layer1.weight" in finding.details["outliers"]
        assert "[fallback" in finding.message

    def test_fewer_than_two_weight_matrices_reports_nothing_to_compare(self) -> None:
        tensors = {"only.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100))}
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.passed
        for stat in ("mean", "std", "kurtosis", "entropy"):
            finding = _finding(report, f"{stat}_outliers")
            assert finding.passed
            assert "nothing to compare" in finding.message

    def test_no_weight_bearing_tensors_at_all(self) -> None:
        tensors = {"scale": _tensor((16,), np.ones(16, dtype=np.float32))}
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.passed
        assert _finding(report, "statistical_coverage").passed


class TestNeverHardFails:
    def test_statistical_report_has_no_hard_fail_field(self) -> None:
        model = _model({"only.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100))})
        report = run_statistical_check(model)
        assert not hasattr(report, "hard_fail")

    def test_fallback_severity_is_never_high_or_critical(self) -> None:
        tensors = {
            "layer0.weight": _tensor((10, 10), np.random.default_rng(0).normal(0, 0.1, 100)),
            "layer1.weight": _tensor((10, 10), np.full(100, 999.0, dtype=np.float32)),  # wildly out of range
        }
        model = _model(tensors)

        report = run_statistical_check(model)

        assert report.mode == "absolute_fallback"
        assert not report.passed
        assert all(f.severity in (Severity.MEDIUM, Severity.INFO) for f in report.findings)


class TestFullBenchmarkMatrix:
    """All 5x3 Phase 0 benchmark variants, with expectations scoped per
    the module docstring above — see PHASE2.md for the full rationale."""

    @pytest.mark.parametrize(
        "variant", ["clean", "noisy", "steganographic", "backdoored", "combined"]
    )
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_variant_reaches_relative_mode_on_this_benchmark(self, benchmark_dir, variant, suffix) -> None:
        """TinyCNN has 7 conv/linear weight matrices in every format
        (BatchNorm gets algebraically fused into the preceding conv's
        weight+bias in the ONNX export, but the conv/linear tensor count
        is unaffected) — a deterministic, structural fact independent of
        the benchmark's random weight values, so it holds regardless of
        Phase 0's generator non-determinism. This is what unlocks Stage
        3's primary detection mode on the real benchmark at all."""
        model = load_model(str(benchmark_dir / f"{variant}.{suffix}"))
        report = run_statistical_check(model)
        assert report.mode == "relative_outlier"
        assert report.metadata["num_layers_analyzed"] == 7

    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_clean_has_no_findings(self, benchmark_dir, suffix) -> None:
        """False-positive control. This did NOT hold before the log-scale
        std / normalized-entropy corrections were added — the smallest-
        fan-in layer (conv1.weight) registered as a spurious outlier on
        this exact clean model purely from architectural scale, not
        tampering. See PHASE2.md."""
        model = load_model(str(benchmark_dir / f"clean.{suffix}"))
        report = run_statistical_check(model)
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]

    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_steganographic_is_not_detected_here_by_design(self, benchmark_dir, suffix) -> None:
        """Must NOT be caught by Stage 3 — mantissa-LSB steganography is
        specifically invisible at this level. Bit-level tests are Stage 4's
        job. If this ever starts failing, that's a signal to investigate
        Phase 0's embedding density (per the task brief), not a win."""
        model = load_model(str(benchmark_dir / f"steganographic.{suffix}"))
        report = run_statistical_check(model)
        assert report.passed, (
            "Stage 3 detected the steganographic variant — investigate whether Phase 0's "
            f"embedding density is too crude to be statistically invisible: {[f.to_dict() for f in report.findings if not f.passed]}"
        )

    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_noisy_shows_no_detection_on_this_benchmark(self, benchmark_dir, suffix) -> None:
        """Reported honestly, not forced: noise injection's affected
        layers shift by less than this 7-layer architecture's own
        natural fan-in-driven layer-to-layer spread, so relative-outlier
        detection does not catch it here even after the false-positive
        corrections. See PHASE2.md for the layer-by-layer z-score
        analysis this is based on. This is NOT asserted as a desirable
        outcome — it's the measured one."""
        model = load_model(str(benchmark_dir / f"noisy.{suffix}"))
        report = run_statistical_check(model)
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]

    @pytest.mark.parametrize("variant", ["backdoored", "combined"])
    def test_backdoored_and_combined_show_no_detection_on_pytorch_native_formats(
        self, benchmark_dir, variant
    ) -> None:
        """safetensors and pt (both carry the un-fused BatchNorm layers,
        excluded from Stage 3's comparison population) show zero detection
        for backdoor training on this benchmark."""
        for suffix in ("safetensors", "pt"):
            model = load_model(str(benchmark_dir / f"{variant}.{suffix}"))
            report = run_statistical_check(model)
            assert report.passed, (suffix, [f.to_dict() for f in report.findings if not f.passed])

    @pytest.mark.parametrize("variant", ["backdoored", "combined"])
    def test_backdoored_and_combined_show_weak_onnx_only_signal(self, benchmark_dir, variant) -> None:
        """A weak, real (not forced) partial signal: ONNX's BN-fusion
        changes conv4's effective weight values, and backdoor training
        (which has no single "target layer" — the whole model is
        retrained on poisoned data) shifts conv4's kurtosis just past the
        MEDIUM threshold. Never observed to reach HIGH, and never observed
        on safetensors/pt for the identical variant — recorded as the
        weak, format-inconsistent signal it is, not oversold."""
        model = load_model(str(benchmark_dir / f"{variant}.onnx"))
        report = run_statistical_check(model)
        failed = [f for f in report.findings if not f.passed]
        assert all(f.severity == Severity.MEDIUM for f in failed)
