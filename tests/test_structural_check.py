"""Tests for Stage 2 (Structural Consistency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from peekaboo.loaders import load_model
from peekaboo.loaders.common import LoadedModel, TensorInfo
from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.pipeline.structural_check import ArchitectureSpec, LayerSpec, run_structural_check
from peekaboo.schema import Severity


def _tensor(shape: tuple[int, ...], dtype: str = "float32", name: str = "") -> TensorInfo:
    arr = np.zeros(shape, dtype=dtype)
    return TensorInfo(name=name, shape=tuple(shape), dtype=str(arr.dtype), array=arr)


def _model(tensors: dict[str, tuple], source_format: str = "safetensors") -> LoadedModel:
    """Build a synthetic LoadedModel. `tensors` maps name -> (shape, dtype)."""
    built = {}
    for name, spec in tensors.items():
        shape = spec[0]
        dtype = spec[1] if len(spec) > 1 else "float32"
        built[name] = _tensor(shape, dtype=dtype, name=name)
    return LoadedModel(source_path="synthetic", source_format=source_format, tensors=built)


def _clean_cnn_tensors() -> dict[str, tuple]:
    """Mirrors Phase 0's TinyCNN shapes: conv1(1->8) -> conv2(8->16) ->
    flatten(16*4*4=256) -> fc(256->4)."""
    return {
        "conv1.weight": ((8, 1, 3, 3),),
        "conv1.bias": ((8,),),
        "bn1.weight": ((8,),),
        "bn1.bias": ((8,),),
        "bn1.running_mean": ((8,),),
        "bn1.running_var": ((8,),),
        "bn1.num_batches_tracked": ((), "int64"),
        "conv2.weight": ((16, 8, 3, 3),),
        "conv2.bias": ((16,),),
        "bn2.weight": ((16,),),
        "bn2.bias": ((16,),),
        "bn2.running_mean": ((16,),),
        "bn2.running_var": ((16,),),
        "bn2.num_batches_tracked": ((), "int64"),
        "fc.weight": ((4, 256),),
        "fc.bias": ((4,),),
    }


def _finding(report, check_name):
    return next(f for f in report.findings if f.check == check_name)


class TestBenchmarkModelsPassStructuralCheck:
    """None of Phase 0's tampering (noise, stego, backdoor) touches shape,
    dtype, or structure, only weight *values* — so every benchmark variant,
    in every format, must pass Stage 2 self-consistency cleanly. This is
    what proves Stage 2 isn't accidentally reacting to Stage 3+ signals."""

    @pytest.mark.parametrize(
        "variant", ["clean", "noisy", "steganographic", "backdoored", "combined"]
    )
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_variant_passes_metadata_and_structural(
        self, benchmark_dir: Path, variant: str, suffix: str
    ) -> None:
        path = str(benchmark_dir / f"{variant}.{suffix}")

        metadata_report = run_metadata_check(path)
        assert metadata_report.passed
        assert not metadata_report.hard_fail

        model = load_model(path)
        structural_report = run_structural_check(model)
        assert structural_report.mode == "self_consistency"
        assert structural_report.passed, [
            f.to_dict() for f in structural_report.findings if not f.passed
        ]


class TestCleanSyntheticModelPasses:
    def test_clean_cnn_shape_passes(self) -> None:
        model = _model(_clean_cnn_tensors())
        report = run_structural_check(model)
        assert report.mode == "self_consistency"
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]


class TestShapeChainBreak:
    def test_fc_input_does_not_match_conv_output_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        # 100 is not a multiple of conv1's out_channels (8) or conv2's (16),
        # so this cannot be explained by a legitimate flatten operation.
        tensors["fc.weight"] = ((4, 100),)
        model = _model(tensors)

        report = run_structural_check(model)
        assert not report.passed
        finding = _finding(report, "consecutive_dimension_chain")
        assert not finding.passed
        assert finding.severity == Severity.HIGH
        assert "fc.weight" in finding.details["mismatches"]

    def test_conv_to_conv_channel_mismatch_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        # conv2 now expects 5 input channels, but conv1 only produces 8.
        tensors["conv2.weight"] = ((16, 5, 3, 3),)
        model = _model(tensors)

        report = run_structural_check(model)
        assert not report.passed
        finding = _finding(report, "consecutive_dimension_chain")
        assert not finding.passed


class TestStrayDtypeOutlier:
    def test_float64_tensor_among_float32_model_is_flagged_high(self) -> None:
        tensors = _clean_cnn_tensors()
        tensors["fc.bias"] = ((4,), "float64")
        model = _model(tensors)

        report = run_structural_check(model)
        assert not report.passed
        finding = _finding(report, "dtype_uniformity")
        assert not finding.passed
        assert finding.severity == Severity.HIGH
        assert finding.details["outliers"] == {"fc.bias": "float64"}

    def test_num_batches_tracked_int64_is_not_flagged(self) -> None:
        """bn*.num_batches_tracked is legitimately int64 in an otherwise
        float32 model (this is exactly what all 15 benchmark files contain)
        — it must be exempted, not flagged as a dtype anomaly."""
        model = _model(_clean_cnn_tensors())
        report = run_structural_check(model)
        finding = _finding(report, "dtype_uniformity")
        assert finding.passed


class TestDegenerateShape:
    def test_zero_sized_dimension_is_flagged(self) -> None:
        tensors = _clean_cnn_tensors()
        tensors["conv2.weight"] = ((16, 8, 3, 0),)
        model = _model(tensors)

        report = run_structural_check(model)
        assert not report.passed
        finding = _finding(report, "no_degenerate_shapes")
        assert not finding.passed
        assert finding.severity == Severity.CRITICAL
        assert "conv2.weight" in finding.details["tensors"]
        # CRITICAL here is a severity/triage label only — confirm nothing
        # about the report shape implies control-flow / hard-fail semantics.
        assert not hasattr(report, "hard_fail")


class TestOrphanedTensorNamingHeuristic:
    def test_stray_unrelated_tensor_is_surfaced(self) -> None:
        tensors = _clean_cnn_tensors()
        tensors["debug_blob"] = ((17, 3),)
        model = _model(tensors)

        report = run_structural_check(model)
        # Informational only — never fails the report.
        finding = _finding(report, "naming_pattern_outliers")
        assert finding.passed
        assert finding.severity == Severity.INFO
        assert "debug_blob" in finding.details["outliers"]
        # The overall report may still pass; this check never contributes
        # to report.passed being False.
        assert report.passed


class TestUnusualButValidArchitecture:
    """A branching/skip-style architecture where two layers both consume
    the same upstream layer's output (not literally sequential to each
    other) must not be flagged as a shape-chain mismatch."""

    def test_two_branches_from_same_conv_output_does_not_false_positive(self) -> None:
        tensors = {
            "conv1.weight": ((8, 1, 3, 3),),
            "conv1.bias": ((8,),),
            # Both branches consume conv1's flattened output (8 * 4 * 4 = 128)
            # but have different, unrelated output widths.
            "branch_a.weight": ((32, 128),),
            "branch_a.bias": ((32,),),
            "branch_b.weight": ((16, 128),),
            "branch_b.bias": ((16,),),
        }
        model = _model(tensors)

        report = run_structural_check(model)
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]
        chain_finding = _finding(report, "consecutive_dimension_chain")
        assert chain_finding.passed

    def test_deep_uniform_width_mlp_does_not_false_positive(self) -> None:
        """An unusually deep stack of same-width linear layers (not the
        conv->linear pattern the benchmark uses) should chain cleanly."""
        tensors = {}
        for i in range(6):
            tensors[f"layer{i}.weight"] = ((64, 64),)
            tensors[f"layer{i}.bias"] = ((64,),)
        model = _model(tensors)

        report = run_structural_check(model)
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]


class TestSpecDiffMode:
    def _spec_from_tensors(self, tensors: dict[str, tuple]) -> ArchitectureSpec:
        layers = {}
        for name, spec in tensors.items():
            shape = spec[0]
            dtype = spec[1] if len(spec) > 1 else "float32"
            layers[name] = LayerSpec(shape=shape, dtype=dtype)
        return ArchitectureSpec(layers=layers)

    def test_matching_model_passes_spec_diff(self) -> None:
        tensors = _clean_cnn_tensors()
        model = _model(tensors)
        spec = self._spec_from_tensors(tensors)

        report = run_structural_check(model, spec=spec)
        assert report.mode == "spec_diff"
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]

    def test_shape_mismatch_vs_spec_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        spec = self._spec_from_tensors(tensors)

        tensors["fc.weight"] = ((4, 256 + 1),)  # only the model deviates
        model = _model(tensors)

        report = run_structural_check(model, spec=spec)
        assert report.mode == "spec_diff"
        assert not report.passed
        finding = _finding(report, "spec_shape_match")
        assert not finding.passed
        assert finding.severity == Severity.HIGH
        assert "fc.weight" in finding.details["mismatches"]

    def test_missing_layer_vs_spec_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        spec = self._spec_from_tensors(tensors)

        del tensors["fc.bias"]
        model = _model(tensors)

        report = run_structural_check(model, spec=spec)
        assert not report.passed
        finding = _finding(report, "spec_missing_layers")
        assert not finding.passed
        assert finding.severity == Severity.HIGH
        assert "fc.bias" in finding.details["missing_layers"]

    def test_extra_layer_vs_spec_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        spec = self._spec_from_tensors(tensors)

        tensors["extra.weight"] = ((2, 2),)
        model = _model(tensors)

        report = run_structural_check(model, spec=spec)
        assert not report.passed
        finding = _finding(report, "spec_extra_layers")
        assert not finding.passed
        assert finding.severity == Severity.MEDIUM
        assert "extra.weight" in finding.details["extra_layers"]

    def test_dtype_mismatch_vs_spec_fails(self) -> None:
        tensors = _clean_cnn_tensors()
        spec = self._spec_from_tensors(tensors)

        tensors["fc.bias"] = ((4,), "float64")
        model = _model(tensors)

        report = run_structural_check(model, spec=spec)
        assert not report.passed
        finding = _finding(report, "spec_dtype_match")
        assert not finding.passed
        assert finding.severity == Severity.HIGH  # float64 involved
        assert "fc.bias" in finding.details["mismatches"]

    def test_real_benchmark_model_matches_its_own_spec(self, benchmark_dir: Path) -> None:
        """Build a spec directly from the clean model and confirm it
        diffs cleanly against itself — exercises spec_diff against a real
        loaded model, not just synthetic tensors."""
        model = load_model(str(benchmark_dir / "clean.safetensors"))
        spec = ArchitectureSpec(
            layers={
                name: LayerSpec(shape=t.shape, dtype=t.dtype) for name, t in model.tensors.items()
            }
        )

        report = run_structural_check(model, spec=spec)
        assert report.mode == "spec_diff"
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]


class TestNoHardFailFieldOnStructuralReport:
    def test_structural_report_has_no_hard_fail_field(self) -> None:
        model = _model(_clean_cnn_tensors())
        report = run_structural_check(model)
        assert not hasattr(report, "hard_fail")
