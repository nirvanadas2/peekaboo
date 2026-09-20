"""Tests for the Phase 1 pipeline gate: run_pre_checks."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from peekaboo.loaders.common import LoadedModel
from peekaboo.pipeline import ArchitectureSpec, LayerSpec, PreCheckResult, run_pre_checks
from peekaboo.schema import MetadataReport, StructuralReport


class TestHardFailShortCircuits:
    """The core contract: on a Stage 1 hard-fail, Stage 2 must never run
    and the model must never be loaded — not just "the result looks like
    it wasn't run", but verified by making both raise if called."""

    def test_unsafe_pickle_short_circuits_before_load_and_structural(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("should not be called when metadata hard-fails")

        monkeypatch.setattr("peekaboo.pipeline.gate.load_model", _must_not_be_called)
        monkeypatch.setattr("peekaboo.pipeline.gate.run_structural_check", _must_not_be_called)

        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned') or {}",))

        evil_path = tmp_path / "evil.pt"
        with open(evil_path, "wb") as f:
            pickle.dump({"state": Evil()}, f)

        result = run_pre_checks(str(evil_path))

        assert isinstance(result, PreCheckResult)
        assert result.metadata.hard_fail
        assert result.stopped_at_metadata
        assert result.structural is None
        assert result.loaded_model is None

    def test_corrupted_safetensors_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("should not be called when metadata hard-fails")

        monkeypatch.setattr("peekaboo.pipeline.gate.load_model", _must_not_be_called)
        monkeypatch.setattr("peekaboo.pipeline.gate.run_structural_check", _must_not_be_called)

        import struct

        path = tmp_path / "corrupt.safetensors"
        path.write_bytes(struct.pack("<Q", 3_000_000) + b"\x00" * 32)

        result = run_pre_checks(str(path))

        assert result.metadata.hard_fail
        assert result.structural is None
        assert result.loaded_model is None


class TestNormalFlowRunsBothStages:
    def test_clean_model_runs_both_stages(self, benchmark_dir: Path) -> None:
        result = run_pre_checks(str(benchmark_dir / "clean.safetensors"))

        assert isinstance(result, PreCheckResult)
        assert isinstance(result.metadata, MetadataReport)
        assert isinstance(result.structural, StructuralReport)
        assert isinstance(result.loaded_model, LoadedModel)
        assert not result.stopped_at_metadata

        assert not result.metadata.hard_fail
        assert result.metadata.passed
        assert result.structural.passed
        assert result.structural.mode == "self_consistency"
        assert len(result.loaded_model) > 0

    def test_soft_fail_metadata_still_runs_structural(self, tmp_path: Path) -> None:
        """A metadata finding that fails but isn't CRITICAL (e.g. an extra
        safetensors header key) must NOT stop the pipeline — only
        hard_fail does."""
        import json
        import struct

        path = tmp_path / "extra_key.safetensors"
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
                "hidden_payload": "sneaky",
            }
        }
        header_bytes = json.dumps(header).encode("utf-8")
        data = b"\x00" * 16
        path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)

        result = run_pre_checks(str(path))

        assert not result.metadata.hard_fail
        assert not result.metadata.passed  # the extra-key finding still fails the report
        assert result.structural is not None  # but Stage 2 still ran
        assert result.loaded_model is not None


class TestSpecPassthrough:
    def test_spec_is_forwarded_to_structural_check(self, benchmark_dir: Path) -> None:
        from peekaboo.loaders import load_model

        model = load_model(str(benchmark_dir / "clean.safetensors"))
        spec = ArchitectureSpec(
            layers={name: LayerSpec(shape=t.shape, dtype=t.dtype) for name, t in model.tensors.items()}
        )

        result = run_pre_checks(str(benchmark_dir / "clean.safetensors"), spec=spec)

        assert result.structural.mode == "spec_diff"
        assert result.structural.passed


class TestPreCheckResultToDict:
    def test_to_dict_on_short_circuited_result(self, tmp_path: Path) -> None:
        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned') or {}",))

        evil_path = tmp_path / "evil.pt"
        with open(evil_path, "wb") as f:
            pickle.dump({"state": Evil()}, f)

        result = run_pre_checks(str(evil_path))
        d = result.to_dict()
        assert d["stopped_at_metadata"] is True
        assert d["structural"] is None
        assert d["metadata"]["hard_fail"] is True

    def test_to_dict_on_full_result(self, benchmark_dir: Path) -> None:
        result = run_pre_checks(str(benchmark_dir / "clean.safetensors"))
        d = result.to_dict()
        assert d["stopped_at_metadata"] is False
        assert d["structural"]["mode"] == "self_consistency"
        assert d["metadata"]["passed"] is True


class TestFullBenchmarkMatrixThroughTheGate:
    @pytest.mark.parametrize(
        "variant", ["clean", "noisy", "steganographic", "backdoored", "combined"]
    )
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_variant_passes_end_to_end(self, benchmark_dir: Path, variant: str, suffix: str) -> None:
        result = run_pre_checks(str(benchmark_dir / f"{variant}.{suffix}"))
        assert not result.stopped_at_metadata
        assert result.metadata.passed
        assert result.structural.passed
