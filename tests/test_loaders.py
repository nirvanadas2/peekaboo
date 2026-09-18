"""Tests that each loader correctly parses its format into the common
LoadedModel representation, and that the dispatcher/security behavior works."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from peekaboo.loaders import (
    LoadedModel,
    UnsafeCheckpointError,
    load_model,
    load_onnx,
    load_pytorch_pickle,
    load_safetensors,
)


def _assert_valid_loaded_model(lm: LoadedModel, expected_format: str) -> None:
    assert isinstance(lm, LoadedModel)
    assert lm.source_format == expected_format
    assert len(lm) > 0
    assert lm.total_params() > 0
    for name, tensor in lm.tensors.items():
        assert tensor.name == name
        assert isinstance(tensor.array, np.ndarray)
        assert tensor.shape == tuple(tensor.array.shape)
        assert tensor.array.size > 0


class TestSafetensorsLoader:
    def test_loads_clean_model(self, benchmark_dir: Path) -> None:
        lm = load_safetensors(str(benchmark_dir / "clean.safetensors"))
        _assert_valid_loaded_model(lm, "safetensors")

    def test_known_layer_present(self, benchmark_dir: Path) -> None:
        lm = load_safetensors(str(benchmark_dir / "clean.safetensors"))
        assert "conv1.weight" in lm.tensors
        assert lm.tensors["conv1.weight"].shape == (8, 1, 3, 3)


class TestPytorchPickleLoader:
    def test_loads_clean_model(self, benchmark_dir: Path) -> None:
        lm = load_pytorch_pickle(str(benchmark_dir / "clean.pt"))
        _assert_valid_loaded_model(lm, "pytorch_pickle")

    def test_known_layer_present(self, benchmark_dir: Path) -> None:
        lm = load_pytorch_pickle(str(benchmark_dir / "clean.pt"))
        assert "conv1.weight" in lm.tensors
        assert lm.tensors["conv1.weight"].shape == (8, 1, 3, 3)

    def test_refuses_arbitrary_code_execution(self, tmp_path: Path) -> None:
        """A pickle whose __reduce__ tries to run a shell command must be
        rejected, not executed, by the safe loader."""

        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned') or {}",))

        evil_path = tmp_path / "evil.pt"
        with open(evil_path, "wb") as f:
            pickle.dump({"state": Evil()}, f)

        with pytest.raises(UnsafeCheckpointError):
            load_pytorch_pickle(str(evil_path))

    def test_rejects_non_tensor_checkpoint(self, tmp_path: Path) -> None:
        empty_path = tmp_path / "empty.pt"
        torch.save({"just_a_string": "no tensors here"}, str(empty_path))
        with pytest.raises(UnsafeCheckpointError):
            load_pytorch_pickle(str(empty_path))


class TestOnnxLoader:
    def test_loads_clean_model(self, benchmark_dir: Path) -> None:
        lm = load_onnx(str(benchmark_dir / "clean.onnx"))
        _assert_valid_loaded_model(lm, "onnx")

    def test_metadata_present(self, benchmark_dir: Path) -> None:
        lm = load_onnx(str(benchmark_dir / "clean.onnx"))
        assert "opset_imports" in lm.metadata
        assert lm.metadata["num_nodes"] > 0


class TestDispatch:
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_load_model_dispatches_by_extension(self, benchmark_dir: Path, suffix: str) -> None:
        lm = load_model(str(benchmark_dir / f"clean.{suffix}"))
        assert len(lm) > 0

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "model.xyz"
        bogus.write_text("not a model")
        with pytest.raises(ValueError):
            load_model(str(bogus))


class TestAllVariantsLoadInAllFormats:
    """Every benchmark variant must be loadable in every format it was saved in."""

    @pytest.mark.parametrize(
        "variant", ["clean", "noisy", "steganographic", "backdoored", "combined"]
    )
    @pytest.mark.parametrize("suffix", ["safetensors", "pt", "onnx"])
    def test_variant_loads(self, benchmark_dir: Path, variant: str, suffix: str) -> None:
        lm = load_model(str(benchmark_dir / f"{variant}.{suffix}"))
        assert len(lm) > 0
        assert lm.total_params() > 0
