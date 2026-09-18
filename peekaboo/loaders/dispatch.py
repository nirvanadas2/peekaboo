"""Single entry point that picks the right loader based on file extension."""

from __future__ import annotations

from pathlib import Path

from peekaboo.loaders.common import LoadedModel
from peekaboo.loaders.onnx_loader import load_onnx
from peekaboo.loaders.pytorch_pickle_loader import load_pytorch_pickle
from peekaboo.loaders.safetensors_loader import load_safetensors

_LOADERS_BY_SUFFIX = {
    ".safetensors": load_safetensors,
    ".pt": load_pytorch_pickle,
    ".pth": load_pytorch_pickle,
    ".onnx": load_onnx,
}


def load_model(path: str) -> LoadedModel:
    """Load any supported model file into the common LoadedModel representation."""
    suffix = Path(path).suffix.lower()
    loader = _LOADERS_BY_SUFFIX.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported model file extension '{suffix}' for '{path}'. "
            f"Supported: {sorted(_LOADERS_BY_SUFFIX)}"
        )
    return loader(path)
