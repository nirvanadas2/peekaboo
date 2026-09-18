"""Loader for .safetensors files.

SafeTensors is a safe-by-construction format (no arbitrary code execution),
so this loader can use the reference `safetensors` library directly.
"""

from __future__ import annotations

from safetensors import safe_open

from peekaboo.loaders.common import LoadedModel, TensorInfo


def load_safetensors(path: str) -> LoadedModel:
    """Load a .safetensors file into the common LoadedModel representation."""
    tensors: dict[str, TensorInfo] = {}
    file_metadata: dict[str, str] = {}

    with safe_open(path, framework="numpy") as f:
        file_metadata = dict(f.metadata() or {})
        for name in f.keys():
            array = f.get_tensor(name)
            tensors[name] = TensorInfo(
                name=name,
                shape=tuple(array.shape),
                dtype=str(array.dtype),
                array=array,
            )

    return LoadedModel(
        source_path=path,
        source_format="safetensors",
        tensors=tensors,
        metadata=file_metadata,
    )
