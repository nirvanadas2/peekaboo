"""Loader for pickle-based PyTorch checkpoints (.pt / .pth).

Plain `pickle.load`/`torch.load` on an untrusted checkpoint can execute
arbitrary code, because pickle opcodes can invoke arbitrary callables
(via ``__reduce__``) while deserializing. This loader never does that.

Instead it uses ``torch.load(..., weights_only=True)``, which installs
PyTorch's restricted unpickler: only a fixed allowlist of safe globals
(tensors, plain containers, numpy scalars, a handful of known simple
types) may be reconstructed. Any other global reference in the pickle
stream — a class constructor, a function call, an arbitrary object —
causes deserialization to fail closed rather than execute.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from peekaboo.loaders.common import LoadedModel, TensorInfo


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely deserialized."""


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype in (torch.bfloat16,):
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _flatten(obj: Any, prefix: str, tensors: dict[str, TensorInfo], scalars: dict[str, Any]) -> None:
    if isinstance(obj, torch.Tensor):
        array = _tensor_to_numpy(obj)
        name = prefix or "tensor"
        tensors[name] = TensorInfo(
            name=name,
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            array=array,
        )
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(value, child_prefix, tensors, scalars)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten(value, child_prefix, tensors, scalars)
        return

    if isinstance(obj, (int, float, bool, str)) or obj is None:
        if prefix:
            scalars[prefix] = obj
        return

    # Anything else that survived weights_only=True but isn't something we
    # know how to flatten (rare) — keep a repr so it isn't silently dropped.
    if prefix:
        scalars[prefix] = repr(obj)


def load_pytorch_pickle(path: str) -> LoadedModel:
    """Load a .pt/.pth checkpoint into the common LoadedModel representation.

    Raises UnsafeCheckpointError if the file contains pickled objects
    outside PyTorch's weights-only allowlist (i.e. it is not a plain
    tensor/state-dict checkpoint and cannot be safely loaded).
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises various error types here
        raise UnsafeCheckpointError(
            f"Refusing to load '{path}': not a safe weights-only checkpoint ({exc})"
        ) from exc

    tensors: dict[str, TensorInfo] = {}
    scalars: dict[str, Any] = {}
    _flatten(obj, "", tensors, scalars)

    if not tensors:
        raise UnsafeCheckpointError(
            f"'{path}' contained no tensors after safe deserialization"
        )

    return LoadedModel(
        source_path=path,
        source_format="pytorch_pickle",
        tensors=tensors,
        metadata={"scalars": scalars},
    )
