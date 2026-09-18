"""Format-agnostic internal representation of a loaded model.

Every loader in peekaboo.loaders (safetensors, pickle/.pt, onnx) returns a
LoadedModel so that downstream analysis code never needs to know which
serialization format the weights originally came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TensorInfo:
    """One named weight tensor plus its metadata, in a format-neutral form."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    array: np.ndarray

    @property
    def num_params(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


@dataclass
class LoadedModel:
    """Common representation produced by every peekaboo loader."""

    source_path: str
    source_format: str  # "safetensors" | "pytorch_pickle" | "onnx"
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def layer_names(self) -> list[str]:
        return list(self.tensors.keys())

    def total_params(self) -> int:
        return sum(t.num_params for t in self.tensors.values())

    def __len__(self) -> int:
        return len(self.tensors)
