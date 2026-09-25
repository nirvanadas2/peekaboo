"""Caller-side `forward_fn` builders for the benchmark's own TinyCNN files.

VALIDATION/CALIBRATION ONLY -- never the shipped general path. Stage 5
(`peekaboo.pipeline.behavioral_probe`) requires a caller-supplied
`forward_fn` because safetensors/pickle carry no computation graph (see
PHASE4.md Sec 1). For this project's benchmark the caller legitimately
knows the architecture -- `TinyCNN` is a fixed class this repo controls --
so building one here is fair. It must never be mistaken for how Stage 5
would behave on an unknown public-hub upload.

For `.onnx` files the graph itself is executed via `onnx.reference`
(the pure-Python reference evaluator that ships inside the `onnx` package,
already a dependency) -- no `onnxruntime`. It's slow (~50s per default-
budget Stage 5 run on TinyCNN vs <1s via torch), but needs nothing new.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from peekaboo.benchmark.models import TinyCNN
from peekaboo.loaders import LoadedModel

TINYCNN_INPUT_SHAPE: tuple[int, ...] = (1, 16, 16)
TINYCNN_NUM_CLASSES = 4


def tinycnn_forward_fn(model: LoadedModel) -> Callable[[np.ndarray], np.ndarray]:
    """Batch-in/logits-out forward_fn: loads a safetensors/pt `LoadedModel`'s
    tensors into a fresh `TinyCNN` (strict `load_state_dict`, so a tensor
    mismatch raises rather than running a half-loaded model)."""
    net = TinyCNN()
    net.load_state_dict({k: torch.from_numpy(np.array(t.array)) for k, t in model.tensors.items()})
    net.eval()

    def forward_fn(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))).numpy()

    return forward_fn


def onnx_reference_forward_fn(onnx_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Batch-in/logits-out forward_fn executing the .onnx file's own graph
    with `onnx.reference.ReferenceEvaluator`. The benchmark's export has a
    fixed batch dim of 1 (no `dynamic_axes` in `io_utils.save_onnx`), so
    samples are run one at a time."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    proto = onnx.load(str(onnx_path))
    session = ReferenceEvaluator(proto)
    input_name = proto.graph.input[0].name

    def forward_fn(batch: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        return np.concatenate(
            [session.run(None, {input_name: batch[i : i + 1]})[0] for i in range(batch.shape[0])]
        )

    return forward_fn


def benchmark_forward_fn(model: LoadedModel) -> Callable[[np.ndarray], np.ndarray]:
    """Dispatch on the benchmark file's format."""
    if Path(model.source_path).suffix == ".onnx":
        return onnx_reference_forward_fn(model.source_path)
    return tinycnn_forward_fn(model)
