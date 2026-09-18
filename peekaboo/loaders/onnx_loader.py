"""Loader for ONNX models.

ONNX is a protobuf-based format — parsing it does not execute arbitrary
code, so this loader can use the `onnx` library directly to read the
graph's initializers (the trained weight tensors).
"""

from __future__ import annotations

import onnx
from onnx import numpy_helper

from peekaboo.loaders.common import LoadedModel, TensorInfo


def load_onnx(path: str) -> LoadedModel:
    """Load an .onnx model's weight tensors into the common representation."""
    model = onnx.load(path, load_external_data=True)

    tensors: dict[str, TensorInfo] = {}
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        tensors[initializer.name] = TensorInfo(
            name=initializer.name,
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            array=array,
        )

    metadata = {
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "ir_version": model.ir_version,
        "opset_imports": [
            {"domain": imp.domain, "version": imp.version}
            for imp in model.opset_import
        ],
        "graph_name": model.graph.name,
        "num_nodes": len(model.graph.node),
    }

    return LoadedModel(
        source_path=path,
        source_format="onnx",
        tensors=tensors,
        metadata=metadata,
    )
