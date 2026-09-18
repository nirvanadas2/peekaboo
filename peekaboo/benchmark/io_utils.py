"""Conversions between torch state dicts, plain numpy state dicts, and the
three on-disk formats Peekaboo's loaders understand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_safetensors_file


def state_dict_to_numpy(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def load_numpy_state_dict(model: torch.nn.Module, state: dict[str, np.ndarray]) -> None:
    torch_state = {k: torch.from_numpy(v.copy()) for k, v in state.items()}
    model.load_state_dict(torch_state)


def save_safetensors(state: dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous arrays; state_dict_to_numpy already copies.
    save_safetensors_file(state, path)


def save_pt(state: dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch_state = {k: torch.from_numpy(v.copy()) for k, v in state.items()}
    torch.save(torch_state, path)


def save_onnx(model: torch.nn.Module, state: dict[str, np.ndarray], path: str, dummy_input: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    load_numpy_state_dict(model, state)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )
