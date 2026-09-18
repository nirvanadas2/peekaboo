"""Ground-truth-tracked tampering operations applied to a trained state dict:
Gaussian noise injection and mantissa-bit steganographic payload embedding."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from peekaboo.benchmark.stego import embed_lsb

NOISE_TARGET_LAYERS = ("conv1.weight", "fc.weight")
STEGO_TARGET_LAYERS = ("conv2.weight", "fc.weight")
STEGO_PAYLOAD = b"PEEKABOO-STEGO-PAYLOAD-v1::this text is hidden in the mantissa bits"
STEGO_BITS_PER_VALUE = 4


def add_gaussian_noise(
    state: dict[str, np.ndarray],
    layer_names: tuple[str, ...] = NOISE_TARGET_LAYERS,
    std: float = 0.05,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Add N(0, std^2) noise to the given layers. Returns (new_state, ground_truth)."""
    rng = np.random.default_rng(seed)
    new_state = {k: v.copy() for k, v in state.items()}
    affected = [name for name in layer_names if name in state]

    for name in affected:
        arr = new_state[name].astype(np.float32)
        noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
        new_state[name] = arr + noise

    ground_truth = {
        "tamper_type": "gaussian_noise",
        "affected_layers": affected,
        "noise_std": std,
        "seed": seed,
    }
    return new_state, ground_truth


def embed_steganographic_payload(
    state: dict[str, np.ndarray],
    layer_names: tuple[str, ...] = STEGO_TARGET_LAYERS,
    payload: bytes = STEGO_PAYLOAD,
    bits_per_value: int = STEGO_BITS_PER_VALUE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Embed `payload` into the low-order mantissa bits of the given layers,
    splitting it across them in order until it is fully embedded or capacity
    runs out. Returns (new_state, ground_truth)."""
    new_state = {k: v.copy() for k, v in state.items()}
    candidates = [name for name in layer_names if name in state]

    remaining = payload
    per_layer: dict[str, int] = {}
    modified_layers: list[str] = []
    for name in candidates:
        if not remaining:
            break
        arr = new_state[name].astype(np.float32)
        modified, n_bits = embed_lsb(arr, remaining, bits_per_value=bits_per_value)
        if n_bits == 0:
            continue
        new_state[name] = modified
        per_layer[name] = n_bits
        modified_layers.append(name)

        n_bytes_consumed = n_bits // 8
        remaining = remaining[n_bytes_consumed:] if n_bits % 8 == 0 else b""

    ground_truth = {
        "tamper_type": "steganographic_payload",
        "affected_layers": modified_layers,
        "bits_per_value": bits_per_value,
        "bits_embedded_per_layer": per_layer,
        "payload_length_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_preview": payload[:32].decode("utf-8", errors="replace"),
    }
    return new_state, ground_truth
