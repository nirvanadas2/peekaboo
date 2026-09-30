"""Ground-truth-tracked tampering operations applied to a trained state dict:
Gaussian noise injection and mantissa-bit steganographic payload embedding."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from peekaboo.benchmark.stego import embed_lsb

NOISE_TARGET_LAYERS = ("conv1.weight", "fc1.weight")
STEGO_TARGET_LAYERS = ("conv3.weight", "fc2.weight")
STEGO_PAYLOAD = b"PEEKABOO-STEGO-PAYLOAD-v1::this text is hidden in the mantissa bits"
STEGO_BITS_PER_VALUE = 4
# Deliberately a layer no other variant touches (noise: conv1/fc1, stego:
# conv3/fc2), so this variant's Stage 3 signal can't be confused with theirs.
OUTLIER_TARGET_LAYERS = ("conv4.weight",)


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


def inflate_layer_tails(
    state: dict[str, np.ndarray],
    layer_names: tuple[str, ...] = OUTLIER_TARGET_LAYERS,
    fraction: float = 0.01,
    scale: float = 4.0,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Multiply a random `fraction` of each layer's values by `scale`,
    giving that layer heavy tails (a Stage 3 kurtosis/entropy outlier).

    `scale` must be a power of two: that changes only the float32 exponent,
    so every mantissa bit -- what Stage 4 tests -- is left untouched, and
    this variant exercises Stage 3 alone. Returns (new_state, ground_truth)."""
    mantissa, _ = np.frexp(scale)
    if scale <= 0 or mantissa != 0.5:
        raise ValueError(f"scale must be a positive power of two, got {scale}")

    rng = np.random.default_rng(seed)
    new_state = {k: v.copy() for k, v in state.items()}
    affected = [name for name in layer_names if name in state]
    n_scaled: dict[str, int] = {}

    for name in affected:
        flat = new_state[name].astype(np.float32).ravel()
        count = int(round(fraction * flat.size))
        idx = rng.choice(flat.size, size=count, replace=False)
        flat[idx] *= np.float32(scale)
        new_state[name] = flat.reshape(state[name].shape)
        n_scaled[name] = count

    ground_truth = {
        "tamper_type": "statistical_outlier",
        "affected_layers": affected,
        "fraction": fraction,
        "scale": scale,
        "values_scaled_per_layer": n_scaled,
        "seed": seed,
    }
    return new_state, ground_truth
