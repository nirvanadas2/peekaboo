"""LSB-style steganographic payload embedding in float32 mantissa bits.

Embeds a byte payload into the low-order `bits_per_value` bits of each
float32 value's bit pattern (which fall within the 23-bit mantissa),
which perturbs each value by a relative amount on the order of 2^-19
or smaller — far below normal training noise, but detectable by a
targeted bit-level scan.
"""

from __future__ import annotations

import numpy as np

MAX_BITS_PER_VALUE = 8


def _bytes_to_bitarray(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bitarray_to_bytes(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    return np.packbits(bits).tobytes()


def capacity_bits(array: np.ndarray, bits_per_value: int) -> int:
    return array.size * bits_per_value


def embed_lsb(array: np.ndarray, payload: bytes, bits_per_value: int = 4) -> tuple[np.ndarray, int]:
    """Embed `payload` into the low bits of `array` (must be float32).

    Returns (modified_array, num_bits_actually_embedded). If the payload
    doesn't fit in the array's capacity, it is truncated.
    """
    if array.dtype != np.float32:
        raise ValueError("embed_lsb only supports float32 arrays")
    if not (1 <= bits_per_value <= MAX_BITS_PER_VALUE):
        raise ValueError(f"bits_per_value must be between 1 and {MAX_BITS_PER_VALUE}")

    flat_u32 = array.reshape(-1).view(np.uint32).copy()
    n_slots = flat_u32.size
    capacity = n_slots * bits_per_value

    payload_bits = _bytes_to_bitarray(payload)
    n_bits = min(len(payload_bits), capacity)
    if n_bits == 0:
        return array.copy(), 0

    n_values = -(-n_bits // bits_per_value)  # ceil division
    bits_used = payload_bits[:n_bits]
    pad = n_values * bits_per_value - n_bits
    if pad:
        bits_used = np.concatenate([bits_used, np.zeros(pad, dtype=bits_used.dtype)])
    bits_grid = bits_used.reshape(n_values, bits_per_value)

    values = np.zeros(n_values, dtype=np.uint32)
    for b in range(bits_per_value):
        values = (values << 1) | bits_grid[:, b].astype(np.uint32)

    mask = np.uint32((1 << bits_per_value) - 1)
    flat_u32[:n_values] = (flat_u32[:n_values] & ~mask) | values

    modified = flat_u32.view(np.float32).reshape(array.shape)
    return modified, n_bits


def extract_lsb(array: np.ndarray, n_bits: int, bits_per_value: int = 4) -> bytes:
    """Inverse of embed_lsb: recover the payload bytes given how many bits
    were embedded and the bits-per-value setting used."""
    if array.dtype != np.float32:
        raise ValueError("extract_lsb only supports float32 arrays")

    flat_u32 = array.reshape(-1).view(np.uint32)
    n_values = -(-n_bits // bits_per_value)
    mask = np.uint32((1 << bits_per_value) - 1)
    values = (flat_u32[:n_values] & mask).astype(np.uint8)

    bits = np.zeros((n_values, bits_per_value), dtype=np.uint8)
    for b in range(bits_per_value):
        shift = bits_per_value - 1 - b
        bits[:, b] = (values >> shift) & 1

    flat_bits = bits.reshape(-1)[:n_bits]
    return _bitarray_to_bytes(flat_bits)
