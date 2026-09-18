# Peekaboo

Peekaboo is a pre-deployment scanner that inspects AI model weight files (SafeTensors, pickle-based PyTorch checkpoints, and ONNX) to detect steganographic payloads hidden in low-order mantissa bits, backdoor triggers that cause targeted misbehavior on specific inputs, and statistically abnormal parameter distributions — producing a Model Risk Score with per-layer flags and explanations before a model is trusted and deployed.

## Status

Phase 0: Setup & Ground Truth. Currently provides format-agnostic model loaders, a synthetic benchmark generator for ground-truth testing (clean / noisy / steganographic / backdoored / combined variants), and a stub schema for the eventual Model Risk Score. Detection/scoring logic is not yet implemented.

## Layout

- `peekaboo/loaders/` — load `.safetensors`, `.pt`/`.pth`, and `.onnx` files into a common internal representation (layer names, shapes, dtypes, raw tensors).
- `peekaboo/benchmark/` — generates small synthetic models with known ground-truth tampering for testing detectors.
- `peekaboo/schema/` — data classes for the Model Risk Score output.
- `tests/` — unit tests for loaders and the benchmark generator.

## Install

```
pip install -e ".[dev]"
```

## Run tests

```
pytest
```
