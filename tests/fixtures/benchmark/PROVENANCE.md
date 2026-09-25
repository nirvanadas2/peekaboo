# Canonical benchmark fixture

`generate_benchmark("tests/fixtures/benchmark", seed=0)`, generated once and
committed so every machine measures the **same bytes**.

## Why a committed fixture

Regenerating with `seed=0` is deterministic *within* one environment. It is
NOT bit-identical across torch/CPU builds. Weights agree to about 3 decimal
places, and the lowest mantissa bits differ. Stage 4 tests exactly those bits,
so its per-model results changed from machine to machine. PHASE3.md's original
table did not reproduce on a second machine; see PHASE3.md "Reproducibility".
Every benchmark-backed test reads these files (`tests/conftest.py::benchmark_dir`).
The generator itself is still exercised separately against a fresh generation
(`generated_benchmark_dir`).

## Generated with

- trigger: 3×3, value 6.0, top-left, **target class 3** (retargeted from 0; see PHASE4.md "Finding 1")
- torch 2.14.0+cpu, numpy 2.5.3, onnx 1.23.0, safetensors 0.8.0
- Python 3.13.7, Windows 11

## Regenerating

Only regenerate this fixture on purpose, when the generator changes. Every
benchmark-derived number in PHASE2-4.md must then be re-measured, and this
file updated.
