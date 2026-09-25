"""Shared pytest fixtures.

`benchmark_dir`/`manifest` point at the committed canonical benchmark in
tests/fixtures/benchmark/ -- NOT a fresh generation -- because regenerated
weights differ at the mantissa-LSB level across torch/CPU builds, which
made Stage 4's measured results machine-dependent. See
tests/fixtures/benchmark/PROVENANCE.md.

`generated_benchmark_dir`/`generated_manifest` generate the benchmark once
per session into a temp directory, for tests of the generator itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peekaboo.benchmark.generate import generate_benchmark

FIXTURE_BENCHMARK_DIR = Path(__file__).parent / "fixtures" / "benchmark"


@pytest.fixture(scope="session")
def benchmark_dir() -> Path:
    return FIXTURE_BENCHMARK_DIR


@pytest.fixture(scope="session")
def manifest(benchmark_dir: Path) -> list[dict]:
    return json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def generated_benchmark_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("peekaboo_benchmark")
    generate_benchmark(str(out_dir), seed=0)
    return out_dir


@pytest.fixture(scope="session")
def generated_manifest(generated_benchmark_dir: Path) -> list[dict]:
    return json.loads((generated_benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
