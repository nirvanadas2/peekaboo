"""Shared pytest fixtures: generate the synthetic benchmark once per test
session into a temp directory, and hand out its manifest + directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from peekaboo.benchmark.generate import generate_benchmark


@pytest.fixture(scope="session")
def benchmark_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("peekaboo_benchmark")
    generate_benchmark(str(out_dir), seed=0)
    return out_dir


@pytest.fixture(scope="session")
def manifest(benchmark_dir: Path) -> list[dict]:
    import json

    return json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
