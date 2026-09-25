"""
tests/test_report.py

Stage 7 (explainable report + CLI). Rendering only -- detection results
are locked in by the stage tests.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from peekaboo.__main__ import main
from peekaboo.benchmark.runnable import TINYCNN_INPUT_SHAPE, TINYCNN_NUM_CLASSES, tinycnn_forward_fn
from peekaboo.loaders import load_model
from peekaboo.pipeline.gate import run_pre_checks
from peekaboo.report import LIMITATIONS, exit_code, render_json, render_markdown


def _probed(path: Path):
    return run_pre_checks(
        str(path),
        forward_fn=tinycnn_forward_fn(load_model(str(path))),
        input_shape=TINYCNN_INPUT_SHAPE,
        num_classes=TINYCNN_NUM_CLASSES,
    )


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


@pytest.fixture
def evil_pickle(tmp_path: Path) -> Path:
    path = tmp_path / "evil.pt"
    with open(path, "wb") as f:
        pickle.dump({"w": _Evil()}, f)
    return path


class TestMarkdown:
    def test_backdoor_report_explains_the_trigger(self, benchmark_dir):
        md = render_markdown(_probed(benchmark_dir / "backdoored.safetensors"))
        assert "## Verdict: HIGH RISK" in md
        assert "behavioral_trigger_island" in md
        assert "position [0, 0]" in md and "class 3" in md
        assert "The position is a hint" in md
        assert "## Not assessed" not in md
        for item in LIMITATIONS:
            assert item in md

    def test_unprobed_report_says_backdoors_not_assessed(self, benchmark_dir):
        result = run_pre_checks(str(benchmark_dir / "backdoored.safetensors"))
        md = render_markdown(result)
        assert "## Not assessed" in md
        assert "**not assessed**" in md
        assert "not evidence of safety" in md
        assert "## Verdict: No evidence of tampering from the checks that ran" in md

    def test_noisy_report_lists_layer_risk(self, benchmark_dir):
        md = render_markdown(run_pre_checks(str(benchmark_dir / "noisy.safetensors")))
        assert "## Per-layer risk" in md and "`fc1.weight`" in md
        assert "bit_balance_chi_square" in md

    def test_report_only_stage3_is_marked(self):
        path = Path(__file__).parent / "fixtures" / "final_validation" / "seed28" / "clean.safetensors"
        md = render_markdown(_probed(path))
        assert "## Report-only observations (not scored)" in md
        assert "does not affect the score" in md
        assert "## Verdict: No evidence of tampering" in md

    def test_unsafe_pickle_is_not_analyzed(self, evil_pickle):
        result = run_pre_checks(str(evil_pickle))
        md = render_markdown(result)
        assert "## Verdict: UNSAFE / NOT ANALYZED" in md
        assert "not loaded" in md
        assert exit_code(result) == 3


class TestJsonAndExitCodes:
    def test_json_round_trips(self, benchmark_dir):
        doc = json.loads(render_json(_probed(benchmark_dir / "backdoored.safetensors")))
        assert doc["verdict"] == "high"
        assert doc["scan"]["risk_score"]["pillars"]["behavioral"]["status"] == "flagged"
        assert doc["scan"]["risk_score"]["pillars"]["statistical"]["weight"] == 0.0
        assert doc["limitations"] == LIMITATIONS

    def test_exit_codes(self, benchmark_dir):
        assert exit_code(_probed(benchmark_dir / "backdoored.safetensors")) == 2
        # seed-0 clean carries the known bn4.running_var MEDIUM false positive (PHASE3.md)
        assert exit_code(run_pre_checks(str(benchmark_dir / "clean.safetensors"))) == 1
        clean_fresh = Path(__file__).parent / "fixtures" / "final_validation" / "seed20" / "clean.safetensors"
        assert exit_code(_probed(clean_fresh)) == 0


class TestCli:
    def test_scan_writes_reports_and_returns_code(self, benchmark_dir, tmp_path):
        md, js = tmp_path / "r.md", tmp_path / "r.json"
        code = main(["scan", str(benchmark_dir / "backdoored.safetensors"), "--tinycnn", "--md", str(md), "--json", str(js)])
        assert code == 2
        assert "HIGH RISK" in md.read_text(encoding="utf-8")
        assert json.loads(js.read_text(encoding="utf-8"))["verdict"] == "high"

    def test_scan_unsafe_file(self, evil_pickle, capsys):
        assert main(["scan", str(evil_pickle)]) == 3
        assert "UNSAFE / NOT ANALYZED" in capsys.readouterr().out
