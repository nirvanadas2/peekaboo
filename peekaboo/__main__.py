"""Command-line entry point.

    python -m peekaboo scan MODEL [--md OUT.md] [--json OUT.json] [--tinycnn]

Prints the Markdown report to stdout unless --md/--json are given.
Exit code: 0 = no scored evidence, 1 = medium, 2 = high, 3 = unsafe
file (see peekaboo.report.exit_code) -- usable as a CI gate.

--tinycnn builds a forward_fn from this repo's benchmark architecture so
Stage 5 can probe the benchmark's own files. It is for the synthetic
benchmark ONLY; an unknown model's architecture can't be assumed, and
without a forward_fn backdoors are reported as not assessed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from peekaboo.pipeline.gate import run_pre_checks
from peekaboo.report import exit_code, render_json, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peekaboo", description="Pre-deployment scanner for AI model weights")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan one model file")
    scan.add_argument("model", help=".safetensors / .pt / .pth / .onnx file")
    scan.add_argument("--md", help="write the Markdown report here")
    scan.add_argument("--json", help="write the JSON report here")
    scan.add_argument(
        "--tinycnn",
        action="store_true",
        help="probe behavior using the benchmark TinyCNN architecture (synthetic benchmark only)",
    )
    args = parser.parse_args(argv)

    kwargs = {}
    if args.tinycnn:
        from peekaboo.benchmark.runnable import TINYCNN_INPUT_SHAPE, TINYCNN_NUM_CLASSES, benchmark_forward_fn
        from peekaboo.loaders import load_model

        kwargs = dict(
            forward_fn=benchmark_forward_fn(load_model(args.model)),
            input_shape=TINYCNN_INPUT_SHAPE,
            num_classes=TINYCNN_NUM_CLASSES,
        )
    result = run_pre_checks(args.model, **kwargs)

    markdown = render_markdown(result)
    if args.md:
        Path(args.md).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(render_json(result), encoding="utf-8")
    if not args.md and not args.json:
        if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy codepage
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.write(markdown)
    return exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
