"""
peekaboo/api.py

A minimal local API server wrapping run_pre_checks for the dashboard's
live-scan upload flow (frontend/src/components/dashboard/ReportSourcePanel.tsx).
Not part of the CLI -- run separately:

    uvicorn peekaboo.api:app --reload

See PHASE7.md for why this exists as a persistent server rather than a
CLI subprocess call (import overhead), and its known limitations.

By default never builds a forward_fn for an uploaded model: Peekaboo has
no way to reconstruct a runnable forward pass for an architecture it
doesn't already know (see --tinycnn's own docstring in __main__.py, and
PHASE4.md/PHASE6.md). A live scan's behavioral pillar is therefore
honestly not_run -- this is a limitation of the technique, not something
this endpoint should try to paper over.

The one exception mirrors the CLI's --tinycnn flag: when the uploader
explicitly declares `architecture=tinycnn` (this repo's own benchmark
class), the uploaded tensors are loaded into a real TinyCNN and Stage 5
probes it with real forward passes. If they don't fit TinyCNN, the scan
falls back to the default (no forward_fn), and the X-Peekaboo-Forward-Fn
response header says which of the three outcomes happened.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response

from peekaboo.loaders import load_model
from peekaboo.pipeline.gate import run_pre_checks
from peekaboo.pipeline.metadata_check import run_metadata_check
from peekaboo.report import render_json

app = FastAPI(title="Peekaboo scan API")

_ALLOWED_SUFFIXES = {".safetensors", ".pt", ".pth", ".onnx"}
# Scan time beyond this is untested -- measured timings (PHASE7.md) are all
# on the 92 KB TinyCNN benchmark file. Chosen as a sane upper bound for a
# local dev tool, not a validated performance ceiling.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

_ARCH_UNKNOWN = "unknown"
_ARCH_TINYCNN = "tinycnn"
_ALLOWED_ARCHITECTURES = {_ARCH_UNKNOWN, _ARCH_TINYCNN}


def _tinycnn_behavioral_kwargs(tmp_path: str) -> dict | None:
    """Stage 5 kwargs for a file declared to be TinyCNN, or None if it
    can't be probed as one (Stage 1 hard-fails, or its tensors don't fit)."""
    # Stage 1 before load_model(): an unsafe pickle must never reach a
    # loader, and run_pre_checks only loads after its own Stage 1 passes.
    # Building the forward_fn first would skip that gate. (Stage 1 then
    # runs again inside run_pre_checks -- a hash and a header parse, cheap.)
    if run_metadata_check(tmp_path).hard_fail:
        return None

    from peekaboo.benchmark.runnable import TINYCNN_INPUT_SHAPE, TINYCNN_NUM_CLASSES, tinycnn_forward_fn

    try:
        # Always the TinyCNN class, even for .onnx (the CLI runs an .onnx
        # file's own graph instead, ~50s per scan): the uploader declared
        # the architecture, so the weights must load into it.
        forward_fn = tinycnn_forward_fn(load_model(tmp_path))
    except (RuntimeError, TypeError):
        # load_state_dict is strict: missing/unexpected keys or a shape
        # mismatch raise RuntimeError; a dtype torch can't take raises
        # TypeError. Either way this isn't a TinyCNN -- don't guess.
        return None
    return dict(forward_fn=forward_fn, input_shape=TINYCNN_INPUT_SHAPE, num_classes=TINYCNN_NUM_CLASSES)


@app.post("/scan")
async def scan(file: UploadFile, architecture: str = Form(_ARCH_UNKNOWN)) -> Response:
    if architecture not in _ALLOWED_ARCHITECTURES:
        # Rejected rather than treated as unknown, so a typo can't quietly
        # turn a requested behavioral probe into a not_run one.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported architecture '{architecture}'. Supported: {sorted(_ALLOWED_ARCHITECTURES)}",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Supported: {sorted(_ALLOWED_SUFFIXES)}",
        )

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    # load_model() (peekaboo/loaders/dispatch.py) dispatches on the path's
    # suffix and expects a real file, not bytes -- write to a temp file
    # with the same suffix rather than the upload's own filename, so a
    # crafted filename can't do anything odd with the temp path.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # No forward_fn unless declared TinyCNN: see module docstring. Stage
        # 1 hard-gates unsafe files internally, but Stage 2+ loading can
        # still raise for a file that passed Stage 1's broader format check
        # yet fails to actually load (documented gap, metadata_check.py's
        # module docstring) -- caught here so a malformed upload gets a
        # clean error instead of an unhandled 500 with a stack trace.
        try:
            behavioral_kwargs = None
            if architecture == _ARCH_TINYCNN:
                behavioral_kwargs = _tinycnn_behavioral_kwargs(tmp_path)
                forward_fn_status = "tinycnn" if behavioral_kwargs else "tinycnn-load-failed"
            else:
                forward_fn_status = "none"
            result = run_pre_checks(tmp_path, **(behavioral_kwargs or {}))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not scan this file: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # metadata.model_path otherwise shows the server's temp path, which
    # means nothing to the person who uploaded it -- swap in the filename
    # they actually uploaded for display. The scan itself already ran
    # against the real temp file; this only affects what's reported.
    if file.filename:
        result.metadata.model_path = file.filename

    return Response(
        content=render_json(result),
        media_type="application/json",
        headers={"X-Peekaboo-Forward-Fn": forward_fn_status},
    )
