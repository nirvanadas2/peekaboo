"""
peekaboo/api.py

A minimal local API server wrapping run_pre_checks for the dashboard's
live-scan upload flow (frontend/src/components/dashboard/ReportSourcePanel.tsx).
Not part of the CLI -- run separately:

    uvicorn peekaboo.api:app --reload

See PHASE7.md for why this exists as a persistent server rather than a
CLI subprocess call (import overhead), and its known limitations.

Never builds a forward_fn for an uploaded model: Peekaboo has no way to
reconstruct a runnable forward pass for an architecture it doesn't
already know (see --tinycnn's own docstring in __main__.py, and
PHASE4.md/PHASE6.md). Every live scan's behavioral pillar is therefore
honestly not_run -- this is a limitation of the technique, not something
this endpoint should try to paper over.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response

from peekaboo.pipeline.gate import run_pre_checks
from peekaboo.report import render_json

app = FastAPI(title="Peekaboo scan API")

_ALLOWED_SUFFIXES = {".safetensors", ".pt", ".pth", ".onnx"}
# Scan time beyond this is untested -- measured timings (PHASE7.md) are all
# on the 92 KB TinyCNN benchmark file. Chosen as a sane upper bound for a
# local dev tool, not a validated performance ceiling.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@app.post("/scan")
async def scan(file: UploadFile) -> Response:
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
        # No forward_fn: see module docstring. Stage 1 hard-gates unsafe
        # files internally, but Stage 2+ loading can still raise for a file
        # that passed Stage 1's broader format check yet fails to actually
        # load (documented gap, metadata_check.py's module docstring) --
        # caught here so a malformed upload gets a clean error instead of
        # an unhandled 500 with a stack trace.
        try:
            result = run_pre_checks(tmp_path)
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

    return Response(content=render_json(result), media_type="application/json")
