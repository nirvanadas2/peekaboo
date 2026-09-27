# Phase 7 — Live-scan API for the dashboard

## Status: a minimal local FastAPI server wraps `run_pre_checks` so the dashboard can scan an uploaded model directly, in addition to loading pre-generated JSON reports

## 1. What this is

`peekaboo/api.py`: one endpoint, `POST /scan`, accepting a multipart file
upload (`.safetensors`/`.pt`/`.pth`/`.onnx`). It writes the upload to a
temp file, calls `run_pre_checks` with **no `forward_fn`**, and returns
`render_json`'s output directly — byte-for-byte the same shape the CLI's
`--json` flag produces. The frontend's `types/report.ts` and `useReport`
hook needed no schema changes; `useReport` just gained a third loader that
POSTs instead of reading a local file or fetching a bundled demo.

Not part of the CLI. Run it separately:

```
uvicorn peekaboo.api:app --reload
```

(`pip install -e ".[api]"` for the FastAPI/uvicorn/python-multipart
dependencies — kept as an optional extra so CLI-only installs don't pull
in a web framework.)

## 2. Why a persistent server, not a CLI subprocess

Measured on `tests/fixtures/benchmark/backdoored.safetensors`:

| | Wall time |
|---|---|
| `python -m peekaboo scan ... --tinycnn` (full CLI process) | 1.17s |
| same, without `--tinycnn` | 0.79s |
| `import torch, numpy, safetensors, onnx` alone, isolated | 0.76s |

Import overhead alone accounts for the large majority of a CLI
invocation's time; actual scan logic is ~0.03–0.4s once warm. A design
that shelled out to the CLI per request would re-pay that ~0.76s tax on
every scan. The FastAPI server imports once at startup and stays warm:
measured round-trip for the same file through the running server was
**0.06s**.

## 3. No `forward_fn` for uploaded models — ever

`--tinycnn` only works because it knows the file is TinyCNN and
reconstructs a matching PyTorch module (`peekaboo/benchmark/runnable.py`).
There is no general way to build a `forward_fn` for an arbitrary uploaded
model — that requires knowing its exact architecture, which an uploaded
`.safetensors`/`.pt`/`.onnx` file doesn't carry on its own. Every live scan
through this API therefore has `pillars.behavioral.status == "not_run"`,
with no exception. This is not a gap in the endpoint; it's the same
limitation `--tinycnn`'s own docstring and PHASE4.md/PHASE6.md already
describe, now hitting real uploads instead of only the synthetic
benchmark. The dashboard's existing "not assessed" UI (built and verified
in the dashboard phase, before this one) already covers it correctly: a
`not_run` pillar never renders as clean.

One thing this phase changed on the frontend: the "not assessed" banner's
copy quotes a specific AUC figure (0.50, chance) that PHASE6.md measured
*only* on the TinyCNN benchmark. Quoting it for an arbitrary uploaded
model would overclaim precision that doesn't transfer. Live-scan reports
now show the qualitative claim only ("no static check detects backdoors
without a runnable model"); the quantified AUC figure is still shown for
demo/benchmark reports, where it was actually measured.

## 4. Known limitations (not yet resolved)

- **Large-file scan time is untested.** Every timing above is on the
  92 KB TinyCNN benchmark file. A multi-GB real-world checkpoint's scan
  time — Stage 4's bit-level tests in particular scan every tensor's low
  mantissa bits — has not been measured. The API's 500 MB upload cap is a
  sane-sounding limit for a local dev tool, not a validated ceiling; it
  may need to come down (or the UI may need real progress reporting
  instead of a single loading spinner) once this is measured.
- **Stage 2's documented `.bin`/`.ckpt` gap** (metadata_check.py's own
  module docstring) applies here too: a file that passes Stage 1 but
  fails to actually load raises inside `run_pre_checks`. The endpoint
  catches this and returns HTTP 422 with the underlying error rather than
  a raw traceback, but it's a caught failure, not a fixed one.
- **No auth, no rate limiting.** This is a local dev tool; the server is
  meant to run on localhost only, proxied by the Vite dev server.
