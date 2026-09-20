# Phase 1: Metadata Integrity & Structural Consistency

Phase 1 sits between Phase 0's loaders and the deep-analysis pillars of
later phases (statistical / steganographic / behavioral). It answers one
question before any of that expensive work runs: **is this file safe and
sane enough to bother analyzing, and does its structure look normal?**

Everything here lives in `peekaboo/pipeline/`:

- `metadata_check.py` — Stage 1
- `structural_check.py` — Stage 2
- `gate.py` — wires the two together (`run_pre_checks`)

New schema types live in `peekaboo/schema/reports.py`: `Finding`,
`MetadataReport`, `StructuralReport` (exported from `peekaboo.schema`
alongside Phase 0's `ModelRiskScore`).

## Stage 1: Metadata Integrity (`run_metadata_check`)

File-level and format-level checks, run before any tensors are trusted:

- **File hash (SHA-256) and size.**
- **Format detection** — an extension-independent structural sniff (a real
  safetensors header parse; zip/pickle magic bytes) is compared against
  what the file extension claims. A mismatch (renamed/mismatched file)
  fails immediately, before the format-specific parser even runs.
- **Safetensors** — the 8-byte header-length prefix + JSON header is
  parsed manually (not via the `safetensors` library, so lenient parsing
  can't hide a problem): every tensor entry must have exactly
  `{dtype, shape, data_offsets}` (extra keys flagged), and `data_offsets`
  must fall inside the file.
- **Pickle checkpoints (`.pt`/`.pth`/`.bin`/`.ckpt`)** — Phase 0's
  `load_pytorch_pickle` (weights-only restricted unpickler) is invoked
  here as the hard gate. Anything that isn't a plain tensor/state-dict
  checkpoint raises `UnsafeCheckpointError`, which becomes the report's
  defining hard-fail case.
- **ONNX** — `onnx.load` + `onnx.checker.check_model`, plus an opset
  sanity range check (`[1, 30]`).

Every check appends a `Finding` — pass or fail — so the report is a full
audit trail, not just a list of problems.

### Known extension-allowlist gap (`.bin` / `.ckpt`)

Stage 1's extension map is intentionally **broader** than Phase 0's loader
dispatch (`peekaboo/loaders/dispatch.py::_LOADERS_BY_SUFFIX`, which only
recognizes `.safetensors` / `.pt` / `.pth` / `.onnx`). Stage 1 also
recognizes `.bin` (HuggingFace's legacy `pytorch_model.bin`) and `.ckpt`
(PyTorch Lightning) as `pytorch_pickle` content, because both are common,
legitimate, safe-to-validate real-world conventions that predate/sit
alongside safetensors.

**Consequence:** a `.bin`/`.ckpt` file can pass Stage 1 cleanly and still
fail to load in Stage 2+ with a `ValueError` from `load_model()`, because
Phase 0's dispatcher doesn't know those extensions. This is a **known,
temporary gap, not a bug** — flagged rather than fixed, per the standing
rule not to change Phase 0's public interfaces without approval.
`metadata_check.py` carries a `TODO(flagged, not yet approved)` to extend
`_LOADERS_BY_SUFFIX` once that's decided. `tests/test_metadata_check.py::TestBroadenedExtensionAllowlist::test_known_gap_passes_stage1_but_phase0_dispatch_cannot_load_it`
locks in and documents this exact behavior.

## Stage 2: Structural Consistency (`run_structural_check`)

Operates on an already-loaded `LoadedModel` (format-neutral: safetensors,
pickle, or ONNX). Two modes:

### Self-consistency (default — the primary, real-world use case)

Peekaboo's actual job is scanning unknown models from public hubs with no
access to the original training pipeline, so this mode infers what
"normal" should look like purely from the tensors:

- **Consecutive-dimension chain** — walks `.weight` tensors in state-dict
  order (linear = 2D, conv2d = 4D) and checks each layer's input dimension
  against **every prior layer's output**, not just the immediate
  predecessor. This is what lets branching/skip architectures (two layers
  both consuming one upstream layer's output) pass without a false
  positive, while a layer whose input matches nothing anyone produced is
  still caught. conv→conv and linear→linear require an exact match;
  conv→linear allows any integer multiple (the real flatten invariant —
  `in_features = out_channels * spatial_h * spatial_w`); linear→conv is
  treated as inconclusive rather than risking a false positive.
- **dtype uniformity** — majority-vote dtype across tensors (excluding
  `num_batches_tracked`, a legitimate int64 counter present in every
  benchmark file). Outliers involving float64 are HIGH; others MEDIUM.
- **Degenerate shapes** — any tensor with a literal `0` in its shape.
- **Weight/bias shape consistency** — a bias tensor's shape must equal its
  weight's output dimension.
- **Param-count profile** and **naming-pattern outliers** — both always
  informational (`INFO`, `passed=True`, never fail the report). The
  naming-pattern check is a coarse, digit-normalized template-clustering
  heuristic standing in for "no orphaned/unreferenced tensors" on formats
  with no real computation graph (safetensors/pickle state dicts don't
  carry one, unlike ONNX) — it will typically flag a legitimate one-off
  layer (e.g. `fc`, `classifier`) alongside any genuinely stray/injected
  tensor. That's intentional: better to surface too much for a human or a
  later pillar to triage than to silently miss a real smuggled tensor.

### Spec-diff (secondary — caller supplies an `ArchitectureSpec`)

A straightforward **exact-match diff** of actual vs. expected layer
names/shapes/dtypes, using the same comparison logic as self-consistency
(the same `_dtype_pair_severity` helper decides HIGH vs. MEDIUM in both
modes). Deliberately simple: no partial-spec handling, no spec
versioning, no fuzzy/approximate matching.

### `StructuralReport` has no `hard_fail`

This is the load-bearing design decision for Stage 2. The models Peekaboo
most needs to catch — backdoored, steganographic — are, by construction,
perfectly loadable and structurally normal; only their weight *values*
are compromised. A structural anomaly must never skip the deep-analysis
pillars that would actually characterize a real threat. So severity in
`StructuralReport` findings is a **triage label only, never a control-flow
signal** — even a finding labeled `CRITICAL` (a degenerate zero-size
shape) only sets that finding's `passed=False`; it cannot stop anything.

## Severity policy (applies across both stages)

| Severity | Meaning | Gates the pipeline? |
|---|---|---|
| `CRITICAL` | Stage 1: unsafe (pickle code-exec) or too malformed to trust at all (corrupt header, offsets out of bounds, unparseable file, mismatched extension). Stage 2: a triage label only (e.g. degenerate shape) — never gates. | Only in Stage 1, via `MetadataReport.hard_fail = any finding failed at CRITICAL`. |
| `HIGH` | A real, load-bearing anomaly, but the file can still be safely/correctly loaded for deeper analysis (shape-chain mismatch, spec shape mismatch, float64 dtype outlier). | No |
| `MEDIUM` | A softer anomaly (non-float64 dtype outlier, extra spec layer, ONNX external data not found). | No |
| `LOW` / `INFO` | Informational — passing checks, and always-`passed=True` heuristics (param-count profile, naming-pattern outliers). | No |

### Two deliberate CRITICAL→HIGH/MEDIUM exceptions in Stage 1

Documented in a single policy comment in `metadata_check.py` (above
`_check_safetensors`), because both follow the same principle: **don't
hard-fail (and thereby skip all downstream analysis) when the file can
still be safely and correctly loaded.**

1. **`safetensors_no_extra_keys` → HIGH, not CRITICAL.** An extra key in a
   tensor header entry is a plausible steganographic smuggling channel —
   close to Peekaboo's core threat model — but it doesn't affect whether
   the real tensor data (validated separately via `data_offsets`) can be
   correctly located, and safetensors is safe-by-construction regardless.
   Hard-failing here would skip exactly the statistical/steganographic
   analysis that would characterize the payload, and would miss any
   co-occurring threat in the same file (e.g. a payload plus a weight-level
   backdoor, mirroring the benchmark's `combined` variant). It's still
   flagged loudly (`passed=False`, `HIGH`) — nothing is silently ignored.
2. **`onnx_checker` failure → HIGH, not CRITICAL.** A model that fails
   ONNX's strict `checker.check_model` (e.g. a dangling node reference)
   may still have perfectly extractable, analyzable initializer tensors —
   Peekaboo's loader only reads `graph.initializer`, it doesn't require a
   checker-valid graph.

A related, narrower fix in the same spirit: ONNX's "external data file not
found" failure (a `.onnx` graph descriptor scanned without its companion
weights file, e.g. `model.onnx.data` — a common, legitimate export
convention) is detected specifically (`external_data_not_found`, `MEDIUM`)
and kept distinct from a genuinely corrupted/mismatched file
(`onnx_parses`, `CRITICAL`).

## The pipeline gate (`run_pre_checks`)

```python
def run_pre_checks(model_path: str, spec: ArchitectureSpec | None = None) -> PreCheckResult:
    metadata_report = run_metadata_check(model_path)

    if metadata_report.hard_fail:
        return PreCheckResult(metadata=metadata_report, structural=None, loaded_model=None)

    loaded_model = load_model(model_path)
    structural_report = run_structural_check(loaded_model, spec=spec)

    return PreCheckResult(metadata=metadata_report, structural=structural_report, loaded_model=loaded_model)
```

Returns a single `PreCheckResult` dataclass (`peekaboo/pipeline/gate.py`),
not a bare tuple — `metadata: MetadataReport`, `structural:
StructuralReport | None`, `loaded_model: LoadedModel | None`, plus a
`stopped_at_metadata` property and a `to_dict()`. `structural` and
`loaded_model` are `None` exactly when Stage 1 hard-failed. On the normal
path, `loaded_model` is handed back so a later phase (e.g. Phase 2's
statistical analysis) can reuse the tensors Stage 2 already loaded from
disk, without needing to know Phase 1 is internally split into two stages
or reloading the file itself.

Only `hard_fail` gates the short-circuit — a Stage 1 finding that fails
but isn't `CRITICAL` (e.g. the extra-key case above) still lets Stage 2
run. `tests/test_gate.py::TestHardFailShortCircuits` verifies this by
monkeypatching `load_model` and `run_structural_check` to raise if called
at all, not just by inspecting the result shape.

## Testing

139 tests pass across the full suite (Phase 0's original 42 plus Phase 1's
97). Phase 1-specific coverage:

- **`tests/test_metadata_check.py`** (44) — the full 5×3 Phase 0 benchmark
  matrix passes cleanly; malformed fixtures for truncated files (mid-header,
  mid-data, mid-protobuf), every mismatched-extension permutation across
  the three formats, corrupted/non-JSON safetensors headers, out-of-bounds
  offsets, an injected extra header key, insane ONNX opsets, a
  checker-invalid-but-loadable ONNX graph, the `.bin`/`.ckpt` allowlist
  broadening (plus a test locking in the documented Phase-0-dispatch gap),
  and ONNX external-data-missing handling.
- **`tests/test_structural_check.py`** (31) — the same 5×3 benchmark matrix
  passes structural checks too (proving Stage 2 isn't accidentally reacting
  to Phase 0's weight-*value*-only tampering); synthetic fixtures for a
  shape-chain break, a stray float64 tensor, a degenerate zero-dim shape,
  and an orphaned tensor; two "unusual but valid" architectures (a
  branching/skip split, an unusually deep uniform-width stack) confirmed
  *not* to false-positive; full spec-diff coverage.
- **`tests/test_gate.py`** (22) — the short-circuit contract (verified via
  monkeypatching, not just result inspection), the non-hard-fail
  pass-through case, spec forwarding, `to_dict()`, and the full 5×3
  benchmark matrix end-to-end through `run_pre_checks`.
