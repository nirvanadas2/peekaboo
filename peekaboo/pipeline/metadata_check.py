"""Stage 1: Metadata Integrity.

File-level and format-level checks that must pass before Peekaboo trusts a
model file enough to load its full tensors for deeper analysis:

  - file hash (SHA-256) and size
  - format detection: does the file extension match the actual content?
    (catches mismatched/renamed files, whether accidental or adversarial)
  - safetensors: header structure validated against the spec, including
    rejecting unexpected extra keys/segments in tensor entries
  - pytorch pickle (.pt/.pth): Phase 0's `load_pytorch_pickle` is invoked
    here as a hard gate — any checkpoint whose pickle stream contains
    anything outside the weights-only allowlist is rejected before
    anything downstream runs
  - onnx: validated against `onnx.checker`, and opset versions are
    sanity-checked

Every check appends a `Finding` (pass or fail) to the report, so the
report is a full audit trail, not just a list of problems. A finding
that fails at CRITICAL severity marks the report `hard_fail=True`,
which is the signal the pipeline gate (`run_pre_checks`) uses to skip
Stage 2 entirely — see peekaboo/pipeline/gate.py.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import onnx

from peekaboo.loaders.pytorch_pickle_loader import UnsafeCheckpointError, load_pytorch_pickle
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import Finding, MetadataReport, compute_hard_fail, compute_passed

# NOTE: this mapping is intentionally broader than Phase 0's loader dispatch
# (peekaboo/loaders/dispatch.py::_LOADERS_BY_SUFFIX), which only recognizes
# .safetensors/.pt/.pth/.onnx. Stage 1's job is to validate *content*
# safety/integrity independent of what today's loader happens to accept, so
# .bin (HuggingFace's legacy `pytorch_model.bin` convention) and .ckpt
# (PyTorch Lightning) are both recognized here as pytorch_pickle format.
#
# Consequence: a .bin/.ckpt file can PASS Stage 1 cleanly and still fail to
# load in Stage 2+ with a ValueError raised by `load_model()`, because Phase
# 0's dispatcher doesn't know those extensions. This is a known, temporary
# gap, not a bug in either module.
# TODO(flagged, not yet approved): extend _LOADERS_BY_SUFFIX in Phase 0's
# loaders/dispatch.py to include .bin/.ckpt (both pytorch_pickle) so a file
# that passes Stage 1 is guaranteed loadable in Stage 2+.
_EXTENSION_FORMATS = {
    ".safetensors": "safetensors",
    ".pt": "pytorch_pickle",
    ".pth": "pytorch_pickle",
    ".bin": "pytorch_pickle",
    ".ckpt": "pytorch_pickle",
    ".onnx": "onnx",
}

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_PICKLE_PROTOCOL_MAGIC = b"\x80"

# ONNX opset versions in the wild run roughly 1-23 as of ONNX 1.15+. Anything
# outside this band is either a malformed/tampered opset_import or generated
# by tooling far outside anything Peekaboo has been validated against.
_MIN_SANE_OPSET = 1
_MAX_SANE_OPSET = 30

_SAFETENSORS_ALLOWED_TENSOR_KEYS = {"dtype", "shape", "data_offsets"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_safetensors_header(path: Path, file_size: int) -> tuple[dict[str, Any], int]:
    """Return (header_dict, data_region_start_offset), or raise ValueError
    if the file does not have a structurally valid safetensors header."""
    with open(path, "rb") as f:
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise ValueError("file is too small to contain a safetensors header length")
        header_len = struct.unpack("<Q", len_bytes)[0]
        if header_len <= 0 or 8 + header_len > file_size:
            raise ValueError(
                f"header length {header_len} is out of bounds for file size {file_size}"
            )
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError("file is truncated: header claims more bytes than are present")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("header JSON is not an object")

    return header, 8 + header_len


def _sniff_cheap(path: Path, file_size: int) -> str | None:
    """Cheap, extension-independent structural sniff. Returns a format name
    when confident, or None when inconclusive (caller falls through to the
    declared format's own parser, whose success/failure is itself
    informative)."""
    try:
        _parse_safetensors_header(path, file_size)
        return "safetensors"
    except ValueError:
        pass

    with open(path, "rb") as f:
        magic = f.read(8)
    if magic[:4] in _ZIP_MAGICS or magic[:1] == _PICKLE_PROTOCOL_MAGIC:
        return "pytorch_pickle"

    return None


# --- CRITICAL vs. HIGH severity policy ---------------------------------
#
# CRITICAL (and therefore `hard_fail=True`, which short-circuits the entire
# pipeline before Stage 2+ ever runs) is reserved for cases where continuing
# is *unsafe* (pickle code-exec risk) or the file is too malformed to even
# correctly locate its own tensor data (unparseable header, offsets outside
# the file, protobuf that won't parse at all). In both cases there is
# nothing legitimate for Stage 2+ to analyze.
#
# HIGH is used when a check fails but the file can still be safely and
# correctly loaded for deeper analysis. Two checks apply this deliberately:
#
#   - safetensors_no_extra_keys: an extra key in a tensor header entry is a
#     plausible steganographic smuggling channel — exactly the kind of thing
#     Peekaboo exists to catch. But it doesn't affect whether the real
#     tensor data (validated separately via data_offsets) can be correctly
#     located, and safetensors is safe-by-construction regardless of what
#     extra JSON sits next to it. Hard-failing here would skip the
#     statistical/steganographic/behavioral analysis that would actually
#     characterize the payload — and would miss any co-occurring threat in
#     the same file (e.g. a payload plus a weight-level backdoor, mirroring
#     the benchmark's `combined` variant). So: flagged loudly (passed=False,
#     HIGH), but not hard-failed.
#   - onnx_checker: a model that fails ONNX's strict `checker.check_model`
#     (e.g. a dangling node reference, a non-topologically-sorted graph)
#     may still have perfectly extractable, analyzable initializer tensors
#     — Peekaboo's loader only reads `graph.initializer`, it doesn't require
#     a checker-valid graph. Hard-failing here would skip deep analysis on
#     exactly the kind of malformed-but-loadable file that most needs it.
#
# Neither of these findings is "silently ignored": both mark the report
# `passed=False` so nothing is silently approved, they just don't set
# `hard_fail=True`.
# -------------------------------------------------------------------------

_EXTERNAL_DATA_ERROR_HINTS = ("should be stored in", "not regular file", "external data", "externaldata")


def _looks_like_missing_external_data(exc: Exception) -> bool:
    """Best-effort detection of ONNX's "external data file not found" failure
    mode, distinct from a genuinely corrupted/non-ONNX file. ONNX models can
    reference tensor data stored in a companion file (e.g. model.onnx.data);
    scanning the .onnx graph descriptor in isolation from that companion
    file is a common, legitimate scenario, not evidence of tampering."""
    message = str(exc).lower()
    return any(hint in message for hint in _EXTERNAL_DATA_ERROR_HINTS)


def _check_safetensors(path: Path, file_size: int, findings: list[Finding]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        header, data_start = _parse_safetensors_header(path, file_size)
    except ValueError as exc:
        findings.append(
            Finding(
                check="safetensors_header_valid",
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Malformed safetensors header: {exc}",
            )
        )
        return metadata

    findings.append(
        Finding(
            check="safetensors_header_valid",
            severity=Severity.INFO,
            passed=True,
            message="Header parses as valid JSON within file bounds",
        )
    )

    data_size = file_size - data_start
    unexpected_keys: dict[str, list[str]] = {}
    structure_errors: list[str] = []

    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or not all(isinstance(v, str) for v in entry.values()):
                structure_errors.append("__metadata__ must be a flat string-to-string map")
            continue

        if not isinstance(entry, dict):
            structure_errors.append(f"tensor entry '{name}' is not an object")
            continue

        extra = sorted(set(entry.keys()) - _SAFETENSORS_ALLOWED_TENSOR_KEYS)
        missing = sorted(_SAFETENSORS_ALLOWED_TENSOR_KEYS - set(entry.keys()))
        if extra:
            unexpected_keys[name] = extra
        if missing:
            structure_errors.append(f"tensor '{name}' is missing required keys {missing}")
            continue

        offsets = entry["data_offsets"]
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(o, int) for o in offsets)):
            structure_errors.append(f"tensor '{name}' has a malformed data_offsets field")
            continue

        start, end = offsets
        if not (0 <= start <= end <= data_size):
            structure_errors.append(
                f"tensor '{name}' data_offsets {offsets} fall outside the data region (size {data_size})"
            )

    if unexpected_keys:
        findings.append(
            Finding(
                check="safetensors_no_extra_keys",
                severity=Severity.HIGH,
                passed=False,
                message=f"Unexpected extra keys found in tensor header entries: {unexpected_keys}",
                details={"unexpected_keys": unexpected_keys},
            )
        )
    else:
        findings.append(
            Finding(
                check="safetensors_no_extra_keys",
                severity=Severity.INFO,
                passed=True,
                message="No unexpected keys in tensor header entries",
            )
        )

    if structure_errors:
        findings.append(
            Finding(
                check="safetensors_offsets_valid",
                severity=Severity.CRITICAL,
                passed=False,
                message="; ".join(structure_errors),
            )
        )
    else:
        findings.append(
            Finding(
                check="safetensors_offsets_valid",
                severity=Severity.INFO,
                passed=True,
                message="All tensor data_offsets are well-formed and within file bounds",
            )
        )

    metadata["safetensors_file_metadata"] = header.get("__metadata__", {})
    metadata["num_tensor_entries"] = len([k for k in header if k != "__metadata__"])
    return metadata


def _check_onnx(path: Path, findings: list[Finding]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        model = onnx.load(str(path))
    except Exception as exc:
        if _looks_like_missing_external_data(exc):
            findings.append(
                Finding(
                    check="external_data_not_found",
                    severity=Severity.MEDIUM,
                    passed=False,
                    message=(
                        "ONNX file references external tensor data that could not be found "
                        f"alongside it: {exc}. This commonly happens when a .onnx graph "
                        "descriptor is copied or scanned separately from its companion weights "
                        "file (e.g. model.onnx.data) — the graph structure itself may still be "
                        "entirely legitimate."
                    ),
                )
            )
            return metadata
        findings.append(
            Finding(
                check="onnx_parses",
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Failed to parse ONNX file: {exc}",
            )
        )
        return metadata

    findings.append(
        Finding(
            check="onnx_parses",
            severity=Severity.INFO,
            passed=True,
            message="File parses as a valid ONNX protobuf model",
        )
    )

    try:
        onnx.checker.check_model(model)
        findings.append(
            Finding(
                check="onnx_checker",
                severity=Severity.INFO,
                passed=True,
                message="Model passes onnx.checker.check_model",
            )
        )
    except Exception as exc:
        # HIGH, not CRITICAL/hard_fail — see the severity policy comment
        # above _check_safetensors: a checker-invalid graph may still have
        # perfectly extractable initializer tensors for deep analysis.
        findings.append(
            Finding(
                check="onnx_checker",
                severity=Severity.HIGH,
                passed=False,
                message=f"onnx.checker.check_model rejected the model: {exc}",
            )
        )

    opset_imports = {(imp.domain or "ai.onnx"): imp.version for imp in model.opset_import}
    metadata["opset_imports"] = opset_imports
    main_opset = opset_imports.get("ai.onnx")

    if main_opset is None:
        findings.append(
            Finding(
                check="onnx_opset_sane",
                severity=Severity.HIGH,
                passed=False,
                message="No default (ai.onnx) opset import found",
            )
        )
    elif not (_MIN_SANE_OPSET <= main_opset <= _MAX_SANE_OPSET):
        findings.append(
            Finding(
                check="onnx_opset_sane",
                severity=Severity.HIGH,
                passed=False,
                message=(
                    f"Opset version {main_opset} is outside the sane range "
                    f"[{_MIN_SANE_OPSET}, {_MAX_SANE_OPSET}]"
                ),
            )
        )
    else:
        findings.append(
            Finding(
                check="onnx_opset_sane",
                severity=Severity.INFO,
                passed=True,
                message=f"Opset version {main_opset} is within the sane range",
            )
        )

    return metadata


def _check_pytorch_pickle(path: Path, findings: list[Finding]) -> dict[str, Any]:
    try:
        load_pytorch_pickle(str(path))
    except UnsafeCheckpointError as exc:
        findings.append(
            Finding(
                check="pickle_weights_only_safe",
                severity=Severity.CRITICAL,
                passed=False,
                message=str(exc),
            )
        )
        return {}

    findings.append(
        Finding(
            check="pickle_weights_only_safe",
            severity=Severity.INFO,
            passed=True,
            message="Checkpoint deserializes safely under torch's weights-only restricted unpickler",
        )
    )
    return {}


def _finalize(
    model_path: str,
    declared_format: str,
    detected_format: str,
    file_size: int,
    file_hash: str,
    findings: list[Finding],
    metadata: dict[str, Any],
) -> MetadataReport:
    return MetadataReport(
        model_path=model_path,
        declared_format=declared_format,
        detected_format=detected_format,
        file_size=file_size,
        file_hash=file_hash,
        passed=compute_passed(findings),
        hard_fail=compute_hard_fail(findings),
        findings=findings,
        metadata=metadata,
    )


def run_metadata_check(model_path: str) -> MetadataReport:
    """Run Stage 1 (Metadata Integrity) checks on a single model file."""
    path = Path(model_path)
    findings: list[Finding] = []
    metadata: dict[str, Any] = {}
    declared_format = _EXTENSION_FORMATS.get(path.suffix.lower(), "unknown")

    if not path.is_file():
        findings.append(
            Finding(
                check="file_exists",
                severity=Severity.CRITICAL,
                passed=False,
                message=f"File does not exist: {model_path}",
            )
        )
        return _finalize(model_path, declared_format, "unknown", 0, "", findings, metadata)

    file_size = path.stat().st_size
    file_hash = _sha256_file(path)
    findings.append(
        Finding(
            check="file_hash_sha256",
            severity=Severity.INFO,
            passed=True,
            message=f"SHA-256: {file_hash}",
            details={"sha256": file_hash, "file_size": file_size},
        )
    )

    if file_size == 0:
        findings.append(
            Finding(
                check="file_not_empty",
                severity=Severity.CRITICAL,
                passed=False,
                message="File is empty (0 bytes)",
            )
        )
        return _finalize(model_path, declared_format, "unknown", file_size, file_hash, findings, metadata)

    if declared_format == "unknown":
        findings.append(
            Finding(
                check="known_extension",
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Unrecognized file extension '{path.suffix}'; cannot select a validator",
            )
        )
        detected_format = _sniff_cheap(path, file_size) or "unknown"
        return _finalize(model_path, declared_format, detected_format, file_size, file_hash, findings, metadata)

    sniffed = _sniff_cheap(path, file_size)
    if sniffed is not None and sniffed != declared_format:
        findings.append(
            Finding(
                check="extension_matches_content",
                severity=Severity.CRITICAL,
                passed=False,
                message=(
                    f"File extension '{path.suffix}' implies format '{declared_format}', but the "
                    f"content's structural signature matches '{sniffed}' instead — possible "
                    f"mismatched or renamed file"
                ),
            )
        )
        return _finalize(model_path, declared_format, sniffed, file_size, file_hash, findings, metadata)

    if sniffed == declared_format:
        findings.append(
            Finding(
                check="extension_matches_content",
                severity=Severity.INFO,
                passed=True,
                message=f"File extension matches the detected content format ('{sniffed}')",
            )
        )

    detected_format = declared_format

    if declared_format == "safetensors":
        metadata.update(_check_safetensors(path, file_size, findings))
    elif declared_format == "onnx":
        metadata.update(_check_onnx(path, findings))
    elif declared_format == "pytorch_pickle":
        metadata.update(_check_pytorch_pickle(path, findings))

    # If the format-specific parse itself failed, that's the strongest
    # signal of a mismatched/corrupted file even when the cheap sniff was
    # inconclusive (this is the common case for onnx, since protobuf has
    # no cheap magic-byte signature).
    format_check_failed = any(
        (not f.passed) and f.severity == Severity.CRITICAL
        and f.check in ("safetensors_header_valid", "onnx_parses", "pickle_weights_only_safe")
        for f in findings
    )
    if sniffed is None and format_check_failed:
        detected_format = "unknown"

    return _finalize(model_path, declared_format, detected_format, file_size, file_hash, findings, metadata)
