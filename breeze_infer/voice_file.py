"""Voice file v1: the on-disk codec and validation for one saved voice.

data-model.md "Voice file v1" (R13): one JSON file per saved voice,
``<voices_dir>/<id>.voice.json``. Codes are base64 of int16 little-endian, row-major
``[frames][codebooks]`` -- numpy's own int16 byte layout, so encoding is a plain
``.tobytes()``/``np.frombuffer()`` round trip with no byte-swapping step (every
platform this runs on is little-endian).

A consumer rejects any file whose ``format`` or ``version`` it doesn't recognise, so a
future version 2 is never silently misread as version 1. ``decode()`` validates in the
exact order data-model.md's "Voice file v1" skip-rule list gives, so the first rule a
file breaks is the one reported: JSON/schema error (which includes every field's exact
type, a blank or over-long ``ref_text``, ``frames`` below 1, and a ``codebooks`` other
than the model's), stem-vs-id mismatch, bad name (or a reserved ``v_`` prefix), a codes
length or checksum mismatch, an out-of-range code, then a codec fingerprint mismatch.
The one skip rule this module does *not* check -- "a case-duplicate of a file that sorts
earlier" -- needs to compare across files, so it's `voice_store.py`'s job.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from breeze_infer.limits import MAX_REF_TEXT_CHARS

FORMAT = "breeze-tts-voice"
VERSION = 1

# BC-26: the C++ `valid_voice_name` shape (1-64 letters/digits/dash/underscore), minus
# the reserved `v_` prefix. The prefix check is case-insensitive -- matching
# `http_fields._is_valid_voice_id`'s own `value[:2].lower() == "v_"` -- since BC-26 makes
# name uniqueness case-insensitive too: a name spelled "V_..." must be just as reserved
# as one spelled "v_...", or it would dodge the reservation by case alone.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}", re.ASCII)

_REQUIRED_FIELDS = (
    "format",
    "version",
    "id",
    "ref_text",
    "frames",
    "codebooks",
    "codes",
    "codes_sha256",
    "codec_fingerprint",
    "encode_ms",
    "created_at",
)


def is_valid_name(name: object) -> bool:
    """The saved-voice name shape (contracts/http-api.md `POST /v1/voices` `name`):
    1-64 letters/digits/dash/underscore, not starting with `v_` (case-insensitive)."""
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        return False
    return name[:2].lower() != "v_"


class VoiceFileError(ValueError):
    """A voice file failed one of data-model.md "Voice file v1"'s skip rules."""


@dataclass(frozen=True)
class VoiceFile:
    """One saved voice's file content (data-model.md "Voice" plus "Voice file v1")."""

    id: str
    ref_text: str
    frames: int
    codebooks: int
    codes: np.ndarray  # int16 [frames, codebooks]
    codes_sha256: str
    codec_fingerprint: str
    encode_ms: int
    created_at: str  # ISO-8601 UTC, e.g. "2026-09-24T20:15:00Z"


def _codes_bytes(codes: np.ndarray) -> bytes:
    """Row-major int16 little-endian bytes: what encoding writes and what the length
    and sha256 checks in `decode` both operate on."""
    return np.ascontiguousarray(codes, dtype="<i2").tobytes()


def encode(
    *,
    id: str,
    ref_text: str,
    codes: np.ndarray,
    codec_fingerprint: str,
    encode_ms: int,
    created_at: str,
) -> bytes:
    """Build one voice file's JSON bytes.

    `codes` must be `[frames, codebooks]`; its dtype and value range are the caller's
    responsibility here -- the caller already knows both from the codec that produced
    them. `decode` below is what validates an untrusted file read back from disk, so it
    re-checks both.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D [frames, codebooks], got shape {codes.shape}")
    frames, codebooks = codes.shape
    payload = _codes_bytes(codes)
    record = {
        "format": FORMAT,
        "version": VERSION,
        "id": id,
        "ref_text": ref_text,
        "frames": int(frames),
        "codebooks": int(codebooks),
        "codes": base64.b64encode(payload).decode("ascii"),
        "codes_sha256": hashlib.sha256(payload).hexdigest(),
        "codec_fingerprint": codec_fingerprint,
        "encode_ms": int(encode_ms),
        "created_at": created_at,
    }
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _require_int(payload: dict, field: str) -> int:
    """A JSON integer: `bool` is refused although it subclasses `int` (`true` would
    otherwise read as 1), and so is a float, even a whole one like `4.0` -- the writer
    only ever writes integers, so anything else is a hand-edited or foreign file."""
    value = payload[field]
    if type(value) is not int:
        raise VoiceFileError(f"{field} must be an integer, got {type(value).__name__}")
    return value


def _require_str(payload: dict, field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise VoiceFileError(f"{field} must be a string, got {type(value).__name__}")
    return value


def decode(
    raw: bytes, *, expected_id: str, codebooks: int, codebook_size: int, codec_fingerprint: str
) -> VoiceFile:
    """Parse and validate one voice file's bytes.

    `expected_id` is the filename's stem; `codebooks` and `codebook_size` are the loaded
    model's codebook count and per-codebook size; `codec_fingerprint` is the running
    server's own (`breeze_infer.audio.codec_fingerprint`). Raises `VoiceFileError`, and
    only `VoiceFileError`, naming the first rule the file breaks, in data-model.md's own
    order: `VoiceStore.scan()` skips a file on that exception alone, so any other
    exception escaping here would stop the server from starting over one bad file
    (BC-25). Each check below is explicit for that reason, rather than one blanket
    `except` that could also hide a real bug in this module.
    """
    # data-model.md: the file is UTF-8 JSON. Decoding it ourselves, rather than letting
    # json.loads(bytes) sniff UTF-16/32, turns bad bytes into one targeted error.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VoiceFileError(f"not UTF-8: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        # JSONDecodeError, plus the plain ValueError json raises for an integer longer
        # than Python's int-digit limit.
        raise VoiceFileError(f"invalid JSON: {exc}") from exc
    except RecursionError as exc:
        # json's C scanner recurses once per nesting level; `[[[[...` runs out of stack.
        raise VoiceFileError("invalid JSON: nested too deeply") from exc

    # -- JSON/schema errors (data-model.md's first skip rule)
    if not isinstance(payload, dict):
        raise VoiceFileError("record root must be an object")
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise VoiceFileError(f"missing fields: {missing}")
    # The type check comes first: `True == 1`, so an equality check alone would accept
    # `"version": true`, and `1.0 == 1` would accept a float.
    if (
        not isinstance(payload["format"], str)
        or type(payload["version"]) is not int
        or payload["format"] != FORMAT
        or payload["version"] != VERSION
    ):
        raise VoiceFileError(
            f"unknown format/version: {payload['format']!r}/{payload['version']!r}"
        )
    voice_id = _require_str(payload, "id")
    ref_text = _require_str(payload, "ref_text")
    frames = _require_int(payload, "frames")
    file_codebooks = _require_int(payload, "codebooks")
    codes_b64 = _require_str(payload, "codes")
    codes_sha256 = _require_str(payload, "codes_sha256")
    file_fingerprint = _require_str(payload, "codec_fingerprint")
    encode_ms = _require_int(payload, "encode_ms")
    created_at = _require_str(payload, "created_at")

    # The same ref_text rule POST /v1/voices applies (contracts/http-api.md): a saved
    # voice can't hold a transcript the API itself would never have accepted.
    if not ref_text.strip():
        raise VoiceFileError("ref_text is blank")
    if len(ref_text) > MAX_REF_TEXT_CHARS:
        raise VoiceFileError(f"ref_text is longer than {MAX_REF_TEXT_CHARS} characters")
    if frames < 1:
        raise VoiceFileError(f"frames must be at least 1, got {frames}")
    # Checked against the model's own count, not merely for being positive: codes with
    # frames and codebooks swapped have the same byte length and sha256, and only this
    # tells them apart.
    if file_codebooks != codebooks:
        raise VoiceFileError(f"codebooks is {file_codebooks}, the model has {codebooks}")

    # -- stem vs id, then the name itself
    if voice_id != expected_id:
        raise VoiceFileError(f"id {voice_id!r} does not match file name {expected_id!r}")
    if not is_valid_name(voice_id):
        raise VoiceFileError(f"invalid voice name: {voice_id!r}")

    # -- codes: length, then checksum
    try:
        codes_bytes = base64.b64decode(codes_b64, validate=True)
    except ValueError as exc:
        # binascii.Error (bad base64) and the ValueError for a non-ASCII str are both
        # ValueErrors.
        raise VoiceFileError(f"codes is not valid base64: {exc}") from exc
    expected_len = frames * codebooks * 2
    if len(codes_bytes) != expected_len:
        raise VoiceFileError(
            f"codes byte length {len(codes_bytes)} does not match "
            f"frames*codebooks*2 ({expected_len})"
        )
    if hashlib.sha256(codes_bytes).hexdigest() != codes_sha256:
        raise VoiceFileError("codes_sha256 does not match the decoded bytes")

    # -- code range. frames >= 1 and codebooks matching the model make this reshape safe.
    codes = np.frombuffer(codes_bytes, dtype="<i2").reshape(frames, codebooks)
    if int(codes.min()) < 0 or int(codes.max()) >= codebook_size:
        raise VoiceFileError(
            f"code out of range [0, {codebook_size}): "
            f"[{int(codes.min())}, {int(codes.max())}]"
        )

    # -- codec fingerprint
    if file_fingerprint != codec_fingerprint:
        raise VoiceFileError("codec_fingerprint does not match the running codec")

    return VoiceFile(
        id=voice_id,
        ref_text=ref_text,
        frames=frames,
        codebooks=codebooks,
        codes=codes.copy(),  # detach from the base64-decode buffer before returning
        codes_sha256=codes_sha256,
        codec_fingerprint=file_fingerprint,
        encode_ms=encode_ms,
        created_at=created_at,
    )
