"""Voice file v1: the on-disk codec and validation for one saved voice.

data-model.md "Voice file v1" (R13): one JSON file per saved voice,
``<voices_dir>/<id>.voice.json``. Codes are base64 of int16 little-endian, row-major
``[frames][codebooks]`` -- numpy's own int16 byte layout, so encoding is a plain
``.tobytes()``/``np.frombuffer()`` round trip with no byte-swapping step (every
platform this runs on is little-endian).

A consumer rejects any file whose ``format`` or ``version`` it doesn't recognise, so a
future version 2 is never silently misread as version 1. ``decode()`` validates in the
exact order data-model.md's "Voice file v1" skip-rule list gives, so the first rule a
file breaks is the one reported: JSON/schema error, stem-vs-id mismatch, bad name (or a
reserved ``v_`` prefix), a codes length or checksum mismatch, an out-of-range code, then
a codec fingerprint mismatch. The one skip rule this module does *not* check --
"a case-duplicate of a file that sorts earlier" -- needs to compare across files, so
it's `voice_store.py`'s job.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

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


def decode(
    raw: bytes, *, expected_id: str, codebook_size: int, codec_fingerprint: str
) -> VoiceFile:
    """Parse and validate one voice file's bytes.

    `expected_id` is the filename's stem; `codec_fingerprint` is the running server's own
    (`breeze_infer.audio.codec_fingerprint`). Raises `VoiceFileError`, naming the first
    rule the file breaks, in data-model.md's own order.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VoiceFileError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VoiceFileError("record root must be an object")
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise VoiceFileError(f"missing fields: {missing}")
    if payload["format"] != FORMAT or payload["version"] != VERSION:
        raise VoiceFileError(
            f"unknown format/version: {payload['format']!r}/{payload['version']!r}"
        )

    voice_id = payload["id"]
    if voice_id != expected_id:
        raise VoiceFileError(f"id {voice_id!r} does not match file name {expected_id!r}")
    if not is_valid_name(voice_id):
        raise VoiceFileError(f"invalid voice name: {voice_id!r}")

    try:
        frames = int(payload["frames"])
        codebooks = int(payload["codebooks"])
        codes_bytes = base64.b64decode(payload["codes"], validate=True)
    except (TypeError, ValueError) as exc:
        raise VoiceFileError(f"malformed field: {exc}") from exc
    except binascii.Error as exc:
        raise VoiceFileError(f"codes is not valid base64: {exc}") from exc

    expected_len = frames * codebooks * 2
    if len(codes_bytes) != expected_len:
        raise VoiceFileError(
            f"codes byte length {len(codes_bytes)} does not match "
            f"frames*codebooks*2 ({expected_len})"
        )
    if hashlib.sha256(codes_bytes).hexdigest() != payload["codes_sha256"]:
        raise VoiceFileError("codes_sha256 does not match the decoded bytes")

    codes = np.frombuffer(codes_bytes, dtype="<i2").reshape(frames, codebooks)
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= codebook_size):
        raise VoiceFileError(
            f"code out of range [0, {codebook_size}): "
            f"[{int(codes.min())}, {int(codes.max())}]"
        )

    if payload["codec_fingerprint"] != codec_fingerprint:
        raise VoiceFileError("codec_fingerprint does not match the running codec")

    return VoiceFile(
        id=voice_id,
        ref_text=str(payload["ref_text"]),
        frames=frames,
        codebooks=codebooks,
        codes=codes.copy(),  # detach from the base64-decode buffer before returning
        codes_sha256=payload["codes_sha256"],
        codec_fingerprint=payload["codec_fingerprint"],
        encode_ms=int(payload["encode_ms"]),
        created_at=str(payload["created_at"]),
    )
