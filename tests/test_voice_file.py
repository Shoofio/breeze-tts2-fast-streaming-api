"""tests/test_voice_file.py -- breeze_infer/voice_file.py (T055, T061).

Pure: no filesystem, no GPU. The v1 codec (data-model.md "Voice file v1") and every
rejection its module docstring promises, each isolated to one broken field so the test
names the rule it exercises.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from breeze_infer import voice_file

CODEBOOK_SIZE = 2048
FINGERPRINT = "f" * 64


def _codes(frames: int = 4, codebooks: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, CODEBOOK_SIZE, size=(frames, codebooks), dtype=np.int16)


def _encode(**overrides: object) -> tuple[bytes, np.ndarray]:
    """`voice_file.encode()` for a well-formed voice, with `overrides` applied to the
    resulting JSON record afterward -- so a test can corrupt exactly one field without
    the encode step silently "fixing" it back up (e.g. re-deriving `codes_sha256`)."""
    codes = overrides.pop("codes", None)
    if codes is None:
        codes = _codes()
    raw = voice_file.encode(
        id="alice",
        ref_text="hello there",
        codes=codes,
        codec_fingerprint=FINGERPRINT,
        encode_ms=812,
        created_at="2026-09-24T20:15:00Z",
    )
    if overrides:
        record = json.loads(raw)
        record.update(overrides)
        raw = json.dumps(record).encode("utf-8")
    return raw, codes


def _decode(raw: bytes, *, expected_id: str = "alice", codebook_size: int = CODEBOOK_SIZE,
            codec_fingerprint: str = FINGERPRINT) -> voice_file.VoiceFile:
    return voice_file.decode(
        raw, expected_id=expected_id, codebook_size=codebook_size, codec_fingerprint=codec_fingerprint
    )


def test_round_trip():
    raw, codes = _encode()
    voice = _decode(raw)
    assert voice.id == "alice"
    assert voice.ref_text == "hello there"
    assert voice.frames == codes.shape[0]
    assert voice.codebooks == codes.shape[1]
    assert np.array_equal(voice.codes, codes)
    assert voice.codes.dtype == np.int16
    assert voice.codec_fingerprint == FINGERPRINT
    assert voice.encode_ms == 812
    assert voice.created_at == "2026-09-24T20:15:00Z"


def test_unknown_format_is_rejected():
    raw, _ = _encode(format="not-breeze-tts-voice")
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_unknown_version_is_rejected():
    raw, _ = _encode(version=2)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_stem_differing_from_id_is_rejected():
    raw, _ = _encode()  # id == "alice"
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw, expected_id="bob")


def test_bad_name_is_rejected():
    raw, _ = _encode(id="bad name!")
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw, expected_id="bad name!")


def test_v_prefixed_name_is_rejected():
    raw, _ = _encode(id="v_0123456789abcdef")
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw, expected_id="v_0123456789abcdef")


def test_v_prefixed_name_is_rejected_case_insensitively():
    raw, _ = _encode(id="V_0123456789ABCDEF")
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw, expected_id="V_0123456789ABCDEF")


def test_codes_length_mismatch_is_rejected():
    # frames now disagrees with the actual (unchanged) base64 codes payload.
    raw, _ = _encode(frames=999)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_codes_sha256_mismatch_is_rejected():
    raw, _ = _encode(codes_sha256="0" * 64)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_code_out_of_range_is_rejected():
    codes = _codes()
    codes[0, 0] = CODEBOOK_SIZE  # exactly one past the valid range
    raw, _ = _encode(codes=codes)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_negative_code_is_rejected():
    codes = _codes().astype(np.int16)
    codes[-1, -1] = -1
    raw, _ = _encode(codes=codes)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_fingerprint_mismatch_is_rejected():
    raw, _ = _encode()
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw, codec_fingerprint="a" * 64)


def test_malformed_json_is_rejected():
    with pytest.raises(voice_file.VoiceFileError):
        _decode(b"not json at all")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("alice", True),
        ("Alice-2_final", True),
        ("v_deadbeef", False),
        ("V_DEADBEEF", False),
        ("", False),
        ("a" * 65, False),
        ("bad name!", False),
    ],
)
def test_is_valid_name(name: str, expected: bool):
    assert voice_file.is_valid_name(name) is expected
