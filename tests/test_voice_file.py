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
CODEBOOKS = 16
FINGERPRINT = "f" * 64


def _codes(frames: int = 4, codebooks: int = CODEBOOKS) -> np.ndarray:
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


def _decode(raw: bytes, *, expected_id: str = "alice", codebooks: int = CODEBOOKS,
            codebook_size: int = CODEBOOK_SIZE, codec_fingerprint: str = FINGERPRINT) -> voice_file.VoiceFile:
    return voice_file.decode(
        raw,
        expected_id=expected_id,
        codebooks=codebooks,
        codebook_size=codebook_size,
        codec_fingerprint=codec_fingerprint,
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


# ------------------------------------------------ malformed files (review findings #1, #7)
#
# Every one of these must surface as VoiceFileError, never as some other exception:
# VoiceStore.scan() skips a file only on VoiceFileError (or OSError), so anything else
# escaping decode() would stop the server from starting over one bad file (BC-25).


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":"\xff"}',  # not UTF-8: json.loads raised UnicodeDecodeError
        b"[" * 100_000,  # deep nesting: json.loads raised RecursionError
        b'{"frames": ' + b"1" * 5000 + b"}",  # past Python's int-digit limit: plain ValueError
        b"\"abc\"",  # valid JSON, but not an object
    ],
    ids=["invalid-utf8", "deep-nesting", "huge-int", "not-an-object"],
)
def test_unparseable_bytes_raise_voice_file_error(raw: bytes):
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # ints must be real ints: not null, not text, not fractional, not bool.
        ("encode_ms", None),
        ("encode_ms", "abc"),
        ("encode_ms", 1.5),
        ("encode_ms", True),
        ("frames", 4.0),
        ("frames", 4.5),
        ("frames", "4"),
        ("frames", True),
        ("codebooks", 16.0),
        ("codebooks", True),
        ("version", True),  # True == 1 in Python, so an equality check alone let it pass
        ("version", 1.0),
        # strings must be real strings.
        ("format", 5),
        ("id", 5),
        ("codes", 5),
        ("codes", None),
        ("codes_sha256", None),
        ("codec_fingerprint", 5),
        ("created_at", 5),
        ("created_at", None),
        # ref_text: a non-blank string of at most MAX_REF_TEXT_CHARS.
        ("ref_text", None),
        ("ref_text", 5),
        ("ref_text", ""),
        ("ref_text", "   "),
        pytest.param("ref_text", "x" * 2001, id="ref_text-2001-chars"),
    ],
)
def test_wrongly_typed_or_out_of_range_field_is_rejected(field: str, value: object):
    # Edits the JSON record directly rather than going through `_encode(**overrides)`,
    # which treats a `codes` override as the array to encode, not the field's value.
    record = json.loads(_encode()[0])
    record[field] = value
    raw = json.dumps(record).encode("utf-8")
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_ref_text_at_the_limit_is_accepted():
    raw, _ = _encode(ref_text="x" * 2000)
    assert len(_decode(raw).ref_text) == 2000


def test_negative_frames_and_codebooks_with_a_matching_sha_are_rejected():
    """(-2) * (-8) * 2 == 32 bytes, exactly one 16-code frame, with a correct sha256:
    only an explicit frames/codebooks check stops this reaching reshape(-2, -8)."""
    raw, _ = _encode(codes=_codes(frames=1), frames=-2, codebooks=-8)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_zero_frames_is_rejected():
    raw, _ = _encode(codes=np.zeros((0, CODEBOOKS), dtype=np.int16))
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_swapped_frames_and_codebooks_are_rejected():
    """[4, 16] codes relabelled as frames=16, codebooks=4: the byte length and sha256
    still match, so only the check against the model's codebook count catches it."""
    raw, _ = _encode(frames=CODEBOOKS, codebooks=4)
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)


def test_a_codebook_count_other_than_the_models_is_rejected():
    raw, _ = _encode(codes=_codes(codebooks=8))  # self-consistent, but 8 != 16
    with pytest.raises(voice_file.VoiceFileError):
        _decode(raw)
