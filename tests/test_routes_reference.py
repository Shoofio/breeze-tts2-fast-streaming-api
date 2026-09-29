"""Route-level reference-audio validation tests (specs/003-cpp-compatible-api/tasks.md
T045; BC-11 to BC-16).

`tests/test_reference_audio.py` already covers `reference_audio.decode()` as a pure
unit; this file proves `POST /v1/audio/speech` wires that decoder up correctly end to
end -- through `read_fields`/`parse_speech`'s reference-rule checks, the route's own
order, and the response envelope a real client actually sees -- using a real
`TestClient` (`FakeRuntime`, no GPU) and real `soundfile`-written audio, plus a few
WAV headers built by hand with `struct` for cases no valid encoder can produce (BC-15:
a bogus data-chunk length, a truncated `fmt` chunk, a zero/absurd channel count, 0
bits per sample, and an unrecognized format tag).

Reuses `tests/test_routes_speech.py`'s app-wiring helpers and fixtures instead of
rebuilding them.
"""

# Every fixture imported below (`client`, `components`, `readiness`, `ready_client`) is
# reused, by pytest's own name-based discovery, as a same-named parameter on the tests
# in this file -- the standard way to share fixtures across modules without a
# `conftest.py`. Ruff's F811 ("redefinition") otherwise fires on every one of those
# parameters, since it can't tell a fixture parameter from an accidental shadow.
# ruff: noqa: F811

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from breeze_infer.limits import MAX_REF_SECONDS
from tests.fakes import CODEC_SAMPLE_RATE
from tests.test_routes_speech import (  # noqa: F401 -- fixtures reused by name
    SPEECH_PATH,
    _wav_bytes,
    client,
    components,
    readiness,
    ready_client,
)

# --- WAV headers built by hand (BC-15): the malformed shapes below aren't producible
# through any real encoder, so `soundfile` can't generate them -- only `struct` can. ---


def _build_wav(
    *,
    format_tag: int = 1,  # WAVE_FORMAT_PCM
    channels: int = 1,
    sample_rate: int = 16000,
    bits_per_sample: int = 16,
    data: bytes = bytes(1600),  # a small, fixed data payload -- deliberately never
    # scaled by `channels`/`bits_per_sample` above: those fields are exactly what's
    # under test, and an absurd one (e.g. 65,535 channels) must not blow up the
    # actual file size just because a real encoder would have scaled it that way.
) -> bytearray:
    """A minimal, otherwise well-formed RIFF/WAVE file: `RIFF` + size, `WAVE`, one
    `fmt ` chunk (declared 16 bytes) and one `data` chunk. Every malformed variant
    below starts here and changes exactly one thing, so each test isolates one
    defect rather than several at once.
    """
    block_align = (max(channels, 0) * (bits_per_sample // 8)) & 0xFFFF
    byte_rate = (sample_rate * block_align) & 0xFFFFFFFF
    fmt_chunk = struct.pack(
        "<HHIIHH",
        format_tag & 0xFFFF,
        channels & 0xFFFF,
        sample_rate & 0xFFFFFFFF,
        byte_rate,
        block_align,
        bits_per_sample & 0xFFFF,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    chunks += b"data" + struct.pack("<I", len(data)) + data
    riff = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    return bytearray(riff)


def _with_bogus_data_length() -> bytes:
    """The `data` chunk declares a length of `0xFFFFFFFF` -- far past both the real
    bytes present (1,600, well under the 80 ms/1,920-sample minimum at 16 kHz) and any
    sane file size -- without the actual byte count changing at all: the length a naive
    reader would use to seek or allocate, not the length the file genuinely has. A safe
    decoder either refuses to trust it (`invalid_audio`) or reads only the real bytes
    present, in which case the result is a genuinely short clip (`audio_too_short`) --
    either is a safe, structured `400`; only a `200`, a crash or a hang would be BC-15's
    defect made concrete again."""
    blob = _build_wav()
    idx = blob.index(b"data")
    struct.pack_into("<I", blob, idx + 4, 0xFFFFFFFF)
    return bytes(blob)


def _truncated_fmt_chunk() -> bytes:
    """The `fmt ` chunk declares 16 bytes, but the file ends 4 bytes into it -- no
    `data` chunk, no rest of `fmt` at all."""
    header = b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    return header + b"\x01\x00\x01\x00"


# Every case's blob, plus which error code(s) count as "rejected safely" for it: most
# must be exactly invalid_audio (the header itself is nonsense), but a data length that
# merely *overstates* the real payload can be safely resolved either by refusing it or
# by reading only what's really there (see _with_bogus_data_length's docstring).
_MALFORMED_WAVS: list[tuple[str, bytes, frozenset[str]]] = [
    ("bogus_data_chunk_length", _with_bogus_data_length(), frozenset({"invalid_audio", "audio_too_short"})),
    ("truncated_fmt_chunk", _truncated_fmt_chunk(), frozenset({"invalid_audio"})),
    ("zero_channels", bytes(_build_wav(channels=0)), frozenset({"invalid_audio"})),
    ("zero_bits_per_sample", bytes(_build_wav(bits_per_sample=0)), frozenset({"invalid_audio"})),
    ("zero_hz_sample_rate", bytes(_build_wav(sample_rate=0)), frozenset({"invalid_audio"})),
    ("one_hz_sample_rate", bytes(_build_wav(sample_rate=1)), frozenset({"invalid_audio"})),
    ("65535_channels", bytes(_build_wav(channels=65_535)), frozenset({"invalid_audio"})),
    ("unknown_format_tag", bytes(_build_wav(format_tag=9999)), frozenset({"invalid_audio"})),
]


def _sine_wav(seconds: float, sample_rate: int = 16000, *, subtype: str = "PCM_16") -> bytes:
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float64)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype=subtype)
    return buf.getvalue()


def _post_with_reference(
    client: TestClient, *, ref_audio: bytes, ref_text: str = "a reference transcript", **extra: str
) -> object:
    data = {"text": "hello there", "ref_text": ref_text, **extra}
    return client.post(
        SPEECH_PATH, data=data, files={"ref_audio": ("ref.wav", ref_audio, "audio/wav")}
    )


# --- BC-11: undecodable or empty ref_audio ------------------------------------------


def test_bc_11_undecodable_or_empty_ref_audio_gets_400(ready_client: TestClient) -> None:
    """C++ silently turned bad/empty audio into voice-design speech (a random voice);
    this must be a 400 instead."""
    empty = _post_with_reference(ready_client, ref_audio=b"")
    assert empty.status_code == 400
    assert empty.json()["code"] == "invalid_audio"

    garbage = _post_with_reference(ready_client, ref_audio=b"not audio, just noise" * 50)
    assert garbage.status_code == 400
    assert garbage.json()["code"] == "invalid_audio"


# --- BC-12/BC-13/BC-14: reference consistency ---------------------------------------


def test_bc_12_ref_audio_without_ref_text_gets_400(ready_client: TestClient) -> None:
    """BC-12: `ref_audio` without `ref_text` is `400 ref_text_required` -- the C++ server
    ignored the upload silently instead.
    """
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello there"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text is required with ref_audio",
        "code": "ref_text_required",
    }


def test_bc_13_ref_text_without_reference_gets_400(ready_client: TestClient) -> None:
    """BC-13: a stray `ref_text` with no `ref_audio`/`voice_id` is `400
    reference_required` on HTTP -- the C++ server ignored it silently instead.
    """
    response = ready_client.post(
        SPEECH_PATH, data={"text": "hello there", "ref_text": "a stray transcript"}
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text needs ref_audio or voice_id",
        "code": "reference_required",
    }


def test_bc_14_voice_id_with_ref_audio_gets_400(ready_client: TestClient) -> None:
    """BC-14: sending both `voice_id` and `ref_audio` is `400 reference_conflict`
    (mutually exclusive) -- the C++ server silently ignored the upload instead.
    """
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello there", "voice_id": "alice", "ref_text": "override text"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "voice_id and ref_audio cannot be used together",
        "code": "reference_conflict",
    }


# --- BC-15: malformed WAVs are rejected safely --------------------------------------


@pytest.mark.parametrize(
    "name,blob,expected_codes", _MALFORMED_WAVS, ids=[n for n, _, _ in _MALFORMED_WAVS]
)
def test_bc_15_malformed_wavs_are_rejected_safely(
    ready_client: TestClient, name: str, blob: bytes, expected_codes: frozenset[str]
) -> None:
    """BC-15: the C++ server's hand-written WAV parser could over-read memory or divide by
    zero on each of these. Here they must produce an ordinary 400, and the
    server must still be answering /health afterwards -- proof the process itself was
    unaffected."""
    response = _post_with_reference(ready_client, ref_audio=blob)

    assert response.status_code == 400, name
    assert response.json()["code"] in expected_codes, name

    health = ready_client.get("/health")
    assert health.status_code == 200, f"server unresponsive after {name}"


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_24", "FLOAT"])
def test_bc_15_pcm_and_float_wavs_now_decode(ready_client: TestClient, subtype: str) -> None:
    """BC-15's other half: the C++ server decoded 8/24-bit PCM as silence; now they decode
    (and reach a real 200), same as 16-bit."""
    response = _post_with_reference(ready_client, ref_audio=_sine_wav(0.5, subtype=subtype))

    assert response.status_code == 200
    assert len(response.content) > 0


# --- BC-16: reference clip length ----------------------------------------------------


def test_bc_16_too_long_or_too_short_reference_gets_400(ready_client: TestClient) -> None:
    """BC-16: a reference clip over 30 seconds, or under one full codec frame (80 ms), is
    `400` -- the C++ server allowed unlimited length and silently ignored a clip too short
    for any frame.
    """
    too_long = _post_with_reference(
        ready_client, ref_audio=_sine_wav(MAX_REF_SECONDS + 1, sample_rate=CODEC_SAMPLE_RATE)
    )
    assert too_long.status_code == 400
    assert too_long.json() == {
        "error": "ref_audio is longer than 30 seconds",
        "code": "audio_too_long",
    }

    # 500 samples at 24 kHz is ~20.8 ms, well under the 80 ms (1,920-sample) minimum.
    short_wav = np.zeros(500, dtype=np.float64)
    buf = io.BytesIO()
    sf.write(buf, short_wav, CODEC_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    too_short = _post_with_reference(ready_client, ref_audio=buf.getvalue())
    assert too_short.status_code == 400
    assert too_short.json() == {"error": "ref_audio is too short", "code": "audio_too_short"}
