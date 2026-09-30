"""`GET /v1/audio/speech.wav` tests (specs/004-browser-wav-stream/tasks.md T004/T005).

The app, components and fake runtime come from `tests/test_routes_speech.py`'s helpers, so the
GET route is exercised on exactly the setup the POST route's tests use.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import wav_header
from tests.test_routes_speech import (
    SPEECH_PATH,
    _build_components,
    _client_for,
    _fake_runtime,
)

WAV_PATH = "/v1/audio/speech.wav"


def test_wav_header_matches_the_contract_at_24000_hz() -> None:
    expected = (
        b"RIFF" b"\xff\xff\xff\xff" b"WAVE"
        b"fmt " b"\x10\x00\x00\x00"  # fmt chunk size 16
        b"\x01\x00" b"\x01\x00"  # PCM, mono
        b"\xc0\x5d\x00\x00"  # 24000 Hz
        b"\x80\xbb\x00\x00"  # byte rate 48000
        b"\x02\x00" b"\x10\x00"  # block align 2, 16 bits
        b"data" b"\xff\xff\xff\xff"
    )

    assert wav_header(24000) == expected


def _first_response(send: Callable[[TestClient], httpx.Response]) -> httpx.Response:
    """`send`'s response from a fresh app and runtime: `FakeRuntime`'s samples depend on how
    many calls it has served, so only each one's first request is comparable to another's."""
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        readiness.mark_ready(_fake_runtime())
        return send(_client_for(components))
    finally:
        components.gpu.shutdown()


def test_get_streams_the_post_body_behind_a_wav_header_and_ignores_range() -> None:
    fields = {"text": "hello there", "seed": "7"}

    response = _first_response(
        lambda client: client.get(WAV_PATH, params=fields, headers={"Range": "bytes=0-"})
    )
    pcm = _first_response(lambda client: client.post(SPEECH_PATH, data=fields))

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["accept-ranges"] == "none"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-sample-rate"] == "24000"
    assert pcm.status_code == 200
    assert response.content == wav_header(24000) + pcm.content
