"""`GET /v1/audio/speech.wav` tests (specs/004-browser-wav-stream/tasks.md T004/T005).

The app, components and fake runtime come from `tests/test_routes_speech.py`'s helpers, so the
GET route is exercised on exactly the setup the POST route's tests use.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from breeze_infer import api, routes_speech
from breeze_infer.gpu import GpuCloseTimeout, GpuSession
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import wav_header
from tests.fakes import RecordingEvents
from tests.test_routes_speech import (
    SPEECH_PATH,
    _build_components,
    _client_for,
    _fake_runtime,
    _gate_is_free,
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


def _second_response(send: Callable[[TestClient], httpx.Response]) -> httpx.Response:
    """`send`'s second response from a fresh app and runtime.

    `FakeRuntime`'s samples are `call_index / 100`, so two runtimes are only comparable at the
    same call count. The first call's samples are all zero, which would hide which audio a
    route served, so the first response is a warm-up and the second is compared.
    """
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        readiness.mark_ready(_fake_runtime())
        client = _client_for(components)
        assert send(client).status_code == 200
        return send(client)
    finally:
        components.gpu.shutdown()


def test_get_streams_the_post_body_behind_a_wav_header_and_ignores_range() -> None:
    fields = {"text": "hello there", "seed": "7"}

    response = _second_response(
        lambda client: client.get(WAV_PATH, params=fields, headers={"Range": "bytes=0-"})
    )
    pcm = _second_response(lambda client: client.post(SPEECH_PATH, data=fields))

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["accept-ranges"] == "none"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-sample-rate"] == "24000"
    assert pcm.status_code == 200
    assert pcm.content.strip(b"\x00"), "the compared audio must not be silence"
    assert response.content == wav_header(24000) + pcm.content


class _FirstCloseTimesOut(GpuSession[Any]):
    """The first close times out, as a stuck GPU thread would; a later close succeeds,
    because the stuck close has finished by the time the response closes again."""

    _timed_out = False

    async def aclose(self) -> None:
        await super().aclose()
        if not _FirstCloseTimesOut._timed_out:
            _FirstCloseTimesOut._timed_out = True
            raise GpuCloseTimeout("still closing")


def test_a_close_timeout_at_the_end_of_generation_is_reported_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Phase 4 review: the producer's close at DONE timed out, the response's final close then
    # succeeded, and the request was reported as completed on what would be a poisoned gate.
    monkeypatch.setattr(routes_speech, "GpuSession", _FirstCloseTimesOut)
    monkeypatch.setattr(_FirstCloseTimesOut, "_timed_out", False)
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        readiness.mark_ready(_fake_runtime())
        # The headers went out before the failure, so the status is still 200; TestClient
        # returns the body cut short rather than raising, unlike a real connection.
        response = _client_for(components).get(WAV_PATH, params={"text": "hello there"})
        assert response.status_code == 200
        assert _gate_is_free(components)
    finally:
        components.gpu.shutdown()

    names = [name for name, _ in events.calls]
    assert "speech.completed" not in names
    assert "speech.generated" not in names
    [failed] = [fields for name, fields in events.calls if name == "speech.failed"]
    assert failed["reason"] == "gpu_close_timeout"
    assert failed["format"] == "wav"


def test_a_get_that_waits_too_long_for_the_gpu_gets_503_busy_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `create_app` installs the route with production's 60 s wait; a short one keeps this fast.
    monkeypatch.setattr(
        api, "install_speech", functools.partial(routes_speech.install_speech, wav_gpu_wait=0.2)
    )
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        readiness.mark_ready(_fake_runtime())
        client = _client_for(components)
        # Held from this thread: safe, because the waiter times out and leaves the queue
        # before this release, so the release never resolves its future from the wrong thread.
        lease = components.gate.try_acquire()
        assert lease is not None
        try:
            response = client.get(WAV_PATH, params={"text": "hello there"})
        finally:
            lease.release()
    finally:
        components.gpu.shutdown()

    assert response.status_code == 503
    assert response.json() == {"error": "the GPU stayed busy", "code": "busy_timeout"}
    # Only that it was emitted: `waited_s` and asyncio's timer use different clocks, and on
    # Windows the timer's (about 15.6 ms resolution) can fire a tick early.
    assert [name for name, _ in events.calls if name == "speech.queued_timeout"] == [
        "speech.queued_timeout"
    ]
