"""`GET /v1/audio/speech.wav` buffered delivery (specs/004-browser-wav-stream/tasks.md T007;
research.md R4).

These run the real app on a real uvicorn (`tests/test_speech_abort.py`'s `LiveServer`), because
what is under test is how generation relates to a client that stops reading: only a real server
blocks in `send()` once the socket buffers are full. The components and fake runtime come from
`tests/test_routes_speech.py`'s helpers, as in `tests/test_speech_wav.py`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from breeze_infer.api import Components, create_app
from breeze_infer.limits import TCP_USER_TIMEOUT_MS
from breeze_infer.routes_health import Readiness
from tests.fakes import CODEC_SAMPLES_PER_FRAME, FakeRuntime, RecordingEvents
from tests.test_routes_speech import _build_components, _fake_runtime, _gate_is_free
from tests.test_speech_abort import LiveServer, wait_until

FIELDS = {"text": "hello there"}
# 150 chunks of 5 frames: 750 frames, the fake's default per-piece cap, so none is cut. As
# s16le that is 2.88 MB, several times what loopback socket buffers hold for a client that
# doesn't read (about 0.8 MB here) plus uvicorn's write buffer, so without buffered delivery
# the server's `send()` blocks long before generation ends.
CHUNKS = 150
FRAMES_PER_CHUNK = 5
PCM_BYTES = CHUNKS * FRAMES_PER_CHUNK * CODEC_SAMPLES_PER_FRAME * 2
WAV_HEADER_BYTES = 44
OUTCOME_EVENTS = ("speech.completed", "speech.aborted", "speech.failed")


@contextmanager
def _serving(
    runtime: FakeRuntime,
) -> Iterator[tuple[Components, RecordingEvents, LiveServer, str]]:
    """The real app over `runtime`, on a live uvicorn: its components, the recorded events,
    the server, and the route's URL."""
    events = RecordingEvents()
    readiness = Readiness()
    components = _build_components(readiness, events=events)
    readiness.mark_ready(runtime)
    # Production's kernel timeout, not LiveServer's 2 s default: Linux applies it to a zero
    # receive window too, so a client that deliberately reads nothing would be reset.
    server = LiveServer(
        create_app(components),  # type: ignore[arg-type]
        user_timeout_ms=TCP_USER_TIMEOUT_MS,
    )
    try:
        yield components, events, server, f"http://127.0.0.1:{server.port}/v1/audio/speech.wav"
    finally:
        if runtime.gate is not None:
            runtime.gate.set()  # never leave the GPU thread parked on a failed test
        try:
            server.stop()
        finally:
            assert components.gpu.shutdown(timeout=5)


def _gate_free_on_server(components: Components, server: LiveServer) -> bool:
    # `GpuGate` is loop-bound (gpu.py): probe it on the server's loop, never from this thread.
    return server.call(lambda: _gate_is_free(components))


def _named(events: RecordingEvents, *names: str) -> list[tuple[str, dict[str, object]]]:
    return [(name, fields) for name, fields in events.calls if name in names]


def test_a_client_that_stops_reading_does_not_hold_the_gpu() -> None:
    runtime = _fake_runtime(chunks=CHUNKS, frames_per_chunk=FRAMES_PER_CHUNK)
    with (
        _serving(runtime) as (components, events, server, url),
        httpx.Client(timeout=30) as client,
        client.stream("GET", url, params=FIELDS) as response,
    ):
        assert response.status_code == 200
        # Nothing is read past the headers: generation must still finish and free the GPU.
        wait_until(lambda: bool(_named(events, "speech.generated")), timeout=10)
        assert _gate_free_on_server(components, server)

        body = response.read()

    assert len(body) == WAV_HEADER_BYTES + PCM_BYTES
    [(_, generated)] = _named(events, "speech.generated")
    assert generated["audio_seconds"] == PCM_BYTES / 2 / runtime.sample_rate
    assert generated["format"] == "wav"
    wait_until(lambda: bool(_named(events, *OUTCOME_EVENTS)))
    [(name, fields)] = _named(events, *OUTCOME_EVENTS)
    assert name == "speech.completed"
    assert fields["format"] == "wav"


def test_a_disconnect_mid_generation_stops_it_and_frees_the_gpu() -> None:
    hold = threading.Event()
    reached = threading.Event()
    # Chunk 0 is primed before the `200`; the step for chunk 2 then blocks on `hold`, so the
    # disconnect lands while a step is in flight on the GPU thread.
    runtime = _fake_runtime(
        chunks=CHUNKS,
        frames_per_chunk=FRAMES_PER_CHUNK,
        gate=hold,
        gate_at=2,
        gate_reached=reached,
    )
    with _serving(runtime) as (components, events, server, url):
        with httpx.Client(timeout=10) as client, client.stream("GET", url, params=FIELDS) as r:
            assert r.status_code == 200
            assert reached.wait(timeout=5), "generation never reached the held chunk"
        # Leaving both blocks closed the connection. Release the step only once the server
        # has seen the disconnect and had a few loop turns to act on it, or generation could
        # simply run on to the end.
        server.wait_for_no_connections(timeout=5)
        asyncio.run_coroutine_threadsafe(_loop_turns(10), server.loop).result(timeout=5)
        hold.set()

        wait_until(lambda: _gate_free_on_server(components, server), timeout=5)
        wait_until(lambda: bool(_named(events, *OUTCOME_EVENTS)))
        server.wait_for_handlers()

    [(name, fields)] = _named(events, *OUTCOME_EVENTS)
    assert name == "speech.aborted"
    assert fields["reason"] == "client_disconnect"
    assert fields["format"] == "wav"
    # Stopped, not finished: generation never reached its end.
    assert _named(events, "speech.generated", "speech.piece_done") == []


async def _loop_turns(count: int) -> None:
    for _ in range(count):
        await asyncio.sleep(0)
