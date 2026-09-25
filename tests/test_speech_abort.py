"""How a streamed speech response ends: abort, disconnect, stall (T040; research.md R2, R3).

These run a real uvicorn (h11, as in production) on an ephemeral port, because the behaviour
under test lives in the server: whether the chunked terminator is written, when a disconnect
is noticed, and what a blocked `send()` does. The route is a minimal stand-in for
`routes_speech.py`: take the gate, prime the first chunk, return a `SpeechResponse` fed by
`FakeRuntime` through a real `GpuThread`.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response

from breeze_infer.audio import pcm16
from breeze_infer.errors import install_error_handlers
from breeze_infer.gpu import DONE, GpuGate, GpuSession, GpuThread
from breeze_infer.streaming import SpeechResponse

# FakeRuntime imports this lazily on the first step, i.e. on the GPU thread in the middle of a
# request; on a slow mount that takes tens of seconds and would blow every bound below.
from models import fast_streaming  # noqa: F401
from tests.fakes import CODEC_SAMPLES_PER_FRAME, FakeRuntime, RecordingEvents

T = TypeVar("T")

CHUNK_BYTES = CODEC_SAMPLES_PER_FRAME * 2  # one default FakeRuntime chunk as s16le
TERMINATOR = b"0\r\n\r\n"
REQUEST = b"POST /speech HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n"
TERMINAL_EVENTS = ("speech.completed", "speech.aborted", "speech.failed")


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(0.01)


class TrackedGeneration:
    """Wraps a FakeRuntime generation to record how far it got and which thread ended it."""

    def __init__(self, chunks: Iterator[Any]) -> None:
        self._chunks = chunks
        self.yielded = 0
        self.ended_on: str | None = None

    def run(self) -> Generator[Any, None, None]:
        try:
            for chunk in self._chunks:
                self.yielded += 1
                yield chunk
        finally:
            self._chunks.close()  # type: ignore[attr-defined]
            self.ended_on = threading.current_thread().name


class ObservedSession(GpuSession[T]):
    """A GpuSession that lets the test see the moment the response asked for the close."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.close_requested = threading.Event()

    async def aclose(self) -> None:
        self.close_requested.set()
        await super().aclose()


async def rest_of_audio(session: GpuSession[Any]) -> AsyncGenerator[bytes, None]:
    while True:
        chunk = await session.step()
        if chunk is DONE:
            return
        yield pcm16(chunk.audio)


@dataclass
class Rig:
    runtime: FakeRuntime
    send_timeout: float = 30.0
    events: RecordingEvents = field(default_factory=RecordingEvents)
    gate: GpuGate = field(default_factory=GpuGate)
    gpu: GpuThread = field(default_factory=lambda: GpuThread("cpu", lambda _device: None))
    generations: list[TrackedGeneration] = field(default_factory=list)
    sessions: list[ObservedSession[Any]] = field(default_factory=list)

    def app(self) -> FastAPI:
        app = FastAPI()
        install_error_handlers(app, self.events)  # type: ignore[arg-type]

        @app.post("/speech")
        async def speech(_: Request) -> Response:
            lease = self.gate.try_acquire()
            assert lease is not None, "a previous test request still holds the gate"
            generation = TrackedGeneration(self.runtime.iter_audio_chunks({}))
            self.generations.append(generation)
            session = ObservedSession(lease, self.gpu, generation.run())
            self.sessions.append(session)
            try:
                first = await session.step()
            except BaseException:
                await session.aclose()
                raise  # the app's error handlers turn this into 500 internal_error
            assert first is not DONE
            return SpeechResponse(
                first_chunk=pcm16(first.audio),
                body=rest_of_audio(session),
                session=session,
                events=self.events,
                request_id="req-test",
                sample_rate=self.runtime.sample_rate,
                clock=time.monotonic,
                started_at=time.monotonic(),
                send_timeout=self.send_timeout,
            )

        return app

    def terminal_events(self) -> list[tuple[str, dict[str, object]]]:
        return [(name, fields) for name, fields in self.events.calls if name in TERMINAL_EVENTS]


class LiveServer:
    """uvicorn serving `app` on 127.0.0.1:<ephemeral> from its own thread and event loop."""

    def __init__(self, app: FastAPI) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/speech"
        self.loop = asyncio.new_event_loop()
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                lifespan="off",
                http="h11",
                log_config=None,
                access_log=False,
                timeout_graceful_shutdown=2,
            )
        )
        self._thread = threading.Thread(
            target=self.loop.run_until_complete,
            args=(self.server.serve(sockets=[sock]),),
            daemon=True,
        )
        self._thread.start()
        wait_until(lambda: self.server.started)

    def call(self, fn: Callable[[], T]) -> T:
        """Run `fn` on the server's event loop, where the gate lives."""

        async def run() -> T:
            return fn()

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout=5)

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=10)
        self.loop.close()


def gate_is_free(rig: Rig, server: LiveServer) -> bool:
    def probe() -> bool:
        lease = rig.gate.try_acquire()
        if lease is None:
            return False
        lease.release()
        return True

    return server.call(probe)


@pytest.fixture
def serve() -> Iterator[Callable[[Rig], LiveServer]]:
    started: list[tuple[Rig, LiveServer]] = []

    def start(rig: Rig) -> LiveServer:
        server = LiveServer(rig.app())
        started.append((rig, server))
        return server

    yield start
    for rig, server in started:
        if rig.runtime.gate is not None:
            rig.runtime.gate.set()  # never leave the GPU thread parked on a failed test
        server.stop()
        assert rig.gpu.shutdown(timeout=5)


def read_until_closed(sock: socket.socket) -> bytes:
    received = bytearray()
    while data := sock.recv(65536):
        received += data
    return bytes(received)


def test_bc_17_failure_after_streaming_starts_aborts_the_response(
    serve: Callable[[Rig], LiveServer],
) -> None:
    """C++ ends a stream that failed mid-generation exactly like a successful one (the
    chunked terminator is written), so a client can't tell truncated audio from complete
    audio. Here the connection is dropped without the terminator."""
    rig = Rig(FakeRuntime(chunks=10, fail_after=3))
    server = serve(rig)

    received = bytearray()
    with httpx.Client(timeout=10) as client, client.stream("POST", server.url) as response:
        assert response.status_code == 200
        with pytest.raises(httpx.RemoteProtocolError):
            for data in response.iter_raw():
                received += data
    assert len(received) == 3 * CHUNK_BYTES  # everything produced before the failure
    wait_until(lambda: gate_is_free(rig, server))

    # The same failure seen on the wire, as curl would: headers, some body, then EOF with
    # no terminating zero-length chunk.
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
        sock.sendall(REQUEST)
        raw = read_until_closed(sock)
    assert raw.startswith(b"HTTP/1.1 200 ")
    assert b"transfer-encoding: chunked" in raw.lower()
    assert not raw.endswith(TERMINATOR)
    wait_until(lambda: gate_is_free(rig, server))

    assert [name for name, _ in rig.terminal_events()] == ["speech.failed", "speech.failed"]
    for _, fields in rig.terminal_events():
        assert fields["level"] == "error"
        assert fields["reason"] == "generation_error"
        assert "RuntimeError" in str(fields["error"])
    assert all(generation.ended_on.startswith("breeze-gpu") for generation in rig.generations)


def test_bc_17_failure_before_first_chunk_returns_an_error_status(
    serve: Callable[[Rig], LiveServer],
) -> None:
    """C++ answers `200` with an empty body when generation fails before any audio. Here the
    first chunk is produced before the status is chosen, so the client gets a JSON 500."""
    rig = Rig(FakeRuntime(chunks=4, fail_after=0))
    server = serve(rig)

    response = httpx.post(server.url, timeout=10)

    assert response.status_code == 500
    assert response.json() == {"error": "internal error", "code": "internal_error"}
    assert gate_is_free(rig, server)
    assert rig.runtime.closed == 1
    assert rig.terminal_events() == []  # no response was ever built
    assert [name for name, _ in rig.events.calls] == ["request.failed"]


def test_client_disconnect_releases_the_gate_within_one_chunk(
    serve: Callable[[Rig], LiveServer],
) -> None:
    # Generation blocks before chunk 2, so the disconnect lands while a step is in flight.
    hold = threading.Event()
    rig = Rig(FakeRuntime(chunks=10, gate=hold, gate_at=2))
    server = serve(rig)

    with httpx.Client(timeout=10) as client, client.stream("POST", server.url) as response:
        assert response.status_code == 200
        received = 0
        for data in response.iter_raw():
            received += len(data)
            if received >= 2 * CHUNK_BYTES:
                break
    # Leaving both blocks closed the connection mid-body.

    [session] = rig.sessions
    assert session.close_requested.wait(timeout=5), "the disconnect was not noticed"
    # The close queues behind the step still running on the GPU thread, and only then is
    # the gate released: the next request must not overlap a step that is still running.
    assert not gate_is_free(rig, server)
    hold.set()
    wait_until(lambda: gate_is_free(rig, server))

    [generation] = rig.generations
    assert generation.yielded == 3  # the two sent, plus the one in flight at the disconnect
    assert generation.ended_on is not None and generation.ended_on.startswith("breeze-gpu")
    assert rig.runtime.closed == 1
    assert rig.terminal_events() == [
        ("speech.aborted", {"level": "info", "request_id": "req-test",
                            "audio_seconds": 2 * CODEC_SAMPLES_PER_FRAME / 24000,
                            "reason": "client_disconnect"})
    ]


def test_gate_released_if_body_never_starts() -> None:
    """Cleanup in the body generator's `finally` would never run here: the request is
    cancelled while the headers are still being sent, before the body is first iterated."""
    runtime = FakeRuntime(chunks=4)
    events = RecordingEvents()
    body_started = False

    async def scenario() -> None:
        nonlocal body_started
        gate = GpuGate()
        gpu = GpuThread("cpu", lambda _device: None)
        lease = gate.try_acquire()
        assert lease is not None
        generation = TrackedGeneration(runtime.iter_audio_chunks({}))
        session = GpuSession(lease, gpu, generation.run())
        first = await session.step()
        assert first is not DONE

        async def body() -> AsyncGenerator[bytes, None]:
            nonlocal body_started
            body_started = True
            async for chunk in rest_of_audio(session):
                yield chunk

        response = SpeechResponse(
            first_chunk=pcm16(first.audio),
            body=body(),
            session=session,
            events=events,
            request_id="req-test",
            sample_rate=runtime.sample_rate,
            clock=time.monotonic,
            started_at=time.monotonic(),
        )
        headers_sending = asyncio.Event()

        async def receive() -> dict[str, Any]:
            await asyncio.Event().wait()  # the client never sends or disconnects
            raise AssertionError("unreachable")

        async def send(message: dict[str, Any]) -> None:
            headers_sending.set()
            await asyncio.Event().wait()  # a peer that never accepts the headers

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}
        call = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(headers_sending.wait(), timeout=5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, timeout=5)

        # aclose() has run (not just been started): the gate is free right away.
        free = gate.try_acquire()
        assert free is not None
        free.release()
        assert generation.ended_on is not None and generation.ended_on.startswith("breeze-gpu")
        assert await asyncio.to_thread(gpu.shutdown, 5)

    asyncio.run(scenario())

    assert not body_started
    assert runtime.closed == 1
    assert [(name, fields["reason"]) for name, fields in events.calls] == [
        ("speech.aborted", "cancelled")
    ]


def test_stalled_reader_hits_send_timeout(serve: Callable[[Rig], LiveServer]) -> None:
    # ~380 KB chunks, ~38 MB in all: far more than the socket buffers can absorb, so the
    # server's send() blocks once the client stops reading.
    rig = Rig(FakeRuntime(chunks=100, frames_per_chunk=100), send_timeout=0.5)
    server = serve(rig)

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # before connect
        sock.settimeout(10)
        sock.connect(("127.0.0.1", server.port))
        sock.sendall(REQUEST)
        assert sock.recv(4096).startswith(b"HTTP/1.1 200 ")
        # ...and never read again. The connection stays open, so this isn't a disconnect.

        wait_until(lambda: gate_is_free(rig, server), timeout=10)

        [(name, fields)] = rig.terminal_events()
        assert name == "speech.aborted"
        assert fields["reason"] == "send_timeout"
        [generation] = rig.generations
        assert generation.yielded < rig.runtime.chunks  # stopped early, not run to the end
        assert generation.ended_on is not None and generation.ended_on.startswith("breeze-gpu")
        assert rig.runtime.closed == 1
