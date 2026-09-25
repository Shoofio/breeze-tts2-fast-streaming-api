"""How a streamed speech response ends: abort, disconnect, stall (T040; research.md R2, R3).

These run a real uvicorn (h11, as in production) on an ephemeral port, because the behaviour
under test lives in the server: whether the chunked terminator is written, when a disconnect
is noticed, and what a blocked `send()` does. The route is a minimal stand-in for
`routes_speech.py`: take the gate, prime the first chunk, return a `SpeechResponse` fed by
`FakeRuntime` through a real `GpuThread`.
"""

from __future__ import annotations

import asyncio
import gc
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
from breeze_infer.errors import StreamAborted, install_error_handlers
from breeze_infer.gpu import DONE, GpuGate, GpuSession, GpuThread
from breeze_infer.streaming import SendTimeout, SpeechResponse

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


def wait_for_events(rig: Rig, count: int = 1) -> None:
    """Wait for `count` outcome events: each is emitted after the gate is released."""
    wait_until(lambda: len(rig.terminal_events()) >= count)


async def eventually(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        await asyncio.sleep(0.01)


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
                headers={"X-Request-Id": "req-test"},
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
        try:
            wait_until(lambda: self.server.started or not self._thread.is_alive())
            if not self.server.started:
                raise AssertionError("uvicorn exited without starting")
        except BaseException:
            self.stop()
            sock.close()  # uvicorn closes it once serving; not if it never got that far
            raise

    def call(self, fn: Callable[[], T]) -> T:
        """Run `fn` on the server's event loop, where the gate lives."""

        async def run() -> T:
            return fn()

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout=5)

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            # The loop is still running: closing it now would fail and hide this error.
            raise AssertionError("uvicorn did not stop within 10 s")
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
        try:
            server.stop()
        finally:
            assert rig.gpu.shutdown(timeout=5)




def read_until_closed(sock: socket.socket) -> bytes:
    received = bytearray()
    while data := sock.recv(65536):
        received += data
    return bytes(received)


def test_a_finished_stream_completes_with_the_contract_headers(
    serve: Callable[[Rig], LiveServer],
) -> None:
    rig = Rig(FakeRuntime(chunks=4))
    server = serve(rig)

    response = httpx.post(server.url, timeout=10)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-sample-format"] == "s16le"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == "req-test"  # extra headers still pass through
    assert len(response.content) == 4 * CHUNK_BYTES
    wait_for_events(rig)
    [(name, fields)] = rig.events.calls
    assert name == "speech.completed"
    assert fields["audio_seconds"] == 4 * CODEC_SAMPLES_PER_FRAME / 24000
    assert isinstance(fields["rtf"], float) and fields["rtf"] > 0
    assert gate_is_free(rig, server)


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
    wait_for_events(rig, 1)

    # The same failure seen on the wire, as curl would: headers, some body, then EOF with
    # no terminating zero-length chunk.
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
        sock.sendall(REQUEST)
        raw = read_until_closed(sock)
    assert raw.startswith(b"HTTP/1.1 200 ")
    assert b"transfer-encoding: chunked" in raw.lower()
    assert not raw.endswith(TERMINATOR)
    wait_for_events(rig, 2)

    # One outcome event per request, and nothing else: no second `request.failed` from the
    # app's catch-all handler for an error the response already reported.
    assert [name for name, _ in rig.events.calls] == ["speech.failed", "speech.failed"]
    for _, fields in rig.events.calls:
        assert fields["level"] == "error"
        assert fields["reason"] == "generation_error"
        assert "RuntimeError" in str(fields["error"])
    assert all(generation.ended_on.startswith("breeze-gpu") for generation in rig.generations)
    assert gate_is_free(rig, server)


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
    assert [name for name, _ in rig.events.calls] == ["request.failed"]  # no response built
    assert gate_is_free(rig, server)
    assert rig.runtime.closed == 1


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
    wait_for_events(rig)

    assert rig.events.calls == [
        (
            "speech.aborted",
            {
                "level": "info",
                "request_id": "req-test",
                "audio_seconds": 2 * CODEC_SAMPLES_PER_FRAME / 24000,
                "reason": "client_disconnect",
            },
        )
    ]
    [generation] = rig.generations
    assert generation.yielded == 3  # the two sent, plus the one in flight at the disconnect
    assert generation.ended_on is not None and generation.ended_on.startswith("breeze-gpu")
    assert rig.runtime.closed == 1
    assert gate_is_free(rig, server)


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

        wait_for_events(rig)

        [(name, fields)] = rig.events.calls  # and no `request.failed` on top
        assert name == "speech.aborted"
        assert fields["reason"] == "send_timeout"
        assert gate_is_free(rig, server)
        [generation] = rig.generations
        assert generation.yielded < rig.runtime.chunks  # stopped early, not run to the end
        assert generation.ended_on is not None and generation.ended_on.startswith("breeze-gpu")
        assert rig.runtime.closed == 1


# --- Direct ASGI calls, for orderings a real client can't produce on demand --------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Direct:
    """A primed generation holding the gate, as the route leaves it just before responding."""

    runtime: FakeRuntime
    events: RecordingEvents
    gate: GpuGate
    gpu: GpuThread
    generation: TrackedGeneration
    session: GpuSession[Any]
    first_chunk: bytes

    @classmethod
    async def open(cls, runtime: FakeRuntime) -> Direct:
        gate = GpuGate()
        gpu = GpuThread("cpu", lambda _device: None)
        lease = gate.try_acquire()
        assert lease is not None
        generation = TrackedGeneration(runtime.iter_audio_chunks({}))
        session = GpuSession(lease, gpu, generation.run())
        first = await session.step()
        assert first is not DONE
        return cls(runtime, RecordingEvents(), gate, gpu, generation, session, pcm16(first.audio))

    def response(self, **kwargs: Any) -> SpeechResponse:
        options: dict[str, Any] = {
            "body": rest_of_audio(self.session),
            "clock": time.monotonic,
            "started_at": time.monotonic(),
        }
        options.update(kwargs)
        return SpeechResponse(
            first_chunk=self.first_chunk,
            session=self.session,
            events=self.events,
            request_id="req-test",
            sample_rate=self.runtime.sample_rate,
            **options,
        )

    def gate_is_free(self) -> bool:
        lease = self.gate.try_acquire()
        if lease is None:
            return False
        lease.release()
        return True

    async def shut_down(self) -> None:
        assert await asyncio.to_thread(self.gpu.shutdown, 5)


SCOPE = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}


async def never_disconnects() -> dict[str, Any]:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def test_gate_released_if_body_never_starts() -> None:
    """Cleanup in the body generator's `finally` would never run here: the request is
    cancelled while the headers are still being sent, before the body is first iterated."""
    body_started = False

    async def scenario() -> Direct:
        direct = await Direct.open(FakeRuntime(chunks=4))

        async def body() -> AsyncGenerator[bytes, None]:
            nonlocal body_started
            body_started = True
            async for chunk in rest_of_audio(direct.session):
                yield chunk

        headers_sending = asyncio.Event()

        async def send(message: dict[str, Any]) -> None:
            headers_sending.set()
            await asyncio.Event().wait()  # a peer that never accepts the headers

        call = asyncio.create_task(direct.response(body=body())(SCOPE, never_disconnects, send))
        await asyncio.wait_for(headers_sending.wait(), timeout=5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, timeout=5)

        # The cancelled call waited for the close: the gate is free right away.
        assert direct.gate_is_free()
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    assert not body_started
    assert direct.generation.ended_on is not None
    assert direct.generation.ended_on.startswith("breeze-gpu")
    assert direct.runtime.closed == 1
    assert [(name, fields["reason"]) for name, fields in direct.events.calls] == [
        ("speech.aborted", "cancelled")
    ]


def test_a_response_that_is_never_called_still_releases_the_gate() -> None:
    async def scenario() -> Direct:
        direct = await Direct.open(FakeRuntime(chunks=4))
        response = direct.response()
        del response
        gc.collect()

        await eventually(lambda: bool(direct.events.calls))
        assert direct.gate_is_free()
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    assert direct.runtime.closed == 1
    assert direct.generation.ended_on is not None
    assert direct.generation.ended_on.startswith("breeze-gpu")
    assert [(name, fields["reason"]) for name, fields in direct.events.calls] == [
        ("speech.aborted", "not_sent")
    ]


def test_a_trickling_reader_is_aborted_as_too_slow() -> None:
    # Each send "takes" 1 s of the injected clock for 0.08 s of audio: 0.08x real time.
    clock = FakeClock()

    async def send(message: dict[str, Any]) -> None:
        clock.now += 1.0

    async def scenario() -> Direct:
        direct = await Direct.open(FakeRuntime(chunks=20))
        response = direct.response(clock=clock, started_at=0.0, min_rate_grace=2.0)
        with pytest.raises(StreamAborted) as aborted:
            await response(SCOPE, never_disconnects, send)
        assert isinstance(aborted.value.__cause__, SendTimeout)
        assert direct.gate_is_free()
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    # Budget = 2 + audio / 0.5 - blocked. Headers: blocked 1. Chunk 1: budget 1, blocked 2.
    # Chunk 2: budget 2.16 - 2 = 0.16, blocked 3. Chunk 3: budget 2.32 - 3 < 0, aborted.
    [(name, fields)] = direct.events.calls
    assert name == "speech.aborted"
    assert fields["reason"] == "too_slow"
    assert fields["audio_seconds"] == 2 * CODEC_SAMPLES_PER_FRAME / 24000
    assert direct.generation.yielded < direct.runtime.chunks
    assert direct.runtime.closed == 1


def test_a_reader_above_the_minimum_rate_completes() -> None:
    # 0.05 s of the injected clock per 0.08 s of audio: 1.6x real time.
    clock = FakeClock()

    async def send(message: dict[str, Any]) -> None:
        clock.now += 0.05

    async def scenario() -> Direct:
        direct = await Direct.open(FakeRuntime(chunks=20))
        response = direct.response(clock=clock, started_at=0.0, min_rate_grace=0.1)
        await response(SCOPE, never_disconnects, send)
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    [(name, fields)] = direct.events.calls
    assert name == "speech.completed"
    assert fields["audio_seconds"] == 20 * CODEC_SAMPLES_PER_FRAME / 24000


def test_a_slow_generator_with_a_fast_reader_completes() -> None:
    """Generation at 0.016x real time (5 s per 0.08 s chunk) is the server's slowness, not
    the client's: only time blocked in send() counts, so this never becomes `too_slow`."""
    clock = FakeClock()

    async def send(message: dict[str, Any]) -> None:
        clock.now += 0.001

    async def scenario() -> Direct:
        direct = await Direct.open(FakeRuntime(chunks=20))

        async def slow_body() -> AsyncGenerator[bytes, None]:
            async for chunk in rest_of_audio(direct.session):
                clock.now += 5.0
                yield chunk

        response = direct.response(
            body=slow_body(), clock=clock, started_at=0.0, min_rate_grace=0.1
        )
        await response(SCOPE, never_disconnects, send)
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    [(name, fields)] = direct.events.calls
    assert name == "speech.completed"
    assert fields["audio_seconds"] == 20 * CODEC_SAMPLES_PER_FRAME / 24000


def test_a_send_blocked_past_the_minimum_rate_is_aborted_as_too_slow() -> None:
    """The rate also bounds a send in progress, well before the (here 30 s) send timeout."""

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            await asyncio.Event().wait()  # the first chunk never drains

    async def scenario() -> tuple[Direct, float]:
        direct = await Direct.open(FakeRuntime(chunks=4))
        response = direct.response(min_rate_grace=0.2)
        started = time.monotonic()
        with pytest.raises(StreamAborted):
            await asyncio.wait_for(response(SCOPE, never_disconnects, send), timeout=10)
        elapsed = time.monotonic() - started
        await direct.shut_down()
        return direct, elapsed

    direct, elapsed = asyncio.run(scenario())

    assert elapsed < 5
    assert [(name, fields["reason"]) for name, fields in direct.events.calls] == [
        ("speech.aborted", "too_slow")
    ]


def test_a_disconnect_seen_before_the_stream_finishes_is_not_a_completion() -> None:
    """uvicorn drops sends after a disconnect without an error. Here the last chunk and the
    terminator go out after the listener has seen the disconnect: Starlette can't cancel the
    stream in time (anyio skips a task whose awaited future is already done), so only the
    flag keeps this from being reported as completed, with bytes the client never got."""
    client_gone = False

    async def scenario() -> Direct:
        nonlocal client_gone
        direct = await Direct.open(FakeRuntime(chunks=1))
        body_waiting = asyncio.Event()
        last_chunk: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def body() -> AsyncGenerator[bytes, None]:
            body_waiting.set()
            await last_chunk
            yield b"\0\0" * CODEC_SAMPLES_PER_FRAME

        async def receive() -> dict[str, Any]:
            nonlocal client_gone
            await body_waiting.wait()
            # The body's wait ends in the same step as the disconnect is reported.
            last_chunk.set_result(None)
            client_gone = True
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if client_gone:
                return  # uvicorn: `if self.disconnected: return`

        await direct.response(body=body())(SCOPE, receive, send)
        await direct.shut_down()
        return direct

    direct = asyncio.run(scenario())

    [(name, fields)] = direct.events.calls
    assert name == "speech.aborted"
    assert fields["reason"] == "client_disconnect"
    assert fields["audio_seconds"] == CODEC_SAMPLES_PER_FRAME / 24000  # the primed chunk only
