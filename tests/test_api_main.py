"""The composition root in `breeze_infer.api` (T022; research R3, R5, R14).

No GPU and no model: the `GpuThread` gets a stub `set_device`, and the model loader is a stub
returning `FakeRuntime`. `serve()` runs a real uvicorn on an ephemeral port, and the shutdown
tests send the process real signals.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import socket
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from breeze_infer import __version__, api
from breeze_infer.api import (
    Components,
    ServeOutcome,
    bind_http_sockets,
    create_app,
    request_exit,
    serve,
)
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuSession, GpuThread
from breeze_infer.limits import MAX_BODY_BYTES, TCP_USER_TIMEOUT_MS
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.settings import settings_from_args
from tests.fakes import FakeRuntime

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it


def _components(sink: io.StringIO) -> Components:
    events = Emitter(sink, lambda: 0.0)
    readiness = Readiness()
    return Components(
        settings=settings_from_args([MODEL_DIR]),
        events=events,
        gate=GpuGate(on_poisoned=lambda: api._gpu_unresponsive(events, readiness)),
        gpu=GpuThread(
            "cpu", lambda _device: None, lambda error: api._report_close_failed(events, error)
        ),
        readiness=readiness,
        ws_port=lambda: 0,
    )


def _events(sink: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in sink.getvalue().splitlines()]


# --- the socket pre-bind ---------------------------------------------------------------


def _bind_one(host: str, port: int) -> socket.socket:
    [sock] = bind_http_sockets(host, port)
    return sock


@pytest.mark.skipif(
    not hasattr(socket, "TCP_USER_TIMEOUT"), reason="TCP_USER_TIMEOUT is Linux-only"
)
def test_bound_socket_has_tcp_user_timeout() -> None:
    with _bind_one("127.0.0.1", 0) as sock:
        value = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)

    assert value == TCP_USER_TIMEOUT_MS


@pytest.mark.skipif(os.name != "posix", reason="SO_REUSEADDR is set on POSIX only")
def test_bound_socket_has_reuseaddr_and_is_listening() -> None:
    with _bind_one("127.0.0.1", 0) as sock:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1
        # Listening already: a client can connect before uvicorn starts accepting.
        with socket.create_connection(sock.getsockname(), timeout=2):
            pass


def test_bind_works_where_the_platform_has_no_tcp_user_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(socket, "TCP_USER_TIMEOUT", raising=False)

    with _bind_one("127.0.0.1", 0) as sock:
        assert sock.getsockname()[1] > 0


def _has_ipv6_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _resolving_to(
    monkeypatch: pytest.MonkeyPatch, *addresses: tuple[socket.AddressFamily, tuple]
) -> None:
    """Make every getaddrinfo() return `addresses`, as a dual-stack or odd host would."""
    infos = [
        (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)
        for family, address in addresses
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kw: infos)


@pytest.mark.skipif(not _has_ipv6_loopback(), reason="no IPv6 loopback here")
def test_every_resolved_address_is_bound_with_ipv6_kept_off_the_ipv4_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    _resolving_to(
        monkeypatch,
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    )

    sockets = bind_http_sockets("localhost", port)
    try:
        assert [sock.family for sock in sockets] == [socket.AF_INET, socket.AF_INET6]
        assert {sock.getsockname()[1] for sock in sockets} == {port}
        # Without V6ONLY a wildcard IPv6 bind also claims the IPv4 port, and the two clash.
        assert sockets[1].getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
        assert [api._address(sock) for sock in sockets] == [
            f"127.0.0.1:{port}",
            f"[::1]:{port}",
        ]
    finally:
        for sock in sockets:
            sock.close()


def test_an_address_the_host_does_not_have_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 192.0.2.1 is TEST-NET-1: never a local address, so bind() fails with EADDRNOTAVAIL.
    _resolving_to(
        monkeypatch,
        (socket.AF_INET, ("192.0.2.1", 0)),
        (socket.AF_INET, ("127.0.0.1", 0)),
    )

    [sock] = bind_http_sockets("somehost", 0)
    with sock:
        assert sock.getsockname()[0] == "127.0.0.1"


def test_a_taken_port_on_one_address_closes_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[socket.socket] = []
    real_socket = socket.socket

    def recording_socket(*args: Any) -> socket.socket:
        sock = real_socket(*args)
        opened.append(sock)
        return sock

    with real_socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        _resolving_to(
            monkeypatch,
            (socket.AF_INET, ("127.0.0.2", port)),
            (socket.AF_INET, ("127.0.0.1", port)),
        )
        monkeypatch.setattr(socket, "socket", recording_socket)

        with pytest.raises(OSError):
            bind_http_sockets("somehost", port)

    assert len(opened) == 2
    assert all(sock.fileno() == -1 for sock in opened)


def test_main_exits_non_zero_with_a_message_when_the_port_is_taken(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]

        with pytest.raises(SystemExit) as exited:
            api.main([MODEL_DIR, "--port", str(port)])

    assert f"cannot listen on 127.0.0.1:{port}" in str(exited.value.code)
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["event"] == "server.bind_failed"
    assert event["level"] == "error"
    assert event["port"] == port


# --- middleware order --------------------------------------------------------------------


async def _call_asgi(app: Any, method: str, path: str, headers: list) -> list[dict]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)
    return sent


def test_version_header_is_outside_the_body_limit() -> None:
    """A 413 is sent by BodyLimitMiddleware before the app runs; it still gets the header."""
    components = _components(io.StringIO())
    app = create_app(components)
    try:
        sent = asyncio.run(
            _call_asgi(
                app,
                "POST",
                "/health",
                [(b"content-length", str(MAX_BODY_BYTES + 1).encode())],
            )
        )
    finally:
        components.gpu.shutdown()

    start = sent[0]
    assert start["status"] == 413
    assert (b"x-breeze-version", __version__.encode()) in start["headers"]


# --- device and signals -------------------------------------------------------------------


def test_cuda_device_follows_local_rank_then_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)

    assert api._cuda_device({}) == "cuda:0"
    assert api._cuda_device({"RANK": "2"}) == "cuda:2"
    assert api._cuda_device({"RANK": "2", "LOCAL_RANK": "1"}) == "cuda:1"

    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: False)
    assert api._cuda_device({"LOCAL_RANK": "1"}) == "cpu"


def test_first_signal_exits_gracefully_and_second_forces() -> None:
    server = uvicorn.Server(uvicorn.Config(lambda *_: None))

    request_exit(server)
    assert (server.should_exit, server.force_exit) == (True, False)

    request_exit(server)
    assert server.force_exit is True


# --- serve(): background load, readiness, shutdown ---------------------------------------


async def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode())
    raw = await reader.read()
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body)


async def _wait_until(condition: Any, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


posix_only = pytest.mark.skipif(
    os.name != "posix", reason="sends the process POSIX signals with os.kill"
)


def _loaded() -> LoadedModel:
    return LoadedModel(runtime=FakeRuntime(), report={"device": "cpu"})


async def _serve(
    components: Components,
    app: Any,
    sockets: list[socket.socket],
    load: Any,
    outcome: ServeOutcome | None = None,
) -> ServeOutcome:
    outcome = ServeOutcome() if outcome is None else outcome
    await serve(components, app, sockets, load, outcome)
    return outcome


def _serving(
    components: Components, load: Any, app: Any = None
) -> tuple[asyncio.Task[ServeOutcome], int]:
    sock = _bind_one("127.0.0.1", 0)
    app = create_app(components) if app is None else app
    port = sock.getsockname()[1]
    return asyncio.create_task(_serve(components, app, [sock], load)), port


@pytest.fixture()
def keep_sigint() -> Any:
    """A hard-exit outcome leaves SIGINT ignored (main() is about to os._exit); undo it."""
    before = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, before)


@posix_only
def test_serve_loads_in_the_background_then_stops_cleanly_on_sigterm() -> None:
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> tuple[ServeOutcome, int, tuple, tuple]:
        serving, port = _serving(components, _loaded)
        await _wait_until(lambda: components.readiness.runtime is not None)
        ready = await _get(port, "/health")
        missing = await _get(port, "/nope")
        # If uvicorn's own handler were still installed, its re-raise after serve() would
        # kill the test process with the default SIGTERM action.
        os.kill(os.getpid(), signal.SIGTERM)
        return await asyncio.wait_for(serving, 10), port, ready, missing

    outcome, port, ready, missing = asyncio.run(scenario())

    assert outcome == ServeOutcome(0)
    assert ready == (200, {"status": "ok", "sample_rate": 24000, "ws_port": 0})
    assert missing[0] == 404
    names = [event["event"] for event in _events(sink)]
    assert names == ["server.started", "model.loaded"]
    started, loaded = _events(sink)
    assert "port" not in started  # one port can't describe several sockets
    assert started["addresses"] == [f"127.0.0.1:{port}"]
    assert loaded["sample_rate"] == 24000
    assert loaded["device"] == "cpu"
    with pytest.raises(RuntimeError):  # serve() shut the GPU thread down
        asyncio.run(components.gpu.run(lambda: None))


def test_serve_answers_loading_while_the_model_loads() -> None:
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> tuple[ServeOutcome, tuple]:
        unblock = asyncio.get_running_loop().create_future()
        loop = asyncio.get_running_loop()

        def load() -> LoadedModel:
            # Runs on the GPU thread: block until the test has seen the 503.
            asyncio.run_coroutine_threadsafe(asyncio.wait_for(unblock, 10), loop).result()
            raise RuntimeError("stop here")

        serving, port = _serving(components, load)
        loading = await _get(port, "/health")
        unblock.set_result(None)
        return await asyncio.wait_for(serving, 10), loading

    outcome, loading = asyncio.run(scenario())

    assert loading == (
        503,
        {"status": "loading", "error": "model is loading", "code": "loading"},
    )
    assert outcome == ServeOutcome(1)


def test_a_failed_load_is_reported_and_the_server_exits_non_zero() -> None:
    sink = io.StringIO()
    components = _components(sink)

    def load() -> LoadedModel:
        raise FileNotFoundError("no checkpoint here")

    async def scenario() -> ServeOutcome:
        serving, _ = _serving(components, load)
        return await serving

    outcome = asyncio.run(scenario())

    assert outcome == ServeOutcome(1)
    failed = _events(sink)[-1]
    assert failed["event"] == "model.load_failed"
    assert failed["level"] == "error"
    assert "no checkpoint here" in failed["error"]
    assert "FileNotFoundError" in failed["traceback"]
    assert components.readiness.runtime is None


@pytest.mark.parametrize(
    "load",
    [
        # A BaseException from the loader (sys.exit in a library, say).
        pytest.param(lambda: (_ for _ in ()).throw(SystemExit(3)), id="system-exit"),
        # A loader result that breaks marking ready / reporting it.
        pytest.param(lambda: LoadedModel(runtime=object(), report={}), id="bad-result"),
    ],
)
def test_any_failure_in_the_load_task_ends_the_server(load: Any) -> None:
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> ServeOutcome:
        serving, _ = _serving(components, load)
        return await asyncio.wait_for(serving, 10)

    assert asyncio.run(scenario()) == ServeOutcome(1)
    assert _events(sink)[-1]["event"] == "model.load_failed"


# --- shutdown: graceful, forced, during the load -------------------------------------------


async def _slow_stream(scope: dict, receive: Any, send: Any) -> None:
    """A 5 s streaming response, the reviewer's case for the forced exit."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    for _ in range(50):
        await send({"type": "http.response.body", "body": b"x", "more_body": True})
        await asyncio.sleep(0.1)
    await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _start_request(port: int) -> asyncio.StreamWriter:
    """Open a streaming GET and return once the response has started."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET / HTTP/1.1\r\nHost: t\r\n\r\n")
    await reader.readuntil(b"\r\n\r\n")
    return writer


@posix_only
def test_a_second_signal_stops_at_once_despite_an_open_stream() -> None:
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> tuple[ServeOutcome, float]:
        serving, port = _serving(components, _loaded, app=_slow_stream)
        await _wait_until(lambda: components.readiness.runtime is not None)
        writer = await _start_request(port)

        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.3)
        assert not serving.done()  # graceful: the open response is allowed to finish

        os.kill(os.getpid(), signal.SIGTERM)
        forced_at = time.monotonic()
        outcome = await asyncio.wait_for(serving, 10)
        writer.close()
        return outcome, time.monotonic() - forced_at

    outcome, after_second = asyncio.run(scenario())

    assert outcome == ServeOutcome(0)
    assert after_second < 1.0


def _frames(closed: list[str]) -> Generator[bytes, None, None]:
    try:
        while True:
            time.sleep(0.01)  # a "GPU step", on the GPU thread
            yield b"x"
    finally:
        closed.append(threading.current_thread().name)


@posix_only
@pytest.mark.parametrize("signals", [1, 2], ids=["graceful-timeout", "forced"])
def test_each_open_generation_is_closed_on_the_gpu_thread_before_it_stops(
    monkeypatch: pytest.MonkeyPatch, signals: int
) -> None:
    """After the graceful timeout uvicorn cancels the request without waiting for it; on a
    forced exit it doesn't cancel it at all. Either way serve() must see the close through."""
    monkeypatch.setattr(api, "GRACEFUL_SHUTDOWN_SECONDS", 0.3)
    sink = io.StringIO()
    components = _components(sink)
    closed: list[str] = []

    async def generating(scope: dict, receive: Any, send: Any) -> None:
        lease = components.gate.try_acquire()
        assert lease is not None
        session = GpuSession(lease, components.gpu, _frames(closed))
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            while True:
                chunk = await session.step()
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await session.aclose()

    async def scenario() -> ServeOutcome:
        serving, port = _serving(components, _loaded, app=generating)
        await _wait_until(lambda: components.readiness.runtime is not None)
        writer = await _start_request(port)
        for _ in range(signals):  # the stream never ends on its own
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.05)
        outcome = await asyncio.wait_for(serving, 10)
        writer.close()
        return outcome

    outcome = asyncio.run(scenario())

    assert outcome == ServeOutcome(0)
    assert len(closed) == 1 and closed[0].startswith("breeze-gpu")
    assert components.gate.try_acquire() is not None
    assert "gpu.close_failed" not in [event["event"] for event in _events(sink)]


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_signal_during_the_load_asks_main_for_a_hard_exit() -> None:
    sink = io.StringIO()
    components = _components(sink)
    started, release = threading.Event(), threading.Event()

    def load() -> LoadedModel:
        started.set()
        assert release.wait(10)  # a load the signal can't interrupt
        return _loaded()

    async def scenario() -> ServeOutcome:
        serving, _ = _serving(components, load)
        assert await asyncio.to_thread(started.wait, 10)
        os.kill(os.getpid(), signal.SIGTERM)
        return await asyncio.wait_for(serving, 5)

    try:
        outcome = asyncio.run(scenario())
    finally:
        release.set()
        components.gpu.shutdown()

    assert outcome == ServeOutcome(0, hard_exit=True)
    stopping = _events(sink)[-1]
    assert stopping["event"] == "server.stopping"
    assert stopping["level"] == "warning"
    assert stopping["reason"] == "load in progress"
    # From the decision on, a Ctrl+C can't turn the hard exit into a KeyboardInterrupt.
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN


@pytest.mark.usefixtures("keep_sigint")
def test_serve_raising_during_the_load_hard_exits_non_zero_with_the_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """os._exit would swallow the exception: a crash must not look like a clean stop (0)."""
    sink = io.StringIO()
    components = _components(sink)
    started, release = threading.Event(), threading.Event()
    outcome = ServeOutcome()

    def load() -> LoadedModel:
        started.set()
        assert release.wait(10)
        return _loaded()

    async def broken_main_loop(self: Any) -> None:
        await asyncio.to_thread(started.wait, 10)
        raise RuntimeError("uvicorn broke")

    monkeypatch.setattr(api._Server, "main_loop", broken_main_loop)

    async def scenario() -> None:
        app = create_app(components)
        await serve(components, app, [_bind_one("127.0.0.1", 0)], load, outcome)

    try:
        with pytest.raises(RuntimeError, match="uvicorn broke"):
            asyncio.run(asyncio.wait_for(scenario(), 10))
    finally:
        release.set()
        components.gpu.shutdown()
    assert outcome == ServeOutcome(1, hard_exit=True)
    stderr = capsys.readouterr().err
    assert "Traceback" in stderr
    assert "RuntimeError: uvicorn broke" in stderr
    crashed = _events(sink)[-1]
    assert (crashed["event"], crashed["level"]) == ("server.stopping", "error")
    assert crashed["reason"] == "serve raised"
    assert "uvicorn broke" in crashed["error"]
    assert "RuntimeError" in crashed["traceback"]


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_serve_raising_with_a_stuck_drain_keeps_ex_software(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(api, "GPU_DRAIN_SECONDS", 0.5)
    sink = io.StringIO()
    components = _components(sink)
    proceed, cancelled, streaming = (threading.Event() for _ in range(3))
    outcome = ServeOutcome()
    app = _stuck_close_app(components, proceed, cancelled, streaming)

    async def broken_main_loop(self: Any) -> None:
        await asyncio.to_thread(streaming.wait, 10)  # a request holds the GPU
        raise RuntimeError("uvicorn broke")

    monkeypatch.setattr(api._Server, "main_loop", broken_main_loop)

    async def scenario() -> None:
        sock = _bind_one("127.0.0.1", 0)
        port = sock.getsockname()[1]
        serving = asyncio.create_task(serve(components, app, [sock], _loaded, outcome))
        await _wait_until(lambda: components.readiness.runtime is not None)
        writer = await _start_request(port)
        try:
            with pytest.raises(RuntimeError, match="uvicorn broke"):
                await asyncio.wait_for(serving, 10)
        finally:
            proceed.set()  # unstick the GPU before asyncio.run's cleanup
            writer.close()

    try:
        asyncio.run(scenario())
    finally:
        proceed.set()
        components.gpu.shutdown()
    assert outcome == ServeOutcome(70, hard_exit=True)
    assert "RuntimeError: uvicorn broke" in capsys.readouterr().err
    reasons = [event.get("reason") for event in _events(sink)]
    assert reasons[-2:] == ["gpu drain timed out", "serve raised"]


def test_hard_exit_survives_a_broken_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenStdout(io.StringIO):
        def flush(self) -> None:
            raise BrokenPipeError("gone")

    exits: list[int] = []
    monkeypatch.setattr(api.sys, "stdout", BrokenStdout())
    monkeypatch.setattr(api.os, "_exit", exits.append)

    api._hard_exit(70)

    assert exits == [70]


# --- stuck GPU work while draining -------------------------------------------------------


def _stuck_close_app(
    components: Components,
    proceed: threading.Event,
    cancelled: threading.Event,
    streaming: threading.Event | None = None,
) -> Any:
    """A streaming request whose gen.close() is stuck behind other GPU work."""

    def hold() -> None:
        assert proceed.wait(20)

    async def app(scope: dict, receive: Any, send: Any) -> None:
        lease = components.gate.try_acquire()
        assert lease is not None
        session = GpuSession(lease, components.gpu, _frames([]))
        blocker = asyncio.ensure_future(components.gpu.run(hold))
        try:
            await session.step()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if streaming is not None:
                streaming.set()
            await asyncio.sleep(60)
        finally:
            cancelled.set()
            await session.aclose()
            await blocker

    return app


def _run_with_stuck_close(
    components: Components, extra_signal: bool
) -> tuple[ServeOutcome, float]:
    proceed, cancelled = threading.Event(), threading.Event()

    async def scenario() -> tuple[ServeOutcome, float]:
        app = _stuck_close_app(components, proceed, cancelled)
        serving, port = _serving(components, _loaded, app=app)
        await _wait_until(lambda: components.readiness.runtime is not None)
        writer = await _start_request(port)
        os.kill(os.getpid(), signal.SIGTERM)
        assert await asyncio.to_thread(cancelled.wait, 10)  # uvicorn gave up on the request
        await asyncio.sleep(0.2)  # uvicorn has returned; serve() is draining
        signalled_at = time.monotonic()
        if extra_signal:
            os.kill(os.getpid(), signal.SIGTERM)
        outcome = await asyncio.wait_for(serving, 10)
        elapsed = time.monotonic() - signalled_at
        # Unstick the GPU before asyncio.run's cleanup, which waits for the abandoned drain
        # (main() would have hard-exited instead).
        proceed.set()
        writer.close()
        return outcome, elapsed

    try:
        return asyncio.run(scenario())
    finally:
        proceed.set()
        components.gpu.shutdown()


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_signal_while_draining_hard_exits_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "GRACEFUL_SHUTDOWN_SECONDS", 0.3)
    monkeypatch.setattr(api, "GPU_DRAIN_SECONDS", 8.0)
    sink = io.StringIO()
    components = _components(sink)

    outcome, after_signal = _run_with_stuck_close(components, extra_signal=True)

    assert outcome == ServeOutcome(0, hard_exit=True)
    assert after_signal < 1.0
    stopping = _events(sink)[-1]
    assert (stopping["event"], stopping["reason"]) == ("server.stopping", "signal during drain")


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_drain_past_its_bound_hard_exits_with_ex_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "GRACEFUL_SHUTDOWN_SECONDS", 0.3)
    monkeypatch.setattr(api, "GPU_DRAIN_SECONDS", 0.5)
    sink = io.StringIO()
    components = _components(sink)

    outcome, _ = _run_with_stuck_close(components, extra_signal=False)

    assert outcome == ServeOutcome(70, hard_exit=True)
    stopping = _events(sink)[-1]
    assert (stopping["event"], stopping["reason"]) == ("server.stopping", "gpu drain timed out")


# --- a close that never finishes: the gate is poisoned and the server unhealthy ------------


def test_a_poisoned_gate_marks_the_server_unhealthy_and_reports_it() -> None:
    sink = io.StringIO()
    components = _components(sink)
    components.readiness.mark_ready(FakeRuntime())
    try:
        lease = components.gate.try_acquire()
        assert lease is not None
        lease.poison()
    finally:
        components.gpu.shutdown()

    assert components.readiness.runtime is None
    event = _events(sink)[-1]
    assert (event["event"], event["level"]) == ("gpu.close_timeout", "error")


@pytest.mark.skipif(not _has_ipv6_loopback(), reason="no IPv6 loopback here")
def test_server_started_lists_each_sockets_own_port_for_a_dual_stack_port_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolving_to(
        monkeypatch,
        (socket.AF_INET, ("127.0.0.1", 0)),
        (socket.AF_INET6, ("::1", 0, 0, 0)),
    )
    sockets = bind_http_sockets("localhost", 0)
    monkeypatch.undo()
    expected = [f"127.0.0.1:{sockets[0].getsockname()[1]}", f"[::1]:{sockets[1].getsockname()[1]}"]
    sink = io.StringIO()
    components = _components(sink)

    def load() -> LoadedModel:
        raise RuntimeError("no model needed")

    asyncio.run(_serve(components, create_app(components), sockets, load))

    started = _events(sink)[0]
    assert started["event"] == "server.started"
    assert started["addresses"] == expected


# --- signal handler installation -----------------------------------------------------------


def test_uvicorn_never_captures_signals_itself() -> None:
    server = api._Server(uvicorn.Config(lambda *_: None))
    before = signal.getsignal(signal.SIGTERM)

    with server.capture_signals():
        assert signal.getsignal(signal.SIGTERM) is before


@posix_only
@pytest.mark.parametrize("refusal", [NotImplementedError, RuntimeError])
def test_without_loop_signal_handlers_signal_signal_still_stops_the_server(
    refusal: type[Exception],
) -> None:
    """Windows loops raise NotImplementedError; a loop off the main thread, RuntimeError."""
    sink = io.StringIO()
    components = _components(sink)
    before = signal.getsignal(signal.SIGTERM)

    def refuse(*_args: Any) -> None:
        raise refusal("no add_signal_handler here")

    async def scenario() -> ServeOutcome:
        asyncio.get_running_loop().add_signal_handler = refuse  # type: ignore[method-assign]
        serving, _ = _serving(components, _loaded)
        await _wait_until(lambda: components.readiness.runtime is not None)
        assert signal.getsignal(signal.SIGTERM) is not before  # the fallback is in place
        os.kill(os.getpid(), signal.SIGTERM)
        return await asyncio.wait_for(serving, 10)

    assert asyncio.run(scenario()) == ServeOutcome(0)
    assert signal.getsignal(signal.SIGTERM) is before  # and removed again


def test_off_the_main_thread_no_handlers_are_installed() -> None:
    server = uvicorn.Server(uvicorn.Config(lambda *_: None))
    before = signal.getsignal(signal.SIGTERM)
    results: list[BaseException | None] = []

    def in_thread() -> None:
        loop = asyncio.new_event_loop()
        try:
            api._install_signal_handlers(loop, lambda: request_exit(server))()  # and undo
            results.append(None)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            results.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=in_thread)
    thread.start()
    thread.join(5)

    assert results == [None]
    assert signal.getsignal(signal.SIGTERM) is before
