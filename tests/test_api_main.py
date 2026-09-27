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

from breeze_infer import __version__, api, ws_server
from breeze_infer.api import (
    Components,
    ServeOutcome,
    bind_http_sockets,
    create_app,
    request_exit,
    serve,
)
from breeze_infer.events import Emitter
from breeze_infer.gpu import (
    GpuGate,
    GpuSession,
    GpuThread,
    GpuUnavailable,
    report_close_failed,
)
from breeze_infer.limits import MAX_BODY_BYTES, TCP_USER_TIMEOUT_MS
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from tests.fakes import (
    FakeRuntime,
    FakeTokenizer,
    model_with_codec_facts,
    open_no_voices,
)

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it


def _components(sink: io.StringIO) -> Components:
    events = Emitter(sink, lambda: 0.0)
    readiness = Readiness()
    return Components(
        settings=settings_from_args([MODEL_DIR]),
        events=events,
        gate=GpuGate(on_poisoned=lambda: api._gpu_unresponsive(events, readiness)),
        gpu=GpuThread(
            "cpu", lambda _device: None, lambda error: report_close_failed(events, error)
        ),
        readiness=readiness,
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_no_voices,
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


def test_bc_30_ws_sockets_bind_only_the_configured_host() -> None:
    """C++ binds the WebSocket to 0.0.0.0 whenever `--host` isn't an IPv4 literal, so it
    listens on every interface; here it binds the configured host only, as HTTP does."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    settings = settings_from_args([MODEL_DIR, "--host", "127.0.0.1", "--ws-port", str(free)])

    sockets = api.bind_ws_sockets(settings, Emitter(io.StringIO(), lambda: 0.0))
    try:
        assert [sock.getsockname() for sock in sockets] == [("127.0.0.1", free)]
        assert api.bound_port(sockets) == free
    finally:
        for sock in sockets:
            sock.close()


def test_bc_30_ws_sockets_cover_every_address_the_host_resolves_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`localhost` is both ::1 and 127.0.0.1: both are bound, each served by its own
    `ws_server.serve()` (T078), and nothing else."""
    if not _has_ipv6_loopback():
        pytest.skip("no IPv6 loopback here")
    port = _free_port()
    _resolving_to(
        monkeypatch,
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    )
    settings = settings_from_args([MODEL_DIR, "--host", "localhost", "--ws-port", str(port)])

    sockets = api.bind_ws_sockets(settings, Emitter(io.StringIO(), lambda: 0.0))
    try:
        assert [api._address(sock) for sock in sockets] == [
            f"127.0.0.1:{port}",
            f"[::1]:{port}",
        ]
    finally:
        for sock in sockets:
            sock.close()


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
    runtime = FakeRuntime()
    runtime.model = model_with_codec_facts()  # the voice prefix cache is sized from its config
    return LoadedModel(
        runtime=runtime,
        report={"device": "cpu"},
        cpu_tokenizer=FakeTokenizer(),
        sizing_tokenizer=FakeTokenizer(),
    )


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
    with pytest.raises(GpuUnavailable):  # and the CPU tokenizer's executor, which answers 503
        asyncio.run(components.cpu_tokenizer.run(lambda _tokenizer: None))


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
        pytest.param(
            lambda: LoadedModel(
                runtime=object(),
                report={},
                cpu_tokenizer=FakeTokenizer(),
                sizing_tokenizer=FakeTokenizer(),
            ),
            id="bad-result",
        ),
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


def _stopping_events(sink: io.StringIO) -> list[dict[str, Any]]:
    return [event for event in _events(sink) if event["event"] == "server.stopping"]


def _serve_with(
    monkeypatch: pytest.MonkeyPatch,
    components: Components,
    load: Any,
    main_loop_raises: BaseException | None,
    drain_raises: BaseException | None = None,
    wait_until: Any = None,
    cancel_when: Any = None,
) -> tuple[ServeOutcome, BaseException | None, list[asyncio.Task[Any]]]:
    """Run serve() with uvicorn's main loop (once `wait_until()` holds) returning or raising
    `main_loop_raises`, and the GPU drain optionally raising `drain_raises` (the real
    `_stop_gpu` still runs around it). With `cancel_when`, the task awaiting serve() is
    cancelled once `cancel_when()` holds. Returns the outcome, whatever serve() raised, and the
    tasks still pending once it had."""

    async def main_loop(self: Any) -> None:
        if wait_until is not None:
            await _wait_until(wait_until)
        if main_loop_raises is not None:
            raise main_loop_raises

    monkeypatch.setattr(api._Server, "main_loop", main_loop)
    if drain_raises is not None:

        async def broken_drain(*_args: Any) -> bool:
            raise drain_raises

        monkeypatch.setattr(api, "_drain_gpu", broken_drain)
    outcome = ServeOutcome()
    raised: list[BaseException] = []
    pending: list[asyncio.Task[Any]] = []

    async def scenario() -> None:
        app = create_app(components)
        # serve() awaited in this task, not its own: SystemExit out of a task would escape the
        # event loop, so a cancel has to reach this task instead.
        serving = asyncio.current_task()
        assert serving is not None

        async def cancel_once() -> None:
            await _wait_until(cancel_when)
            serving.cancel()

        if cancel_when is not None:
            asyncio.create_task(cancel_once())
        try:
            await serve(components, app, [_bind_one("127.0.0.1", 0)], load, outcome)
        except BaseException as exc:  # noqa: BLE001 - returned for the test to inspect
            raised.append(exc)
        current = asyncio.current_task()
        pending.extend(t for t in asyncio.all_tasks() if t is not current and not t.done())

    asyncio.run(scenario())
    return outcome, (raised[0] if raised else None), pending


def _loaded_in(components: Components) -> Any:
    """For `wait_until`: the load has finished, so `_stop_gpu` goes on to drain the GPU."""
    return lambda: components.readiness.runtime is not None


@pytest.mark.parametrize(
    ("crash", "exit_code"),
    [
        (RuntimeError("boom"), 1),
        (SystemExit(3), 3),
        (SystemExit(None), 0),
        (SystemExit("bye"), 1),  # Python prints a non-int code and exits 1
        (KeyboardInterrupt(), 130),
    ],
    ids=["exception", "system-exit-3", "system-exit-none", "system-exit-str", "sigint"],
)
def test_crash_exit_codes_follow_python(crash: BaseException, exit_code: int) -> None:
    assert api._crash_exit_code(crash) == exit_code


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [
        (257, 1),
        (-1, 255),
        (512, 0),
        (-256, 0),
        (2**31 + 1, 1),  # past a C int: os._exit itself would raise OverflowError
        (2**32, 0),
        (-(2**31) - 1, 255),
    ],
)
def test_on_posix_a_crash_code_is_the_8_bit_status_the_process_exits_with(
    monkeypatch: pytest.MonkeyPatch, code: int, exit_code: int
) -> None:
    monkeypatch.setattr(api, "_WINDOWS", False)

    assert api._crash_exit_code(SystemExit(code)) == exit_code


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [
        (257, 257),  # Windows keeps the whole code
        (-256, -256),
        (2**31 - 1, 2**31 - 1),
        (-(2**31), -(2**31)),
        (2**31, 1),  # past a C int: os._exit would raise OverflowError instead of exiting
        (2**31 + 1, 1),
        (2**32, 1),
        (-(2**31) - 1, 1),
    ],
)
def test_on_windows_a_crash_code_past_a_c_int_exits_1(
    monkeypatch: pytest.MonkeyPatch, code: int, exit_code: int
) -> None:
    monkeypatch.setattr(api, "_WINDOWS", True)

    assert api._crash_exit_code(SystemExit(code)) == exit_code


def test_outcome_equality_includes_gpu_failed() -> None:
    assert ServeOutcome(70, hard_exit=True, gpu_failed=True) != ServeOutcome(70, hard_exit=True)


@pytest.mark.usefixtures("keep_sigint")
@pytest.mark.parametrize(
    ("crash", "exit_code"),
    [
        (RuntimeError("uvicorn broke"), 1),
        (SystemExit(3), 3),  # SystemExit's own int code is kept
        (SystemExit(0), 0),  # a load in progress is not a GPU failure: 0 stands
        (SystemExit(None), 0),
        (SystemExit(70), 70),  # 70 from the crash itself; still not a GPU failure
        (SystemExit(2**31 + 1), 1),  # past a C int: 1 on every platform
        (KeyboardInterrupt(), 130),
    ],
    ids=[
        "exception",
        "system-exit",
        "system-exit-0",
        "system-exit-none",
        "system-exit-70",
        "system-exit-past-c-int",
        "keyboard-interrupt",
    ],
)
def test_a_crash_during_the_load_hard_exits_with_its_code_and_one_error_event(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    crash: BaseException,
    exit_code: int,
) -> None:
    """os._exit would swallow the exception: a crash must not look like a clean stop (0)."""
    sink = io.StringIO()
    components = _components(sink)
    started, release = threading.Event(), threading.Event()

    def load() -> LoadedModel:
        started.set()
        assert release.wait(10)
        return _loaded()

    try:
        outcome, raised, _ = _serve_with(monkeypatch, components, load, crash, wait_until=started.is_set)
    finally:
        release.set()
        components.gpu.shutdown()

    assert raised is crash
    assert outcome == ServeOutcome(exit_code, hard_exit=True)
    assert type(crash).__name__ in capsys.readouterr().err
    [stopping] = _stopping_events(sink)  # one event, not a warning plus an error
    assert (stopping["level"], stopping["reason"]) == ("error", "load in progress")
    assert type(crash).__name__ in stopping["crash"]
    assert "stop_error" not in stopping


@pytest.mark.usefixtures("keep_sigint")
def test_a_cancelled_serve_during_the_load_is_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = io.StringIO()
    components = _components(sink)
    started, release = threading.Event(), threading.Event()

    def load() -> LoadedModel:
        started.set()
        assert release.wait(10)
        return _loaded()

    try:
        outcome, raised, _ = _serve_with(
            monkeypatch, components, load, asyncio.CancelledError(), wait_until=started.is_set
        )
    finally:
        release.set()
        components.gpu.shutdown()

    assert isinstance(raised, asyncio.CancelledError)
    assert outcome == ServeOutcome(0, hard_exit=True)
    assert capsys.readouterr().err == ""
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("warning", "load in progress")
    assert "crash" not in stopping


def test_a_crash_without_a_hard_exit_is_reported_and_propagates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = io.StringIO()
    components = _components(sink)
    crash = RuntimeError("uvicorn broke")

    def loaded() -> bool:  # the load is over, so the GPU drains normally
        return components.readiness.runtime is not None

    outcome, raised, _ = _serve_with(monkeypatch, components, _loaded, crash, wait_until=loaded)

    assert raised is crash
    # No hard exit: the exception reaches main() and Python prints it and exits non-zero.
    assert outcome.hard_exit is False
    assert capsys.readouterr().err == ""
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "serve raised")
    assert "uvicorn broke" in stopping["crash"]


@pytest.mark.usefixtures("keep_sigint")
def test_a_failing_gpu_stop_hard_exits_with_ex_software(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It may have failed with the GPU thread still busy: exiting normally could hang."""
    sink = io.StringIO()
    components = _components(sink)
    stop_error = OSError("stop broke")

    try:
        outcome, raised, pending = _serve_with(
            monkeypatch,
            components,
            _loaded,
            None,
            drain_raises=stop_error,
            wait_until=_loaded_in(components),
        )
    finally:
        components.gpu.shutdown()

    assert raised is stop_error
    assert pending == []
    assert outcome == ServeOutcome(70, hard_exit=True, gpu_failed=True, drain_error=stop_error)
    assert "OSError: stop broke" in capsys.readouterr().err
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "gpu stop failed")
    assert "stop broke" in stopping["stop_error"]
    assert "crash" not in stopping


@pytest.mark.usefixtures("keep_sigint")
@pytest.mark.parametrize(
    ("crash", "exit_code"),
    [
        (RuntimeError("uvicorn broke"), 1),
        (KeyboardInterrupt(), 130),
        (SystemExit(4), 4),
        (SystemExit(0), 70),  # a GPU failure never exits 0
        (SystemExit(None), 70),
        (SystemExit(2**31 + 1), 1),  # past a C int: 1 on every platform, not 0
    ],
    ids=[
        "exception",
        "sigint",
        "system-exit",
        "system-exit-0",
        "system-exit-none",
        "system-exit-past-c-int",
    ],
)
def test_a_crash_and_a_failing_gpu_stop_keep_the_crash_code_and_report_both(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    crash: BaseException,
    exit_code: int,
) -> None:
    sink = io.StringIO()
    components = _components(sink)
    stop_error = OSError("stop broke")

    try:
        outcome, raised, pending = _serve_with(
            monkeypatch,
            components,
            _loaded,
            crash,
            drain_raises=stop_error,
            wait_until=_loaded_in(components),
        )
    finally:
        components.gpu.shutdown()

    assert raised is crash
    assert pending == []
    assert outcome == ServeOutcome(
        exit_code, hard_exit=True, gpu_failed=True, drain_error=stop_error
    )
    stderr = capsys.readouterr().err
    assert type(crash).__name__ in stderr
    assert "OSError: stop broke" in stderr
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "gpu stop failed")
    assert type(crash).__name__ in stopping["crash"]
    assert "stop broke" in stopping["stop_error"]


@pytest.mark.usefixtures("keep_sigint")
def test_a_cancelled_serve_with_a_failing_gpu_stop_re_raises_the_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation always propagates; the stop failure is still reported and exits 70."""
    sink = io.StringIO()
    components = _components(sink)
    stop_error = OSError("stop broke")

    try:
        outcome, raised, pending = _serve_with(
            monkeypatch,
            components,
            _loaded,
            asyncio.CancelledError(),
            drain_raises=stop_error,
            wait_until=_loaded_in(components),
        )
    finally:
        components.gpu.shutdown()

    assert isinstance(raised, asyncio.CancelledError)
    assert pending == []
    assert outcome == ServeOutcome(70, hard_exit=True, gpu_failed=True, drain_error=stop_error)
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "gpu stop failed")
    assert "crash" not in stopping


@pytest.mark.usefixtures("keep_sigint")
def test_cancelling_serve_while_the_gpu_drains_is_a_cancel_not_a_stop_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = io.StringIO()
    components = _components(sink)
    draining = threading.Event()

    async def stuck_drain(*_args: Any) -> bool:
        draining.set()
        await asyncio.Event().wait()  # never finishes on its own
        return True

    monkeypatch.setattr(api._Server, "main_loop", _returns_once(_loaded_in(components)))
    monkeypatch.setattr(api, "_drain_gpu", stuck_drain)
    outcome = ServeOutcome()

    async def scenario() -> list[asyncio.Task[Any]]:
        app = create_app(components)
        serving = asyncio.create_task(
            serve(components, app, [_bind_one("127.0.0.1", 0)], _loaded, outcome)
        )
        await _wait_until(draining.is_set)
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serving
        current = asyncio.current_task()
        return [t for t in asyncio.all_tasks() if t is not current and not t.done()]

    try:
        pending = asyncio.run(scenario())
    finally:
        components.gpu.shutdown()

    assert pending == []
    # The GPU may still be busy, so still a hard exit, but with the normal code.
    assert outcome == ServeOutcome(0, hard_exit=True)
    assert capsys.readouterr().err == ""
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("warning", "gpu stop cancelled")
    assert "stop_error" not in stopping


@pytest.mark.usefixtures("keep_sigint")
def test_a_crash_exiting_0_and_a_cancelled_gpu_stop_exit_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cancel is someone stopping us, not a GPU failure: the crash's 0 stands."""
    sink = io.StringIO()
    components = _components(sink)
    draining = threading.Event()
    crash = SystemExit(0)

    async def stuck_drain(*_args: Any) -> bool:
        draining.set()
        await asyncio.Event().wait()  # never finishes on its own
        return True

    monkeypatch.setattr(api, "_drain_gpu", stuck_drain)
    try:
        outcome, raised, pending = _serve_with(
            monkeypatch,
            components,
            _loaded,
            crash,
            wait_until=_loaded_in(components),
            cancel_when=draining.is_set,
        )
    finally:
        components.gpu.shutdown()

    assert raised is crash
    assert pending == []
    assert outcome == ServeOutcome(0, hard_exit=True, gpu_failed=False)
    assert "SystemExit" in capsys.readouterr().err
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "gpu stop cancelled")
    assert "SystemExit" in stopping["crash"]
    assert "stop_error" not in stopping


def _serve_with_a_cancel_after_the_drain(
    monkeypatch: pytest.MonkeyPatch, components: Components, drain_ends: bool | BaseException
) -> tuple[ServeOutcome, BaseException | None, list[asyncio.Task[Any]]]:
    """`_serve_with`, but the drain ends at once (returning `drain_ends`, or raising it), and
    then a cancel lands on `_stop_gpu`'s cleanup, while it waits for its tasks."""
    real_gather = asyncio.gather
    drain_ended: list[bool] = []

    async def drain(*_args: Any) -> bool:
        drain_ended.append(True)
        if isinstance(drain_ends, BaseException):
            raise drain_ends
        return drain_ends

    def gather_cancelled_once_the_drain_ended(*awaitables: Any, **kwargs: Any) -> Any:
        if drain_ended:  # only `_stop_gpu`'s wait for its tasks, after the drain ended
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
        return real_gather(*awaitables, **kwargs)

    monkeypatch.setattr(api, "_drain_gpu", drain)
    monkeypatch.setattr(api.asyncio, "gather", gather_cancelled_once_the_drain_ended)
    try:
        result = _serve_with(
            monkeypatch, components, _loaded, None, wait_until=_loaded_in(components)
        )
    finally:
        components.gpu.shutdown()
    assert drain_ended == [True]
    return result


@pytest.mark.usefixtures("keep_sigint")
def test_a_cancel_after_the_drain_timed_out_still_exits_ex_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drain has already timed out when a cancel lands on `_stop_gpu`'s cleanup: the GPU
    is stuck all the same, so the process must not exit 0."""
    sink = io.StringIO()
    components = _components(sink)

    outcome, raised, pending = _serve_with_a_cancel_after_the_drain(
        monkeypatch, components, False
    )

    assert isinstance(raised, asyncio.CancelledError)
    assert pending == []
    assert outcome == ServeOutcome(70, hard_exit=True, gpu_failed=True)
    [stopping] = _stopping_events(sink)
    # The timeout is what happened to the GPU, so an alert on it still fires.
    assert (stopping["level"], stopping["reason"]) == ("warning", "gpu drain timed out")


@pytest.mark.usefixtures("keep_sigint")
def test_a_cancel_after_the_drain_failed_is_still_a_gpu_stop_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The drain has already raised when a cancel lands on `_stop_gpu`'s cleanup: the failure,
    not the cancel, is what happened to the GPU. The cancel still propagates."""
    sink = io.StringIO()
    components = _components(sink)
    drain_error = RuntimeError("drain broke")

    outcome, raised, pending = _serve_with_a_cancel_after_the_drain(
        monkeypatch, components, drain_error
    )

    assert isinstance(raised, asyncio.CancelledError)
    assert pending == []
    assert outcome == ServeOutcome(
        70, hard_exit=True, gpu_failed=True, drain_error=drain_error
    )
    assert "RuntimeError: drain broke" in capsys.readouterr().err
    [stopping] = _stopping_events(sink)
    assert (stopping["level"], stopping["reason"]) == ("error", "gpu stop failed")
    assert "RuntimeError: drain broke" in stopping["stop_error"]
    assert "crash" not in stopping


def test_stopping_the_gpu_clears_a_reused_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    async def drained(*_args: Any) -> bool:
        return True

    monkeypatch.setattr(api, "_drain_gpu", drained)
    # Left from an earlier run.
    outcome = ServeOutcome(70, hard_exit=True, gpu_failed=True, drain_error=OSError("old"))

    async def scenario() -> str | None:
        async def loaded() -> bool:
            return True

        loading = asyncio.create_task(loaded())
        await loading
        # No components or server: the drain is replaced, and nothing else uses them.
        no_components: Any = None
        return await api._stop_gpu(no_components, None, loading, asyncio.Event(), outcome)

    assert asyncio.run(scenario()) is None
    assert outcome == ServeOutcome(0)


def _returns_once(condition: Any) -> Any:
    async def main_loop(self: Any) -> None:
        await _wait_until(condition)

    return main_loop


@posix_only
@pytest.mark.usefixtures("keep_sigint")
@pytest.mark.parametrize(
    ("crash", "signal_while_draining", "exit_code", "reason"),
    [
        (RuntimeError("uvicorn broke"), False, 1, "gpu drain timed out"),
        (KeyboardInterrupt(), False, 130, "gpu drain timed out"),
        (SystemExit(0), False, 70, "gpu drain timed out"),  # a GPU failure never exits 0
        (SystemExit(None), False, 70, "gpu drain timed out"),
        # This test runs on POSIX only, where the process exits with the code's low 8 bits:
        # these would all exit 0, so they count as 0 too.
        (SystemExit(256), False, 70, "gpu drain timed out"),
        (SystemExit(512), False, 70, "gpu drain timed out"),
        (SystemExit(-256), False, 70, "gpu drain timed out"),
        (SystemExit(2**32), False, 70, "gpu drain timed out"),
        # Past a C int, with low 8 bits 1: exits 1, not 70 and not an OverflowError.
        (SystemExit(2**31 + 1), False, 1, "gpu drain timed out"),
        # An operator stopped the drain: the GPU may be busy, but it didn't fail. 0 stands.
        (SystemExit(0), True, 0, "signal during drain"),
    ],
    ids=[
        "exception",
        "sigint",
        "system-exit-0",
        "system-exit-none",
        "system-exit-256",
        "system-exit-512",
        "system-exit-minus-256",
        "system-exit-2-to-the-32",
        "system-exit-past-c-int",
        "system-exit-0-signal-during-drain",
    ],
)
def test_a_crash_with_a_stuck_drain_exits_with_the_crash_code_but_never_0(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    crash: BaseException,
    signal_while_draining: bool,
    exit_code: int,
    reason: str,
) -> None:
    # With a signal, the drain must still be running when it arrives.
    monkeypatch.setattr(api, "GPU_DRAIN_SECONDS", 8.0 if signal_while_draining else 0.5)
    sink = io.StringIO()
    components = _components(sink)
    proceed, cancelled, streaming = (threading.Event() for _ in range(3))
    outcome = ServeOutcome()
    app = _stuck_close_app(components, proceed, cancelled, streaming)
    raised: list[BaseException] = []

    async def broken_main_loop(self: Any) -> None:
        await asyncio.to_thread(streaming.wait, 10)  # a request holds the GPU
        raise crash

    monkeypatch.setattr(api._Server, "main_loop", broken_main_loop)

    async def scenario() -> None:
        sock = _bind_one("127.0.0.1", 0)
        port = sock.getsockname()[1]

        async def client() -> asyncio.StreamWriter:
            await _wait_until(lambda: components.readiness.runtime is not None)
            return await _start_request(port)

        async def signal_once_draining() -> None:
            # Requests are cancelled only once uvicorn has returned and the drain began. Never
            # signal otherwise: with no handler left, SIGTERM would kill the whole test run.
            assert await asyncio.to_thread(cancelled.wait, 10)
            os.kill(os.getpid(), signal.SIGTERM)

        requesting = asyncio.create_task(client())
        if signal_while_draining:
            signalling = asyncio.create_task(signal_once_draining())
        # serve() awaited in this task, not its own: SystemExit and KeyboardInterrupt out of
        # a task escape the event loop instead of reaching the `except` here.
        try:
            await serve(components, app, [sock], _loaded, outcome)
        except BaseException as exc:  # noqa: BLE001 - inspected below
            raised.append(exc)
        finally:
            proceed.set()  # unstick the GPU before asyncio.run's cleanup
            (await requesting).close()
            if signal_while_draining:
                await signalling

    try:
        asyncio.run(scenario())
    finally:
        proceed.set()
        components.gpu.shutdown()
    assert raised == [crash]
    assert outcome == ServeOutcome(
        exit_code, hard_exit=True, gpu_failed=not signal_while_draining
    )
    assert type(crash).__name__ in capsys.readouterr().err
    [stopping] = _stopping_events(sink)
    # `reason` stays the hard-exit reason, so an alert on "gpu drain timed out" still fires.
    assert (stopping["level"], stopping["reason"]) == ("error", reason)
    assert type(crash).__name__ in stopping["crash"]


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

    assert outcome == ServeOutcome(70, hard_exit=True, gpu_failed=True)
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


# --- the WebSocket stop in serve() (review 48 #1, #2) -------------------------------------


async def _stuck_websocket_shutdown(self: ws_server.ConnectionRegistry) -> None:
    """A WebSocket shutdown that never finishes, as one held by a hung `gen.close()` would."""
    self.shutting_down = True
    await asyncio.Event().wait()


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_second_signal_cuts_a_stuck_websocket_stop_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_server.ConnectionRegistry, "shutdown", _stuck_websocket_shutdown)
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> tuple[ServeOutcome, float]:
        serving, _port = _serving(components, _loaded)
        await _wait_until(lambda: components.readiness.runtime is not None)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(1.0)  # uvicorn has returned; serve() waits for the WebSocket stop
        signalled_at = time.monotonic()
        os.kill(os.getpid(), signal.SIGTERM)
        outcome = await asyncio.wait_for(serving, 10)
        return outcome, time.monotonic() - signalled_at

    try:
        outcome, after_signal = asyncio.run(scenario())
    finally:
        components.gpu.shutdown()

    assert outcome.hard_exit
    assert after_signal < 1.0
    stopping = _events(sink)[-1]
    assert (stopping["event"], stopping["reason"]) == ("server.stopping", "signal during drain")


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_cancel_during_the_websocket_stop_still_concludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CancelledError is not an Exception: it must not skip deciding how serve() ends."""
    monkeypatch.setattr(ws_server.ConnectionRegistry, "shutdown", _stuck_websocket_shutdown)
    sink = io.StringIO()
    components = _components(sink)
    outcome = ServeOutcome()

    async def scenario() -> None:
        sock = _bind_one("127.0.0.1", 0)
        serving = asyncio.create_task(
            _serve(components, create_app(components), [sock], _loaded, outcome)
        )
        await _wait_until(lambda: components.readiness.runtime is not None)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(1.0)  # serve() now waits for the WebSocket stop
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(serving, 10)

    try:
        asyncio.run(scenario())
    finally:
        components.gpu.shutdown()

    assert outcome.hard_exit
    stopping = _events(sink)[-1]
    assert (stopping["event"], stopping["reason"]) == ("server.stopping", "gpu stop cancelled")


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_a_signal_while_the_websocket_servers_start_closes_them_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One `ws_server.serve()` per address: a signal that lands between two of them must not
    leave the later ones listening."""
    real_serve = ws_server.serve
    calls = 0

    async def signal_during_the_first(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.1)  # the handler runs: the shutdown starts
        return await real_serve(*args)

    monkeypatch.setattr(ws_server, "serve", signal_during_the_first)
    sink = io.StringIO()
    components = _components(sink)
    ws_sockets = [_bind_one("127.0.0.1", 0), _bind_one("127.0.0.1", 0)]

    async def scenario() -> list[bool]:
        http_sock = _bind_one("127.0.0.1", 0)
        # The load may still be running when the signal lands (a hard exit): only the sockets
        # matter here.
        await asyncio.wait_for(
            serve(
                components,
                create_app(components),
                [http_sock],
                _loaded,
                ServeOutcome(),
                ws_sockets=ws_sockets,
            ),
            15,
        )
        return [sock.fileno() == -1 for sock in ws_sockets]

    try:
        closed = asyncio.run(scenario())
    finally:
        for sock in ws_sockets:
            sock.close()
        components.gpu.shutdown()

    assert closed == [True, True]


@posix_only
@pytest.mark.usefixtures("keep_sigint")
def test_two_quick_signals_cut_a_stuck_websocket_stop_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two Ctrl+C always end it: a second signal that lands during uvicorn's graceful
    shutdown, before serve() is waiting on the WebSocket stop, still cuts that wait short."""
    monkeypatch.setattr(ws_server.ConnectionRegistry, "shutdown", _stuck_websocket_shutdown)
    sink = io.StringIO()
    components = _components(sink)

    async def scenario() -> ServeOutcome:
        serving, _port = _serving(components, _loaded)
        await _wait_until(lambda: components.readiness.runtime is not None)
        # Both handlers run in the same loop step, while uvicorn is still serving.
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
        return await asyncio.wait_for(serving, 10)

    try:
        outcome = asyncio.run(scenario())
    finally:
        components.gpu.shutdown()

    assert outcome.exit_code == 0
