"""The composition root in `breeze_infer.api` (T022; research R3, R5, R14).

No GPU and no model: the `GpuThread` gets a stub `set_device`, and the model loader is a stub
returning `FakeRuntime`. `serve()` runs a real uvicorn on an ephemeral port.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import socket
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from breeze_infer import __version__, api
from breeze_infer.api import (
    Components,
    bind_http_socket,
    create_app,
    request_exit,
    serve,
)
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import MAX_BODY_BYTES, TCP_USER_TIMEOUT_MS
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.settings import settings_from_args
from tests.fakes import FakeRuntime

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it


def _components(sink: io.StringIO) -> Components:
    return Components(
        settings=settings_from_args([MODEL_DIR]),
        events=Emitter(sink, lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=Readiness(),
        ws_port=lambda: 0,
    )


def _events(sink: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in sink.getvalue().splitlines()]


# --- the socket pre-bind ---------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(socket, "TCP_USER_TIMEOUT"), reason="TCP_USER_TIMEOUT is Linux-only"
)
def test_bound_socket_has_tcp_user_timeout() -> None:
    with bind_http_socket("127.0.0.1", 0) as sock:
        value = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)

    assert value == TCP_USER_TIMEOUT_MS


@pytest.mark.skipif(os.name != "posix", reason="SO_REUSEADDR is set on POSIX only")
def test_bound_socket_has_reuseaddr_and_is_listening() -> None:
    with bind_http_socket("127.0.0.1", 0) as sock:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1
        # Listening already: a client can connect before uvicorn starts accepting.
        with socket.create_connection(sock.getsockname(), timeout=2):
            pass


def test_bind_works_where_the_platform_has_no_tcp_user_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(socket, "TCP_USER_TIMEOUT", raising=False)

    with bind_http_socket("127.0.0.1", 0) as sock:
        assert sock.getsockname()[1] > 0


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


def test_cuda_device_follows_local_rank_then_rank(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.skipif(os.name != "posix", reason="loop.add_signal_handler is POSIX-only")
def test_serve_loads_in_the_background_then_stops_cleanly_on_sigterm() -> None:
    sink = io.StringIO()
    components = _components(sink)
    sock = bind_http_socket("127.0.0.1", 0)
    port = sock.getsockname()[1]

    def load() -> LoadedModel:
        return LoadedModel(runtime=FakeRuntime(), report={"device": "cpu"})

    async def scenario() -> tuple[int, tuple, tuple]:
        serving = asyncio.create_task(serve(components, create_app(components), sock, load))
        await _wait_until(lambda: components.readiness.runtime is not None)
        ready = await _get(port, "/health")
        missing = await _get(port, "/nope")
        # If uvicorn's own handler were still installed, its re-raise after serve() would
        # kill the test process with the default SIGTERM action.
        os.kill(os.getpid(), signal.SIGTERM)
        return await asyncio.wait_for(serving, 10), ready, missing

    exit_code, ready, missing = asyncio.run(scenario())

    assert exit_code == 0
    assert ready == (200, {"status": "ok", "sample_rate": 24000, "ws_port": 0})
    assert missing[0] == 404
    names = [event["event"] for event in _events(sink)]
    assert names == ["server.started", "model.loaded"]
    started, loaded = _events(sink)
    assert started["port"] == port
    assert loaded["sample_rate"] == 24000
    assert loaded["device"] == "cpu"


def test_serve_answers_loading_while_the_model_loads() -> None:
    sink = io.StringIO()
    components = _components(sink)
    sock = bind_http_socket("127.0.0.1", 0)
    port = sock.getsockname()[1]

    async def scenario() -> tuple[int, tuple]:
        unblock = asyncio.get_running_loop().create_future()
        loop = asyncio.get_running_loop()

        def load() -> LoadedModel:
            # Runs on the GPU thread: block until the test has seen the 503.
            asyncio.run_coroutine_threadsafe(asyncio.wait_for(unblock, 10), loop).result()
            raise RuntimeError("stop here")

        serving = asyncio.create_task(serve(components, create_app(components), sock, load))
        loading = await _get(port, "/health")
        unblock.set_result(None)
        return await asyncio.wait_for(serving, 10), loading

    exit_code, loading = asyncio.run(scenario())

    assert loading == (
        503,
        {"status": "loading", "error": "model is loading", "code": "loading"},
    )
    assert exit_code == 1


def test_a_failed_load_is_reported_and_the_server_exits_non_zero() -> None:
    sink = io.StringIO()
    components = _components(sink)
    sock = bind_http_socket("127.0.0.1", 0)

    def load() -> LoadedModel:
        raise FileNotFoundError("no checkpoint here")

    exit_code = asyncio.run(serve(components, create_app(components), sock, load))

    assert exit_code == 1
    failed = _events(sink)[-1]
    assert failed["event"] == "model.load_failed"
    assert failed["level"] == "error"
    assert "no checkpoint here" in failed["error"]
    assert "FileNotFoundError" in failed["traceback"]
    assert components.readiness.runtime is None
