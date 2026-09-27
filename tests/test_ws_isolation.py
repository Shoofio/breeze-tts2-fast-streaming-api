"""SC-006: a WebSocket client that stops reading doesn't hold up anyone else (T074).

The whole server as `api.serve()` runs it: uvicorn and the WebSocket server on one event loop,
sharing one `GpuGate` and `GpuThread`, with `FakeRuntime` at the model edge and a real SIGTERM to
stop it.

Interface pinned here for T078 (written before it exists): `api.serve()` gains a keyword
`ws_sockets`, the pre-bound WebSocket listening sockets. It builds one
`ws_server.ConnectionRegistry()`, runs one `ws_server.serve(settings, components, sock,
registry)` per socket on uvicorn's loop, and the one signal handler stops both servers
(`registry.shutdown()` for the WebSocket side). `main()` binds the sockets (every address
`settings.host` resolves to, like the HTTP sockets), so a test passes its own:

    await api.serve(components, app, [http_sock], load, outcome, ws_sockets=[ws_sock])
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from breeze_infer import api, ws_server
from breeze_infer.api import Components, ServeOutcome, bind_http_sockets, create_app
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread, report_close_failed
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from tests.fakes import FakeTokenizer, open_no_voices
from tests.ws_helpers import (
    BIG_PIECE_CHUNKS,
    BIG_START,
    SMALL_SNDBUF,
    TCP_ESTABLISHED,
    TINY_RCVBUF,
    PacedRuntime,
    RawWs,
    big_runtime,
    tcp_state,
    wait_until,
)

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it
SPEECH_PATH = "/v1/audio/speech"
# Client A's piece takes at least this long on the GPU thread (150 chunks of 10 ms each).
CHUNK_DELAY = 0.01
PIECE_SECONDS = BIG_PIECE_CHUNKS * CHUNK_DELAY
# Shrunk so A's eviction is quick to observe; A is dropped within their sum plus a margin.
SEND_TIMEOUT = 1.0
CLOSE_TIMEOUT = 1.0
DROP_MARGIN = 3.0

pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="stops the server with a real SIGTERM"),
    pytest.mark.skipif(not hasattr(socket, "TCP_INFO"), reason="TCP_INFO is Linux-only"),
]


def _components(sink: io.StringIO) -> Components:
    events = Emitter(sink, time.time)
    return Components(
        settings=settings_from_args([MODEL_DIR]),
        events=events,
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None, lambda error: report_close_failed(events, error)),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_no_voices,
    )


def _loaded(runtime: PacedRuntime) -> LoadedModel:
    return LoadedModel(
        runtime=runtime,
        report={"device": "cpu"},
        cpu_tokenizer=FakeTokenizer(),
        sizing_tokenizer=FakeTokenizer(),
    )


def _events(sink: io.StringIO, name: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    return [record for record in records if record["event"] == name]


def _handshake_status(port: int) -> int:
    client = RawWs(port)
    try:
        return client.handshake().status
    finally:
        client.close()


@dataclass
class Observed:
    stalled_at: float
    status_while_stalled: int
    busy_answers: int
    first_audio_at: float
    dropped_at: float


def _drive(http_port: int, ws_port: int) -> Observed:
    """Runs off the server's loop. Client A starts a big piece and stops reading once it is
    speaking; client B asks for speech over HTTP until it streams (a `409 busy` while A's piece
    holds the GPU is retried, as a client would)."""
    client_a = RawWs(ws_port, rcvbuf=TINY_RCVBUF)
    try:
        client_a.open()
        client_a.send_json(BIG_START)
        client_a.send_json({"type": "end", "text": "Client A reads no further than this."})
        client_a.frames_until_event("speaking")
        stalled_at = time.monotonic()
        # A holds the only WebSocket slot (WS_MAX_CONNECTIONS is 1 here).
        status_while_stalled = _handshake_status(ws_port)

        busy_answers = 0
        url = f"http://127.0.0.1:{http_port}{SPEECH_PATH}"
        form = {"text": "Client B speaks.", "max_new_tokens": "20"}
        with httpx.Client(timeout=10) as http:
            while True:
                with http.stream("POST", url, data=form) as response:
                    if response.status_code == 409:
                        response.read()
                        busy_answers += 1
                    else:
                        assert response.status_code == 200, response.read()
                        next(response.iter_raw())
                        first_audio_at = time.monotonic()
                        break
                if time.monotonic() > stalled_at + PIECE_SECONDS + 10:
                    raise AssertionError(f"client B never streamed ({busy_answers} x 409)")
                time.sleep(0.02)

        # A never reads again: it is dropped, not left holding its slot.
        wait_until(
            lambda: tcp_state(client_a.sock) != TCP_ESTABLISHED,
            timeout=max(0.0, stalled_at + SEND_TIMEOUT + CLOSE_TIMEOUT + DROP_MARGIN - time.monotonic()),
            what="client A is dropped",
        )
        dropped_at = time.monotonic()
        wait_until(
            lambda: _handshake_status(ws_port) == 101, timeout=5, what="client A's slot is freed"
        )
        return Observed(stalled_at, status_while_stalled, busy_answers, first_audio_at, dropped_at)
    finally:
        client_a.close()


async def _wait_until(condition: Any, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def test_sc_006_http_streams_while_a_websocket_client_has_stopped_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-006: while WebSocket client A has stopped reading mid-piece, HTTP client B starts
    streaming within A's in-flight piece plus 1 s. (In C++ the stalled client holds the GPU and
    blocks every other client for as long as it stays stalled, BC-42.) A, which never reads
    again, is dropped with `ws.closed{code: 1008, aborted: true}` within
    `WS_SEND_TIMEOUT_SECONDS` plus `WS_CLOSE_TIMEOUT_SECONDS` (and a margin) of stalling, and
    its slot is freed."""
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 1)
    monkeypatch.setattr(ws_server, "WS_SEND_TIMEOUT_SECONDS", SEND_TIMEOUT)
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", CLOSE_TIMEOUT)
    sink = io.StringIO()
    components = _components(sink)
    runtime = big_runtime(delay=CHUNK_DELAY)
    [http_sock] = bind_http_sockets("127.0.0.1", 0)
    [ws_sock] = bind_http_sockets("127.0.0.1", 0)
    # Accepted sockets inherit it: A's backlog lands in the server's 2 MiB outbox, not in
    # megabytes of autotuned kernel buffer (see test_ws_server.SMALL_SNDBUF).
    ws_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SMALL_SNDBUF)
    http_port = http_sock.getsockname()[1]
    ws_port = ws_sock.getsockname()[1]

    async def scenario() -> tuple[ServeOutcome, Observed]:
        outcome = ServeOutcome()
        serving = asyncio.create_task(
            api.serve(
                components,
                create_app(components),
                [http_sock],
                lambda: _loaded(runtime),
                outcome,
                ws_sockets=[ws_sock],
            )
        )
        try:
            await _wait_until(lambda: components.readiness.runtime is not None or serving.done())
            if serving.done():
                serving.result()  # raises what serve() raised
                raise AssertionError("serve() returned before the model was ready")
            observed = await asyncio.to_thread(_drive, http_port, ws_port)
        finally:
            if not serving.done():
                os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.wait_for(serving, 30)
        return outcome, observed

    try:
        outcome, observed = asyncio.run(scenario())
    finally:
        http_sock.close()
        ws_sock.close()

    assert observed.status_while_stalled == 503
    waited = observed.first_audio_at - observed.stalled_at
    assert waited <= PIECE_SECONDS + 1.0, (waited, observed.busy_answers)
    assert runtime.yielded_per_call[0] < BIG_PIECE_CHUNKS  # A's piece was cut short
    dropped = observed.dropped_at - observed.stalled_at
    assert dropped <= SEND_TIMEOUT + CLOSE_TIMEOUT + DROP_MARGIN, dropped
    [closed] = [e for e in _events(sink, "ws.closed") if e["code"] == 1008]
    assert closed["aborted"] is True
    assert closed["reason"] == "client too slow"
    assert outcome == ServeOutcome(0)
