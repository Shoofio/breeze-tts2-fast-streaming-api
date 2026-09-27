"""The WebSocket server on a real socket (T073; contracts/ws-api.md; research.md R4, R15;
research/ws-prototype.md).

A real `websockets` server on 127.0.0.1:<ephemeral>, fed by `FakeRuntime` through a real
`GpuThread` and `GpuGate`. Message-level tests use the `websockets` sync client; protocol and
slow-client probes use raw sockets, because the behaviour under test is what goes over the wire
(the exact handshake bytes, close frames, a peer that stops reading).

Interface pinned here for T077 (written before `breeze_infer/ws_server.py` exists):

- `registry = ws_server.ConnectionRegistry()`: the one object every `serve()` instance shares
  (the connection set, the cap of `WS_MAX_CONNECTIONS`, the shutdown flag).
- `server = await ws_server.serve(settings, components, sock, registry)` starts serving on the
  pre-bound, listening `sock` and returns once it is accepting. It reads the model from
  `components.readiness`, takes the GPU through `components.gate` / `components.gpu`, and emits
  through `components.events`.
- `await registry.shutdown()` is the whole shutdown of every server started with that registry:
  set the flag (new handshakes get `503 shutting_down`), stop listening (the pre-bound sockets
  are closed), close every tracked connection with 1001 concurrently, each bounded by
  `WS_CLOSE_TIMEOUT_SECONDS` (then aborted), and return once every connection handler is done.
- `ws_server` reads these limits from its own module globals at call time (when the registry is
  built, `serve()` is called or a connection opens; never as default arguments), so tests
  shrink them with `monkeypatch.setattr(ws_server, NAME, value)` before building the rig:
  `WS_OUTBOX_BYTES`, `WS_MAX_CONNECTIONS`, `WS_HANDSHAKE_SECONDS`, `WS_SEND_TIMEOUT_SECONDS`
  and `WS_CLOSE_TIMEOUT_SECONDS` (the last two are new in `limits.py`, review 44).
- `ws.closed` carries `code` (the code actually sent, `ws.protocol.close_sent`), `aborted`
  (bool) and `reason`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import numpy as np
import pytest
import torch
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from breeze_infer import __version__, gpu, ws_server
from breeze_infer.api import Components, bind_http_sockets
from breeze_infer.gpu import GpuGate, GpuLease, GpuThread, GpuUnavailable
from breeze_infer.http_fields import DEFAULT_CFG_SCALE, DEFAULT_INSTRUCTION
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from breeze_infer.synthesis import CodesRef, prepare_piece
from breeze_infer.templates import prepare_prefix_inputs

# FakeRuntime imports this lazily on its first step, i.e. on the GPU thread mid-piece; on a slow
# mount that takes long enough to blow the timing bounds below.
from models import fast_streaming  # noqa: F401
from tests.fakes import (
    CODEC_CODEBOOK_SIZE,
    CODEC_CODEBOOKS,
    FakeStreamingConfig,
    FakeTokenizer,
    RecordingEvents,
    model_with_codec_facts,
    open_no_voices,
)
from tests.ws_helpers import (
    BIG_PIECE_CHUNKS,
    BIG_START,
    SMALL_SNDBUF,
    TCP_ESTABLISHED,
    TINY_RCVBUF,
    HttpResponse,
    PacedRuntime,
    RawWs,
    big_runtime,
    close_payload,
    frame,
    tcp_state,
    wait_until,
)

T = TypeVar("T")

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it
MIB = 1024 * 1024
ALLOWED_ORIGIN = "http://127.0.0.1:8000"
CORS_ON = (f"--cors={ALLOWED_ORIGIN}",)


def build_components(argv: Sequence[str], events: RecordingEvents) -> Components:
    """Components as `api.main()` builds them, minus the model load: the tests mark readiness
    themselves. No voices (voice design only)."""
    cpu_tokenizer = CpuTokenizer()
    cpu_tokenizer.install(FakeTokenizer(), FakeTokenizer())
    components = Components(
        settings=settings_from_args([MODEL_DIR, *argv]),
        events=events,  # type: ignore[arg-type]
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=cpu_tokenizer,
        open_voices=open_no_voices,
    )
    components.voices.install(open_no_voices(None))
    return components


needs_tcp_info = pytest.mark.skipif(
    not hasattr(socket, "TCP_INFO"), reason="TCP_INFO is Linux-only"
)


# --- the server under test ----------------------------------------------------------------


@dataclass
class WsRig:
    """`ws_server.serve()` on its own thread and event loop, as uvicorn would share it."""

    components: Components
    runtime: PacedRuntime
    events: RecordingEvents
    sndbuf: int | None = None
    raw_clients: list[RawWs] = field(default_factory=list)
    leases: list[GpuLease] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        self.sockets: list[socket.socket] = []
        self.servers: list[Any] = []
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="ws-test-loop", daemon=True)
        self.thread.start()
        try:
            self.registry = ws_server.ConnectionRegistry()
            self.port = self.add_server()
        except BaseException:
            self.stop()
            raise
        self.sock = self.sockets[0]
        self.url = f"ws://127.0.0.1:{self.port}/"

    def add_server(self) -> int:
        """Another `serve()` on its own socket, sharing the registry (as T078 serves each
        address the host resolves to). Returns its port."""
        [sock] = bind_http_sockets("127.0.0.1", 0)
        self.sockets.append(sock)
        if self.sndbuf is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
        port = sock.getsockname()[1]  # read first: a server started after a shutdown is closed
        self.servers.append(
            self.run(
                ws_server.serve(self.components.settings, self.components, sock, self.registry)
            )
        )
        return port

    def run(self, coro: Any, timeout: float = 10.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, fn: Callable[[], T]) -> T:
        """Run `fn` on the server's loop, where the gate lives."""

        async def run() -> T:
            return fn()

        return self.run(run())

    def gate_is_free(self) -> bool:
        def probe() -> bool:
            lease = self.components.gate.try_acquire()
            if lease is None:
                return False
            lease.release()
            return True

        return self.call(probe)

    def hold_gate(self) -> GpuLease:
        lease = self.call(self.components.gate.try_acquire)
        assert lease is not None, "the gate is busy"
        self.leases.append(lease)
        return lease

    def release(self, lease: GpuLease) -> None:
        self.call(lease.release)

    def raw(self, *, port: int | None = None, **kwargs: Any) -> RawWs:
        client = RawWs(self.port if port is None else port, **kwargs)
        self.raw_clients.append(client)
        return client

    def client(self, **kwargs: Any) -> ClientConnection:
        return connect(self.url, open_timeout=5, close_timeout=2, compression=None, **kwargs)

    def ws_closed_events(self) -> list[dict[str, object]]:
        return [fields for name, fields in list(self.events.calls) if name == "ws.closed"]

    def shutdown(self, timeout: float = 15.0) -> float:
        """`await registry.shutdown()`; returns how long it took."""
        started = time.monotonic()
        self.closed = True
        self.run(self.registry.shutdown(), timeout)
        return time.monotonic() - started

    def stop(self) -> None:
        try:
            for lease in self.leases:
                if self.call(lambda lease=lease: lease.held):
                    self.release(lease)
            if self.servers and not self.closed:
                self.shutdown()
        finally:
            for client in self.raw_clients:
                client.close()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            if not self.thread.is_alive():
                self.loop.close()
            for sock in self.sockets:
                sock.close()
            assert self.components.gpu.shutdown(timeout=5)


@pytest.fixture
def start_ws() -> Iterator[Callable[..., WsRig]]:
    rigs: list[WsRig] = []

    def start(
        runtime: PacedRuntime | None = None,
        *,
        argv: Sequence[str] = (),
        ready: bool = True,
        sndbuf: int | None = None,
        readiness: Readiness | None = None,
    ) -> WsRig:
        runtime = PacedRuntime() if runtime is None else runtime
        events = RecordingEvents()
        components = build_components(argv, events)
        if readiness is not None:
            components = replace(components, readiness=readiness)
        if ready:
            components.readiness.mark_ready(runtime)
        rig = WsRig(components, runtime, events, sndbuf)
        rigs.append(rig)
        return rig

    yield start
    failures: list[BaseException] = []
    for rig in rigs:
        try:
            rig.stop()
        except BaseException as exc:  # noqa: BLE001 - stop every rig, then report the first
            failures.append(exc)
    if failures:
        raise failures[0]


# --- sync-client helpers -------------------------------------------------------------------


def read_ready(ws: ClientConnection) -> dict[str, Any]:
    ready = json.loads(ws.recv(timeout=5))
    assert ready["type"] == "ready", ready
    return ready


def send(ws: ClientConnection, kind: str, **fields: Any) -> None:
    ws.send(json.dumps({"type": kind, **fields}))


def collect_until(ws: ClientConnection, kind: str, timeout: float = 10.0) -> list[Any]:
    """Events (dicts) and audio frames (bytes) up to and including the first `kind` event."""
    deadline = time.monotonic() + timeout
    items: list[Any] = []
    while True:
        item = ws.recv(timeout=max(0.01, deadline - time.monotonic()))
        if isinstance(item, bytes):
            items.append(item)
            continue
        event = json.loads(item)
        items.append(event)
        if event["type"] == kind:
            return items


def kinds(items: list[Any]) -> list[str]:
    """Event types in order, with each run of audio frames shown once as "audio"."""
    out: list[str] = []
    for item in items:
        kind = "audio" if isinstance(item, bytes) else item["type"]
        if not (kind == "audio" and out and out[-1] == "audio"):
            out.append(kind)
    return out


def stall_mid_piece(rig: WsRig, text: str = "A long piece of speech.") -> RawWs:
    """A raw client with a tiny receive buffer that starts one big piece, reads up to its
    `speaking`, and then never reads again."""
    client = rig.raw(rcvbuf=TINY_RCVBUF)
    client.open()
    client.send_json(BIG_START)
    client.send_json({"type": "end", "text": text})
    client.frames_until_event("speaking")
    return client


def assert_json_refusal(response: HttpResponse, status: int, code: str | None = None) -> None:
    assert response.status == status, response
    assert response.values("Content-Type") == ["application/json"], response.headers
    assert response.values("X-Breeze-Version") == [__version__], response.headers
    body = response.json()
    assert isinstance(body.get("error"), str) and body["error"], body
    assert isinstance(body.get("code"), str) and body["code"], body
    if code is not None:
        assert body["code"] == code, body


# --- handshake ------------------------------------------------------------------------------


def test_bc_33_ready_comes_first_with_the_runtimes_sample_rate(
    start_ws: Callable[..., WsRig],
) -> None:
    """C++ hard-codes `ready.sample_rate` to 24000 whatever the model produces; here it is the
    loaded runtime's rate, sent before any client message."""
    runtime = PacedRuntime()
    runtime.sample_rate = 16000
    rig = start_ws(runtime)

    with rig.client() as ws:
        first = json.loads(ws.recv(timeout=5))

    assert first == {"type": "ready", "sample_rate": 16000, "format": "s16le"}


@pytest.mark.parametrize(
    ("case", "argv", "ready", "headers", "request_kwargs", "status"),
    [
        ("accepted", (), True, (), {}, 101),
        ("disallowed origin", CORS_ON, True, (("Origin", "http://evil.example"),), {}, 403),
        ("loading", (), False, (), {}, 503),
        ("plain GET", (), True, (), {"upgrade": False}, 426),
        ("unsupported version", (), True, (), {"version": "8"}, 400),
    ],
)
def test_every_handshake_response_carries_the_version_header_once(
    start_ws: Callable[..., WsRig],
    case: str,
    argv: tuple[str, ...],
    ready: bool,
    headers: tuple[tuple[str, str], ...],
    request_kwargs: dict[str, Any],
    status: int,
) -> None:
    """FR-037a: accepted or refused, including the library's own refusals. Exactly once:
    `Headers.__setitem__` appends, so a set without a delete would duplicate it."""
    rig = start_ws(argv=argv, ready=ready)

    response = rig.raw().handshake(headers, **request_kwargs)

    assert response.status == status, (case, response)
    assert response.values("X-Breeze-Version") == [__version__], (case, response.headers)


def test_bc_31_disallowed_origin_gets_403_json(start_ws: Callable[..., WsRig]) -> None:
    """C++ runs no Origin check on the handshake, so any web page can open a session
    (cross-site WebSocket hijacking); here a disallowed Origin gets a JSON 403."""
    rig = start_ws(argv=CORS_ON)

    response = rig.raw().handshake([("Origin", "http://evil.example")])

    assert_json_refusal(response, 403, "origin_not_allowed")


def test_bc_31_with_cors_off_every_browser_origin_is_refused(
    start_ws: Callable[..., WsRig],
) -> None:
    """C++ accepts every browser origin; with CORS off (the default) none is allowed here."""
    rig = start_ws()

    response = rig.raw().handshake([("Origin", ALLOWED_ORIGIN)])

    assert_json_refusal(response, 403, "origin_not_allowed")


@pytest.mark.parametrize("argv", [(), CORS_ON], ids=["cors off", "cors on"])
def test_a_handshake_without_origin_is_accepted(
    start_ws: Callable[..., WsRig], argv: tuple[str, ...]
) -> None:
    """Non-browser clients (SillyTavern's server plugin, scripts) send no Origin."""
    rig = start_ws(argv=argv)

    assert rig.raw().open()["type"] == "ready"


def test_an_allowed_origin_is_accepted(start_ws: Callable[..., WsRig]) -> None:
    rig = start_ws(argv=CORS_ON)

    assert rig.raw().open([("Origin", ALLOWED_ORIGIN)])["type"] == "ready"


def _other_loopback_reachable() -> bool:
    """Whether 127.0.0.2 reaches a wildcard listener here (all of 127/8 is loopback on Linux,
    not on macOS)."""
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        try:
            with socket.create_connection(("127.0.0.2", listener.getsockname()[1]), timeout=2):
                return True
        except OSError:
            return False


def test_bc_30_binds_configured_host_only(start_ws: Callable[..., WsRig]) -> None:
    """C++ binds the WebSocket to 0.0.0.0 whenever the host isn't an IPv4 literal, exposing it
    on every interface; here the server listens on the socket bound to the configured host
    and on nothing else."""
    if not _other_loopback_reachable():
        pytest.skip("127.0.0.2 doesn't reach a wildcard listener on this platform")
    rig = start_ws()

    assert rig.raw().open()["type"] == "ready"
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.2", rig.port), timeout=2).close()


def test_503_loading_until_the_model_is_ready(start_ws: Callable[..., WsRig]) -> None:
    rig = start_ws(ready=False)

    assert_json_refusal(rig.raw().handshake(), 503, "loading")

    rig.call(lambda: rig.components.readiness.mark_ready(rig.runtime))
    assert rig.raw().open()["type"] == "ready"


def test_503_gpu_unavailable_once_the_gpu_stopped_responding(
    start_ws: Callable[..., WsRig],
) -> None:
    """As on HTTP: a poisoned GPU gate is not "loading"; only a restart recovers it."""
    rig = start_ws()

    rig.call(rig.components.readiness.mark_unhealthy)

    assert_json_refusal(rig.raw().handshake(), 503, "gpu_unavailable")


def test_17_concurrent_handshakes_give_exactly_one_503_too_many_connections(
    start_ws: Callable[..., WsRig],
) -> None:
    """Every request is on the wire before any response is read, so the handshakes race.
    `server.connections` counts a connection only once its 101 is written (prototype gotcha
    6); the server's own connection set must hold the cap at 16 anyway."""
    rig = start_ws()
    clients = [rig.raw() for _ in range(17)]
    for client in clients:
        client.send_request()

    responses = [client.read_response() for client in clients]

    statuses = sorted(response.status for response in responses)
    assert statuses == [101] * 16 + [503], statuses
    [refused] = [r for r in responses if r.status == 503]
    assert_json_refusal(refused, 503, "too_many_connections")

    # Closing one frees its slot.
    accepted = next(c for c, r in zip(clients, responses) if r.status == 101)
    accepted.send_frame(0x8, close_payload(1000))
    _, close = accepted.read_until_close(reply=False)
    assert close is not None and close.close_code == 1000

    def a_new_client_is_accepted() -> bool:
        return rig.raw().handshake().status == 101

    wait_until(a_new_client_is_accepted, timeout=5, what="the closed connection's slot is freed")


def _abandon_after_the_request(client: RawWs) -> None:
    """Send a complete handshake request, then reset the connection before reading the 101."""
    client.send_request()
    client.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


@pytest.mark.parametrize(
    "attempt",
    [
        lambda client: client.handshake(upgrade=False),  # plain GET: the library's 426
        lambda client: client.handshake(key="not-a-base64-key"),  # the library's 400
        _abandon_after_the_request,  # gone before the 101
    ],
    ids=["plain GET", "bad key", "disconnect before the 101"],
)
def test_refused_and_abandoned_handshakes_do_not_leak_connection_slots(
    start_ws: Callable[..., WsRig], attempt: Callable[[RawWs], Any]
) -> None:
    """`process_request` reserves a slot before the library refuses the handshake or the client
    goes, and the library never runs the handler for those, so the slot must be released when
    the transport closes. 17 of them would fill all 16 slots if it leaked."""
    rig = start_ws()
    for _ in range(17):
        client = rig.raw()
        attempt(client)
        client.close()

    wait_until(
        lambda: rig.raw().handshake().status == 101, timeout=5, what="a normal handshake succeeds"
    )


def test_one_registry_caps_and_shuts_down_every_server(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T078 runs one `serve()` per bound address (localhost is ::1 and 127.0.0.1); the cap and
    the shutdown cover all of them together."""
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 2)
    rig = start_ws()
    other_port = rig.add_server()

    rig.raw().open()
    rig.raw(port=other_port).open()

    for port in (rig.port, other_port):
        assert_json_refusal(rig.raw(port=port).handshake(), 503, "too_many_connections")

    rig.shutdown()
    assert all(sock.fileno() == -1 for sock in rig.sockets)


@pytest.mark.parametrize(
    ("request_kwargs", "status", "code"),
    [
        ({"upgrade": False}, 426, "upgrade_required"),
        ({"version": "8"}, 400, "bad_handshake"),
        ({"key": "not-a-base64-key"}, 400, "bad_handshake"),
    ],
    ids=["plain GET", "unsupported version", "bad key"],
)
def test_the_librarys_own_refusals_come_back_as_the_json_envelope(
    start_ws: Callable[..., WsRig], request_kwargs: dict[str, Any], status: int, code: str
) -> None:
    """`websockets` answers these itself in text/plain; `process_response` rewrites them."""
    rig = start_ws()

    response = rig.raw().handshake(**request_kwargs)

    assert_json_refusal(response, status, code)


@pytest.mark.parametrize(
    "sent",
    [b"", b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"],
    ids=["silent", "half-sent request"],
)
def test_a_handshake_that_does_not_complete_in_time_is_dropped(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch, sent: bytes
) -> None:
    """BC-42's handshake limit (C++ has none: a silent client holds a thread forever)."""
    monkeypatch.setattr(ws_server, "WS_HANDSHAKE_SECONDS", 0.5)
    rig = start_ws()
    client = rig.raw()
    client.sock.sendall(sent)
    started = time.monotonic()

    try:
        received = client.sock.recv(65536)
    except ConnectionResetError:
        received = b""
    elapsed = time.monotonic() - started

    assert received == b""  # dropped without an HTTP response
    assert 0.3 <= elapsed <= 3.0, elapsed


# --- frames and close codes -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        (frame(0x8, close_payload(1000, "bye")), 1000),
        (frame(0x1, b'{"type": "cancel"}', masked=False), 1002),
        (frame(0x1, b"\xff\xfe"), 1007),
        (frame(0x1, b" " * (MIB + 1)), 1009),
    ],
    ids=["client close echoed", "unmasked frame", "bad UTF-8", "over 1 MiB"],
)
def test_bc_43_close_codes(
    start_ws: Callable[..., WsRig], sent: bytes, expected: int
) -> None:
    """C++ doesn't answer a client close (the browser sees 1006) and accepts unmasked frames,
    invalid UTF-8 and unbounded messages; here each gets the RFC 6455 close code."""
    rig = start_ws()
    client = rig.raw()
    client.open()

    try:
        client.sock.sendall(sent)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the server may close before a 1 MiB frame is fully sent
    _, close = client.read_until_close(reply=expected != 1000)

    assert close is not None, "dropped without a close frame"
    assert close.close_code == expected, (close.close_code, close.close_reason)
    # The code the library actually sent (`close_sent`), not the 1000 the handler's own
    # bounded close would pass afterwards.
    wait_until(lambda: rig.ws_closed_events() != [], what="ws.closed")
    [closed] = rig.ws_closed_events()
    assert closed["code"] == expected
    assert closed["aborted"] is False


def test_a_message_of_exactly_1_mib_is_not_too_big(start_ws: Callable[..., WsRig]) -> None:
    rig = start_ws()
    client = rig.raw()
    client.open()

    client.send_frame(0x1, b" " * MIB)  # at the limit, and not JSON
    event = client.recv_frame().event

    assert event is not None and event["type"] == "error", event
    assert event["code"] == "invalid_json"


def test_bc_44_speaking_text_is_exact(start_ws: Callable[..., WsRig]) -> None:
    """C++ strips tabs and carriage returns from `speaking.text`; here it is the piece text
    exactly."""
    rig = start_ws()
    text = "Hello\tthere\rfriend"

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text=text)
        items = collect_until(ws, "speaking")
        send(ws, "end")
        collect_until(ws, "done")

    assert items[-1]["text"] == text


def test_bc_45_binary_frame_gets_error(start_ws: Callable[..., WsRig]) -> None:
    """C++ silently ignores a binary frame from the client; here it gets an
    `unsupported_binary` error, and the session carries on."""
    rig = start_ws()

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        collect_until(ws, "started")
        ws.send(b"\x00\x01")
        error = json.loads(ws.recv(timeout=5))
        send(ws, "end", text="Still here.")
        after = collect_until(ws, "done")

    assert error["type"] == "error"
    assert error["code"] == "unsupported_binary"
    assert error["request_type"] is None  # a binary frame has no message type
    assert kinds(after) == ["speaking", "audio", "done"]


def test_bc_41_generation_failure_is_an_error_event_and_session_continues(
    start_ws: Callable[..., WsRig],
) -> None:
    """In C++ a generation error inside a session terminates the whole server process; here it
    is an `error{code: generation_failed}` event, and the session, the GPU and the server all
    carry on."""
    runtime = PacedRuntime(fail_first_call_after=1)
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text="First piece.")
        failed = collect_until(ws, "error")
        send(ws, "end", text="Second piece.")
        after = collect_until(ws, "done")

    assert failed[-1]["code"] == "generation_failed"
    assert kinds(after) == ["speaking", "audio", "done"]
    assert after[0]["text"] == "Second piece."
    wait_until(rig.gate_is_free, what="gate released")
    assert runtime.ended == 2
    with rig.client() as ws:  # the server still accepts sessions
        assert read_ready(ws)["type"] == "ready"


# --- slow clients (BC-42) --------------------------------------------------------------------


def test_bc_42_slow_client_closed_1008_and_gpu_freed(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """In C++ a client that stops reading holds the GPU and blocks every other client. Here
    the outbox overflows, the piece in flight is cancelled (the GPU freed), and a client that
    resumes reading promptly receives the 1008 close frame. Small socket buffers on both ends
    keep the close frame from being stuck behind megabytes."""
    monkeypatch.setattr(ws_server, "WS_OUTBOX_BYTES", 256 * 1024)
    runtime = big_runtime()
    rig = start_ws(runtime, sndbuf=SMALL_SNDBUF)

    client = stall_mid_piece(rig)
    wait_until(lambda: runtime.ended == 1, timeout=10, what="the piece is cancelled")
    assert runtime.yielded < BIG_PIECE_CHUNKS  # cut short, not run to the end
    assert rig.gate_is_free()

    frames, close = client.read_until_close()  # at once: well within the 2 s close bound

    assert close is not None, "dropped without the 1008 close frame"
    assert (close.close_code, close.close_reason) == (1008, "client too slow")
    events = [f.event["type"] for f in frames if f.event is not None]
    assert "done" not in events
    wait_until(lambda: rig.ws_closed_events() != [], what="ws.closed")
    [closed] = rig.ws_closed_events()
    assert (closed["code"], closed["aborted"], closed["reason"]) == (
        1008,
        False,
        "client too slow",
    )


@needs_tcp_info
def test_bc_42_a_client_that_never_reads_again_is_dropped_and_its_slot_freed(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C++ keeps a stalled client's thread and the GPU forever. Here the piece is cancelled,
    other clients are served at once, and a client that never reads the 1008 is dropped
    after `WS_CLOSE_TIMEOUT_SECONDS` (it would see 1006), freeing its connection slot."""
    monkeypatch.setattr(ws_server, "WS_OUTBOX_BYTES", 256 * 1024)
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 2)
    runtime = big_runtime()
    rig = start_ws(runtime, sndbuf=SMALL_SNDBUF)

    stalled = stall_mid_piece(rig)
    wait_until(lambda: runtime.ended == 1, timeout=10, what="the piece is cancelled")
    cancelled_at = time.monotonic()
    assert runtime.yielded < BIG_PIECE_CHUNKS
    assert rig.gate_is_free()

    with rig.client() as other:  # the second of two slots
        read_ready(other)
        send(other, "start", max_new_tokens=20)
        send(other, "end", text="Another client.")
        served = collect_until(other, "done")
        assert kinds(served) == ["started", "speaking", "audio", "done"]

        wait_until(
            lambda: tcp_state(stalled.sock) != TCP_ESTABLISHED,
            timeout=max(0.0, cancelled_at + 1.0 + 3.0 - time.monotonic()),
            what="the stalled client is dropped",
        )
        wait_until(lambda: rig.ws_closed_events() != [], what="ws.closed")
        [closed] = rig.ws_closed_events()
        assert (closed["code"], closed["aborted"], closed["reason"]) == (
            1008,
            True,
            "client too slow",
        )

        # `other` still holds one slot; the stalled client's is free again.
        wait_until(
            lambda: rig.raw().handshake().status == 101,
            timeout=3,
            what="the dropped client's slot is freed",
        )


@needs_tcp_info
def test_a_client_stalled_under_the_outbox_limit_is_evicted_by_the_stall_watchdog(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 44: a piece smaller than the outbox never overflows it, and the keepalive ping
    waits in `drain()` behind the same full buffer, so only the stall watchdog (a write buffer
    non-empty and not shrinking for `WS_SEND_TIMEOUT_SECONDS`) can evict this client."""
    monkeypatch.setattr(ws_server, "WS_SEND_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 1)
    runtime = big_runtime()
    rig = start_ws(runtime, sndbuf=SMALL_SNDBUF)

    client = rig.raw(rcvbuf=TINY_RCVBUF)
    client.open()
    # 300 frames: 1.15 MB, under the 2 MiB outbox but far over what the socket buffers hold.
    client.send_json({"type": "start", "max_new_tokens": 300})
    client.send_json({"type": "end", "text": "A short piece."})
    client.frames_until_event("speaking")
    stalled_at = time.monotonic()

    # The piece finishes into the outbox: the GPU is never held waiting on the socket.
    wait_until(lambda: runtime.ended == 1, what="the piece finishes")
    assert rig.gate_is_free()
    assert_json_refusal(rig.raw().handshake(), 503, "too_many_connections")  # still connected

    wait_until(lambda: rig.ws_closed_events() != [], timeout=2.0 + 1.0 + 3.0, what="ws.closed")
    evicted_after = time.monotonic() - stalled_at
    [closed] = rig.ws_closed_events()
    assert (closed["code"], closed["aborted"], closed["reason"]) == (
        1008,
        True,
        "client too slow",
    )
    assert evicted_after >= 1.0, "evicted before the watchdog could have fired"
    wait_until(
        lambda: tcp_state(client.sock) != TCP_ESTABLISHED, what="the stalled client is dropped"
    )
    wait_until(
        lambda: rig.raw().handshake().status == 101, timeout=3, what="its slot is freed"
    )


@needs_tcp_info
def test_an_idle_client_that_stops_reading_but_keeps_pinging_is_evicted(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 45: with no session work at all, the library's automatic pongs fill the write
    buffer (they are written without `drain()`), and our keepalive ping then blocks in
    `drain()` with its timeout never started. The stall watchdog evicts this client too."""
    monkeypatch.setattr(ws_server, "WS_SEND_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 1)
    rig = start_ws(sndbuf=SMALL_SNDBUF)
    client = rig.raw(rcvbuf=TINY_RCVBUF)
    client.open()
    ping = frame(0x9, b"p" * 125)  # the largest control frame

    started = time.monotonic()
    deadline = started + 1.0 + 1.0 + 4.0
    try:
        while not rig.ws_closed_events() and time.monotonic() < deadline:
            client.sock.sendall(ping * 50)  # ~6 KB of pongs owed per round, never read
            time.sleep(0.01)
    except (BrokenPipeError, ConnectionResetError):
        pass  # aborted while we were still pinging
    wait_until(lambda: rig.ws_closed_events() != [], timeout=2, what="ws.closed")
    evicted_after = time.monotonic() - started

    [closed] = rig.ws_closed_events()
    assert (closed["code"], closed["aborted"], closed["reason"]) == (
        1008,
        True,
        "client too slow",
    )
    assert 0.5 <= evicted_after <= 1.0 + 1.0 + 4.0, evicted_after
    wait_until(
        lambda: tcp_state(client.sock) != TCP_ESTABLISHED, what="the pinging client is dropped"
    )
    wait_until(
        lambda: rig.raw().handshake().status == 101, timeout=3, what="its slot is freed"
    )


# --- shutdown ------------------------------------------------------------------------------


@needs_tcp_info
def test_shutdown_with_stalled_peers_finishes_within_the_close_bound(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T070 check 4: `Server.close()` awaited `close(1001)` on a peer whose writes were blocked,
    and never returned. Here the whole shutdown takes about one `WS_CLOSE_TIMEOUT_SECONDS`,
    however many peers are stalled: one blocked mid-piece on a full socket, four that never
    answer the close. Closing them one after another (or walking a set that shrinks
    underneath) would take five bounds. A reading client still gets 1001, and the piece in
    flight is cancelled."""
    close_bound = 1.0
    monkeypatch.setattr(ws_server, "WS_OUTBOX_BYTES", 64 * MIB)  # no overflow: only a stall
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", close_bound)
    runtime = big_runtime(delay=0.01)  # a 1.5 s piece, still running at shutdown
    rig = start_ws(runtime, sndbuf=SMALL_SNDBUF)
    stalled = [stall_mid_piece(rig)]
    for _ in range(4):
        silent = rig.raw(rcvbuf=TINY_RCVBUF)
        silent.open()  # and then never reads or answers again
        stalled.append(silent)

    with rig.client() as idle:
        read_ready(idle)
        time.sleep(0.3)  # the first peer's sender is now blocked on a full socket

        elapsed = rig.shutdown()

        with pytest.raises(ConnectionClosed) as closed:
            idle.recv(timeout=5)
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1001
    assert elapsed < close_bound + 2.0, elapsed
    assert all(sock.fileno() == -1 for sock in rig.sockets)  # the listening sockets are closed
    assert runtime.ended == 1 and runtime.yielded < BIG_PIECE_CHUNKS
    assert rig.gate_is_free()
    for peer in stalled:
        wait_until(lambda peer=peer: tcp_state(peer.sock) != TCP_ESTABLISHED, timeout=2)
    events = rig.ws_closed_events()
    assert len(events) == 6
    assert {event["code"] for event in events} == {1001}
    assert sorted(event["aborted"] for event in events) == [False] + [True] * 5


def test_a_handshake_during_shutdown_gets_503_shutting_down(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-037a with no exception: the library's own shutdown 503 is plain text without
    `X-Breeze-Version` (prototype check 2), so the server refuses first, with its own flag."""
    monkeypatch.setattr(ws_server, "WS_OUTBOX_BYTES", 64 * MIB)
    monkeypatch.setattr(ws_server, "WS_CLOSE_TIMEOUT_SECONDS", 1.5)
    rig = start_ws(big_runtime(delay=0.01), sndbuf=SMALL_SNDBUF)
    stall_mid_piece(rig)  # holds the shutdown open for the close bound
    late = rig.raw()
    request = late.request()
    late.sock.sendall(request[:20])  # accepted before the shutdown, request not finished
    time.sleep(0.2)

    shutting_down = asyncio.run_coroutine_threadsafe(rig.registry.shutdown(), rig.loop)
    rig.closed = True
    time.sleep(0.3)
    late.sock.sendall(request[20:])
    response = late.read_response()
    shutting_down.result(timeout=10)

    assert_json_refusal(response, 503, "shutting_down")


# --- queueing and seeds ----------------------------------------------------------------------


def test_queued_is_sent_only_when_the_piece_has_to_wait_for_the_gate(
    start_ws: Callable[..., WsRig],
) -> None:
    rig = start_ws()
    lease = rig.hold_gate()  # as an HTTP request would

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text="Hello there.")
        waiting = collect_until(ws, "queued")
        with pytest.raises(TimeoutError):  # nothing is spoken while the gate is held
            ws.recv(timeout=0.3)
        send(ws, "end")
        rig.release(lease)
        spoken = collect_until(ws, "done")

    assert kinds(waiting) == ["started", "queued"]
    assert kinds(spoken) == ["speaking", "audio", "done"]


def test_queued_is_not_sent_when_the_gate_is_free(start_ws: Callable[..., WsRig]) -> None:
    rig = start_ws()

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text="One.")
        send(ws, "end", text="Two.")
        items = collect_until(ws, "done")

    assert kinds(items) == ["started", "speaking", "audio", "speaking", "audio", "done"]


def test_piece_seeds_continue_from_start(start_ws: Callable[..., WsRig]) -> None:
    """Piece i counted from `start` uses (seed + i) mod 2**32, across every `end` in between;
    a new `start` counts from its own seed again."""
    runtime = PacedRuntime()
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", seed=100)
        send(ws, "flush", text="One.")
        send(ws, "end", text="Two.")
        collect_until(ws, "done")
        send(ws, "end", text="Three.")
        collect_until(ws, "done")
        send(ws, "start", seed=4_294_967_295)
        send(ws, "flush", text="Four.")
        send(ws, "end", text="Five.")
        collect_until(ws, "done")

    assert [call["seed"] for call in runtime.calls] == [100, 101, 102, 4_294_967_295, 0]


# --- room and anchoring ------------------------------------------------------------------------

# With the fake tokenizer a piece's prompt is about 50 tokens plus one per character, and an
# anchor made from "Hello there." (two frames) adds 47 more.
SHORT_TEXT = "Hello there."
LONG_TEXT = "x" * 300  # 350 tokens alone, 397 with that anchor


def prompt_lengths(runtime: PacedRuntime) -> list[int]:
    """Each generated piece's prompt length, in order: an anchored piece's prompt carries the
    anchor's codes and text, so it is longer than the same text alone."""
    return [int(call["inputs"]["attention_mask"].shape[1]) for call in runtime.calls]


def anchor_skips(rig: WsRig) -> list[tuple[object, object]]:
    return [
        (fields["piece_index"], fields["reason"])
        for name, fields in list(rig.events.calls)
        if name == "speech.anchor_skipped"
    ]


def test_a_piece_with_no_room_gets_text_too_long_and_the_session_continues(
    start_ws: Callable[..., WsRig],
) -> None:
    """BC-47 on the WebSocket: a piece whose prompt leaves the model no room to generate is
    skipped with `text_too_long`, before anything is spoken, and the session carries on."""
    runtime = PacedRuntime(config=FakeStreamingConfig(max_seq_len=300))
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", split_chars=0)
        send(ws, "flush", text=LONG_TEXT)
        refused = collect_until(ws, "error")
        send(ws, "end", text="Hi.")
        after = collect_until(ws, "done")

    assert kinds(refused) == ["started", "error"]
    assert (refused[-1]["code"], refused[-1]["request_type"]) == ("text_too_long", None)
    assert kinds(after) == ["speaking", "audio", "done"]
    assert rig.gate_is_free()


def test_a_later_piece_is_spoken_without_the_anchor_when_the_anchor_would_shorten_it(
    start_ws: Callable[..., WsRig],
) -> None:
    """Each later piece is checked on its own, as HTTP checks every piece up front: the anchor
    is used unless min(cap, room with it) < min(cap, room without it). With a cap of 20 frames
    in a 400-token context, the long piece has room for 20 frames alone but only 2 with the
    anchor, so it is spoken without it; the short piece after it keeps the anchor."""
    runtime = PacedRuntime(config=FakeStreamingConfig(max_seq_len=400))
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", split_chars=0, max_new_tokens=20)
        send(ws, "flush", text=SHORT_TEXT)  # piece 0: becomes the anchor
        send(ws, "flush", text=LONG_TEXT)  # piece 1: the anchor would cost it frames
        send(ws, "end", text="Hi.")  # piece 2: the anchor fits
        items = collect_until(ws, "done")

    assert kinds(items) == [
        "started", "speaking", "audio", "speaking", "audio", "speaking", "audio", "done"
    ]
    assert prompt_lengths(runtime) == [62, 350, 100]
    assert anchor_skips(rig) == [(1, "no_room")]


def test_a_truncated_first_piece_is_not_an_anchor(start_ws: Callable[..., WsRig]) -> None:
    """A piece that used its whole frame limit may stop mid-word, so it anchors nothing
    (`piece_truncated`, as on HTTP), and the next piece is spoken without an anchor."""
    runtime = PacedRuntime()  # two frames per piece
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", split_chars=0, max_new_tokens=2)
        send(ws, "flush", text=SHORT_TEXT)
        send(ws, "end", text="Hi.")
        collect_until(ws, "done")

    assert prompt_lengths(runtime) == [62, 53]
    assert anchor_skips(rig)[0] == (0, "piece_truncated")


def test_a_disconnect_ends_the_session_quietly(
    start_ws: Callable[..., WsRig], caplog: pytest.LogCaptureFixture
) -> None:
    """After an `end` and a `start` that interrupts nothing, a client that closes normally
    leaves nothing behind: the worker queues nothing more (no `cancelled` for a cancel nobody
    sent), exits without an error, and the GPU is free."""
    runtime = PacedRuntime()
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "end", text="Hi.")
        collect_until(ws, "done")
        send(ws, "start")
        collect_until(ws, "started")

    wait_until(lambda: rig.ws_closed_events() != [], what="ws.closed")
    [closed] = rig.ws_closed_events()
    assert (closed["code"], closed["aborted"]) == (1000, False)
    names = [name for name, _ in rig.events.calls]
    assert "request.failed" not in names
    assert runtime.ended == len(runtime.calls) == 1
    assert rig.gate_is_free()
    assert [r for r in caplog.records if r.levelname in ("ERROR", "CRITICAL")] == []


# --- review 48 ---------------------------------------------------------------------------------


def test_a_server_added_after_the_shutdown_began_is_closed_at_once(
    start_ws: Callable[..., WsRig],
) -> None:
    """T078 starts one server per address; a signal can start the shutdown between two of
    them. The late server must not be left listening."""
    rig = start_ws()
    rig.shutdown()

    late_port = rig.add_server()

    def refused() -> bool:
        try:
            socket.create_connection(("127.0.0.1", late_port), timeout=2).close()
        except ConnectionRefusedError:
            return True
        return False

    # The library closes a server from a task, a loop step after `serve()` has returned.
    wait_until(refused, what="the late server stops listening")
    assert rig.sockets[-1].fileno() == -1


def test_a_close_timeout_leaves_the_lease_to_the_generation(
    start_ws: Callable[..., WsRig],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `gen.close()` past `GPU_CLOSE_TIMEOUT_SECONDS` poisons the gate, and the lease is
    released only when the close really ends (the session's own done-callback), as on HTTP:
    releasing it earlier would free the GPU while it is still busy, then release twice."""
    monkeypatch.setattr(gpu, "GPU_CLOSE_TIMEOUT_SECONDS", 0.3)
    runtime = PacedRuntime(chunks=50, delay=0.02, close_delay=1.0)
    rig = start_ws(runtime)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text=SHORT_TEXT)
        collect_until(ws, "speaking")
        send(ws, "cancel")  # stops the piece; its close then takes 1 s
        items = collect_until(ws, "cancelled", timeout=10)

    assert "error" in kinds(items)  # the piece failed: its close timed out
    time.sleep(1.5)  # the close has finished, and its done-callback has run
    with pytest.raises(GpuUnavailable):
        rig.call(rig.components.gate.try_acquire)
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == []


def test_a_cancel_stops_a_piece_waiting_for_the_gate(start_ws: Callable[..., WsRig]) -> None:
    """A piece queued behind another holder of the GPU stops waiting as soon as it is
    cancelled: `cancelled` comes at once, no `queued` follows it, and the gate is never
    taken."""
    runtime = PacedRuntime()
    rig = start_ws(runtime)
    lease = rig.hold_gate()  # as an HTTP request would

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text=SHORT_TEXT)
        collect_until(ws, "queued")
        started = time.monotonic()
        send(ws, "cancel")
        after = collect_until(ws, "cancelled", timeout=5)
        waited = time.monotonic() - started
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.3)

    assert kinds(after) == ["cancelled"]
    assert waited < 1.0, waited
    assert rig.call(lambda: lease.held)  # still ours: the piece never took it
    assert runtime.calls == []


class _OutOfMemoryPrefixRuntime(PacedRuntime):
    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        self.prefix_builds.append(prefix_inputs)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")


VOICE_ID = "v_0123456789abcdef"
VOICE_TEXT = "stored transcript"
VOICE_CODES = (np.arange(8 * CODEC_CODEBOOKS, dtype=np.int16) % CODEC_CODEBOOK_SIZE).reshape(
    8, CODEC_CODEBOOKS
)


def _add_voice(rig: WsRig) -> None:
    prefix_inputs = prepare_prefix_inputs(
        FakeTokenizer(),
        model_with_codec_facts(),
        {"ref_text": VOICE_TEXT, "ref_audio_codes": VOICE_CODES},
    )
    rig.components.voices.get().registry.register_unnamed(
        id=VOICE_ID,
        ref_text=VOICE_TEXT,
        codes=VOICE_CODES,
        frames=8,
        encode_ms=1,
        prefix_len=int(prefix_inputs["attention_mask"].shape[1]),
    )


def test_an_out_of_memory_prefix_build_falls_back_to_codes_for_the_rest_of_the_session(
    start_ws: Callable[..., WsRig],
) -> None:
    """As on HTTP: the voice's codes path, with `speech.prefix_fallback`. Remembered for the
    session, so each later piece doesn't retry a build that just ran out of memory."""
    runtime = _OutOfMemoryPrefixRuntime()
    rig = start_ws(runtime)
    _add_voice(rig)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", voice_id=VOICE_ID)
        send(ws, "flush", text="One.")
        send(ws, "end", text="Two.")
        items = collect_until(ws, "done")

    assert kinds(items) == ["started", "speaking", "audio", "speaking", "audio", "done"]
    assert len(runtime.prefix_builds) == 1
    assert [call["prefix"] for call in runtime.calls] == [None, None]
    [connected] = [f for name, f in rig.events.calls if name == "ws.connected"]
    [fallback] = [f for name, f in rig.events.calls if name == "speech.prefix_fallback"]
    assert fallback["level"] == "warning"
    assert fallback["session_id"] == connected["session_id"]
    assert fallback["request_id"] == connected["session_id"]
    assert fallback["voice_id"] == VOICE_ID
    assert fallback["reason"] == "out_of_memory"
    assert "CUDA out of memory" in fallback["error"]


class _PoisonedAfter(Readiness):
    """Ready for its first `good_reads` reads of `runtime`, then gone: the gate was poisoned
    in between. The handshake reads it in `process_request` (1), then just before the 101 in
    `process_response` (2); the session reads it when its handler starts (3)."""

    def __init__(self, runtime: Any, good_reads: int) -> None:
        super().__init__()
        self.mark_ready(runtime)
        self.good_reads = good_reads
        self.reads = 0

    @property
    def runtime(self) -> Any:
        self.reads += 1
        return super().runtime if self.reads <= self.good_reads else None


def test_a_gpu_that_stops_responding_during_the_handshake_gets_503(
    start_ws: Callable[..., WsRig],
) -> None:
    """Checked again just before the 101, so a gate poisoned while the handshake was in
    progress still gets the handshake's `503 gpu_unavailable`."""
    runtime = PacedRuntime()
    rig = start_ws(runtime, ready=False, readiness=_PoisonedAfter(runtime, good_reads=1))

    assert_json_refusal(rig.raw().handshake(), 503, "gpu_unavailable")


def test_a_gpu_that_stops_responding_before_the_session_starts_closes_1011(
    start_ws: Callable[..., WsRig],
) -> None:
    """The backstop for the loop step between the 101 and the handler: no `ready` without a
    runtime; the session closes at once with 1011 `gpu is not responding`."""
    runtime = PacedRuntime()
    rig = start_ws(runtime, ready=False, readiness=_PoisonedAfter(runtime, good_reads=2))
    client = rig.raw()

    assert client.handshake().status == 101
    frames, close = client.read_until_close()

    assert frames == []  # no `ready`
    assert close is not None
    assert (close.close_code, close.close_reason) == (1011, "gpu is not responding")


def test_anchor_sizing_queues_on_the_pre_gate_worker(start_ws: Callable[..., WsRig]) -> None:
    """The per-piece anchor check runs before the gate, on the pre-gate worker HTTP's piece-0
    checks use, and waits its turn there (no timeout): with that worker busy, the next piece
    waits, then is spoken with its anchor."""
    runtime = PacedRuntime()
    rig = start_ws(runtime)
    busy, release = threading.Event(), threading.Event()

    def block(_tokenizer: Any) -> None:
        busy.set()
        release.wait(10)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text=SHORT_TEXT)  # piece 0: the anchor, which needs no sizing
        collect_until(ws, "speaking")
        blocker = asyncio.run_coroutine_threadsafe(
            rig.components.cpu_tokenizer.run(block), rig.loop
        )
        assert busy.wait(5)
        send(ws, "end", text="Hi.")  # piece 1: its sizing queues behind `block`
        waiting: list[Any] = []  # whatever arrives while the worker is blocked
        with pytest.raises(TimeoutError):
            while True:
                item = ws.recv(timeout=0.5)
                waiting.append(item if isinstance(item, bytes) else json.loads(item))
        release.set()
        blocker.result(timeout=5)
        after = collect_until(ws, "done")

    assert "speaking" not in kinds(waiting)  # piece 1 waited for the worker
    assert kinds(after) == ["speaking", "audio", "done"]
    assert anchor_skips(rig) == []
    assert prompt_lengths(runtime) == [62, 100]


# A second, valid key: two keys are refused even when each is well formed.
OTHER_KEY = base64.b64encode(b"0123456789abcdef").decode()


@pytest.mark.parametrize(
    ("headers", "request_kwargs", "status", "code"),
    [
        ((), {"upgrade": False}, 426, "upgrade_required"),
        ((), {"version": "8"}, 400, "bad_handshake"),
        # Upgrade must name websocket alone, as the library requires.
        ((("Upgrade", "h2c"),), {}, 426, "upgrade_required"),
        ((("Sec-WebSocket-Key", OTHER_KEY),), {}, 400, "bad_handshake"),
        ((), {"key": "not-a-base64-key"}, 400, "bad_handshake"),
        ((), {"key": base64.b64encode(b"too short").decode()}, 400, "bad_handshake"),
    ],
    ids=[
        "plain GET",
        "unsupported version",
        "h2c alongside websocket",
        "two keys",
        "bad base64 key",
        "key not 16 bytes",
    ],
)
def test_a_malformed_handshake_is_refused_before_readiness_and_the_cap(
    start_ws: Callable[..., WsRig],
    monkeypatch: pytest.MonkeyPatch,
    headers: tuple[tuple[str, str], ...],
    request_kwargs: dict[str, Any],
    status: int,
    code: str,
) -> None:
    """A request that isn't a WebSocket upgrade is told so (426/400), classified as the
    library would, even while the model loads or every slot is taken: never `503`, and it
    never takes a slot."""
    monkeypatch.setattr(ws_server, "WS_MAX_CONNECTIONS", 1)
    rig = start_ws(ready=False)
    refused = rig.raw().handshake(headers, **request_kwargs)
    assert_json_refusal(refused, status, code)  # while loading

    rig.call(lambda: rig.components.readiness.mark_ready(rig.runtime))
    rig.raw().open()  # the only slot
    refused = rig.raw().handshake(headers, **request_kwargs)
    assert_json_refusal(refused, status, code)  # at the cap


# --- review 49 ---------------------------------------------------------------------------------


def test_a_slow_client_at_speaking_does_not_leak_the_gate(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The outbox overflowing on the `speaking` event itself, before any audio: the lease
    must still be released, or the GPU is lost to every client for good."""
    monkeypatch.setattr(ws_server, "WS_OUTBOX_BYTES", 200)  # `ready` fits, `speaking` doesn't
    rig = start_ws()
    client = rig.raw()
    client.open()

    client.send_json({"type": "start", "split_chars": 0})
    client.send_json({"type": "flush", "text": LONG_TEXT})
    _, close = client.read_until_close()

    assert close is not None and close.close_code == 1008
    wait_until(lambda: rig.ws_closed_events() != [], what="ws.closed")
    assert rig.gate_is_free()


def test_shutdown_finishes_when_a_pieces_close_times_out(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stuck GPU: cancelling the piece in flight ends in `GpuCloseTimeout`, which replaces
    the task's cancellation. It must not be mistaken for a failed piece (the session going on,
    waiting for work that never comes) or the shutdown never finishes."""
    monkeypatch.setattr(gpu, "GPU_CLOSE_TIMEOUT_SECONDS", 0.3)
    runtime = PacedRuntime(chunks=50, delay=0.05, close_delay=1.5)
    rig = start_ws(runtime)
    client = rig.raw()
    client.open()
    client.send_json({"type": "start"})
    client.send_json({"type": "flush", "text": SHORT_TEXT})
    client.frames_until_event("speaking")

    elapsed = rig.shutdown(timeout=15)

    assert elapsed < 0.3 + 2.0 + 2.0, elapsed


def test_a_cancel_stops_a_piece_waiting_for_its_anchor_sizing(
    start_ws: Callable[..., WsRig],
) -> None:
    """The anchor check waits its turn on the pre-gate worker; a `cancel` doesn't wait behind
    it: `cancelled` comes at once, and the piece is never generated."""
    runtime = PacedRuntime()
    rig = start_ws(runtime)
    busy, release = threading.Event(), threading.Event()

    def block(_tokenizer: Any) -> None:
        busy.set()
        release.wait(10)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "end", text=SHORT_TEXT)  # piece 0: the anchor
        collect_until(ws, "done")
        blocker = asyncio.run_coroutine_threadsafe(
            rig.components.cpu_tokenizer.run(block), rig.loop
        )
        try:
            assert busy.wait(5)
            send(ws, "flush", text="Hi.")  # piece 1: its sizing queues behind `block`
            time.sleep(0.2)
            started = time.monotonic()
            send(ws, "cancel")
            after = collect_until(ws, "cancelled", timeout=5)
            waited = time.monotonic() - started
        finally:
            release.set()
            blocker.result(timeout=5)

    assert kinds(after) == ["cancelled"]
    assert waited < 1.0, waited
    assert len(runtime.calls) == 1


def test_the_unanchored_prompt_is_skipped_when_the_anchor_leaves_the_full_cap(
    start_ws: Callable[..., WsRig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the anchor the piece still gets its whole cap, so it can't be shorter than without
    it, and the second prompt isn't built."""
    measured: list[str] = []
    real_predicted_room = ws_server.predicted_room

    def counting(runtime: Any, tokenizer: Any, reference: Any, *args: Any) -> Any:
        measured.append(type(reference).__name__)
        return real_predicted_room(runtime, tokenizer, reference, *args)

    monkeypatch.setattr(ws_server, "predicted_room", counting)
    rig = start_ws()

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start")
        send(ws, "flush", text=SHORT_TEXT)  # piece 0: the anchor
        send(ws, "end", text="Hi.")  # piece 1: plenty of room either way
        collect_until(ws, "done")

    assert measured == ["CodesRef"]


def test_a_server_added_after_the_shutdown_returned_is_awaited(
    start_ws: Callable[..., WsRig],
) -> None:
    """`registry.wait_closed()` covers a server closed by `add_server` after `shutdown()` had
    already returned, so nothing is left pending."""
    rig = start_ws()
    rig.shutdown()
    rig.add_server()

    rig.run(rig.registry.wait_closed(), timeout=5)

    assert rig.call(lambda: all(server.close_task.done() for server in rig.servers))


def test_no_room_after_an_out_of_memory_fallback_is_reported_as_no_room(
    start_ws: Callable[..., WsRig],
) -> None:
    """As on HTTP (`speech.prefix_fallback`): `no_room` when the codes path the fallback
    chose has no room either; the piece is then skipped with `text_too_long`."""
    codes_prompt = prepare_piece(
        FakeTokenizer(),
        SimpleNamespace(config=model_with_codec_facts().config, device="cpu"),
        CodesRef(codes=VOICE_CODES, ref_text=VOICE_TEXT),
        "One.",
        DEFAULT_INSTRUCTION,
        DEFAULT_CFG_SCALE,
    )
    context = int(codes_prompt["attention_mask"].shape[1])  # no room left on the codes path
    runtime = _OutOfMemoryPrefixRuntime(config=FakeStreamingConfig(max_seq_len=context))
    rig = start_ws(runtime)
    _add_voice(rig)

    with rig.client() as ws:
        read_ready(ws)
        send(ws, "start", voice_id=VOICE_ID)
        send(ws, "end", text="One.")
        items = collect_until(ws, "done")

    assert [item["code"] for item in items if item["type"] == "error"] == ["text_too_long"]
    [fallback] = [f for name, f in rig.events.calls if name == "speech.prefix_fallback"]
    assert fallback["reason"] == "no_room"
