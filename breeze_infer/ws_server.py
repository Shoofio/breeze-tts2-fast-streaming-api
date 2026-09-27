"""The WebSocket server (contracts/ws-api.md; tasks.md T077; research.md R4, R14, R15;
research/ws-prototype.md "Decision").

`websockets`' native asyncio server, on the same event loop as uvicorn. `serve()` runs one
server on one pre-bound socket. Every server started with the same `ConnectionRegistry`
shares one connection set, one cap and one shutdown flag, so `api.py` can serve each address
the host resolves to and still enforce a single cap of `WS_MAX_CONNECTIONS`.

Per connection (`_Channel`): a reader task parses client frames (`ws_messages.parse`) and
applies them to the session (`ws_session.Session`); one worker coroutine drains the session's
work deque, generating each piece on the `GpuThread` under the `GpuGate`; everything the
client receives, events and PCM alike, goes through one ordered outbox, drained by a sender
task. A watchdog evicts a client whose socket has stopped draining.

Why every server-initiated close is bounded here rather than by the library: `ws.close()`
writes the close frame and then waits in `drain()` for room in the socket, and only after that
does its `close_timeout` start. A peer that stopped reading never makes room, so the close (and
the library's keepalive, and `Server.close()`) would wait forever (T070 check 1). So each close
runs inside `asyncio.timeout(WS_CLOSE_TIMEOUT_SECONDS)`; past it, the connection is aborted with
`SO_LINGER(1, 0)`, which also frees the kernel's copy of the unsent bytes.

The limits are read from this module's globals when they are used, never bound as default
arguments, so tests can shrink them.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import http
import json
import os
import socket
import struct
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as websockets_serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidHeader
from websockets.headers import parse_connection, parse_upgrade
from websockets.http11 import Request, Response

from breeze_infer import __version__
from breeze_infer.cors import CorsPolicy, origin_allowed
from breeze_infer.errors import error_fields
from breeze_infer.gpu import (
    DONE,
    GpuLease,
    GpuSession,
    GpuUnavailable,
    gpu_call_under_lease,
)
from breeze_infer.limits import (
    WS_CLOSE_TIMEOUT_SECONDS,
    WS_HANDSHAKE_SECONDS,
    WS_MAX_CONNECTIONS,
    WS_MAX_MESSAGE_BYTES,
    WS_OUTBOX_BYTES,
    WS_SEND_TIMEOUT_SECONDS,
)
from breeze_infer.routes_speech import PrefixOutOfMemory, voice_prefix
from breeze_infer.synthesis import (
    CodesRef,
    NoRef,
    PieceRoom,
    Reference,
    anchor_codes,
    codec_samples_per_frame,
    generate_piece,
    piece_frame_limit,
    piece_room,
    predicted_room,
    prefix_of,
    prepare_piece,
    voice_reference,
)
from breeze_infer.ws_messages import WsError, parse
from breeze_infer.ws_session import (
    CancelMark,
    EndMark,
    Piece,
    Session,
    SessionConfig,
    StartMark,
)
from models.fast_streaming import NoRoomError

if TYPE_CHECKING:
    from breeze_infer.api import Components
    from breeze_infer.settings import Settings
    from breeze_infer.voice_registry import ResolvedVoice

# Keepalive (contracts/ws-api.md "Frames"): a ping every 20 s, 1011 without a pong in 20 s.
PING_INTERVAL_SECONDS = 20
PING_TIMEOUT_SECONDS = 20

# How often the stall watchdog samples the write buffer, at most: a tenth of the send timeout,
# so an eviction comes within 10% of it.
_WATCHDOG_MAX_INTERVAL_SECONDS = 1.0

# `SO_LINGER` on, with a zero timeout: `close()` then resets the connection and drops the unsent
# bytes, instead of the kernel holding them for a peer that will never read them. The struct is
# two ints on POSIX and two unsigned shorts on Windows.
_LINGER_ZERO = struct.pack("HH", 1, 0) if os.name == "nt" else struct.pack("ii", 1, 0)

# The library's own handshake refusals, rewritten into the JSON envelope (contracts/ws-api.md
# "Handshake"). Any other status it produces is an unexpected failure: `internal_error`.
_LIBRARY_REFUSALS = {
    400: ("bad_handshake", "invalid WebSocket handshake"),
    426: ("upgrade_required", "a WebSocket upgrade is required"),
}
_INTERNAL_ERROR = ("internal_error", "internal error")

_SHUTDOWN_CLOSE = (1001, "server shutting down")
# The gate was poisoned between the handshake's checks and the session's start: too late for
# the handshake's `503 gpu_unavailable`, and the GPU won't come back without a restart, so the
# session ends at once, with 1011 (an unexpected server condition) rather than 1013 (try again
# later), which would invite a retry that can only fail the same way.
_GPU_UNAVAILABLE_CLOSE = (1011, "gpu is not responding")
_SLOW_CLIENT_CLOSE = (1008, "client too slow")
_INTERNAL_ERROR_CLOSE = (1011, "internal error")
_NORMAL_CLOSE = (1000, "")


class _Refusal(NamedTuple):
    """A handshake refusal from `process_request`, with any header its status requires."""

    status: int
    code: str
    message: str
    headers: tuple[tuple[str, str], ...] = ()


_UPGRADE_REQUIRED = _Refusal(
    426,
    "upgrade_required",
    "a WebSocket upgrade is required",
    # RFC 9110: a 426 names the protocol to upgrade to.
    (("Upgrade", "websocket"),),
)
_BAD_HANDSHAKE = _Refusal(400, "bad_handshake", "invalid WebSocket handshake")
# RFC 6455 section 4.4: a version mismatch names the version the server speaks.
_BAD_VERSION = _BAD_HANDSHAKE._replace(headers=(("Sec-WebSocket-Version", "13"),))


def _malformed_handshake(headers: Headers) -> _Refusal | None:
    """A request that isn't a valid WebSocket upgrade, checked before readiness and the cap, so
    it is told what is wrong even while loading or at the cap and never takes a slot.

    The same rules, in the same order and with the same answer, as the library's own check
    (`websockets.server.ServerProtocol.process_request`): its `InvalidUpgrade` is a 426 and any
    other handshake error a 400. Its own header parsers are used, so a malformed header value
    is a 400 here too."""
    try:
        connection = [
            option for value in headers.get_all("Connection") for option in parse_connection(value)
        ]
        if not any(option.lower() == "upgrade" for option in connection):
            return _UPGRADE_REQUIRED
        upgrade = [
            protocol for value in headers.get_all("Upgrade") for protocol in parse_upgrade(value)
        ]
    except InvalidHeader:
        return _BAD_HANDSHAKE
    # Exactly one protocol: `Upgrade: websocket, h2c` is refused too.
    if not (len(upgrade) == 1 and upgrade[0].lower() == "websocket"):
        return _UPGRADE_REQUIRED
    keys = headers.get_all("Sec-WebSocket-Key")
    if len(keys) != 1:
        return _BAD_HANDSHAKE
    try:
        key = base64.b64decode(keys[0].encode(), validate=True)
    except binascii.Error:
        return _BAD_HANDSHAKE
    if len(key) != 16:
        return _BAD_HANDSHAKE
    if headers.get_all("Sec-WebSocket-Version") != ["13"]:
        return _BAD_VERSION
    return None


class _SlowClient(Exception):
    """The outbox would pass `WS_OUTBOX_BYTES`: the client isn't reading fast enough."""


class _Connection(ServerConnection):
    """The library's connection, counting how often its transport resumed writing.

    A resume means the socket drained the write buffer down to its low-water mark, so the
    client is reading. The watchdog needs that signal: sampling the buffer size alone can't tell
    a stalled buffer from one that drained and was refilled by the next send in between.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.writes_resumed = 0

    def resume_writing(self) -> None:
        self.writes_resumed += 1
        super().resume_writing()


def _json_refusal(
    status: int, code: str, message: str, headers: Headers | None = None
) -> Response:
    """A handshake refusal with the JSON envelope. `headers` are kept (for example the
    library's `Sec-WebSocket-Version` on a version mismatch) except for the body's type and
    length."""
    body = json.dumps({"error": message, "code": code}).encode()
    kept = Headers()
    if headers is not None:
        for name, value in headers.raw_items():
            if name.lower() not in ("content-type", "content-length"):
                kept[name] = value
    if "Connection" not in kept:
        kept["Connection"] = "close"
    kept["Content-Type"] = "application/json"
    kept["Content-Length"] = str(len(body))
    return Response(status, http.HTTPStatus(status).phrase, kept, body)


def _abort(connection: ServerConnection) -> None:
    """End the connection now, without a closing handshake, freeing the kernel's buffers too."""
    sock = connection.transport.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _LINGER_ZERO)
    connection.transport.abort()


class ConnectionRegistry:
    """What every `serve()` instance shares: the connection set, the cap and the shutdown flag.

    A connection is added in `process_request`, which reserves its slot before the handshake
    goes on, so concurrent handshakes can't pass the cap between them. It leaves when its
    transport closes, not when its handler ends: the library never runs the handler for a
    handshake it refuses itself or that the client abandons, and those must not keep a slot.
    """

    def __init__(self) -> None:
        self.shutting_down = False
        self._servers: list[Server] = []
        # Each admitted connection, with its channel once its handler has started.
        self._connections: dict[ServerConnection, _Channel | None] = {}

    def admit(self, connection: ServerConnection) -> bool:
        """Reserve a slot for `connection`, or `False` at the cap."""
        if len(self._connections) >= WS_MAX_CONNECTIONS:
            return False
        self._connections[connection] = None
        connection.connection_lost_waiter.add_done_callback(
            lambda _lost: self._connections.pop(connection, None)
        )
        return True

    def attach(self, connection: ServerConnection, channel: _Channel) -> None:
        if connection in self._connections:
            self._connections[connection] = channel

    def add_server(self, server: Server) -> None:
        self._servers.append(server)
        if self.shutting_down:
            # Started after the shutdown began (api.serve starts one server per address, and a
            # signal can land between two): close it now; `shutdown()` waits for it too.
            server.close(close_connections=False)

    async def shutdown(self) -> None:
        """Stop every server: refuse new handshakes (`503 shutting_down`), stop listening, close
        every connection with 1001, concurrently, each within `WS_CLOSE_TIMEOUT_SECONDS`, and
        return once every handler has finished.

        A snapshot of the set, since each close removes its entry. A peer that connected but
        never sent its request isn't in the set; the library drops it at `open_timeout`, which
        is then what bounds this call.
        """
        self.shutting_down = True
        for server in self._servers:
            server.close(close_connections=False)
        closes = [
            self._close(connection, channel)
            for connection, channel in list(self._connections.items())
        ]
        await asyncio.gather(*closes, return_exceptions=True)
        await self.wait_closed()

    async def wait_closed(self) -> None:
        """Wait until every server has closed, including one `add_server` closed after
        `shutdown()` had already returned (api.serve awaits this again, so none is left
        pending)."""
        waited = 0
        while waited < len(self._servers):
            servers = self._servers[waited:]
            waited = len(self._servers)
            await asyncio.gather(
                *(server.wait_closed() for server in servers), return_exceptions=True
            )

    async def _close(self, connection: ServerConnection, channel: _Channel | None) -> None:
        if channel is not None:
            await channel.shut_down()
            return
        # Still in its handshake: it ends by itself (a refusal, or a handler that sees the
        # shutdown flag and closes), or is aborted.
        try:
            async with asyncio.timeout(WS_CLOSE_TIMEOUT_SECONDS):
                await connection.wait_closed()
        except TimeoutError:
            _abort(connection)


async def serve(
    settings: Settings,
    components: Components,
    sock: socket.socket,
    registry: ConnectionRegistry,
) -> Server:
    """Serve WebSocket sessions on `sock`, a bound, listening socket; returns once accepting.
    `registry.shutdown()` stops it, together with every other server sharing `registry`."""
    policy = CorsPolicy(origins=settings.cors)
    events = components.events

    # Both hooks are synchronous on purpose: the library checks `is_serving()` after
    # `process_response`, so an `await` in either would let a shutdown slip in between, and the
    # library would then swap our 101 for its own plain-text 503.
    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        refusal = _refusal_for(request)
        if refusal is None and not registry.admit(connection):
            refusal = _Refusal(503, "too_many_connections", "too many connections")
        if refusal is None:
            return None
        events.emit("ws.rejected", reason=refusal.code)
        return _json_refusal(
            refusal.status, refusal.code, refusal.message, Headers(refusal.headers)
        )

    def _refusal_for(request: Request) -> _Refusal | None:
        """In the contract's order (contracts/ws-api.md "Handshake"); the cap comes last, in
        `process_request`, so only a request that passes all of these takes a slot."""
        if registry.shutting_down:
            return _Refusal(503, "shutting_down", "server is shutting down")
        malformed = _malformed_handshake(request.headers)
        if malformed is not None:
            return malformed
        # `get_all`: two Origin headers must not slip past as one (`get` would raise).
        origins = request.headers.get_all("Origin")
        # No Origin at all is a non-browser client, which CORS doesn't govern (BC-31).
        if origins and not all(origin_allowed(policy, origin) for origin in origins):
            return _Refusal(403, "origin_not_allowed", "origin not allowed")
        if components.readiness.unhealthy:
            return _Refusal(503, "gpu_unavailable", "gpu is not responding")
        if components.readiness.runtime is None:
            return _Refusal(503, "loading", "model is loading")
        return None

    def process_response(
        connection: ServerConnection, request: Request, response: Response
    ) -> Response:
        if response.status_code == http.HTTPStatus.SWITCHING_PROTOCOLS and (
            components.readiness.unhealthy or components.readiness.runtime is None
        ):
            # The GPU stopped responding while this handshake was under way: checked again
            # just before the 101, so the client still gets the handshake's answer rather than
            # a session that closes at once (the handler's 1011 is only the backstop for the
            # loop step after this).
            events.emit("ws.rejected", reason="gpu_unavailable")
            response = _json_refusal(503, "gpu_unavailable", "gpu is not responding")
        elif response.status_code != http.HTTPStatus.SWITCHING_PROTOCOLS and (
            response.headers.get_all("Content-Type") != ["application/json"]
        ):
            # One of the library's own refusals, in text/plain.
            code, message = _LIBRARY_REFUSALS.get(response.status_code, _INTERNAL_ERROR)
            events.emit("ws.rejected", reason=code)
            response = _json_refusal(response.status_code, code, message, response.headers)
        # `Headers.__setitem__` appends: delete first, or the header could appear twice.
        if "X-Breeze-Version" in response.headers:
            del response.headers["X-Breeze-Version"]
        response.headers["X-Breeze-Version"] = __version__
        return response

    async def handler(connection: ServerConnection) -> None:
        channel = _Channel(connection, settings, components, registry)
        registry.attach(connection, channel)
        await channel.run()

    server = await websockets_serve(
        handler,
        sock=sock,
        process_request=process_request,
        process_response=process_response,
        server_header=None,
        open_timeout=WS_HANDSHAKE_SECONDS,
        max_size=WS_MAX_MESSAGE_BYTES,
        ping_interval=PING_INTERVAL_SECONDS,
        ping_timeout=PING_TIMEOUT_SECONDS,
        close_timeout=WS_CLOSE_TIMEOUT_SECONDS,
        # PCM hardly compresses, and deflate would cost CPU on every frame.
        compression=None,
        create_connection=_Connection,
    )
    registry.add_server(server)
    return server


@dataclass
class _PieceResult:
    """What a piece's generator leaves behind for the worker: the anchor it built, if any, and
    whether it was asked for one but ran to its frame limit instead."""

    anchor: CodesRef | None = None
    truncated: bool = False


def _prepare(
    runtime: Any, reference: Reference, text: str, config: SessionConfig
) -> tuple[dict[str, Any], Any]:
    """A piece's inputs and its frame room. GPU-thread only (`prepare_piece` touches the model)."""
    inputs = prepare_piece(
        runtime.tokenizer, runtime.model, reference, text, config.instruction, config.cfg_scale
    )
    return inputs, piece_room(runtime, inputs, config.max_new_tokens, prefix=prefix_of(reference))


def _anchor_shortens(
    tokenizer: Any, runtime: Any, anchor: CodesRef, text: str, config: SessionConfig
) -> bool:
    """Whether `anchor` would give the piece `text` a smaller frame limit, min(cap, room), than
    it has without it: HTTP's `no_room` rule, applied per piece, since a streaming session
    can't see its later pieces up front. Run on the pre-gate worker (`CpuTokenizer.run`) with
    its copy, before the gate, from prompts built on the CPU (`predicted_room`)."""

    def room(reference: Reference) -> PieceRoom:
        return predicted_room(
            runtime,
            tokenizer,
            reference,
            text,
            config.instruction,
            config.cfg_scale,
            config.max_new_tokens,
        )

    anchored = room(anchor)
    if anchored.room >= anchored.cap:
        # The whole cap even with the anchor: it can't be shorter without it, so the second
        # prompt needn't be built.
        return False
    return anchored.room < room(NoRef()).room


def _piece_audio(
    runtime: Any,
    inputs: dict[str, Any],
    piece: Piece,
    config: SessionConfig,
    reference: Reference,
    result: _PieceResult,
    *,
    collect_anchor: bool,
    session_id: str,
    frame_limit: int,
    chunk_first: int,
    chunk_max: int,
    samples_per_frame: int,
) -> Iterator[bytes]:
    """One piece's PCM, stepped on the GPU thread. With `collect_anchor` (no voice and no
    anchor yet), it collects the frames and, if the piece ends on its own rather than at its
    frame limit, where its audio may stop mid-word, leaves them in `result` as the anchor for
    the later pieces. Built here, on the GPU thread, because the frames are device tensors."""
    frames: list[Any] | None = [] if collect_anchor else None
    yield from generate_piece(
        runtime,
        inputs,
        request_id=session_id,
        seed=config.piece_seed(piece.index),
        chunk_first=chunk_first,
        chunk_max=chunk_max,
        samples_per_frame=samples_per_frame,
        prefix=prefix_of(reference),
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_new_tokens=frame_limit,
        token_observer=None if frames is None else frames.append,
    )
    if frames is None:
        return
    if len(frames) >= frame_limit:
        result.truncated = True
        return
    codes = anchor_codes(frames, int(runtime.model.config.codebook_pad_token_id))
    if codes is not None:
        result.anchor = CodesRef(codes=codes, ref_text=piece.text)


class _Channel:
    """One connection's session and tasks."""

    def __init__(
        self,
        connection: ServerConnection,
        settings: Settings,
        components: Components,
        registry: ConnectionRegistry,
    ) -> None:
        self.connection = connection
        self.settings = settings
        self.components = components
        self.registry = registry
        self.events = components.events
        self.runtime = components.readiness.runtime
        self.session_id = uuid.uuid4().hex
        # The prefix cache's token as read when each voice was looked up, by the voice
        # object's identity: a voice deleted after its `start` is then built but not cached.
        self._voice_tokens: dict[int, int] = {}
        self.session = Session(
            lookup_voice=self._lookup_voice, default_split_chars=settings.split_chars
        )
        self._outbox: deque[tuple[str | bytes, int]] = deque()
        self._outbox_bytes = 0
        self._outbox_ready = asyncio.Event()
        self._work_ready = asyncio.Event()
        # Set by the reader after every applied message: a piece waiting for the gate wakes up
        # to see whether a `cancel` or `start` made it stale.
        self._session_changed = asyncio.Event()
        # Voices (by prefix key) whose prefix build ran out of GPU memory in this session: they
        # take the codes path from then on, rather than failing the same build every piece.
        self._codes_fallback: set[tuple[str, str]] = set()
        # An out-of-memory fallback not yet reported: its `speech.prefix_fallback` reason
        # (`out_of_memory` or `no_room`) depends on whether the codes path has room, which the
        # piece's frame limit then tells (`_report_prefix_fallback`). (voice id, build error)
        self._fallback_to_report: tuple[str, str] | None = None
        # Set once `session.close()` has run: the worker has nothing more to wait for.
        self._session_closed = False
        loop = asyncio.get_running_loop()
        # The close code and reason, once anything decides the connection is over.
        self._stop: asyncio.Future[tuple[int, str]] = loop.create_future()
        self._finished: asyncio.Future[None] = loop.create_future()
        self._aborted = False

    # ------------------------------------------------------------------ lifecycle

    def request_close(self, code: int, reason: str) -> None:
        """End the connection with `code`; the first request wins."""
        if not self._stop.done():
            self._stop.set_result((code, reason))

    async def shut_down(self) -> None:
        self.request_close(*_SHUTDOWN_CLOSE)
        await asyncio.shield(self._finished)

    async def run(self) -> None:
        code, reason = _INTERNAL_ERROR_CLOSE
        sender: asyncio.Task[None] | None = None
        others: list[asyncio.Task[None]] = []
        try:
            if self.registry.shutting_down:
                code, reason = _SHUTDOWN_CLOSE
                return
            if self.runtime is None:
                code, reason = _GPU_UNAVAILABLE_CLOSE
                return
            self.events.emit("ws.connected", session_id=self.session_id)
            self._send_event(
                {"type": "ready", "sample_rate": int(self.runtime.sample_rate), "format": "s16le"}
            )
            sender = asyncio.create_task(self._send())
            others = [
                asyncio.create_task(self._read()),
                asyncio.create_task(self._work()),
                asyncio.create_task(self._watch()),
            ]
            for task in (sender, *others):
                task.add_done_callback(self._task_ended)
            code, reason = await asyncio.shield(self._stop)
        except Exception as error:  # noqa: BLE001 - reported; the connection closes with 1011
            self._report_failure(error)
        finally:
            # On disconnect: close the session (a new epoch stops the piece in flight at its
            # next chunk, and nothing queued is sent any more), then stop and join the tasks;
            # a cancelled piece's generator is closed on the GPU thread before the gate is
            # released.
            self.session.close()
            self._session_closed = True
            self._work_ready.set()
            for task in others:
                task.cancel()
            await asyncio.gather(*others, return_exceptions=True)
            await self._bounded_close(code, reason, sender)
            sent = self.connection.protocol.close_sent
            self.events.emit(
                "ws.closed",
                session_id=self.session_id,
                code=None if sent is None else int(sent.code),
                reason="" if sent is None else sent.reason,
                aborted=self._aborted,
            )
            self._finished.set_result(None)

    def _task_ended(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._report_failure(error)
            self.request_close(*_INTERNAL_ERROR_CLOSE)
        else:
            # The reader and the sender end when the connection has closed.
            self.request_close(*_NORMAL_CLOSE)

    def _report_failure(self, error: BaseException) -> None:
        self.events.emit(
            "request.failed", level="error", session_id=self.session_id, **error_fields(error)
        )

    async def _bounded_close(
        self, code: int, reason: str, sender: asyncio.Task[None] | None
    ) -> None:
        """Close with `code`, within `WS_CLOSE_TIMEOUT_SECONDS`, then abort (module
        docstring). The sender goes first, so nothing more is queued behind the close frame."""
        if sender is not None:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        try:
            async with asyncio.timeout(WS_CLOSE_TIMEOUT_SECONDS):
                await self.connection.close(code, reason)
        except TimeoutError:
            self._aborted = True
            _abort(self.connection)
            await self.connection.wait_closed()

    # ------------------------------------------------------------------ outbox

    def _enqueue(self, data: str | bytes) -> None:
        size = len(data) if isinstance(data, bytes) else len(data.encode())
        if self._outbox_bytes + size > WS_OUTBOX_BYTES:
            raise _SlowClient
        self._outbox.append((data, size))
        self._outbox_bytes += size
        self._outbox_ready.set()

    def _send_event(self, event: dict[str, Any]) -> None:
        self._enqueue(json.dumps(event, ensure_ascii=False))

    def _slow_client(self) -> None:
        self.request_close(*_SLOW_CLIENT_CLOSE)

    async def _send(self) -> None:
        while True:
            while not self._outbox:
                self._outbox_ready.clear()
                await self._outbox_ready.wait()
            data, size = self._outbox[0]
            try:
                await self.connection.send(data)
            except ConnectionClosed:
                return
            self._outbox.popleft()
            self._outbox_bytes -= size

    async def _watch(self) -> None:
        """Evict a client whose socket has stopped draining: the write buffer non-empty and
        not shrinking for `WS_SEND_TIMEOUT_SECONDS`, whether or not a send of ours is pending
        (the library's automatic pongs fill it too). A resumed write counts as shrinking."""
        loop = asyncio.get_running_loop()
        transport = self.connection.transport
        connection = self.connection
        assert isinstance(connection, _Connection)
        stalled_since: float | None = None
        last_size = 0
        last_resumes = connection.writes_resumed
        while True:
            await asyncio.sleep(min(_WATCHDOG_MAX_INTERVAL_SECONDS, WS_SEND_TIMEOUT_SECONDS / 10))
            size = transport.get_write_buffer_size()
            resumes = connection.writes_resumed
            now = loop.time()
            if size == 0 or size < last_size or resumes != last_resumes:
                stalled_since = None
            elif stalled_since is None:
                stalled_since = now
            elif now - stalled_since >= WS_SEND_TIMEOUT_SECONDS:
                self._slow_client()
                return
            last_size, last_resumes = size, resumes

    # ------------------------------------------------------------------ reader

    async def _read(self) -> None:
        try:
            async for message in self.connection:
                if isinstance(message, bytes):
                    events: list[dict[str, Any]] = [
                        _error("unsupported_binary", "binary frames are not supported", None)
                    ]
                else:
                    parsed = parse(message)
                    if isinstance(parsed, WsError):
                        events = [_error(parsed.code, parsed.message, parsed.request_type)]
                    else:
                        events = self.session.apply(parsed)
                        self._work_ready.set()
                        self._session_changed.set()
                for event in events:
                    self._send_event(event)
        except ConnectionClosed:
            return
        except _SlowClient:
            self._slow_client()

    def _lookup_voice(self, voice_id: str) -> ResolvedVoice | None:
        services = self.components.voices.get()
        voice = services.registry.lookup(voice_id)
        if voice is not None:
            self._voice_tokens[id(voice)] = services.prefix_cache.token()
        return voice

    # ------------------------------------------------------------------ worker

    async def _work(self) -> None:
        try:
            while True:
                item = self.session.next_item()
                if item is None:
                    if self._session_closed:
                        return  # `session.close()` cleared the deque: nothing will come
                    self._work_ready.clear()
                    await self._work_ready.wait()
                elif isinstance(item, CancelMark):
                    self._send_event({"type": "cancelled"})
                elif isinstance(item, StartMark):
                    self._send_event({"type": "started", "voice_id": item.config.voice_id})
                elif isinstance(item, EndMark):
                    # A cancel that supersedes an EndMark removes it from the deque, so every
                    # one the deque still hands out owes its `done`.
                    self._send_event({"type": "done"})
                else:
                    await self._speak(item)
        except _SlowClient:
            self._slow_client()

    async def _speak(self, piece: Piece) -> None:
        """One piece, however it ends; the session hears of it (`mark_piece_done`) either way.
        A failure is an `error` event and the session goes on (BC-41); so is a piece with no
        room to generate, which is skipped (`text_too_long`, as BC-47 on HTTP)."""
        anchor: CodesRef | None = None
        outcome = "cancelled"
        failure: dict[str, str] = {}
        try:
            anchor, outcome = await self._generate(piece)
        except (_SlowClient, asyncio.CancelledError):
            raise
        except NoRoomError:
            outcome = "no_room"
            self._send_event(_error("text_too_long", "text is too long", None))
        except Exception as error:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                # This task is being cancelled (a disconnect, a shutdown), and the error took
                # the place of its CancelledError: `GpuSession.aclose()` raises
                # `GpuCloseTimeout` for a close that outlasted the cancel. Treating that as a
                # failed piece would let the session go on and swallow the cancel.
                raise
            outcome = "failed"
            failure = error_fields(error)
            self._send_event(_error("generation_failed", "generation failed", None))
        finally:
            self.session.mark_piece_done(anchor)
            self.events.emit(
                "ws.piece",
                level="error" if failure else "info",
                session_id=self.session_id,
                piece_index=piece.index,
                outcome=outcome,
                **failure,
            )

    async def _generate(self, piece: Piece) -> tuple[CodesRef | None, str]:
        if self.session.is_stale(piece):
            return None, "cancelled"
        # The instruction as it is when the piece starts (an `instruction` message changes the
        # session's config for the pieces after it).
        config = self.session.config
        assert config is not None  # a Piece is only ever queued after a `start`
        anchor = await self._usable_anchor(piece, config)
        if self.session.is_stale(piece):  # cancelled while its anchor was being sized
            return None, "cancelled"
        lease = await self._acquire(piece)
        if lease is None:
            return None, "cancelled"
        generation: GpuSession[bytes] | None = None
        try:
            audio, result = await self._prepare_audio(piece, config, anchor, lease)
            # Before the session exists: this can raise `_SlowClient`, and until `async with`
            # has started, only the `finally` below releases the lease.
            self._send_event({"type": "speaking", "text": piece.text})
            # From here the session owns the lease: it releases it once the generator's close
            # has really finished on the GPU thread, even after a close timeout (which poisons
            # the gate), so nothing below may release it too. Entered at once, with nothing in
            # between that could raise.
            generation = GpuSession(lease, self.components.gpu, audio)
            async with generation:
                while not self.session.is_stale(piece):
                    chunk = await generation.step()
                    if chunk is DONE:
                        if result.truncated:
                            self._skip_anchor(piece, "piece_truncated")
                        return result.anchor, "done"
                    self._enqueue(chunk)
            return None, "cancelled"
        finally:
            if generation is None and lease.held:  # not handed over by a cancelled GPU call
                lease.release()

    async def _acquire(self, piece: Piece) -> GpuLease | None:
        """The gate for `piece`, or `None` if the piece went stale first: a `cancel` or `start`
        stops the wait at once, rather than when the current holder is done. `queued` is sent
        only when this really waits (`GpuGate.acquire`'s `on_wait`), and never for a stale
        piece."""

        def on_wait() -> None:
            if not self.session.is_stale(piece):
                self._send_event({"type": "queued"})

        acquiring = asyncio.ensure_future(self.components.gate.acquire(on_wait=on_wait))
        try:
            await self._until_done_or_stale(acquiring, piece)
        except BaseException:
            _give_up(acquiring)
            raise
        if acquiring.done() and not self.session.is_stale(piece):
            return acquiring.result()  # raises GpuUnavailable for a poisoned gate
        _give_up(acquiring)
        return None

    async def _until_done_or_stale(self, waiting: asyncio.Future[Any], piece: Piece) -> None:
        """Wait for `waiting`, or until `piece` goes stale (a `cancel` or `start`, which the
        reader signals through `_session_changed`), whichever comes first. The caller decides
        what to do with `waiting` if it isn't done."""
        while not waiting.done() and not self.session.is_stale(piece):
            self._session_changed.clear()
            changed = asyncio.ensure_future(self._session_changed.wait())
            try:
                await asyncio.wait({waiting, changed}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                changed.cancel()

    async def _usable_anchor(self, piece: Piece, config: SessionConfig) -> CodesRef | None:
        """The session's anchor, if this piece should use it: not with a voice, and not when
        it would shorten this piece (`_anchor_shortens`); then the piece is spoken without it,
        with `speech.anchor_skipped`.

        Measured before the gate, on the CPU tokenizer's pre-gate worker (`CpuTokenizer.run`),
        with HTTP's piece-0 checks: sizing before the gate is that worker's job. Not the
        anchor-sizing worker, which only the lease holder may use, since its timeout changes
        HTTP's behaviour (`sizing_timeout`).

        A `cancel` or `start` doesn't wait behind that worker's queue: the sizing is abandoned
        (a queued call is dropped; a running one finishes harmlessly, unread), and the caller's
        stale check ends the piece."""
        anchor = self.session.anchor
        if config.voice is not None or not isinstance(anchor, CodesRef):
            return None
        sizing = asyncio.ensure_future(
            self.components.cpu_tokenizer.run(
                _anchor_shortens, self.runtime, anchor, piece.text, config
            )
        )
        try:
            await self._until_done_or_stale(sizing, piece)
        finally:
            if not sizing.done():
                sizing.cancel()
        if sizing.cancelled():  # abandoned: the piece went stale
            return None
        try:
            shortens = sizing.result()
        except GpuUnavailable:  # the CPU tokenizer is shutting down with the server
            self._skip_anchor(piece, "shutdown")
            return None
        except Exception as error:  # noqa: BLE001 - reported; the piece goes on without it
            self.events.emit(
                "speech.anchor_sizing_failed",
                level="error",
                session_id=self.session_id,
                piece_index=piece.index,
                **error_fields(error),
            )
            self._skip_anchor(piece, "sizing_failed")
            return None
        if shortens:
            self._skip_anchor(piece, "no_room")
            return None
        return anchor

    def _skip_anchor(self, piece: Piece, reason: str) -> None:
        self.events.emit(
            "speech.anchor_skipped",
            session_id=self.session_id,
            piece_index=piece.index,
            reason=reason,
        )

    async def _prepare_audio(
        self, piece: Piece, config: SessionConfig, anchor: CodesRef | None, lease: GpuLease
    ) -> tuple[Iterator[bytes], _PieceResult]:
        """The piece's reference, inputs and frame limit, then its PCM generator (not started).
        Raises `NoRoomError` for a piece with no room to generate."""
        runtime = self.runtime
        gpu = self.components.gpu
        reference = await self._reference(config, anchor, lease)
        inputs, room = await gpu_call_under_lease(
            gpu, lease, _prepare, runtime, reference, piece.text, config
        )
        try:
            frame_limit = piece_frame_limit(
                room,
                self.events,
                request_id=self.session_id,
                piece_index=piece.index,
                requested=config.max_new_tokens,
            )
        except NoRoomError:
            self._report_prefix_fallback("no_room")
            raise
        self._report_prefix_fallback("out_of_memory")
        result = _PieceResult()
        audio = _piece_audio(
            runtime,
            inputs,
            piece,
            config,
            reference,
            result,
            collect_anchor=config.voice is None and self.session.anchor is None,
            session_id=self.session_id,
            frame_limit=frame_limit,
            chunk_first=self.settings.chunk_first,
            chunk_max=self.settings.chunk_max,
            samples_per_frame=codec_samples_per_frame(runtime),
        )
        return audio, result

    async def _reference(
        self, config: SessionConfig, anchor: CodesRef | None, lease: GpuLease
    ) -> Reference:
        """The piece's reference: the session's voice (its cached prefix, built on a miss, or
        its codes with an overriding transcript), else the anchor it may use, else none."""
        if config.voice is None:
            return NoRef() if anchor is None else anchor
        voice = config.voice
        shape = voice_reference(voice, config.ref_text_override)
        if isinstance(shape, CodesRef):
            return shape
        if voice.prefix_key in self._codes_fallback:
            return shape.codes_path()
        built = await voice_prefix(
            shape,
            voice.prefix_key,
            self.components.voices.get().prefix_cache,
            self.runtime,
            self.components.gpu,
            lease=lease,
            resolved_token=self._voice_tokens.get(id(voice), 0),
            request_id=self.session_id,
        )
        if isinstance(built, PrefixOutOfMemory):
            # As on HTTP: the codes path, with every cached prefix already freed. A piece with
            # no room left on it is then skipped like any other (`text_too_long`). Reported once
            # this piece's room is known (`_report_prefix_fallback`).
            self._codes_fallback.add(voice.prefix_key)
            self._fallback_to_report = (voice.id, built.error)
            return shape.codes_path()
        reference, _warm = built
        return reference

    def _report_prefix_fallback(self, reason: str) -> None:
        """`speech.prefix_fallback` for a fallback `_reference` just took, if any: `no_room`
        when the codes path has no room for the piece either, as HTTP decides it, else
        `out_of_memory`. `request_id` is the session's id, as the prefix cache's own events for
        that build carry it (every request event has one, data-model.md "Events"); `session_id`
        is there as on every WebSocket event."""
        if self._fallback_to_report is None:
            return
        voice_id, error = self._fallback_to_report
        self._fallback_to_report = None
        self.events.emit(
            "speech.prefix_fallback",
            level="warning",
            request_id=self.session_id,
            session_id=self.session_id,
            voice_id=voice_id,
            reason=reason,
            error=error,
        )


def _give_up(acquiring: asyncio.Future[GpuLease]) -> None:
    """Stop waiting for the gate: cancel the wait, or pass on a gate it already won (a gate
    handed over in the same loop step as the cancel)."""
    if not acquiring.done():
        acquiring.cancel()  # `GpuGate.acquire` passes on a gate handed to it as it is cancelled
    elif not acquiring.cancelled() and acquiring.exception() is None:
        acquiring.result().release()


def _error(code: str, message: str, request_type: str | None) -> dict[str, Any]:
    return {"type": "error", "message": message, "code": code, "request_type": request_type}
