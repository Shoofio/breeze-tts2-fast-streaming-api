"""Helpers shared by the WebSocket server tests (tests/test_ws_server.py) and the isolation
test (tests/test_ws_isolation.py): a paced `FakeRuntime`, a raw-socket WebSocket client for
protocol probes and stalled peers, and the socket settings that make a stall show up quickly.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tests.fakes import (
    CODEC_CODEBOOKS,
    FakeCodec,
    FakeRuntime,
    FakeStreamingConfig,
    FakeTokenizer,
    model_with_codec_facts,
)

# A piece far bigger than anything the socket buffers can absorb: 1,500 frames (the frame
# ceiling) of 1,920 samples as s16le is 5.76 MB, in 150 chunks of 38,400 bytes.
BIG_PIECE_CHUNKS = 150
BIG_PIECE_FRAMES_PER_CHUNK = 10
BIG_START = {"type": "start", "max_new_tokens": 1500}
# The server's accepted sockets inherit this from the listening socket. A fixed send buffer
# turns off the kernel's autotuning (up to 4 MB on loopback), so "the client stopped reading"
# backs up into the server's own outbox after ~200 KB instead of after several MB.
SMALL_SNDBUF = 64 * 1024
# The receive buffer of a client that stops reading.
TINY_RCVBUF = 4096

TCP_ESTABLISHED = 1  # tcp_info.tcpi_state, include/net/tcp_states.h


# --- waiting ---------------------------------------------------------------------------


def wait_until(condition: Callable[[], bool], timeout: float = 5.0, what: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"not within {timeout} s: {what or condition}")
        time.sleep(0.01)


# --- the model edge ------------------------------------------------------------------------


class PacedRuntime(FakeRuntime):
    """`FakeRuntime` with the attributes the synthesis path reads off the real runtime
    (`tokenizer`, `model`, `audio_tokenizer`), plus three things these tests need:

    - `delay`: seconds slept before each chunk, so a piece takes a known time on the GPU thread;
    - `close_delay`: seconds a generation's close takes on the GPU thread (a stuck close);
    - `fail_first_call_after`: the first call raises after that many chunks (a mid-piece CUDA
      error once, then healthy calls);
    - `yielded`, `yielded_per_call` and `ended`: chunks yielded in all and per call, and
      generations closed or finished, so a test can see a piece cut short and its generator
      closed.

    Its frames are as wide as the fake model's codebooks, so piece 0 of a session with no
    reference yields a usable anchor for the pieces after it.
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        close_delay: float = 0.0,
        fail_first_call_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Room for 1,500-frame pieces: the default 1,024-token context would clamp them.
        kwargs.setdefault("config", FakeStreamingConfig(max_seq_len=4096))
        chunks = kwargs.get("chunks", 2)
        frames = chunks * kwargs.get("frames_per_chunk", 1)
        kwargs.setdefault("frames", [torch.full((CODEC_CODEBOOKS,), 5) for _ in range(frames)])
        super().__init__(**kwargs)
        self.delay = delay
        self.close_delay = close_delay
        self.fail_first_call_after = fail_first_call_after
        self.yielded = 0
        self.yielded_per_call: list[int] = []
        self.ended = 0
        self._started = 0
        self.tokenizer = FakeTokenizer()
        self.model = model_with_codec_facts()
        self.audio_tokenizer = FakeCodec()

    def iter_audio_chunks(self, inputs: dict[str, Any], **kwargs: Any) -> Iterator[Any]:
        return self._paced(super().iter_audio_chunks(inputs, **kwargs))

    def _paced(self, inner: Iterator[Any]) -> Iterator[Any]:
        call = self._started
        self._started += 1
        self.yielded_per_call.append(0)
        produced = 0
        try:
            for chunk in inner:
                if call == 0 and produced == self.fail_first_call_after:
                    raise RuntimeError("CUDA error: an illegal memory access (fake)")
                if self.delay:
                    time.sleep(self.delay)
                produced += 1
                self.yielded += 1
                self.yielded_per_call[call] += 1
                yield chunk
        finally:
            inner.close()  # type: ignore[attr-defined]
            if self.close_delay:
                time.sleep(self.close_delay)
            self.ended += 1


def big_runtime(**kwargs: Any) -> PacedRuntime:
    return PacedRuntime(
        chunks=BIG_PIECE_CHUNKS, frames_per_chunk=BIG_PIECE_FRAMES_PER_CHUNK, **kwargs
    )


# --- raw WebSocket client ----------------------------------------------------------------


def frame(opcode: int, payload: bytes, *, masked: bool = True, fin: bool = True) -> bytes:
    """One client frame (RFC 6455 section 5.2), masked unless told otherwise."""
    head = bytes([(0x80 if fin else 0) | opcode])
    mask_bit = 0x80 if masked else 0
    size = len(payload)
    if size < 126:
        head += bytes([mask_bit | size])
    elif size < 65536:
        head += bytes([mask_bit | 126]) + struct.pack("!H", size)
    else:
        head += bytes([mask_bit | 127]) + struct.pack("!Q", size)
    if not masked:
        return head + payload
    mask = os.urandom(4)
    data = np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, np.uint8), size)
    return head + mask + data.tobytes()


def close_payload(code: int, reason: str = "") -> bytes:
    return struct.pack("!H", code) + reason.encode()


@dataclass
class Frame:
    opcode: int
    payload: bytes

    @property
    def is_close(self) -> bool:
        return self.opcode == 0x8

    @property
    def close_code(self) -> int | None:
        return struct.unpack("!H", self.payload[:2])[0] if len(self.payload) >= 2 else None

    @property
    def close_reason(self) -> str:
        return self.payload[2:].decode(errors="replace")

    @property
    def event(self) -> dict[str, Any] | None:
        return json.loads(self.payload) if self.opcode == 0x1 else None


@dataclass
class HttpResponse:
    status: int
    headers: list[tuple[str, str]]
    body: bytes

    def values(self, name: str) -> list[str]:
        return [value for key, value in self.headers if key.lower() == name.lower()]

    def json(self) -> Any:
        return json.loads(self.body)


class RawWs:
    """A WebSocket client on a plain socket: sends exactly the bytes a test builds, reads the
    server's frames only when asked (so "stop reading" is simply not calling it)."""

    def __init__(
        self, port: int, *, rcvbuf: int | None = None, host: str = "127.0.0.1"
    ) -> None:
        self.port = port
        self.sock = socket.socket()
        if rcvbuf is not None:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)  # before connect
        self.sock.settimeout(10)
        self.sock.connect((host, port))
        self.buffer = bytearray()

    def request(
        self,
        headers: Sequence[tuple[str, str]] = (),
        *,
        upgrade: bool = True,
        version: str = "13",
        key: str | None = None,
    ) -> bytes:
        lines = ["GET / HTTP/1.1", f"Host: 127.0.0.1:{self.port}"]
        if upgrade:
            if key is None:
                key = base64.b64encode(os.urandom(16)).decode()
            lines += [
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {key}",
                f"Sec-WebSocket-Version: {version}",
            ]
        lines += [f"{name}: {value}" for name, value in headers]
        return ("\r\n".join(lines) + "\r\n\r\n").encode()

    def send_request(self, headers: Sequence[tuple[str, str]] = (), **kwargs: Any) -> None:
        self.sock.sendall(self.request(headers, **kwargs))

    def _fill(self, size: int) -> None:
        while len(self.buffer) < size:
            data = self.sock.recv(65536)
            if not data:
                raise EOFError(f"EOF with {len(self.buffer)} of {size} bytes")
            self.buffer += data

    def read_response(self) -> HttpResponse:
        while b"\r\n\r\n" not in self.buffer:
            self._fill(len(self.buffer) + 1)
        head, _, rest = bytes(self.buffer).partition(b"\r\n\r\n")
        self.buffer = bytearray(rest)
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        headers = [
            (name.strip(), value.strip())
            for name, _, value in (line.partition(":") for line in header_lines)
        ]
        status = int(status_line.split()[1])
        body = b""
        if status != 101:
            [length] = [int(v) for k, v in headers if k.lower() == "content-length"]
            self._fill(length)
            body, self.buffer = bytes(self.buffer[:length]), self.buffer[length:]
        return HttpResponse(status, headers, body)

    def handshake(self, headers: Sequence[tuple[str, str]] = (), **kwargs: Any) -> HttpResponse:
        self.send_request(headers, **kwargs)
        return self.read_response()

    def open(self, headers: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        """Handshake, then read the `ready` event."""
        response = self.handshake(headers)
        assert response.status == 101, response
        ready = self.recv_frame().event
        assert ready is not None and ready["type"] == "ready", ready
        return ready

    def recv_frame(self) -> Frame:
        self._fill(2)
        opcode = self.buffer[0] & 0x0F
        size = self.buffer[1] & 0x7F
        offset = 2
        if size == 126:
            self._fill(4)
            size = struct.unpack("!H", self.buffer[2:4])[0]
            offset = 4
        elif size == 127:
            self._fill(10)
            size = struct.unpack("!Q", self.buffer[2:10])[0]
            offset = 10
        self._fill(offset + size)
        payload = bytes(self.buffer[offset : offset + size])
        del self.buffer[: offset + size]
        return Frame(opcode, payload)

    def send_frame(self, opcode: int, payload: bytes, **kwargs: Any) -> None:
        self.sock.sendall(frame(opcode, payload, **kwargs))

    def send_json(self, message: dict[str, Any]) -> None:
        self.send_frame(0x1, json.dumps(message).encode())

    def frames_until_event(self, kind: str) -> list[Frame]:
        frames: list[Frame] = []
        while True:
            received = self.recv_frame()
            frames.append(received)
            assert not received.is_close, f"closed before {kind!r}: {received}"
            if received.event is not None and received.event["type"] == kind:
                return frames

    def read_until_close(self, *, reply: bool = True) -> tuple[list[Frame], Frame | None]:
        """Read every frame up to the server's close frame, answer it (unless the client closed
        first), read to EOF, and close our end as a client does after the closing handshake
        (the server waits for that FIN). The close frame is None if the connection ended
        without one (the client sees 1006)."""
        frames: list[Frame] = []
        try:
            while True:
                received = self.recv_frame()
                if received.is_close:
                    if reply:
                        self.send_frame(0x8, received.payload[:2])
                    while self.sock.recv(65536):
                        pass
                    self.sock.close()
                    return frames, received
                frames.append(received)
        except (EOFError, ConnectionResetError):
            return frames, None

    def close(self) -> None:
        self.sock.close()


def tcp_state(sock: socket.socket) -> int:
    """The kernel's TCP state for `sock` (first byte of `struct tcp_info`), read without
    consuming any data: ESTABLISHED until the peer's FIN or RST arrives."""
    return sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 8)[0]
