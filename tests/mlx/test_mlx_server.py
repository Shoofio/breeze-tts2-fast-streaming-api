"""The real server with the real MLX runtime, over real HTTP (tasks.md T021, FR-001, FR-005).

One server process serves the whole module: loading the weights takes seconds and a 16 GB Mac has
no room for two models. It runs as a subprocess (`python -m breeze_infer.api`), so these tests
see exactly what a user sees: the events on stdout and the HTTP responses. Every request passes
a `seed`, so a run is repeatable.
"""

from __future__ import annotations

import http.client
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from breeze_infer.settings import DEFAULT_SPLIT_CHARS

pytestmark = pytest.mark.mlx

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 24_000
LOAD_TIMEOUT_S = 180
SHORT_TEXT = "Hello there, this is a short test of the streaming voice."
# Long enough that generation is still running while a test interferes with it.
LONGER_TEXT = " ".join([SHORT_TEXT] * 6)
SENTENCES = (
    "The lighthouse keeper climbed the spiral staircase before dawn.",
    "Fog had settled over the harbor during the night, thick enough to blur the boats.",
    "He lit the lamp and checked the mechanism that turned the great lens.",
    "By midmorning the fog had burned away, and the fishing boats began to leave.",
)


class Server:
    """A running server subprocess: its port, and the events it has printed so far."""

    def __init__(self, process: subprocess.Popen[str], port: int) -> None:
        self.process = process
        self.port = port
        self.events: list[dict[str, Any]] = []
        self.health_statuses: list[int] = []
        threading.Thread(target=self._read_events, daemon=True).start()

    def _read_events(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                self.events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # not an event line

    def events_named(self, name: str, request_id: str | None = None) -> list[dict[str, Any]]:
        return [
            event
            for event in list(self.events)
            if event.get("event") == name
            and (request_id is None or event.get("request_id") == request_id)
        ]

    def wait_for_event(self, name: str, request_id: str, timeout: float = 10) -> dict[str, Any]:
        """Events reach stdout a moment after the response ends, so poll for the one we need."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.events_named(name, request_id)
            if found:
                return found[0]
            time.sleep(0.05)
        raise AssertionError(f"no {name} event for {request_id} within {timeout} s")

    def connection(self, timeout: float = 120) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def health(self) -> tuple[int, dict[str, Any]]:
        connection = self.connection(timeout=5)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def post_speech(
        self, text: str, *, retry_busy_for: float = 0
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """Start a speech request and return it with its status line read, the body unread.

        `retry_busy_for` gives a request that follows an abandoned one time to see the gate free.
        """
        body = urlencode({"text": text, "seed": 1234})
        deadline = time.monotonic() + retry_busy_for
        while True:
            connection = self.connection()
            connection.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/x-www-form-urlencoded"})
            response = connection.getresponse()
            if response.status != 409 or time.monotonic() >= deadline:
                return connection, response
            response.read()
            connection.close()
            time.sleep(0.1)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server(mlx_model: Path) -> Iterator[Server]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "breeze_infer.api",
            str(mlx_model),
            "--backend",
            "mlx",
            "--ws-port",
            "disabled",
            "--port",
            str(port := _free_port()),
            "--host",
            "127.0.0.1",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    running = Server(process, port)
    try:
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        while True:
            assert process.poll() is None, f"server exited with {process.returncode} while loading"
            assert time.monotonic() < deadline, "the model did not load in time"
            try:
                status, _ = running.health()
            except OSError:
                time.sleep(0.05)  # the port is not bound yet
                continue
            running.health_statuses.append(status)
            if status == 200:
                break
            time.sleep(0.05)
        yield running
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _weights_of(snapshot: Path) -> str:
    quantization = json.loads((snapshot / "config.json").read_text()).get("quantization")
    return "8bit" if quantization else "bf16"


def test_health_goes_from_loading_to_ok(server: Server) -> None:
    assert server.health_statuses[0] == 503
    assert server.health_statuses[-1] == 200
    status, body = server.health()
    assert status == 200
    assert body["sample_rate"] == SAMPLE_RATE


def test_model_loaded_event_reports_the_mlx_backend(server: Server, mlx_model: Path) -> None:
    (loaded,) = server.events_named("model.loaded")

    assert loaded["backend"] == "mlx"
    assert loaded["device"] == "mlx:gpu"
    assert loaded["weights"] == _weights_of(mlx_model)
    print(f"\nmodel.loaded: {json.dumps(loaded)}")


def test_speech_streams_pcm_before_generation_ends(server: Server) -> None:
    started = time.monotonic()
    connection, response = server.post_speech(LONGER_TEXT)
    try:
        assert response.status == 200
        first = response.read(4096)
        first_byte_s = time.monotonic() - started
        rest = response.read()
        total_s = time.monotonic() - started
    finally:
        connection.close()

    assert len(first) == 4096
    assert (len(first) + len(rest)) % 2 == 0  # whole s16le samples
    # Streaming means the first bytes came before the response was complete.
    assert first_byte_s < total_s
    print(f"\nfirst bytes after {first_byte_s:.2f} s, response complete after {total_s:.2f} s")


def test_wav_route_starts_with_a_streaming_riff_header(server: Server) -> None:
    connection = server.connection()
    try:
        connection.request("GET", "/v1/audio/speech.wav?" + urlencode({"text": SHORT_TEXT, "seed": 1234}))
        response = connection.getresponse()
        assert response.status == 200
        header = response.read(44)
    finally:
        connection.close()

    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert struct.unpack("<I", header[4:8]) == (0xFFFFFFFF,)
    assert header[36:40] == b"data"
    assert struct.unpack("<I", header[40:44]) == (0xFFFFFFFF,)


def test_a_second_request_while_busy_gets_409(server: Server) -> None:
    connection, response = server.post_speech(LONGER_TEXT, retry_busy_for=10)
    try:
        assert response.status == 200
        response.read(1024)

        second, refused = server.post_speech(SHORT_TEXT)
        try:
            assert refused.status == 409
            assert json.loads(refused.read())["code"] == "busy"
        finally:
            second.close()
    finally:
        connection.close()


def test_closing_the_client_mid_stream_frees_the_gate(server: Server) -> None:
    connection, response = server.post_speech(LONGER_TEXT, retry_busy_for=10)
    assert response.status == 200
    assert response.read(1024)
    connection.close()

    started = time.monotonic()
    connection, response = server.post_speech(SHORT_TEXT, retry_busy_for=5)
    try:
        assert response.status == 200
        assert response.read(1024)
    finally:
        connection.close()
    assert time.monotonic() - started < 5


def test_a_long_text_streams_every_piece(server: Server) -> None:
    text = ""
    while len(text) <= 3 * DEFAULT_SPLIT_CHARS:
        text += " ".join(SENTENCES) + " "

    connection, response = server.post_speech(text.strip(), retry_busy_for=10)
    try:
        assert response.status == 200
        request_id = response.getheader("X-Request-Id")
        assert request_id
        audio = response.read()
    finally:
        connection.close()

    completed = server.wait_for_event("speech.completed", request_id)
    pieces = server.events_named("speech.accepted", request_id)[0]["pieces"]
    done = {e["piece_index"]: e["frames"] for e in server.events_named("speech.piece_done", request_id)}
    assert pieces >= 3
    assert sorted(done) == list(range(pieces))
    assert all(frames > 0 for frames in done.values())
    assert not server.events_named("speech.failed", request_id)
    assert not server.events_named("speech.aborted", request_id)
    # Piece 0's frames come back as the anchor for the later pieces; a skip would mean the
    # path this test exists for did not run.
    assert not server.events_named("speech.anchor_skipped", request_id)
    print(f"\n{pieces} pieces, {len(audio) / 2 / SAMPLE_RATE:.1f} s of audio, {completed}")
