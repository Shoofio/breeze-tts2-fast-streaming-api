"""The real server with the real MLX runtime, over real HTTP (tasks.md T021, T024, T025, T027; FR-001, FR-005, FR-009, FR-010).

One server process serves the whole module: loading the weights takes seconds and a 16 GB Mac has
no room for two models. It runs as a subprocess (`python -m breeze_infer.api`), so these tests
see exactly what a user sees: the events on stdout and the HTTP responses. Every request passes
a `seed`, so a run is repeatable.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from breeze_infer.settings import DEFAULT_SPLIT_CHARS
from tests.ws_helpers import RawWs

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

    def stop(self) -> None:
        """Terminate the process; safe to call again."""
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def events_named(self, name: str, request_id: str | None = None) -> list[dict[str, Any]]:
        return [
            event
            for event in list(self.events)
            if event.get("event") == name
            and (request_id is None or event.get("request_id") == request_id)
        ]

    def wait_for_event(
        self, name: str, request_id: str | None = None, timeout: float = 10
    ) -> dict[str, Any]:
        """Events are parsed on a reader thread a moment after they are printed, so poll."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.events_named(name, request_id)
            if found:
                return found[0]
            time.sleep(0.05)
        which = f" for {request_id}" if request_id else ""
        raise AssertionError(f"no {name} event{which} within {timeout} s")

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


@contextlib.contextmanager
def running_server(mlx_model: Path, *extra_args: str) -> Iterator[Server]:
    """Start the server, wait until it is ready, and always kill it on the way out."""
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
            *extra_args,
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
        running.stop()


@pytest.fixture(scope="module")
def server(mlx_model: Path) -> Iterator[Server]:
    # The WebSocket listener is on for the shared server: the session test needs it, and a
    # second model load would cost seconds and memory a 16 GB Mac does not have to spare. The
    # later `--ws-port disabled` in `running_server` is overridden by this one (the last wins).
    with running_server(mlx_model, "--ws-port", str(_free_port())) as running:
        yield running


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
    loaded = server.wait_for_event("model.loaded")
    assert len(server.events_named("model.loaded")) == 1

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
    # Streaming means the first bytes came well before the end: a server that buffered the
    # whole response would deliver them at about the same moment as the last ones.
    assert first_byte_s < 0.5 * total_s
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


# --- Voice features (T024, T025) -------------------------------------------------------------

REF_TEXT = SHORT_TEXT
MIN_AUDIO_S = 0.5


def _multipart(fields: dict[str, str], wav: bytes | None = None) -> tuple[bytes, str]:
    boundary = "breezeboundary7d3f"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    if wav is not None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="ref_audio"; filename="ref.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + wav
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post(server: Server, path: str, fields: dict[str, str], wav: bytes | None = None) -> tuple[int, bytes]:
    body, content_type = _multipart(fields, wav)
    connection = server.connection()
    try:
        connection.request("POST", path, body, {"Content-Type": content_type})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _speech_seconds(server: Server, fields: dict[str, str], wav: bytes | None = None) -> float:
    status, pcm = _post(server, "/v1/audio/speech", {"text": SHORT_TEXT, "seed": "1234", **fields}, wav)
    assert status == 200, pcm[:300]
    assert len(pcm) % 2 == 0
    return len(pcm) / 2 / SAMPLE_RATE


def _reference_wav(server: Server) -> bytes:
    """A short reference clip made by the server itself, so the tests need no audio file.

    The `.wav` route streams a header with unknown sizes (0xFFFFFFFF), which uploads would
    reject, so the PCM after the 44-byte header is wrapped in a proper WAV.
    """
    connection = server.connection()
    try:
        connection.request("GET", "/v1/audio/speech.wav?" + urlencode({"text": REF_TEXT, "seed": 1234}))
        response = connection.getresponse()
        assert response.status == 200
        pcm = response.read()[44:]
    finally:
        connection.close()
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm)
    return out.getvalue()


def test_voice_clone_design_and_direction_return_audio(server: Server) -> None:
    reference = _reference_wav(server)
    clone = _speech_seconds(server, {"ref_text": REF_TEXT}, reference)
    design = _speech_seconds(
        server, {"instruction": "A deep, calm older male voice.", "cfg_scale": "4"}
    )
    # cfg_scale 4 as in the docs/api.md direction example: the guidance branch then carries the
    # reference too (cfg_negative_input_values), a path neither clone nor design takes.
    direction = _speech_seconds(
        server,
        {
            "ref_text": REF_TEXT,
            "instruction": "Speak slowly with a restrained, serious tone.",
            "cfg_scale": "4",
        },
        reference,
    )

    assert min(clone, design, direction) > MIN_AUDIO_S
    print(f"\nclone {clone:.2f} s, design {design:.2f} s, direction {direction:.2f} s")


# --- WebSocket (T027) ------------------------------------------------------------------------


def test_websocket_session_streams_audio_then_done(server: Server) -> None:
    """The documented order (docs/api.md, WebSocket API): ready, started, speaking, binary PCM
    frames, then done, which comes after every frame of the piece."""
    status, health = server.health()
    assert status == 200
    ws_port = health["ws_port"]
    assert ws_port > 0, "the module server was started with the WebSocket listener on"

    client = RawWs(ws_port)
    try:
        ready = client.open()
        assert (ready["sample_rate"], ready["format"]) == (SAMPLE_RATE, "s16le")

        client.send_json({"type": "start", "seed": 1234})
        started = client.recv_frame().event
        assert started == {"type": "started", "voice_id": ""}

        client.send_json({"type": "end", "text": SHORT_TEXT})
        frames = client.frames_until_event("done")
    finally:
        client.close()

    events = [f.event["type"] for f in frames if f.event is not None]
    assert events == ["speaking", "done"]  # no `queued`, `error` or `cancelled` on the way
    speaking = next(f.event for f in frames if f.event is not None)
    assert speaking["text"] == SHORT_TEXT

    audio_indexes = [i for i, f in enumerate(frames) if f.opcode == 0x2]
    assert audio_indexes, "no binary audio frames arrived"
    # The piece's audio sits between `speaking` and `done`, with nothing else mixed in.
    assert audio_indexes == list(range(1, len(frames) - 1))
    pcm = b"".join(frames[i].payload for i in audio_indexes)
    assert len(pcm) % 2 == 0  # whole s16le samples
    assert len(pcm) / 2 / SAMPLE_RATE > MIN_AUDIO_S
    print(
        f"\nws sequence: ready, started, speaking, {len(audio_indexes)} audio frames, done; "
        f"{len(pcm) / 2 / SAMPLE_RATE:.2f} s of audio"
    )


def test_saved_voices_survive_a_restart_and_a_foreign_fingerprint_is_skipped(
    server: Server, mlx_model: Path, tmp_path: Path
) -> None:
    reference = _reference_wav(server)
    # One 16 GB Mac holds one model: this test owns the machine from here, so the module
    # server stops. It must be the last test in the module.
    server.stop()

    voices_dir = tmp_path / "voices"
    with running_server(mlx_model, "--voices-dir", str(voices_dir)) as first:
        status, body = _post(first, "/v1/voices", {"name": "alice", "ref_text": REF_TEXT}, reference)
        assert status == 200, body
        assert json.loads(body)["saved"] is True
        assert _speech_seconds(first, {"voice_id": "alice"}) > MIN_AUDIO_S

    # The same voice as another codec would have saved it.
    saved = json.loads((voices_dir / "alice.voice.json").read_text())
    saved["id"] = "foreign"
    saved["codec_fingerprint"] = "not-this-codec"
    (voices_dir / "foreign.voice.json").write_text(json.dumps(saved))

    with running_server(mlx_model, "--voices-dir", str(voices_dir)) as second:
        connection = second.connection()
        try:
            connection.request("GET", "/v1/voices")
            listed = [voice["id"] for voice in json.loads(connection.getresponse().read())]
        finally:
            connection.close()
        assert listed == ["alice"]
        assert _speech_seconds(second, {"voice_id": "alice"}) > MIN_AUDIO_S

        skipped = second.wait_for_event("voice.skipped")
        assert skipped["file"] == "foreign.voice.json"
        assert "codec_fingerprint" in skipped["reason"]
        loaded = second.wait_for_event("voices.loaded")
        assert (loaded["loaded"], loaded["skipped"]) == (1, 1)
        print(f"\nvoices.loaded: {json.dumps(loaded)}\nvoice.skipped: {json.dumps(skipped)}")

        status, body = _post(second, "/v1/audio/speech", {"text": SHORT_TEXT, "voice_id": "foreign"})
        assert status == 404
        assert json.loads(body)["code"] == "unknown_voice"
