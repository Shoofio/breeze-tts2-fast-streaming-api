"""`POST`/`GET`/`DELETE /v1/voices` and their startup wiring (specs/003-cpp-compatible-api/
tasks.md T059, T065; contracts/http-api.md; data-model.md "Voice").

Every app here is built the way `main()` builds it: `api.create_app`, then
`api.load_in_background`, which loads the (fake) model, opens the voice directory through
`Components.open_voices` on a worker thread and only then marks the server ready. The voice
directory is a real `tmp_path`; the model edge is `tests/fakes.py`'s GPU-free stand-ins (the
Principle V deviation recorded there).

The SillyTavern extension's needs (research/live-phase3.md) are pinned here too: `GET` lists
`id`, `saved` and `seconds`; `POST` returns the record; a listed id, URL-encoded the way
`encodeURIComponent` does it, is exactly what `DELETE` accepts; every non-2xx has
`body.error`.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import urllib.parse
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from breeze_infer import api
from breeze_infer.api import Components, create_app, load_in_background
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import UNNAMED_VOICE_CAP
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from breeze_infer.voice_registry import unnamed_id
from tests.fakes import (
    CODEC_SAMPLE_RATE,
    CODEC_SAMPLES_PER_FRAME,
    FakeCodec,
    FakeRuntime,
    FakeTokenizer,
    RecordingEvents,
    codec_frame_count,
    model_with_codec_facts,
    open_no_voices,
)

VOICES = "/v1/voices"
MODEL_DIR = str(Path(__file__).parent)  # never opened: the model load is faked
FINGERPRINT = "f" * 64
RECORD_KEYS = {"id", "frames", "seconds", "encode_ms", "saved", "ref_text"}
# One encode, as the injected clock times it: 812.3 ms, reported as the integer 812.
ENCODE_SECONDS = 0.8123


class _EncodeClock:
    """Advances `ENCODE_SECONDS` per reading, so every encode (two readings) lasts exactly
    that long."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        self._now += ENCODE_SECONDS
        return self._now


def _runtime() -> FakeRuntime:
    """A `FakeRuntime` with what the voice routes and `api.open_voices` read off the real
    `FastBreezeStreamingRuntime`: the audio tokenizer, the model config (codebooks, codebook
    size, and the backbone's attention shape for the prefix cache's KV estimate) and `dtype`."""
    runtime = FakeRuntime()
    runtime.tokenizer = FakeTokenizer()
    runtime.model = model_with_codec_facts()
    config = runtime.model.config
    config.num_hidden_layers = 28
    config.num_attention_heads = 16
    config.num_key_value_heads = 8
    config.hidden_size = 2048
    config.head_dim = 128
    runtime.dtype = torch.bfloat16
    runtime.audio_tokenizer = FakeCodec()
    return runtime


def _wav(seconds: float = 0.5, sample_rate: int = 16000, pitch: float = 220.0) -> bytes:
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * pitch * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class _Server:
    """One started app: its components, client, fake runtime and recorded events."""

    def __init__(self, voices_dir: Path, *, runtime: FakeRuntime | None = None) -> None:
        self.voices_dir = voices_dir
        self.events = RecordingEvents()
        self.runtime = runtime if runtime is not None else _runtime()
        nonces = iter(range(1_000_000))
        self.components = Components(
            settings=settings_from_args([MODEL_DIR, "--voices-dir", str(voices_dir)]),
            events=self.events,  # type: ignore[arg-type]
            gate=GpuGate(),
            gpu=GpuThread("cpu", lambda _device: None),
            readiness=Readiness(),
            ws_port=lambda: 0,
            cpu_tokenizer=CpuTokenizer(),
            open_voices=partial(
                api.open_voices,
                voices_dir=voices_dir,
                events=self.events,
                codec_fingerprint=FINGERPRINT,
                now=lambda: datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
                nonce=lambda: f"n{next(nonces)}",
                encode_clock=_EncodeClock(),
            ),
        )
        loaded = LoadedModel(
            runtime=self.runtime,
            report={},
            cpu_tokenizer=FakeTokenizer(),
            sizing_tokenizer=FakeTokenizer(),
        )
        server = SimpleNamespace(should_exit=False)
        assert asyncio.run(load_in_background(self.components, lambda: loaded, server))
        self.client = TestClient(create_app(self.components), raise_server_exceptions=False)

    @property
    def services(self) -> Any:
        return self.components.voices.get()

    def post(
        self,
        *,
        audio: bytes | None = None,
        ref_text: str | None = "hello there",
        name: str | None = None,
        **extra: str,
    ) -> Any:
        data = dict(extra)
        if ref_text is not None:
            data["ref_text"] = ref_text
        if name is not None:
            data["name"] = name
        if audio is None:
            return self.client.post(VOICES, data=data)
        return self.client.post(
            VOICES, data=data, files={"ref_audio": ("ref.wav", audio, "audio/wav")}
        )

    def delete(self, voice_id: str) -> Any:
        return self.client.delete(f"{VOICES}/{_encode_uri_component(voice_id)}")

    def listing(self) -> list[dict[str, Any]]:
        response = self.client.get(VOICES)
        assert response.status_code == 200
        return response.json()

    def gate_is_free(self) -> bool:
        lease = self.components.gate.try_acquire()
        if lease is None:
            return False
        lease.release()
        return True

    def event_names(self) -> list[str]:
        return [name for name, _fields in self.events.calls]

    def events_named(self, name: str) -> list[dict[str, Any]]:
        return [fields for event, fields in self.events.calls if event == name]

    def close(self) -> None:
        self.components.gpu.shutdown()


def _encode_uri_component(value: str) -> str:
    """JavaScript's `encodeURIComponent`: everything but `A-Z a-z 0-9 - _ . ! ~ * ' ( )` is
    percent-encoded (UTF-8), which is what the SillyTavern extension sends in the path."""
    return urllib.parse.quote(value, safe="-_.!~*'()")


@pytest.fixture()
def start(tmp_path: Path) -> Iterator[Any]:
    """`start()` starts a server on `tmp_path / "voices"` (or `voices_dir`); every server a
    test starts is shut down after it."""
    started: list[_Server] = []

    def factory(voices_dir: Path | None = None, **kwargs: Any) -> _Server:
        server = _Server(voices_dir if voices_dir is not None else tmp_path / "voices", **kwargs)
        started.append(server)
        return server

    yield factory
    for server in started:
        server.close()


def _assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """Every non-2xx is the envelope, and SillyTavern reads `body.error`."""
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code
    assert isinstance(body["error"], str) and body["error"]
    return body


def _expected_frames(wav: bytes) -> int:
    samples, sample_rate = sf.read(io.BytesIO(wav))
    return codec_frame_count(len(samples), sample_rate)


def _write_invalid_voice_file(voices_dir: Path, stem: str) -> Path:
    voices_dir.mkdir(parents=True, exist_ok=True)
    path = voices_dir / f"{stem}.voice.json"
    path.write_text("{not json", encoding="utf-8")
    return path


# --------------------------------------------------------------------- POST: the 200 shape


def test_named_post_returns_the_record_with_two_decimal_seconds_and_integer_encode_ms(
    start: Any,
) -> None:
    server = start()
    wav = _wav(0.5)

    response = server.post(audio=wav, name="alice", ref_text="Hello there.")

    assert response.status_code == 200  # 200, not 201: kept from C++
    body = response.json()
    assert set(body) == RECORD_KEYS
    frames = _expected_frames(wav)
    assert body["id"] == "alice"
    assert body["frames"] == frames
    assert body["seconds"] == round(frames * CODEC_SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE, 2)
    assert round(body["seconds"], 2) == body["seconds"]
    assert body["encode_ms"] == 812
    assert isinstance(body["encode_ms"], int)
    assert body["saved"] is True
    assert body["ref_text"] == "Hello there."
    assert server.gate_is_free()


def test_named_post_writes_the_voice_file_with_the_codec_fingerprint(start: Any) -> None:
    server = start()

    assert server.post(audio=_wav(), name="alice").status_code == 200

    stored = json.loads((server.voices_dir / "alice.voice.json").read_text("utf-8"))
    assert stored["id"] == "alice"
    assert stored["codec_fingerprint"] == FINGERPRINT
    assert stored["encode_ms"] == 812
    assert stored["created_at"] == "2026-09-26T12:00:00Z"


def test_unnamed_post_returns_a_v_id_and_writes_no_file(start: Any) -> None:
    server = start()
    wav = _wav()

    response = server.post(audio=wav, ref_text="unnamed one")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == RECORD_KEYS
    assert body["id"] == unnamed_id(wav, "unnamed one")
    assert body["saved"] is False
    assert list(server.voices_dir.glob("*.voice.json")) == []


def test_the_encode_runs_on_the_gpu_thread_through_encode_prompt_waveform(start: Any) -> None:
    server = start()
    threads: list[str] = []
    codec = server.runtime.audio_tokenizer
    original = codec.encode

    def recording_encode(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    codec.encode = recording_encode

    assert server.post(audio=_wav(), name="alice").status_code == 200

    assert len(threads) == 1 and threads[0].startswith("breeze-gpu")


def test_voice_created_is_emitted_with_the_request_id(start: Any) -> None:
    server = start()

    response = server.post(audio=_wav(), name="alice")

    [created] = server.events_named("voice.created")
    assert created["request_id"] == response.headers["x-request-id"]
    assert created["voice_id"] == "alice"
    assert created["saved"] is True


# --------------------------------------------------------------- POST: the order of checks


@pytest.mark.parametrize(
    ("post", "code"),
    [
        ({"audio": None, "ref_text": "hi"}, "voice_fields_required"),
        ({"ref_text": None}, "voice_fields_required"),
        ({"ref_text": "   "}, "voice_fields_required"),
        # Missing fields come before the name (contract step 2 lists them first).
        ({"audio": None, "name": "bad name"}, "voice_fields_required"),
    ],
    ids=["no_ref_audio", "no_ref_text", "blank_ref_text", "missing_before_bad_name"],
)
def test_missing_fields_get_400_voice_fields_required(
    start: Any, post: dict[str, Any], code: str
) -> None:
    server = start()
    post = {"audio": _wav(), **post}
    _assert_error(server.post(**post), 400, code)


@pytest.mark.parametrize(
    "name",
    ["bad name", "v_reserved", "V_reserved", "x" * 65, "café", "a/b", "a.b"],
    ids=["space", "v_prefix", "V_prefix", "too_long", "non_ascii", "slash", "dot"],
)
def test_a_bad_name_gets_400_invalid_name_before_the_audio_is_decoded(
    start: Any, name: str
) -> None:
    server = start()
    _assert_error(server.post(audio=b"not audio", name=name), 400, "invalid_name")


def test_ref_text_with_control_characters_is_a_field_error(start: Any) -> None:
    """BC-46, the same shared rule as speech's `ref_text` (http_fields)."""
    server = start()
    _assert_error(server.post(audio=b"not audio", ref_text="hi\x07"), 400, "invalid_field")


def test_ref_text_over_2000_characters_is_a_field_error(start: Any) -> None:
    server = start()
    _assert_error(server.post(audio=b"not audio", ref_text="a" * 2001), 400, "invalid_field")


def test_a_duplicate_field_gets_400_duplicate_field(start: Any) -> None:
    server = start()
    response = server.client.post(
        f"{VOICES}?name=alice",
        data={"name": "alice", "ref_text": "hi"},
        files={"ref_audio": ("ref.wav", _wav(), "audio/wav")},
    )
    _assert_error(response, 400, "duplicate_field")


@pytest.mark.parametrize("name", ["alice", "ALICE", "Alice"])
def test_bc_27_existing_name_gets_409_voice_exists(start: Any, name: str) -> None:
    """BC-27 (and BC-26's case rule): a name already saved, ignoring case, is refused before
    anything is decoded, encoded or written -- even with bad audio and a busy GPU."""
    server = start()
    assert server.post(audio=_wav(), name="alice", ref_text="first").status_code == 200
    before = (server.voices_dir / "alice.voice.json").read_bytes()
    encodes = server.runtime.audio_tokenizer.encode_calls
    lease = server.components.gate.try_acquire()
    assert lease is not None
    try:
        _assert_error(server.post(audio=b"not audio", name=name), 409, "voice_exists")
    finally:
        lease.release()

    assert server.runtime.audio_tokenizer.encode_calls == encodes
    assert (server.voices_dir / "alice.voice.json").read_bytes() == before
    assert [record["id"] for record in server.listing()] == ["alice"]


def test_a_name_held_by_a_skipped_file_gets_409_voice_exists(start: Any, tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    _write_invalid_voice_file(voices_dir, "bob")
    server = start(voices_dir)

    _assert_error(server.post(audio=_wav(), name="BOB"), 409, "voice_exists")
    assert server.listing() == []


def test_an_unnamed_duplicate_returns_200_without_the_gate_even_while_busy(start: Any) -> None:
    server = start()
    wav = _wav()
    first = server.post(audio=wav, ref_text="same text")
    assert first.status_code == 200
    encodes = server.runtime.audio_tokenizer.encode_calls
    lease = server.components.gate.try_acquire()
    assert lease is not None
    try:
        again = server.post(audio=wav, ref_text="same text")
    finally:
        lease.release()

    assert again.status_code == 200
    assert again.json() == first.json()
    assert server.runtime.audio_tokenizer.encode_calls == encodes
    assert [record["id"] for record in server.listing()] == [first.json()["id"]]
    assert len(server.events_named("voice.created")) == 1  # nothing new was created


def test_an_unnamed_duplicate_is_returned_before_the_audio_is_decoded(start: Any) -> None:
    """Step 4 comes before step 5: the id is a hash of the bytes, so identical bytes that
    registered once are the same voice."""
    server = start()
    wav = _wav()
    assert server.post(audio=wav, ref_text="t").status_code == 200

    def no_decode(_blob: bytes) -> Any:
        raise AssertionError("a dedupe hit must not decode")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("breeze_infer.reference_audio.decode", no_decode)
        assert server.post(audio=wav, ref_text="t").status_code == 200


@pytest.mark.parametrize(
    ("audio", "code"),
    [
        (b"not audio", "invalid_audio"),
        (b"", "invalid_audio"),
        (None, "audio_too_long"),
        (None, "audio_too_short"),
    ],
    ids=["garbage", "empty_part", "too_long", "too_short"],
)
def test_decode_errors_come_before_busy(start: Any, audio: bytes | None, code: str) -> None:
    server = start()
    if code == "audio_too_long":
        audio = _wav(31.0, sample_rate=8000)
    elif code == "audio_too_short":
        audio = _wav(0.01)
    lease = server.components.gate.try_acquire()
    assert lease is not None
    try:
        _assert_error(server.post(audio=audio, name="alice"), 400, code)
    finally:
        lease.release()


def test_409_busy_when_the_gate_is_held(start: Any) -> None:
    server = start()
    lease = server.components.gate.try_acquire()
    assert lease is not None
    try:
        _assert_error(server.post(audio=_wav(), name="alice"), 409, "busy")
        _assert_error(server.post(audio=_wav(), ref_text="unnamed"), 409, "busy")
        # The holder's lease was not touched.
        assert lease.held
    finally:
        lease.release()

    assert server.runtime.audio_tokenizer.encode_calls == 0
    assert server.listing() == []
    assert list(server.voices_dir.glob("*.voice.json")) == []


def test_a_name_taken_on_disk_at_commit_gets_409_voice_exists(start: Any) -> None:
    """The commit-time re-check: a file that appeared after the scan (another process, a
    hand copy) passes step 3's in-memory check, is caught by the store's no-overwrite commit
    under its lock, and nothing is registered. The lease is released."""
    server = start()
    intruder = server.voices_dir / "carol.voice.json"
    intruder.write_text("someone else's", encoding="utf-8")

    _assert_error(server.post(audio=_wav(), name="carol"), 409, "voice_exists")

    assert intruder.read_text("utf-8") == "someone else's"
    assert server.listing() == []
    assert server.gate_is_free()


def test_a_write_failure_gets_500_voice_write_failed_and_closes_the_connection(
    start: Any,
) -> None:
    server = start()

    def failing_create(_voice: Any) -> Any:
        raise OSError(28, "No space left on device")

    server.services.store.create = failing_create

    response = server.post(audio=_wav(), name="alice")

    _assert_error(response, 500, "voice_write_failed")
    assert response.headers["connection"] == "close"  # every 500 closes the connection
    assert server.listing() == []
    assert server.gate_is_free()
    [failed] = server.events_named("request.failed")
    assert failed["request_id"] == response.headers["x-request-id"]
    assert "No space left on device" in failed["error"]


def test_the_store_is_called_off_the_event_loop_under_the_write_lock(start: Any) -> None:
    """VoiceStore holds its lock across a rename and an fsync: never on the event loop. Each
    store change runs under `write_lock`, with the registry update that goes with it."""
    server = start()
    on_loop: list[bool] = []
    locked: list[bool] = []
    store = server.services.store

    def spy(method: Any) -> Any:
        def call(*args: Any) -> Any:
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            locked.append(server.services.write_lock.locked())
            return method(*args)

        return call

    store.create = spy(store.create)
    store.remove = spy(store.remove)

    assert server.post(audio=_wav(), name="alice").status_code == 200
    assert server.delete("alice").status_code == 200
    assert on_loop == [False, False]
    assert locked == [True, True]


def test_evicting_an_unnamed_voice_drops_its_cached_prefix(start: Any) -> None:
    server = start()
    invalidated: list[tuple[str, str | None]] = []
    cache = server.services.prefix_cache
    original = cache.invalidate

    def spy(voice_id: str, *, request_id: str | None = None) -> bool:
        invalidated.append((voice_id, request_id))
        return original(voice_id, request_id=request_id)

    cache.invalidate = spy
    ids = []
    for index in range(UNNAMED_VOICE_CAP + 1):
        response = server.post(audio=_wav(), ref_text=f"voice {index}")
        assert response.status_code == 200
        ids.append(response.json()["id"])

    assert invalidated == [(ids[0], response.headers["x-request-id"])]
    assert [record["id"] for record in server.listing()] == ids[1:]


# ------------------------------------------------------------------------------------ GET


def test_get_lists_saved_voices_sorted_by_id_then_unnamed_in_registration_order(
    start: Any, tmp_path: Path
) -> None:
    voices_dir = tmp_path / "voices"
    _write_invalid_voice_file(voices_dir, "broken")  # skipped: never listed
    server = start(voices_dir)
    unnamed = []
    for text in ("second", "first"):
        response = server.post(audio=_wav(), ref_text=text)
        unnamed.append(response.json()["id"])
    for name in ("zed", "Alice", "bob"):
        assert server.post(audio=_wav(), name=name).status_code == 200

    listing = server.listing()

    assert [record["id"] for record in listing] == ["Alice", "bob", "zed", *unnamed]
    for record in listing:
        assert set(record) == RECORD_KEYS
        assert isinstance(record["seconds"], float)
        assert isinstance(record["encode_ms"], int)
    assert [record["saved"] for record in listing] == [True, True, True, False, False]


def test_get_has_what_sillytavern_reads(start: Any) -> None:
    server = start()
    posted = server.post(audio=_wav(), name="alice").json()

    [listed] = server.listing()

    for key in ("id", "saved", "seconds"):
        assert listed[key] == posted[key]


def test_voice_routes_answer_503_loading_before_the_scan_has_finished(tmp_path: Path) -> None:
    components = Components(
        settings=settings_from_args([MODEL_DIR]),
        events=Emitter(io.StringIO(), lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_no_voices,
    )
    try:
        client = TestClient(create_app(components))
        for response in (
            client.get(VOICES),
            client.post(VOICES, data={"ref_text": "x"}),
            client.delete(f"{VOICES}/alice"),
        ):
            assert response.status_code == 503
            assert response.json()["code"] == "loading"
    finally:
        components.gpu.shutdown()


# --------------------------------------------------------------------------------- DELETE


def test_bc_28_delete_removes_the_file_and_it_stays_gone_after_restart(
    start: Any, tmp_path: Path
) -> None:
    voices_dir = tmp_path / "voices"
    server = start(voices_dir)
    assert server.post(audio=_wav(), name="alice").status_code == 200
    assert server.post(audio=_wav(), name="keep").status_code == 200
    assert (voices_dir / "alice.voice.json").exists()

    response = server.delete("alice")

    assert response.status_code == 200
    assert response.json() == {"deleted": "alice", "file_kept": False}
    assert not (voices_dir / "alice.voice.json").exists()
    assert [path.name for path in voices_dir.iterdir()] == ["keep.voice.json"]

    restarted = start(voices_dir)  # a new app on the same directory
    assert [record["id"] for record in restarted.listing()] == ["keep"]
    _assert_error(restarted.delete("alice"), 404, "unknown_voice")


def test_saved_voices_come_back_after_restart(start: Any, tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    server = start(voices_dir)
    posted = server.post(audio=_wav(), name="alice", ref_text="kept").json()
    assert server.post(audio=_wav(), ref_text="unnamed").status_code == 200

    restarted = start(voices_dir)

    assert restarted.listing() == [posted]  # unnamed voices are gone


@pytest.mark.parametrize("kind", ["saved", "unnamed", "skipped"])
def test_file_kept_is_always_false(start: Any, tmp_path: Path, kind: str) -> None:
    voices_dir = tmp_path / "voices"
    if kind == "skipped":
        _write_invalid_voice_file(voices_dir, "broken")
    server = start(voices_dir)
    if kind == "saved":
        voice_id = server.post(audio=_wav(), name="alice").json()["id"]
    elif kind == "unnamed":
        voice_id = server.post(audio=_wav(), ref_text="t").json()["id"]
    else:
        voice_id = "broken"

    response = server.delete(voice_id)

    assert response.status_code == 200
    assert response.json() == {"deleted": voice_id, "file_kept": False}


@pytest.mark.parametrize("voice_id", ["nobody", "v_0123456789abcdef", "ALICE"])
def test_an_unknown_id_gets_404_unknown_voice(start: Any, voice_id: str) -> None:
    """`ALICE` for a saved `alice`: DELETE matches the listed id exactly; only create
    ignores case."""
    server = start()
    assert server.post(audio=_wav(), name="alice").status_code == 200

    body = _assert_error(server.delete(voice_id), 404, "unknown_voice")

    assert body["error"] == "unknown voice_id"
    assert [record["id"] for record in server.listing()] == ["alice"]


def test_a_second_delete_gets_404(start: Any) -> None:
    server = start()
    voice_id = server.post(audio=_wav(), ref_text="t").json()["id"]
    assert server.delete(voice_id).status_code == 200
    _assert_error(server.delete(voice_id), 404, "unknown_voice")


def test_delete_never_checks_busy(start: Any) -> None:
    server = start()
    assert server.post(audio=_wav(), name="alice").status_code == 200
    lease = server.components.gate.try_acquire()
    assert lease is not None
    try:
        assert server.delete("alice").status_code == 200
    finally:
        lease.release()


def test_deleting_a_skipped_files_name_removes_the_file_and_releases_the_name(
    start: Any, tmp_path: Path
) -> None:
    voices_dir = tmp_path / "voices"
    skipped = _write_invalid_voice_file(voices_dir, "bob")
    server = start(voices_dir)
    _assert_error(server.post(audio=_wav(), name="bob"), 409, "voice_exists")

    assert server.delete("bob").status_code == 200

    assert not skipped.exists()
    assert server.post(audio=_wav(), name="Bob").status_code == 200
    assert [record["id"] for record in server.listing()] == ["Bob"]


def test_a_skipped_directory_gets_500_voice_delete_failed_and_stays_reserved(
    start: Any, tmp_path: Path
) -> None:
    voices_dir = tmp_path / "voices"
    (voices_dir / "bob.voice.json").mkdir(parents=True)
    server = start(voices_dir)

    response = server.delete("bob")

    _assert_error(response, 500, "voice_delete_failed")
    assert response.headers["connection"] == "close"
    assert (voices_dir / "bob.voice.json").is_dir()
    _assert_error(server.post(audio=_wav(), name="bob"), 409, "voice_exists")


def test_a_store_failure_on_delete_gets_500_and_the_voice_stays_registered(start: Any) -> None:
    server = start()
    assert server.post(audio=_wav(), name="alice").status_code == 200

    def failing_remove(_voice_id: str) -> bool:
        raise PermissionError(13, "Permission denied")

    server.services.store.remove = failing_remove

    response = server.delete("alice")

    _assert_error(response, 500, "voice_delete_failed")
    assert response.headers["connection"] == "close"
    assert [record["id"] for record in server.listing()] == ["alice"]
    assert server.event_names().count("voice.deleted") == 0
    [failed] = server.events_named("request.failed")
    assert failed["request_id"] == response.headers["x-request-id"]


def test_delete_drops_the_cached_prefix_with_the_delete_request_id(start: Any) -> None:
    server = start()
    voice_id = server.post(audio=_wav(), ref_text="t").json()["id"]
    invalidated: list[tuple[str, str | None]] = []
    cache = server.services.prefix_cache
    original = cache.invalidate

    def spy(voice_id: str, *, request_id: str | None = None) -> bool:
        invalidated.append((voice_id, request_id))
        return original(voice_id, request_id=request_id)

    cache.invalidate = spy

    response = server.delete(voice_id)

    assert invalidated == [(voice_id, response.headers["x-request-id"])]
    [deleted] = server.events_named("voice.deleted")
    assert deleted["request_id"] == response.headers["x-request-id"]
    assert deleted["voice_id"] == voice_id


# ------------------------------------------------------------ SillyTavern's id round trip


@pytest.mark.parametrize("name", ["Az-09_x", "-", "_", "a" * 64, None])
def test_a_listed_id_url_encoded_like_encodeuricomponent_is_what_delete_accepts(
    start: Any, name: str | None
) -> None:
    """Names are `[A-Za-z0-9_-]` and unnamed ids `v_` + hex, none of which
    `encodeURIComponent` escapes, so the encoded path is the id itself; this pins that the
    round trip works for every character class the rules allow, and for an unnamed id."""
    server = start()
    posted = server.post(audio=_wav(), name=name, ref_text="round trip")
    assert posted.status_code == 200
    [listed] = server.listing()
    assert listed["id"] == posted.json()["id"]
    assert _encode_uri_component(listed["id"]) == listed["id"]

    response = server.client.delete(f"{VOICES}/{_encode_uri_component(listed['id'])}")

    assert response.status_code == 200
    assert response.json()["deleted"] == listed["id"]
    assert server.listing() == []


# ------------------------------------------------------------- T065: startup and wiring


def test_prefix_bytes_per_token_matches_the_checkpoint_estimate() -> None:
    """data-model.md: 114,688 B per token for this checkpoint (28 layers x 8 KV heads x 128
    head_dim x 2 (key, value) x 2 bytes of bfloat16)."""
    assert api.prefix_bytes_per_token(_runtime()) == 114_688


def test_prefix_bytes_per_token_derives_head_dim_when_the_config_has_none() -> None:
    runtime = _runtime()
    config = runtime.model.config
    config.head_dim = None
    config.hidden_size = 4096
    config.num_attention_heads = 32  # 4096 / 32 = 128
    runtime.dtype = torch.float32
    assert api.prefix_bytes_per_token(runtime) == 2 * 28 * 8 * 128 * 4


def test_the_scan_runs_after_the_load_on_a_worker_thread_and_before_ready(tmp_path: Path) -> None:
    events = RecordingEvents()
    readiness = Readiness()
    order: list[str] = []
    scan_started = threading.Event()
    finish_scan = threading.Event()
    runtime = _runtime()

    def load() -> LoadedModel:
        order.append(f"load on {threading.current_thread().name}")
        return LoadedModel(
            runtime=runtime, report={}, cpu_tokenizer=FakeTokenizer(), sizing_tokenizer=FakeTokenizer()
        )

    def open_voices(loaded_runtime: Any) -> Any:
        assert loaded_runtime is runtime
        order.append(f"scan on {threading.current_thread().name}")
        scan_started.set()
        assert finish_scan.wait(10)
        return api.open_voices(
            loaded_runtime,
            voices_dir=tmp_path,
            events=events,
            codec_fingerprint=FINGERPRINT,
            now=lambda: datetime(2026, 9, 26, tzinfo=timezone.utc),
            nonce=lambda: "n",
            encode_clock=_EncodeClock(),
        )

    components = Components(
        settings=settings_from_args([MODEL_DIR]),
        events=events,  # type: ignore[arg-type]
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_voices,
    )

    async def scenario() -> bool:
        loading = asyncio.create_task(
            load_in_background(components, load, SimpleNamespace(should_exit=False))
        )
        while not scan_started.is_set():
            await asyncio.sleep(0.01)
        # The loop is not blocked by the scan, and the server is not ready during it.
        await asyncio.sleep(0.05)
        assert readiness.runtime is None
        finish_scan.set()
        return await loading

    try:
        assert asyncio.run(scenario())
    finally:
        components.gpu.shutdown()

    assert order[0].startswith("load on breeze-gpu")
    assert order[1].startswith("scan on ") and "breeze-gpu" not in order[1]
    assert readiness.runtime is runtime
    names = [name for name, _fields in events.calls]
    assert names.index("voices.loaded") < names.index("model.loaded")


def test_a_failed_scan_stops_the_server_and_it_never_reports_ready(tmp_path: Path) -> None:
    sink = io.StringIO()
    events = Emitter(sink, lambda: 0.0)
    readiness = Readiness()
    runtime = _runtime()

    def open_voices(_runtime: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(tmp_path))

    components = Components(
        settings=settings_from_args([MODEL_DIR]),
        events=events,
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_voices,
    )
    server = SimpleNamespace(should_exit=False)
    loaded = LoadedModel(
        runtime=runtime, report={}, cpu_tokenizer=FakeTokenizer(), sizing_tokenizer=FakeTokenizer()
    )
    try:
        assert not asyncio.run(load_in_background(components, lambda: loaded, server))
    finally:
        components.gpu.shutdown()

    assert server.should_exit
    assert readiness.runtime is None
    [failed] = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert failed["event"] == "model.load_failed"
    assert failed["stage"] == "voices"
    assert "Permission denied" in failed["error"]


def test_a_failed_model_load_reports_the_model_stage_and_never_scans(tmp_path: Path) -> None:
    sink = io.StringIO()
    scanned: list[bool] = []
    components = Components(
        settings=settings_from_args([MODEL_DIR]),
        events=Emitter(sink, lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=lambda _runtime: scanned.append(True),
    )

    def load() -> LoadedModel:
        raise FileNotFoundError("no checkpoint here")

    try:
        assert not asyncio.run(
            load_in_background(components, load, SimpleNamespace(should_exit=False))
        )
    finally:
        components.gpu.shutdown()

    assert scanned == []
    [failed] = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert failed["event"] == "model.load_failed"
    assert failed["stage"] == "model"


def test_open_voices_loads_saved_voices_and_reserves_skipped_names(tmp_path: Path) -> None:
    """The composition root's scan: the store is built from the loaded model's codebook
    facts and the codec fingerprint, the registry from the scan's result."""
    first = _Server(tmp_path)
    try:
        assert first.post(audio=_wav(), name="alice").status_code == 200
    finally:
        first.close()
    _write_invalid_voice_file(tmp_path, "broken")
    events = RecordingEvents()

    services = api.open_voices(
        _runtime(),
        voices_dir=tmp_path,
        events=events,
        codec_fingerprint=FINGERPRINT,
        now=lambda: datetime(2026, 9, 26, tzinfo=timezone.utc),
        nonce=lambda: "n",
        encode_clock=_EncodeClock(),
    )

    assert [record["id"] for record in services.registry.list_records(
        sample_rate=24000, samples_per_frame=1920
    )] == ["alice"]
    assert services.registry.name_taken("BROKEN")
    loaded = [fields for name, fields in events.calls if name == "voices.loaded"]
    assert loaded == [{"loaded": 1, "skipped": 1, "breeze_ignored": 0}]


def test_a_voice_file_from_another_codec_is_skipped(tmp_path: Path) -> None:
    first = _Server(tmp_path)
    try:
        assert first.post(audio=_wav(), name="alice").status_code == 200
    finally:
        first.close()
    events = RecordingEvents()

    services = api.open_voices(
        _runtime(),
        voices_dir=tmp_path,
        events=events,
        codec_fingerprint="0" * 64,
        now=lambda: datetime(2026, 9, 26, tzinfo=timezone.utc),
        nonce=lambda: "n",
        encode_clock=_EncodeClock(),
    )

    assert services.registry.saved_count() == 0
    assert [name for name, _fields in events.calls if name == "voice.skipped"] == ["voice.skipped"]


def test_components_require_open_voices() -> None:
    """Required, like `cpu_tokenizer`: no wiring can forget to open the voices."""
    with pytest.raises(TypeError, match="open_voices"):
        Components(  # type: ignore[call-arg]
            settings=settings_from_args([MODEL_DIR]),
            events=Emitter(io.StringIO(), lambda: 0.0),
            gate=GpuGate(),
            gpu=GpuThread("cpu", lambda _device: None),
            readiness=Readiness(),
            ws_port=lambda: 0,
            cpu_tokenizer=CpuTokenizer(),
        )
