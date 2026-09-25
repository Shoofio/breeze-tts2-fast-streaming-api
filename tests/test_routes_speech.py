"""`POST /v1/audio/speech` tests (specs/003-cpp-compatible-api/tasks.md T041/T042; review-agent
pass 1 findings 1, 4, 5, 7, 8, 9, 10).

Built through `api.create_app` (which now wires `install_speech` in itself, next to
`install_health` -- the same wiring `tests/test_health.py` uses), with `tests/fakes.py`'s
GPU-free stand-ins at the model edge (`FakeRuntime`, `FakeCodec`, `FakeTokenizer`) -- the
Principle V deviation `tests/fakes.py`'s own docstring records. `FakeRuntime` itself carries no
`tokenizer`/`model`/`audio_tokenizer` (most of its other consumers never need them), so
`_fake_runtime` attaches the same fakes `tests/test_synthesis.py` uses directly onto it,
duck-typing the real `FastBreezeStreamingRuntime`'s own attributes (`models/fast_streaming.py`).
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from breeze_infer import routes_speech
from breeze_infer.api import Components, create_app, load_in_background
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuCloseTimeout, GpuGate, GpuSession, GpuThread
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from tests.fakes import (
    FakeCodec,
    FakeRuntime,
    FakeStreamingConfig,
    FakeTokenizer,
    RecordingEvents,
    model_with_codec_facts,
)

SPEECH_PATH = "/v1/audio/speech"


def _fake_runtime(**kwargs: Any) -> FakeRuntime:
    """A `FakeRuntime` wired up with the `tokenizer`/`model`/`audio_tokenizer` attributes
    the real `FastBreezeStreamingRuntime` carries (`models/fast_streaming.py`'s
    `self.tokenizer`/`self.model`/`self.audio_tokenizer`) -- `routes_speech.py` reads
    them straight off the object `Readiness.require_ready` returns.
    """
    runtime = FakeRuntime(**kwargs)
    runtime.tokenizer = FakeTokenizer()
    runtime.model = model_with_codec_facts()
    runtime.audio_tokenizer = FakeCodec()
    return runtime


def _build_components(
    readiness: Readiness, *, split_chars: int | None = None, events: Any = None
) -> Components:
    # The model directory is never opened: nothing loads in these tests (test_health.py's
    # own _components does the same).
    argv = [str(Path(__file__).parent)]
    if split_chars is not None:
        argv += ["--split-chars", str(split_chars)]
    return Components(
        settings=settings_from_args(argv),
        events=events if events is not None else Emitter(io.StringIO(), lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: 0,
        # What the model load installs: its own copy of the runtime's tokenizer. A plain
        # `FakeTokenizer` holds no state, so a fresh one is as good as a copy.
        cpu_tokenizer=CpuTokenizer(FakeTokenizer()),
    )


def _client_for(components: Components) -> TestClient:
    """`api.create_app` now wires `install_speech` in itself (review-agent pass 1, finding
    9): no more hand-rebuilt app wiring here.

    `raise_server_exceptions=False`: a genuinely unhandled exception (the DONE-with-no-audio
    and priming-failure cases below) must come back as the 500 response a real client would
    see (tests/test_api_errors.py and friends use the same flag for the same reason),
    not re-raised into the test itself by Starlette's TestClient.
    """
    return TestClient(create_app(components), raise_server_exceptions=False)


def _gate_is_free(components: Components) -> bool:
    lease = components.gate.try_acquire()
    if lease is None:
        return False
    lease.release()
    return True


def _wav_bytes(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """A real, small WAV -- soundfile for real, as tests/test_reference_audio.py's own
    fixtures do, not a hand-built header."""
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def readiness() -> Readiness:
    return Readiness()


@pytest.fixture()
def components(readiness: Readiness) -> Iterator[Components]:
    comps = _build_components(readiness)
    yield comps
    comps.gpu.shutdown()


@pytest.fixture()
def client(components: Components) -> TestClient:
    return _client_for(components)


@pytest.fixture()
def ready_client(readiness: Readiness, client: TestClient) -> TestClient:
    readiness.mark_ready(_fake_runtime())
    return client


# --- 200 -----------------------------------------------------------------------------


def test_200_has_pcm_headers_and_a_nonempty_even_length_body(ready_client: TestClient) -> None:
    response = ready_client.post(SPEECH_PATH, data={"text": "hello there"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-sample-format"] == "s16le"
    assert response.headers["cache-control"] == "no-store"
    assert len(response.content) > 0
    assert len(response.content) % 2 == 0  # s16le: every sample is 2 bytes


def test_voice_design_with_no_reference_succeeds(ready_client: TestClient) -> None:
    response = ready_client.post(SPEECH_PATH, data={"text": "a voice with no reference"})

    assert response.status_code == 200
    assert len(response.content) > 0


def test_inline_reference_with_a_real_wav_succeeds(ready_client: TestClient) -> None:
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello there", "ref_text": "a reference transcript"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    assert len(response.content) > 0
    assert len(response.content) % 2 == 0


# --- request_id: X-Request-Id and the speech.* events (finding 4, 8, 10) ---------------------


def test_success_emits_accepted_and_first_audio_and_sets_x_request_id() -> None:
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        runtime = _fake_runtime()
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 200
        request_id = response.headers["x-request-id"]
        assert request_id

        accepted = next(fields for name, fields in events.calls if name == "speech.accepted")
        first_audio = next(
            fields for name, fields in events.calls if name == "speech.first_audio"
        )
        assert accepted["request_id"] == request_id
        assert accepted["pieces"] == 1
        assert accepted["reference"] == "none"
        assert first_audio["request_id"] == request_id
        assert first_audio["ttfa_ms"] >= 0
    finally:
        components.gpu.shutdown()


def test_x_request_id_is_present_on_error_responses(ready_client: TestClient) -> None:
    response = ready_client.post(SPEECH_PATH, data={"text": "hello", "voice_id": "alice"})

    assert response.status_code == 404
    assert response.headers["x-request-id"]


# --- 409 busy --------------------------------------------------------------------------


def test_409_busy_while_the_gate_is_held(
    ready_client: TestClient, components: Components
) -> None:
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(SPEECH_PATH, data={"text": "hello"})
    finally:
        lease.release()

    assert response.status_code == 409
    assert response.json() == {"error": "busy", "code": "busy"}


def test_409_busy_while_a_websocket_waiter_is_queued(
    ready_client: TestClient, components: Components
) -> None:
    """FR-016: a queued WebSocket piece must not let an HTTP `try_acquire` steal the gate
    from it -- driven directly against `GpuGate.acquire`, the way gpu.py's own module
    docstring describes the invariant ("whenever there is no owner, no live waiter
    exists"), rather than through a real WebSocket connection.
    """

    async def main() -> None:
        holder = components.gate.try_acquire()
        assert holder is not None
        waits: list[str] = []
        task = asyncio.create_task(
            components.gate.acquire(on_wait=lambda: waits.append("queued"))
        )
        # Let the waiter actually queue (tests/test_gpu_gate.py's own _settle: waking a
        # waiter and running its continuation take separate loop iterations).
        for _ in range(5):
            await asyncio.sleep(0)
        assert waits == ["queued"]
        assert not task.done()

        # A blocking call from inside this coroutine: it runs TestClient's request on its
        # own thread/loop, which never touches this loop or its queued waiter.
        response = ready_client.post(SPEECH_PATH, data={"text": "hello"})

        assert response.status_code == 409
        assert response.json() == {"error": "busy", "code": "busy"}

        holder.release()
        (await task).release()

    asyncio.run(main())


# --- FR-007: every CPU-only check runs before busy (finding 1) -------------------------------


def test_unspeakable_text_is_400_not_409_even_while_the_gate_is_held(
    ready_client: TestClient, components: Components
) -> None:
    """review-agent pass 1, finding 1: split_text's "nothing to speak" check (and any other
    CPU-only check) must run before try_acquire, so a request that would fail validation
    anyway never sees 409 just because the gate happens to be held.
    """
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(SPEECH_PATH, data={"text": "..."})
    finally:
        lease.release()

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


# --- piece seeds -------------------------------------------------------------------------


def test_piece_seeds_increment_per_piece() -> None:
    readiness = Readiness()
    components = _build_components(readiness, split_chars=100)
    try:
        # A frame with every codebook, since piece 0's frames anchor the later pieces.
        runtime = _fake_runtime(chunks=1, frames=[torch.full((16,), 5)])
        readiness.mark_ready(runtime)
        client = _client_for(components)

        # Each sentence is over the 100-char budget and, with any other, over piece 0's
        # 200-char opening budget too, so text_split.py makes exactly 3 pieces (see the
        # route's own docstring on how piece 0's inputs vs. later pieces' are built).
        sentence = (
            "This sentence is long enough on its own to go past the budget of a piece, "
            "which is one hundred characters."
        )
        assert len(sentence) > 100
        response = client.post(
            SPEECH_PATH,
            data={"text": " ".join([sentence] * 3), "seed": "100"},
        )

        assert response.status_code == 200
        assert len(runtime.calls) == 3
        assert [call["seed"] for call in runtime.calls] == [100, 101, 102]
    finally:
        components.gpu.shutdown()


# --- 503 loading ---------------------------------------------------------------------------


def test_503_loading_before_the_model_is_ready(client: TestClient) -> None:
    response = client.post(SPEECH_PATH, data={"text": "hello"})

    assert response.status_code == 503
    assert response.json() == {
        "status": "loading",
        "error": "model is loading",
        "code": "loading",
    }


# --- voice_id -> 404 -----------------------------------------------------------------------


def test_voice_ref_gives_404_unknown_voice(ready_client: TestClient) -> None:
    response = ready_client.post(SPEECH_PATH, data={"text": "hello", "voice_id": "alice"})

    assert response.status_code == 404
    assert response.json() == {"error": "unknown voice_id", "code": "unknown_voice"}


# --- first piece with no room -> 400 text_too_long ------------------------------------------


def test_first_piece_with_no_room_gives_400_text_too_long() -> None:
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        # max_seq_len=1 leaves no context room for any real tokenized prompt, however
        # short -- deterministic, unlike trying to find a text long enough to overflow
        # the default 1024-token context.
        runtime = _fake_runtime(config=FakeStreamingConfig(max_seq_len=1))
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(
            SPEECH_PATH, data={"text": "not enough room for this to generate at all"}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "text is too long", "code": "text_too_long"}
        assert runtime.calls == []  # generation never started
    finally:
        components.gpu.shutdown()


# --- the gate is released after every pre-200 failure (finding 10) ---------------------------


def test_gate_is_released_after_unspeakable_text_400(
    ready_client: TestClient, components: Components
) -> None:
    response = ready_client.post(SPEECH_PATH, data={"text": "..."})

    assert response.status_code == 400
    assert _gate_is_free(components)


def test_gate_is_released_after_text_too_long_400() -> None:
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        runtime = _fake_runtime(config=FakeStreamingConfig(max_seq_len=1))
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "not enough room at all"})

        assert response.status_code == 400
        assert _gate_is_free(components)
    finally:
        components.gpu.shutdown()


def test_gate_is_released_after_a_priming_failure() -> None:
    # fail_after=0: the runtime raises before yielding any audio -- a genuine priming
    # failure (500), distinct from the "no audio at all" DONE case below.
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        runtime = _fake_runtime(fail_after=0)
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 500
        assert _gate_is_free(components)
    finally:
        components.gpu.shutdown()


def test_gate_is_released_when_the_first_step_is_done_with_no_audio() -> None:
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        runtime = _fake_runtime(chunks=0)  # no chunks at all: DONE on the very first step
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        # review-agent pass 1, finding 5: a plain RuntimeError (the unhandled path), so the
        # contract's "every 500 closes the connection" holds -- not a keep-alive ApiError.
        assert response.status_code == 500
        assert response.json() == {"error": "internal error", "code": "internal_error"}
        assert _gate_is_free(components)
    finally:
        components.gpu.shutdown()


# --- the CPU room check has its own tokenizer (HF fast tokenizers aren't shareable) ----------


class _ThreadRecordingTokenizer(FakeTokenizer):
    """Records which thread called it; a deep copy is a fresh recorder, kept in `copies`."""

    def __init__(self) -> None:
        self.threads: list[str] = []
        self.copies: list[_ThreadRecordingTokenizer] = []

    def __call__(self, text: str, **kwargs: Any) -> Any:
        self.threads.append(threading.current_thread().name)
        return super().__call__(text, **kwargs)

    def __deepcopy__(self, memo: dict[int, Any]) -> _ThreadRecordingTokenizer:
        copied = _ThreadRecordingTokenizer()
        self.copies.append(copied)
        return copied


def test_the_cpu_room_check_uses_the_copy_made_at_load_never_the_gpu_threads() -> None:
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        runtime = _fake_runtime()
        tokenizer = _ThreadRecordingTokenizer()
        runtime.tokenizer = tokenizer
        cpu_copy = _ThreadRecordingTokenizer()
        components.cpu_tokenizer.install(cpu_copy)
        readiness.mark_ready(runtime)
        client = _client_for(components)

        for _ in range(2):
            assert client.post(SPEECH_PATH, data={"text": "hello there"}).status_code == 200

        assert tokenizer.threads
        assert all(name.startswith("breeze-gpu") for name in tokenizer.threads)
        assert tokenizer.copies == []  # no request copies it: the load already did
        assert cpu_copy.threads
        assert not any(name.startswith("breeze-gpu") for name in cpu_copy.threads)
    finally:
        components.gpu.shutdown()


class _OverlapDetectingTokenizer(FakeTokenizer):
    """Records whether two threads were ever inside it at once (a HF fast tokenizer's
    "Already borrowed"), holding each call open long enough for a second to arrive."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inside = 0
        self.overlapped = False
        self.calls = 0

    def __call__(self, text: str, **kwargs: Any) -> Any:
        with self._lock:
            self._inside += 1
            self.calls += 1
            self.overlapped |= self._inside > 1
        try:
            time.sleep(0.02)
            return super().__call__(text, **kwargs)
        finally:
            with self._lock:
                self._inside -= 1


def test_concurrent_cpu_room_checks_are_serialized() -> None:
    """Two requests at once both run their CPU room check on worker threads, before the busy
    check, against the one CPU copy: they must take turns on it."""
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        cpu_copy = _OverlapDetectingTokenizer()
        components.cpu_tokenizer.install(cpu_copy)
        readiness.mark_ready(_fake_runtime())
        client = _client_for(components)
        start = threading.Barrier(4)

        def post(_: int) -> int:
            start.wait()
            return client.post(SPEECH_PATH, data={"text": "hello there"}).status_code

        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(post, range(4)))

        assert set(statuses) <= {200, 409}  # never a 500 from a shared tokenizer
        assert cpu_copy.calls >= 4
        assert not cpu_copy.overlapped
    finally:
        components.gpu.shutdown()


def test_the_model_load_copies_the_tokenizer_on_the_gpu_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy is made while loading, so no request ever waits for it on the GPU thread."""
    from breeze_infer import model_loading

    tokenizer = _ThreadRecordingTokenizer()
    copied_on: list[str] = []
    original_deepcopy = model_loading.copy.deepcopy

    def recording_deepcopy(value: Any) -> Any:
        copied_on.append(threading.current_thread().name)
        return original_deepcopy(value)

    monkeypatch.setattr(
        model_loading, "configure_compile_cache", lambda *_args: (Path("cache"), "hit")
    )
    monkeypatch.setattr(
        model_loading, "load_runtime", lambda *_args, **_kwargs: (tokenizer, object(), object())
    )
    monkeypatch.setattr(model_loading, "update_generation_config_for_breeze", lambda _model: None)
    monkeypatch.setattr(
        model_loading,
        "FastBreezeStreamingRuntime",
        lambda *_args, **_kwargs: SimpleNamespace(fast_enabled=False, tokenizer=tokenizer),
    )
    monkeypatch.setattr(model_loading.copy, "deepcopy", recording_deepcopy)
    settings = settings_from_args([str(Path(__file__).parent)])
    gpu = GpuThread("cpu", lambda _device: None)
    try:
        load = partial(model_loading.load_model, settings, "cpu", {})
        loaded = asyncio.run(gpu.run(load))
    finally:
        gpu.shutdown()

    [copied] = tokenizer.copies
    assert loaded.cpu_tokenizer is copied
    assert len(copied_on) == 1 and copied_on[0].startswith("breeze-gpu")


def test_the_cpu_copy_is_installed_before_the_server_reports_ready() -> None:
    readiness = Readiness()
    components = _build_components(readiness)
    runtime = _fake_runtime()
    cpu_copy = FakeTokenizer()
    seen_at_ready: list[bool] = []
    original_mark_ready = readiness.mark_ready

    def mark_ready(ready_runtime: Any) -> None:
        with components.cpu_tokenizer.borrow() as installed:
            seen_at_ready.append(installed is cpu_copy)
        original_mark_ready(ready_runtime)

    readiness.mark_ready = mark_ready  # type: ignore[method-assign]
    server = SimpleNamespace(should_exit=False)
    try:
        loaded = LoadedModel(runtime=runtime, report={}, cpu_tokenizer=cpu_copy)
        assert asyncio.run(load_in_background(components, lambda: loaded, server))
    finally:
        components.gpu.shutdown()

    assert seen_at_ready == [True]
    assert readiness.runtime is runtime


# --- events: frame prediction, pieces ---------------------------------------------------------


class _ExtraFrameCodec(FakeCodec):
    """Encodes one frame more than `reference_audio.predicted_frames` predicts."""

    def encode(self, wav: Any, sr: int, return_dict: bool = True) -> Any:
        output = super().encode(wav, sr)
        codes = output.audio_codes[0]
        output.audio_codes[0] = torch.cat([codes, codes[-1:]])
        return output


def test_a_frame_prediction_mismatch_is_reported_and_the_request_still_succeeds() -> None:
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        runtime = _fake_runtime()
        runtime.audio_tokenizer = _ExtraFrameCodec()
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(
            SPEECH_PATH,
            data={"text": "hello there", "ref_text": "a reference transcript"},
            files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 200
        [mismatch] = [f for name, f in events.calls if name == "speech.frame_prediction_mismatch"]
        assert mismatch["level"] == "warning"
        assert mismatch["request_id"] == response.headers["x-request-id"]
        assert mismatch["actual_frames"] == mismatch["predicted_frames"] + 1
    finally:
        components.gpu.shutdown()


def test_piece_done_is_emitted_for_every_piece() -> None:
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, split_chars=15, events=events)
    try:
        runtime = _fake_runtime(chunks=2)
        readiness.mark_ready(runtime)
        client = _client_for(components)

        # With a reference there is no opening budget: three pieces of one sentence each.
        response = client.post(
            SPEECH_PATH,
            data={"text": "Hi there. Go now yes. See you soon.", "ref_text": "a transcript"},
            files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 200
        done = [f for name, f in events.calls if name == "speech.piece_done"]
        assert [(f["piece_index"], f["frames"]) for f in done] == [(0, 2), (1, 2), (2, 2)]
        assert {f["request_id"] for f in done} == {response.headers["x-request-id"]}
    finally:
        components.gpu.shutdown()


# --- a failing close before the 200 is reported, never raised (finding 6) --------------------


def _session_whose_close_fails(error: BaseException) -> type[GpuSession[Any]]:
    class _FailingCloseSession(GpuSession[Any]):
        async def aclose(self) -> None:
            await super().aclose()  # the gate is released as usual
            raise error

    return _FailingCloseSession


@pytest.mark.parametrize(
    ("runtime_kwargs", "outcome"),
    [
        ({"fail_after": 0}, "request.failed"),  # priming raised: the `except` cleanup
        ({"chunks": 0}, "speech.failed"),  # priming found no audio: the DONE branch
    ],
)
def test_a_close_failure_before_the_200_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, runtime_kwargs: dict[str, Any], outcome: str
) -> None:
    monkeypatch.setattr(
        routes_speech, "GpuSession", _session_whose_close_fails(RuntimeError("close broke"))
    )
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        readiness.mark_ready(_fake_runtime(**runtime_kwargs))
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        # The original failure reaches the client, not the close error.
        assert response.status_code == 500
        request_id = response.headers["x-request-id"]
        [close_failed] = [f for name, f in events.calls if name == "gpu.close_failed"]
        assert close_failed["request_id"] == request_id
        assert "close broke" in str(close_failed["error"])
        assert [f["request_id"] for name, f in events.calls if name == outcome] == [request_id]
        assert _gate_is_free(components)
    finally:
        components.gpu.shutdown()


def test_a_close_timeout_before_the_200_is_left_to_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate's `on_poisoned` reports a close timeout (`gpu.close_timeout`), so the route
    doesn't report it a second time as `gpu.close_failed`."""
    monkeypatch.setattr(
        routes_speech, "GpuSession", _session_whose_close_fails(GpuCloseTimeout("still closing"))
    )
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        readiness.mark_ready(_fake_runtime(chunks=0))
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 500
        assert [name for name, _ in events.calls if name == "gpu.close_failed"] == []
    finally:
        components.gpu.shutdown()
