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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from starlette.requests import Request

from breeze_infer import routes_speech, voice_file
from breeze_infer.api import Components, create_app, load_in_background
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuCloseTimeout, GpuGate, GpuSession, GpuThread
from breeze_infer.limits import ANCHOR_CHARS
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.routes_voices import VoiceServices
from breeze_infer.settings import settings_from_args
from breeze_infer.synthesis import CodesRef, PrefixRef
from breeze_infer.templates import prepare_prefix_inputs
from breeze_infer.text_split import split_text
from breeze_infer.voice_prefix import VoicePrefixCache
from breeze_infer.voice_registry import VoiceRegistry
from breeze_infer.voice_store import VoiceStore
from tests.fakes import (
    CODEC_CODEBOOK_SIZE,
    CODEC_CODEBOOKS,
    FakeCodec,
    FakeRuntime,
    FakeStreamingConfig,
    FakeTokenizer,
    RecordingEvents,
    model_with_codec_facts,
    open_no_voices,
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
    # What the model load installs: its two copies of the runtime's tokenizer. A plain
    # `FakeTokenizer` holds no state, so a fresh one is as good as a copy. Installed up front
    # because most tests here mark `readiness` ready directly, with a runtime alone.
    cpu_tokenizer = CpuTokenizer()
    cpu_tokenizer.install(FakeTokenizer(), FakeTokenizer())
    components = Components(
        settings=settings_from_args(argv),
        events=events if events is not None else Emitter(io.StringIO(), lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: 0,
        cpu_tokenizer=cpu_tokenizer,
        open_voices=open_no_voices,
    )
    # Most tests here mark `readiness` ready directly, which installs no voices; a `voice_id`
    # is still looked up (and not found) in an empty registry, as on a server with none.
    components.voices.install(open_no_voices(None))
    return components


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
        components.cpu_tokenizer.install(cpu_copy, FakeTokenizer())
        readiness.mark_ready(runtime)
        client = _client_for(components)

        for _ in range(2):
            assert client.post(SPEECH_PATH, data={"text": "hello there"}).status_code == 200

        assert tokenizer.threads
        assert all(name.startswith("breeze-gpu") for name in tokenizer.threads)
        assert tokenizer.copies == []  # no request copies it: the load already did
        assert cpu_copy.threads
        # Never a default-pool worker either: the check has its own single-thread executor.
        assert all(name.startswith("breeze-cpu-tokenizer") for name in cpu_copy.threads)
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
    """Two requests at once both run their CPU room check before the busy check, against the
    one CPU copy: its single executor thread makes them take turns on it."""
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        cpu_copy = _OverlapDetectingTokenizer()
        components.cpu_tokenizer.install(cpu_copy, FakeTokenizer())
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

    # Both copies -- the pre-gate check's and the anchor sizing's (review 34 finding 5) -- are
    # made here, on the GPU thread while loading, never on the event loop.
    pre_gate_copy, sizing_copy = tokenizer.copies
    assert loaded.cpu_tokenizer is pre_gate_copy
    assert loaded.sizing_tokenizer is sizing_copy
    assert len(copied_on) == 2 and all(name.startswith("breeze-gpu") for name in copied_on)


def test_the_cpu_copy_is_installed_before_the_server_reports_ready() -> None:
    readiness = Readiness()
    components = replace(_build_components(readiness), cpu_tokenizer=CpuTokenizer())
    runtime = _fake_runtime()
    cpu_copy = FakeTokenizer()
    seen_at_ready: list[bool] = []
    original_mark_ready = readiness.mark_ready

    def mark_ready(ready_runtime: Any) -> None:
        seen_at_ready.append(components.cpu_tokenizer._tokenizer is cpu_copy)
        original_mark_ready(ready_runtime)

    readiness.mark_ready = mark_ready  # type: ignore[method-assign]
    server = SimpleNamespace(should_exit=False)
    try:
        loaded = LoadedModel(
            runtime=runtime, report={}, cpu_tokenizer=cpu_copy, sizing_tokenizer=FakeTokenizer()
        )
        assert asyncio.run(load_in_background(components, lambda: loaded, server))
    finally:
        components.gpu.shutdown()

    assert seen_at_ready == [True]
    assert readiness.runtime is runtime


def test_mark_ready_installs_the_copy_and_the_runtime_in_one_step() -> None:
    """The one way the server (and the GPU tests) become ready: a `LoadedModel` carries both."""
    readiness = Readiness()
    components = replace(_build_components(readiness), cpu_tokenizer=CpuTokenizer())
    try:
        runtime = _fake_runtime()
        cpu_copy = FakeTokenizer()

        sizing_copy = FakeTokenizer()
        components.mark_ready(
            LoadedModel(
                runtime=runtime, report={}, cpu_tokenizer=cpu_copy, sizing_tokenizer=sizing_copy
            ),
            open_no_voices(None),
        )

        assert readiness.runtime is runtime
        assert components.cpu_tokenizer._tokenizer is cpu_copy
        assert components.cpu_tokenizer._sizing_tokenizer is sizing_copy
        client = _client_for(components)
        assert client.post(SPEECH_PATH, data={"text": "hello there"}).status_code == 200
    finally:
        components.gpu.shutdown()


def test_components_and_loaded_model_both_require_a_cpu_tokenizer() -> None:
    readiness = Readiness()
    fields = {
        name: getattr(_build_components(readiness), name)
        for name in ("settings", "events", "gate", "gpu", "readiness", "ws_port")
    }
    with pytest.raises(TypeError, match="cpu_tokenizer"):
        Components(**fields)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="cpu_tokenizer"):
        LoadedModel(runtime=_fake_runtime(), report={})  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="sizing_tokenizer"):
        LoadedModel(  # type: ignore[call-arg]
            runtime=_fake_runtime(), report={}, cpu_tokenizer=FakeTokenizer()
        )


@pytest.mark.parametrize(
    ("pre_gate", "sizing"),
    [(None, FakeTokenizer()), (FakeTokenizer(), None)],
    ids=["no_pre_gate_copy", "no_sizing_copy"],
)
def test_installing_no_cpu_tokenizer_is_rejected(pre_gate: Any, sizing: Any) -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        CpuTokenizer().install(pre_gate, sizing)


def test_installing_one_copy_for_both_workers_is_rejected() -> None:
    """The two workers run at once, so one shared copy would be used by two threads."""
    shared = FakeTokenizer()
    with pytest.raises(ValueError, match="tokenizer"):
        CpuTokenizer().install(shared, shared)


class _UncopyableTokenizer(FakeTokenizer):
    def __deepcopy__(self, memo: dict[int, Any]) -> _UncopyableTokenizer:
        raise AssertionError("install must not copy: the model load already did")


def test_install_uses_the_two_copies_it_is_given_without_copying_either() -> None:
    """Review 34 follow-up: `install` runs on the event loop (`mark_ready`), where a ~700 ms
    deep copy of a real tokenizer would stall every request, so it only takes the two copies
    the model load made -- one per worker."""
    pre_gate, sizing = _UncopyableTokenizer(), _UncopyableTokenizer()
    cpu_tokenizer = CpuTokenizer()
    cpu_tokenizer.install(pre_gate, sizing)
    try:
        used_by_run = asyncio.run(cpu_tokenizer.run(lambda tokenizer: tokenizer))
        used_by_submit = cpu_tokenizer.submit(lambda tokenizer: tokenizer).result(5)
    finally:
        cpu_tokenizer.shutdown()

    assert used_by_run is pre_gate
    assert used_by_submit is sizing


def test_loaded_model_from_runtime_makes_two_distinct_copies() -> None:
    tokenizer = _ThreadRecordingTokenizer()
    runtime = SimpleNamespace(tokenizer=tokenizer)

    loaded = LoadedModel.from_runtime(runtime, {"device": "cpu"})

    assert tokenizer.copies == [loaded.cpu_tokenizer, loaded.sizing_tokenizer]
    assert loaded.cpu_tokenizer is not loaded.sizing_tokenizer
    assert tokenizer not in (loaded.cpu_tokenizer, loaded.sizing_tokenizer)
    assert loaded.runtime is runtime
    assert loaded.report == {"device": "cpu"}


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


# --- speech by voice_id (T060; the A: cases ported from A:tests/test_api_speech.py ~378-521) --
#
# A voice with no `ref_text` override is spoken through its cached KV prefix (data-model.md
# "Reference", the prefix variant): built once on the GPU thread under the request's lease, then
# reused. An override uses the codes path with the given transcript. The voice is resolved, and
# the prefix cache's token read, before the gate (FR-007: an unknown voice is `404` before
# `409 busy`).

NARRATOR_TEXT = "stored transcript"
UNNAMED_ID = "v_0123456789abcdef"
# A deterministic reference: 8 frames of 16 codebooks, every value a valid code.
VOICE_CODES = (np.arange(8 * CODEC_CODEBOOKS, dtype=np.int16) % CODEC_CODEBOOK_SIZE).reshape(
    8, CODEC_CODEBOOKS
)


def _voice_services(voices_dir: Path, events: RecordingEvents) -> VoiceServices:
    """What `api.open_voices` builds, on a real directory, with a 1-byte-per-token prefix
    budget (the default 1 GiB) so the cache holds whatever the tests build."""
    store = VoiceStore(
        voices_dir,
        codebooks=CODEC_CODEBOOKS,
        codebook_size=CODEC_CODEBOOK_SIZE,
        codec_fingerprint="f" * 64,
        events=events,
        clock=lambda: datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
    )
    store.scan()
    return VoiceServices(
        store=store,
        registry=VoiceRegistry(),
        prefix_cache=VoicePrefixCache(bytes_per_token=1, on_event=events.emit),
    )


@pytest.fixture()
def voice_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., SimpleNamespace]]:
    """Builds a ready server with a real voice directory. Every piece's `prepare_piece` call is
    recorded (`prepared`: its reference and text) while the real one still runs."""
    started: list[Components] = []
    prepared: list[SimpleNamespace] = []
    real_prepare_piece = routes_speech.prepare_piece

    def recording_prepare_piece(
        tokenizer: Any, model: Any, reference: Any, text: str, instruction: str, cfg_scale: float
    ) -> dict[str, Any]:
        prepared.append(SimpleNamespace(reference=reference, text=text))
        return real_prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)

    monkeypatch.setattr(routes_speech, "prepare_piece", recording_prepare_piece)

    def start(
        runtime: FakeRuntime | None = None,
        *,
        split_chars: int | None = None,
        cpu_tokenizer: Any = None,
    ) -> SimpleNamespace:
        readiness = Readiness()
        events = RecordingEvents()
        components = _build_components(readiness, split_chars=split_chars, events=events)
        started.append(components)
        if cpu_tokenizer is not None:
            components.cpu_tokenizer.install(cpu_tokenizer, FakeTokenizer())
        services = _voice_services(tmp_path / "voices", events)
        components.voices.install(services)
        runtime = runtime if runtime is not None else _fake_runtime()
        readiness.mark_ready(runtime)
        return SimpleNamespace(
            components=components,
            client=_client_for(components),
            runtime=runtime,
            events=events,
            services=services,
            prepared=prepared,
        )

    yield start
    for components in started:
        components.gpu.shutdown()


def _save_voice(server: SimpleNamespace, name: str = "Narrator", ref_text: str = NARRATOR_TEXT) -> str:
    """Register a saved voice through `POST /v1/voices`, as a client would."""
    response = server.client.post(
        "/v1/voices",
        data={"ref_text": ref_text, "name": name},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _add_unnamed_voice(server: SimpleNamespace, ref_text: str = NARRATOR_TEXT) -> str:
    """Register an unnamed voice with known codes (`VOICE_CODES`) straight into the registry,
    with its prefix length measured as a registration measures it. No room check: a test can
    register a voice `POST /v1/voices` would refuse, as a voice on disk from a larger context."""
    server.services.registry.register_unnamed(
        id=UNNAMED_ID,
        ref_text=ref_text,
        codes=VOICE_CODES,
        frames=8,
        encode_ms=1,
        prefix_len=_prefix_len(ref_text, VOICE_CODES),
    )
    return UNNAMED_ID


def _speech(server: SimpleNamespace, **data: str) -> Any:
    return server.client.post(SPEECH_PATH, data={"text": "hello there", **data})


def _prefix_len(ref_text: str, codes: Any) -> int:
    """The prefix length the real `build_reference_prefix` gives: its inputs' length."""
    inputs = prepare_prefix_inputs(
        FakeTokenizer(), model_with_codec_facts(), {"ref_text": ref_text, "ref_audio_codes": codes}
    )
    return int(inputs["attention_mask"].shape[1])


def _named(server: SimpleNamespace, name: str) -> list[dict[str, Any]]:
    return [fields for event, fields in server.events.calls if event == name]


def test_voice_id_uses_the_prefix_path(voice_server: Callable[..., SimpleNamespace]) -> None:
    server = voice_server()
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 200
    # One prefix, built from the stored transcript and codes, and the piece continues it.
    [prefix_inputs] = server.runtime.prefix_builds
    assert int(prefix_inputs["attention_mask"].shape[1]) == _prefix_len(NARRATOR_TEXT, VOICE_CODES)
    [call] = server.runtime.calls
    assert call["prefix"].prefix_len == _prefix_len(NARRATOR_TEXT, VOICE_CODES)
    assert call["inputs"].get("input_values") is None  # the suffix carries no audio
    [piece] = server.prepared
    assert isinstance(piece.reference, PrefixRef)
    assert piece.reference.ref_text == NARRATOR_TEXT
    [accepted] = _named(server, "speech.accepted")
    assert accepted["reference"] == "voice_prefix"
    assert accepted["warm"] is False
    assert _gate_is_free(server.components)


def test_a_saved_voice_uses_the_prefix_path(voice_server: Callable[..., SimpleNamespace]) -> None:
    server = voice_server()
    _save_voice(server)

    response = _speech(server, voice_id="Narrator")

    assert response.status_code == 200
    assert len(server.runtime.prefix_builds) == 1
    assert server.runtime.calls[-1]["prefix"] is not None


def test_voice_id_with_ref_text_uses_the_codes_path_with_the_override(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """A: `test_ref_text_with_voice_id_overrides_the_transcript_on_the_codes_tier`."""
    server = voice_server()
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id, ref_text="a client transcript")

    assert response.status_code == 200
    assert server.runtime.prefix_builds == []  # no KV prefix for an overridden transcript
    assert server.runtime.calls[-1]["prefix"] is None
    [piece] = server.prepared
    assert isinstance(piece.reference, CodesRef)
    assert piece.reference.ref_text == "a client transcript"
    assert np.array_equal(np.asarray(piece.reference.codes), VOICE_CODES)
    # The override is for this request only: the stored transcript is unchanged.
    [record] = server.client.get("/v1/voices").json()
    assert record["ref_text"] == NARRATOR_TEXT
    [accepted] = _named(server, "speech.accepted")
    assert accepted["reference"] == "voice_codes"


def test_an_unknown_voice_is_404(voice_server: Callable[..., SimpleNamespace]) -> None:
    server = voice_server()
    _save_voice(server)

    # Only case is ignored at create: a lookup is exact (data-model.md "Registry rules").
    for voice_id in ("nobody", "narrator", "v_ffffffffffffffff"):
        response = _speech(server, voice_id=voice_id)
        assert response.status_code == 404
        assert response.json() == {"error": "unknown voice_id", "code": "unknown_voice"}
    assert server.runtime.calls == []
    assert _gate_is_free(server.components)


@pytest.mark.parametrize("saved", [True, False], ids=["saved", "unnamed"])
def test_a_deleted_voice_is_404(voice_server: Callable[..., SimpleNamespace], saved: bool) -> None:
    server = voice_server()
    voice_id = _save_voice(server) if saved else _add_unnamed_voice(server)
    assert _speech(server, voice_id=voice_id).status_code == 200

    assert server.client.delete(f"/v1/voices/{voice_id}").status_code == 200
    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 404
    assert response.json() == {"error": "unknown voice_id", "code": "unknown_voice"}
    assert len(server.services.prefix_cache) == 0  # the DELETE dropped its prefix


def test_a_kv_prefix_is_reused_across_requests(voice_server: Callable[..., SimpleNamespace]) -> None:
    """A: `test_voice_id_with_a_warm_prefix_uses_the_suffix_path`."""
    server = voice_server()
    voice_id = _add_unnamed_voice(server)

    first = _speech(server, voice_id=voice_id)
    second = _speech(server, voice_id=voice_id)

    assert first.status_code == second.status_code == 200
    assert len(server.runtime.prefix_builds) == 1
    first_call, second_call = server.runtime.calls
    assert second_call["prefix"] is first_call["prefix"]
    assert [accepted["warm"] for accepted in _named(server, "speech.accepted")] == [False, True]
    assert all(isinstance(piece.reference, PrefixRef) for piece in server.prepared)


def test_every_piece_of_a_voice_request_reuses_the_one_prefix(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """A: `test_voice_pieces_all_reuse_the_kv_prefix`."""
    server = voice_server(split_chars=15)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id, text="Hi there. Go now yes. See you soon.")

    assert response.status_code == 200
    assert len(server.runtime.prefix_builds) == 1
    assert len(server.runtime.calls) == 3
    assert all(call["prefix"] is server.runtime.calls[0]["prefix"] for call in server.runtime.calls)
    assert all(isinstance(piece.reference, PrefixRef) for piece in server.prepared)


def test_a_voice_request_has_no_opening_piece(voice_server: Callable[..., SimpleNamespace]) -> None:
    """A: `test_first_piece_uses_the_anchor_budget_only_without_a_reference`: a voice already
    fixes the speaker, so the text is split with no short opening piece and nothing anchors."""
    server = voice_server(split_chars=600)
    voice_id = _add_unnamed_voice(server)
    text = " ".join(f"Line {n} of the voice test text." for n in range(40))

    assert _speech(server, text=text).status_code == 200
    no_reference = [piece.text for piece in server.prepared]
    server.prepared.clear()
    assert _speech(server, text=text, voice_id=voice_id).status_code == 200
    with_voice = [piece.text for piece in server.prepared]

    assert no_reference == split_text(text, budget=600, first_budget=ANCHOR_CHARS)
    assert with_voice == split_text(text, budget=600)
    assert all(isinstance(piece.reference, PrefixRef) for piece in server.prepared)


class _BlockingTokenizer(FakeTokenizer):
    """Once `armed` is set, holds the next pre-gate room check open until `release` is set: the
    request has resolved its voice by then, and hasn't taken the gate yet. Unarmed, it
    tokenizes at once (a `POST /v1/voices` measures its prefix on this worker too)."""

    def __init__(self) -> None:
        self.armed = threading.Event()
        self.reached = threading.Event()
        self.release = threading.Event()

    def __call__(self, text: str, **kwargs: Any) -> Any:
        if self.armed.is_set():
            self.reached.set()
            assert self.release.wait(5.0)
        return super().__call__(text, **kwargs)


def _register_again(server: SimpleNamespace, voice: Any) -> None:
    """Register `voice` (a `lookup` result) as a saved voice again, as `POST /v1/voices` commits
    one, but without the route: a `POST` would queue behind the blocked room check on the CPU
    tokenizer's one worker."""
    stored = server.services.store.create(
        voice_file.VoiceFile(
            id=voice.id,
            ref_text=voice.ref_text,
            frames=int(voice.codes.shape[0]),
            codebooks=int(voice.codes.shape[1]),
            codes=voice.codes,
            codes_sha256=voice_file.codes_sha256(voice.codes),
            codec_fingerprint=server.services.store.codec_fingerprint,
            encode_ms=1,
            created_at="",
        )
    )
    server.services.registry.register_saved(stored, prefix_len=voice.prefix_len)


@pytest.mark.parametrize("re_register", [False, True], ids=["deleted", "deleted_and_re_registered"])
def test_a_voice_deleted_between_resolution_and_the_gate_is_not_cached(
    voice_server: Callable[..., SimpleNamespace], re_register: bool
) -> None:
    """A: `test_voice_deleted_between_lookup_and_prefix_build_is_404`, changed by the new
    design (data-model.md "Voice", prefix-cache key): the request already holds the voice it
    resolved, so it is still spoken, but its prefix is never cached -- even when the voice is
    re-registered with the same audio and text, and so the same cache key, before the build."""
    tokenizer = _BlockingTokenizer()
    server = voice_server(cpu_tokenizer=tokenizer)
    _save_voice(server)
    voice = server.services.registry.lookup("Narrator")
    tokenizer.armed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_speech, server, voice_id="Narrator")
        assert tokenizer.reached.wait(5.0)
        assert server.client.delete("/v1/voices/Narrator").status_code == 200
        if re_register:
            _register_again(server, voice)
        tokenizer.release.set()
        response = pending.result(10.0)

    assert response.status_code == 200
    assert len(server.runtime.prefix_builds) == 1
    assert len(server.services.prefix_cache) == 0
    request_id = response.headers["x-request-id"]
    evicted = [f for f in _named(server, "voice.prefix_evicted") if f.get("request_id") == request_id]
    assert evicted == [{"voice_id": "Narrator", "reason": "deleted", "request_id": request_id}]
    later = _speech(server, voice_id="Narrator")
    if re_register:
        assert later.status_code == 200
        assert len(server.runtime.prefix_builds) == 2  # built again: nothing stale was kept
    else:
        assert later.status_code == 404


class _OutOfMemoryRuntime(FakeRuntime):
    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        self.prefix_builds.append(prefix_inputs)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")


def test_an_out_of_memory_prefix_build_falls_back_to_the_codes_path(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    runtime = _OutOfMemoryRuntime()
    runtime.tokenizer = FakeTokenizer()
    runtime.model = model_with_codec_facts()
    runtime.audio_tokenizer = FakeCodec()
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 200
    assert len(response.content) > 0
    assert len(runtime.prefix_builds) == 1
    assert runtime.calls[-1]["prefix"] is None
    [piece] = server.prepared
    assert isinstance(piece.reference, CodesRef)
    assert piece.reference.ref_text == NARRATOR_TEXT
    assert np.array_equal(np.asarray(piece.reference.codes), VOICE_CODES)
    request_id = response.headers["x-request-id"]
    [fallback] = _named(server, "speech.prefix_fallback")
    assert fallback["level"] == "warning"
    assert fallback["request_id"] == request_id
    assert fallback["voice_id"] == voice_id
    assert fallback["reason"] == "out_of_memory"
    assert "CUDA out of memory" in fallback["error"]
    assert _named(server, "request.failed") == []
    [accepted] = _named(server, "speech.accepted")
    assert accepted["reference"] == "voice_codes"
    assert len(server.services.prefix_cache) == 0
    assert _gate_is_free(server.components)


class _SlowPrefixRuntime(FakeRuntime):
    """`build_reference_prefix` blocks (on the GPU thread) until `release` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        self.started.set()
        assert self.release.wait(5.0)
        return super().build_reference_prefix(prefix_inputs)


def _form_request(data: dict[str, str]) -> Request:
    """A Starlette `Request` carrying `data` as a form body, for driving `_serve_speech`."""
    built = httpx.Request("POST", "http://test" + SPEECH_PATH, data=data)
    body = built.read()
    scope = {
        "type": "http",
        "method": "POST",
        "path": SPEECH_PATH,
        "query_string": b"",
        "headers": [(b"content-type", built.headers["content-type"].encode("latin-1"))],
    }
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        more = not sent
        sent = True
        return {"type": "http.request", "body": body if more else b"", "more_body": False}

    return Request(scope, receive)


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


def test_a_request_cancelled_during_the_prefix_build_hands_the_gate_to_the_build(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """A client disconnect while its prefix builds: the GPU work can't be abandoned, so the
    build keeps the gate (`GpuLease.hand_over`) until it has finished, then frees it, and its
    prefix is still cached for the next request."""
    runtime = _SlowPrefixRuntime()
    runtime.tokenizer = FakeTokenizer()
    runtime.model = model_with_codec_facts()
    runtime.audio_tokenizer = FakeCodec()
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    async def main() -> None:
        task = asyncio.ensure_future(
            routes_speech._serve_speech(
                _form_request({"text": "hello there", "voice_id": voice_id}),
                runtime,
                server.components,
                request_id="cancelled-request",
                received_at=0.0,
                clock=lambda: 0.0,
            )
        )
        await _until(runtime.started.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not _gate_is_free(server.components)  # the build still holds it
        runtime.release.set()
        await _until(lambda: _gate_is_free(server.components))

    asyncio.run(main())

    assert len(server.services.prefix_cache) == 1
    assert _named(server, "voice.prefix_build_failed") == []
    assert runtime.calls == []  # nothing was generated for the cancelled request
    # The next request finds it warm.
    assert _speech(server, voice_id=voice_id).status_code == 200
    assert len(runtime.prefix_builds) == 1


# --- review 42 on 8760821: prefix sizing, the out-of-memory fallback, per-voice facts ------


def _voice_runtime(runtime: FakeRuntime) -> FakeRuntime:
    runtime.tokenizer = FakeTokenizer()
    runtime.model = model_with_codec_facts()
    runtime.audio_tokenizer = FakeCodec()
    return runtime


class _RoomyRuntime(FakeRuntime):
    """Reports the full frame cap as every piece's room, however long its prompt: the room
    check alone never refuses, so only the prefix guard can."""

    def max_new_tokens_room(self, requested: Any, inputs: Any, *, prefix_len: int = 0) -> int:
        return self.frame_cap(requested)


def test_42_1_a_stored_prefix_the_runtime_would_not_build_is_refused_before_the_gate(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """A voice already on disk (registered under a larger context, say) whose prefix leaves
    under MIN_SUFFIX_ROOM slots: the short text has room, but `build_reference_prefix` would
    refuse it, a 500 after the gate. It's the pre-gate `400 text_too_long` instead."""
    from models.fast_streaming import MIN_SUFFIX_ROOM

    length = _prefix_len(NARRATOR_TEXT, VOICE_CODES)
    runtime = _voice_runtime(
        _RoomyRuntime(config=FakeStreamingConfig(max_seq_len=length + MIN_SUFFIX_ROOM))
    )
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 400
    assert response.json()["code"] == "text_too_long"
    assert runtime.prefix_builds == []
    assert _gate_is_free(server.components)


class _CodesPathTightRuntime(FakeRuntime):
    """No room on the codes path (a whole prompt carrying the reference audio, no prefix); the
    ordinary room everywhere else, the prefix path included. Records the thread every codes-path
    measurement ran on."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.codes_path_threads: list[str] = []

    def max_new_tokens_room(self, requested: Any, inputs: Any, *, prefix_len: int = 0) -> int:
        if prefix_len == 0 and inputs.get("input_values") is not None:
            self.codes_path_threads.append(threading.current_thread().name)
            return 0
        return super().max_new_tokens_room(requested, inputs, prefix_len=prefix_len)


def test_43_4_a_voice_only_its_prefix_path_fits_is_served(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """Review 43 #4 (reversing review 42 #3): the pre-gate check sizes a voice without an
    override on its prefix path only. The codes path is the out-of-memory fallback's, and is
    measured only when that fallback is needed, so a voice near the limit isn't refused on every
    request because of a rare out-of-memory."""
    runtime = _voice_runtime(_CodesPathTightRuntime())
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 200
    assert len(runtime.prefix_builds) == 1
    assert runtime.calls[-1]["prefix"] is not None
    assert runtime.codes_path_threads == []  # the codes path isn't tokenised per request


def test_43_4_a_voice_with_an_override_still_needs_its_codes_path_to_fit(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    runtime = _voice_runtime(_CodesPathTightRuntime())
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id, ref_text="a client transcript")

    assert response.status_code == 400
    assert response.json()["code"] == "text_too_long"
    assert runtime.prefix_builds == []
    assert runtime.calls == []


class _OutOfMemoryCodesTightRuntime(_CodesPathTightRuntime):
    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        self.prefix_builds.append(prefix_inputs)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")


def test_43_4_an_out_of_memory_build_whose_codes_path_has_no_room_is_a_503(
    voice_server: Callable[..., SimpleNamespace],
) -> None:
    """The fallback's room is measured when the fallback is needed, on the CPU tokenizer's
    worker. With none, the request can't be served right now: a `503` with its own code, before
    the `200`, not a `400` after the gate that blames the text."""
    runtime = _voice_runtime(_OutOfMemoryCodesTightRuntime())
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 503
    assert response.json() == {
        "error": "not enough GPU memory for this voice right now",
        "code": "gpu_out_of_memory",
    }
    assert len(runtime.prefix_builds) == 1
    assert runtime.calls == []
    [thread] = runtime.codes_path_threads
    assert thread.startswith("breeze-cpu-tokenizer")
    [fallback] = _named(server, "speech.prefix_fallback")
    assert fallback["level"] == "warning"
    assert fallback["request_id"] == response.headers["x-request-id"]
    assert fallback["voice_id"] == voice_id
    assert fallback["reason"] == "no_room"
    assert "CUDA out of memory" in fallback["error"]
    assert _named(server, "speech.accepted") == []
    assert _named(server, "request.failed") == []
    assert _gate_is_free(server.components)


class _OutOfMemoryOnDemandRuntime(FakeRuntime):
    """Builds prefixes normally until `out_of_memory` is set, then runs out of memory."""

    def __init__(self) -> None:
        super().__init__()
        self.out_of_memory = False

    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        if not self.out_of_memory:
            return super().build_reference_prefix(prefix_inputs)
        self.prefix_builds.append(prefix_inputs)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")


def test_43_3_an_out_of_memory_build_evicts_every_cached_prefix_before_the_fallback(
    voice_server: Callable[..., SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other voices' cached KV (up to 1 GiB) would still be on the GPU when the codes path,
    which needs more memory than the failed build, starts. So every entry is evicted
    (`reason: oom`) and PyTorch's cache emptied, on the GPU thread, before the fallback runs."""
    runtime = _voice_runtime(_OutOfMemoryOnDemandRuntime())
    server = voice_server(runtime)
    warm_id = _save_voice(server, name="Other")
    assert _speech(server, voice_id=warm_id).status_code == 200
    assert len(server.services.prefix_cache) == 1
    voice_id = _add_unnamed_voice(server)
    runtime.out_of_memory = True
    emptied: list[tuple[int, str]] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: emptied.append((len(server.services.prefix_cache), threading.current_thread().name)),
    )
    emptied_at_fallback: list[list[tuple[int, str]]] = []
    recording_prepare_piece = routes_speech.prepare_piece

    def checking_prepare_piece(*args: Any) -> Any:
        emptied_at_fallback.append(list(emptied))
        return recording_prepare_piece(*args)

    monkeypatch.setattr(routes_speech, "prepare_piece", checking_prepare_piece)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 200
    assert len(server.services.prefix_cache) == 0
    evicted = [fields for fields in _named(server, "voice.prefix_evicted") if fields["reason"] == "oom"]
    assert evicted == [
        {"voice_id": warm_id, "reason": "oom", "request_id": response.headers["x-request-id"]}
    ]
    # The last empty_cache before the fallback ran after the eviction, on the GPU thread.
    cache_size, thread = emptied_at_fallback[0][-1]
    assert cache_size == 0
    assert thread.startswith("breeze-gpu")
    [fallback] = _named(server, "speech.prefix_fallback")
    assert fallback["reason"] == "out_of_memory"
    assert runtime.calls[-1]["prefix"] is None
    assert _gate_is_free(server.components)


class _OutOfMemoryHoldingRuntime(FakeRuntime):
    """`build_reference_prefix` allocates a large object and runs out of memory while it
    holds it, as a real build holds its GPU tensors in the frames of the failing call."""

    def __init__(self) -> None:
        super().__init__()
        self.held: Any = None

    def build_reference_prefix(self, prefix_inputs: dict[str, Any]) -> Any:
        import weakref

        self.prefix_builds.append(prefix_inputs)
        workspace = torch.zeros(1 << 20)  # a stand-in for the build's GPU tensors
        self.held = weakref.ref(workspace)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")


def test_42_2_the_failed_builds_memory_is_freed_before_the_fallback_starts(
    voice_server: Callable[..., SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the cyclic collector (disabled here): the fallback's codes path needs more
    memory than the build that failed, so nothing of that build may still be alive, through
    the exception's traceback or a task that kept the exception, when it starts."""
    import gc

    runtime = _voice_runtime(_OutOfMemoryHoldingRuntime())
    server = voice_server(runtime)
    voice_id = _add_unnamed_voice(server)
    alive_at_fallback: list[bool] = []
    recording_prepare_piece = routes_speech.prepare_piece

    def checking_prepare_piece(*args: Any) -> Any:
        if runtime.held is not None and not alive_at_fallback:
            alive_at_fallback.append(runtime.held() is not None)
        return recording_prepare_piece(*args)

    monkeypatch.setattr(routes_speech, "prepare_piece", checking_prepare_piece)
    gc.disable()
    try:
        response = _speech(server, voice_id=voice_id)
    finally:
        gc.enable()

    assert response.status_code == 200
    assert alive_at_fallback == [False]
    [fallback] = _named(server, "speech.prefix_fallback")
    assert "CUDA out of memory" in fallback["error"]


def test_42_7_a_warm_voice_request_neither_reassembles_the_prefix_nor_rehashes_its_key(
    voice_server: Callable[..., SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix length and cache key are the voice's own, computed once when it was
    registered: a request reads them off the registry record."""
    from breeze_infer import synthesis, templates, voice_registry

    server = voice_server()
    voice_id = _save_voice(server)
    assert _speech(server, voice_id=voice_id).status_code == 200  # builds the prefix
    assemblies: list[int] = []
    hashes: list[int] = []
    real_prefix_inputs = templates.prepare_prefix_inputs
    real_prefix_key = voice_registry.prefix_key

    def counting_prefix_inputs(*args: Any) -> Any:
        assemblies.append(1)
        return real_prefix_inputs(*args)

    def counting_prefix_key(*args: Any) -> Any:
        hashes.append(1)
        return real_prefix_key(*args)

    monkeypatch.setattr(synthesis, "prepare_prefix_inputs", counting_prefix_inputs)
    # The route no longer imports it at all; set anyway, so the check holds either way.
    monkeypatch.setattr(routes_speech, "prefix_key", counting_prefix_key, raising=False)
    monkeypatch.setattr(voice_registry, "prefix_key", counting_prefix_key)

    response = _speech(server, voice_id=voice_id)

    assert response.status_code == 200
    [_cold, warm] = _named(server, "speech.accepted")
    assert warm["warm"] is True
    assert assemblies == []
    assert hashes == []
    voice = server.services.registry.lookup(voice_id)
    assert voice.prefix_len == server.runtime.prefix_builds[0]["attention_mask"].shape[1]


def test_43_3_a_request_cancelled_during_the_cache_release_leaves_the_gate_held_until_it_ends() -> None:
    """The fallback's GPU-thread call (emptying PyTorch's cache) has no slot in the route's
    cleanup, so a cancel while it runs hands the gate to the call, as a prefix build does."""
    gate = GpuGate()
    gpu = GpuThread("cpu", lambda _device: None)
    started = threading.Event()
    release = threading.Event()

    def slow_release() -> None:
        started.set()
        assert release.wait(5.0)

    async def main() -> None:
        lease = gate.try_acquire()
        assert lease is not None
        task = asyncio.ensure_future(routes_speech._gpu_call_under_lease(gpu, lease, slow_release))
        await _until(started.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lease.release()  # the route's own cleanup: a no-op once handed over
        assert gate.try_acquire() is None  # the call still holds it
        release.set()
        await _until(lambda: _gate_is_free(SimpleNamespace(gate=gate)))

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()
