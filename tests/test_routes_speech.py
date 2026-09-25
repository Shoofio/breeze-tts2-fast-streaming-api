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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from breeze_infer.api import Components, create_app
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.routes_health import Readiness
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
    components = _build_components(readiness, split_chars=15)
    try:
        runtime = _fake_runtime(chunks=1)
        readiness.mark_ready(runtime)
        client = _client_for(components)

        # Each sentence is well under the 15-char budget on its own, but any two combined
        # are over it, so text_split.py packs them into exactly 3 pieces (see the route's
        # own docstring on how piece 0's inputs vs. later pieces' are built).
        response = client.post(
            SPEECH_PATH,
            data={"text": "Hi there. Go now yes. See you soon.", "seed": "100"},
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
