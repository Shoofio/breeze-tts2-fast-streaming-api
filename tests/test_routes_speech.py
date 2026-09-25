"""`POST /v1/audio/speech` tests (specs/003-cpp-compatible-api/tasks.md T041/T042).

Built through `api.create_app` plus `routes_speech.install_speech`, the same wiring
`api.py` uses (mirrors `tests/test_health.py`), with `tests/fakes.py`'s GPU-free stand-ins
at the model edge (`FakeRuntime`, `FakeCodec`, `FakeTokenizer`) -- the Principle V
deviation `tests/fakes.py`'s own docstring records. `FakeRuntime` itself carries no
`tokenizer`/`model`/`audio_tokenizer` (most of its other consumers never need them), so
`_fake_runtime` attaches the same fakes `tests/test_synthesis.py` uses directly onto it,
duck-typing the real `FastBreezeStreamingRuntime`'s own attributes
(`models/fast_streaming.py`).
"""

from __future__ import annotations

import asyncio
import io
import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from breeze_infer import __version__
from breeze_infer.api import Components
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.cors import CorsMiddleware, CorsPolicy
from breeze_infer.errors import install_error_handlers
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.routes_health import Readiness, install_health
from breeze_infer.routes_speech import install_speech
from breeze_infer.settings import settings_from_args
from breeze_infer.version_header import VersionHeaderMiddleware
from tests.fakes import (
    FakeCodec,
    FakeRuntime,
    FakeStreamingConfig,
    FakeTokenizer,
    fake_model,
)

SPEECH_PATH = "/v1/audio/speech"


def _model_with_codec_facts() -> Any:
    """`fake_model()` plus the codec facts `templates._codec_facts` requires
    (`codec_config.codebook_size`). Duplicated from `tests/test_synthesis.py`'s own
    helper of the same name and shape: `tests/fakes.py`'s `fake_model()` deliberately
    leaves this out (another module's own test exercises that omission), so every
    caller that needs a model `prepare_piece` can actually use adds it locally.
    """
    model = fake_model()
    model.config.codec_config.codebook_size = 2048
    return model


def _fake_runtime(**kwargs: Any) -> FakeRuntime:
    """A `FakeRuntime` wired up with the `tokenizer`/`model`/`audio_tokenizer` attributes
    the real `FastBreezeStreamingRuntime` carries (`models/fast_streaming.py`'s
    `self.tokenizer`/`self.model`/`self.audio_tokenizer`) -- `routes_speech.py` reads
    them straight off the object `Readiness.require_ready` returns.
    """
    runtime = FakeRuntime(**kwargs)
    runtime.tokenizer = FakeTokenizer()
    runtime.model = _model_with_codec_facts()
    runtime.audio_tokenizer = FakeCodec()
    return runtime


def _request_ids() -> Callable[[], str]:
    counter = itertools.count()
    return lambda: f"req-{next(counter)}"


def _build_components(readiness: Readiness, *, split_chars: int | None = None) -> Components:
    # The model directory is never opened: nothing loads in these tests (test_health.py's
    # own _components does the same).
    argv = [str(Path(__file__).parent)]
    if split_chars is not None:
        argv += ["--split-chars", str(split_chars)]
    return Components(
        settings=settings_from_args(argv),
        events=Emitter(io.StringIO(), lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: 0,
    )


def _client_for(components: Components) -> TestClient:
    """Builds the app the way `api.create_app` does -- `install_speech` needs the raw
    FastAPI `app` (for `app.post(...)`), which `create_app` doesn't expose: it returns
    the fully wrapped ASGI app instead. This mirrors `create_app`'s own steps, with
    `install_speech` slotted in next to `install_health`, where the main session's own
    wiring into `api.py` is expected to put it (routes_speech.py's module docstring).
    """
    app = FastAPI(title="Breeze TTS", docs_url=None, redoc_url=None, openapi_url=None)
    install_error_handlers(app, components.events)
    install_health(app, components.readiness, components.ws_port)
    install_speech(app, components, clock=time.monotonic, new_request_id=_request_ids())

    policy = CorsPolicy(origins=components.settings.cors)
    inner = CorsMiddleware(BodyLimitMiddleware(app), policy, app.router)
    wrapped = VersionHeaderMiddleware(inner, version=__version__)
    return TestClient(wrapped)


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
