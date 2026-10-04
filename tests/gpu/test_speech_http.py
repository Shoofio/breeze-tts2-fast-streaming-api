"""GPU smoke test for `POST /v1/audio/speech` (tasks.md T043, part 1: tests only).

Builds the *whole* app through `breeze_infer.api.create_app` -- the same wiring
`breeze_infer.api.main` uses in production -- on the real, warmed runtime
`tests/gpu/conftest.py`'s session-scoped `gpu_env` fixture loads. The components are
marked ready with that runtime and two CPU copies of its tokenizer through
`Components.mark_ready`, the step `load_in_background` ends with, and each test gets its own `GpuGate` /
`GpuThread` / `RecordingEvents`, so a lease or event held by one test can never
leak into the next.

Threading note (see the task's own instructions): `gpu_env` loads and warms the
model on whatever thread pytest resolves the fixture on -- a synchronous fixture,
so in practice the main test-collection thread -- while `GpuThread` (`breeze_infer/gpu.py`)
then runs every request's GPU work on its *own*, separate worker thread. That is
exactly the split production already has between `main()`'s `load_in_background`
(also run via a `GpuThread`, just a different instance/thread than the one here)
and every later request. The CUDA graph modules under `models/cudagraph/` and
`models/text_encoder_graph.py` capture onto an explicit `torch.cuda.Stream()` and,
on every replay, synchronize it against whichever thread's *current* stream is
calling in (`stream.wait_stream(current_stream)` / `current_stream.wait_stream(stream)`
on both sides) rather than assuming replay happens on the capture thread -- so a
different replay thread is expected to work. This file's own pass/fail is the
actual verification that no such device/stream mismatch exists in practice.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from breeze_infer.api import Components, create_app
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import MAX_TEXT_CHARS
from breeze_infer.model_loading import LoadedModel
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.runtime import resolve_device
from breeze_infer.settings import DEFAULT_CHUNK_MAX, settings_from_args
from breeze_infer.streaming import BYTES_PER_SAMPLE
from tests.fakes import RecordingEvents, open_no_voices

pytestmark = pytest.mark.gpu

SPEECH_PATH = "/v1/audio/speech"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Unset on a machine without the reference voices, which can still run the rest of this
# file; the inline-reference test skips (rather than erroring) when the sample is absent.
REFERENCE_VOICES_DIR = (
    Path(v) if (v := os.environ.get("REFERENCE_VOICES_DIR")) else None
)
VOICE_DIR = REFERENCE_VOICES_DIR / "eric" if REFERENCE_VOICES_DIR else None
SAMPLE_RATE = 24000
# contracts/http-api.md "Chunks grow from --chunk-first to --chunk-max codec frames (1,920
# samples per frame)": the largest possible ramp flush, used as the tolerance for matching
# a completed response's `audio_seconds_sent` against the body it actually sent.
CODEC_FRAME_SAMPLES = 1920
TERMINAL_EVENTS = ("speech.completed", "speech.aborted", "speech.failed")
# `SpeechResponse.__call__`'s own `finally` (breeze_infer/streaming.py) closes the GPU
# session and emits the outcome event only *after* the chunked terminator has already
# reached the client; TestClient's ASGI transport can hand `response.content` back as soon
# as it has seen that terminator, without waiting for the app task's `finally` to actually
# run. So the event can still be a few loop iterations away even once `.post()` has
# returned -- short deadline poll rather than an immediate assert.
EVENT_POLL_TIMEOUT_SECONDS = 5.0


_loaded_model: LoadedModel | None = None


def _shared_loaded_model(runtime: Any) -> LoadedModel:
    """One `LoadedModel.from_runtime(runtime)` -- and so one pair of CPU tokenizer copies, the
    pre-gate check's and the anchor sizing's -- for the whole test session (review 32,
    review-of-2d9070a #7 in the same pass; a pair since review 34 finding 5), not one per
    `_components` call.

    `from_runtime` makes fresh copies every time -- deliberately, in production, where it runs
    once before the server ever reports ready (each deep copy takes about 700 ms). Here, every
    GPU test in this session shares the same session-scoped, already-warmed
    `gpu_env.runtime`, whose tokenizer never changes underneath it, and `_components` is
    called once per test -- `test_speech_long_text.py`'s own multi-run test even calls it once
    per run -- so repeating those deep copies each time only adds test wall-clock for no
    safety benefit: `CpuTokenizer` docstring's "one thread at a time" rule is about concurrent
    use, not object identity, and these tests never run two GPU tests at once. Each test's
    workers do use the shared copies, though, so every teardown ends with
    `shut_down_cpu_tokenizer` (review 33 on 10f0c29): a call still running there must finish
    before the next test's workers touch the same copy.
    """
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = LoadedModel.from_runtime(runtime, {})
    return _loaded_model


def _components(gpu_env) -> tuple[Components, RecordingEvents]:
    # The model directory is never opened here (nothing loads through Settings): the
    # components are marked ready with `gpu_env.runtime` by the same step the server's own
    # load ends with (`Components.mark_ready`), which also installs the speech route's two CPU
    # tokenizer copies -- the session-shared pair above, not fresh copies per call.
    events = RecordingEvents()
    device = resolve_device()
    components = Components(
        settings=settings_from_args([str(REPO_ROOT)]),
        events=events,
        gate=GpuGate(),
        gpu=GpuThread(device, torch.cuda.set_device),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=open_no_voices,
    )
    components.mark_ready(_shared_loaded_model(gpu_env.runtime), open_no_voices(None))
    return components, events


# How long a teardown waits for a CPU tokenizer call still running. Sizing the longest text
# takes well under a second, so a call still running after this is wedged.
CPU_TOKENIZER_JOIN_SECONDS = 30.0


def shut_down_cpu_tokenizer(components: Components) -> None:
    """Stop `components`' CPU tokenizer workers and wait for any call still running, since the
    next test's workers use the same tokenizer copies (`_shared_loaded_model`). The wait is
    bounded (review 34 finding 6): a wedged call fails this teardown instead of hanging the
    whole GPU session."""
    components.cpu_tokenizer.shutdown()
    assert components.cpu_tokenizer.join(CPU_TOKENIZER_JOIN_SECONDS), (
        f"a CPU tokenizer call was still running {CPU_TOKENIZER_JOIN_SECONDS:.0f} s after "
        "shutdown; the next test would share its tokenizer copies"
    )


@pytest.fixture()
def speech_app(gpu_env) -> Iterator[tuple[TestClient, RecordingEvents]]:
    components, events = _components(gpu_env)
    client = TestClient(create_app(components))
    try:
        yield client, events
    finally:
        # Nested, not sequential (review 32, review-of-2d9070a #4/#7 in the same pass): a
        # `gpu.shutdown()` failure must not leave the CPU tokenizer's own executor running
        # past the test.
        try:
            components.gpu.shutdown()
        finally:
            shut_down_cpu_tokenizer(components)


def _pcm_stats(body: bytes) -> tuple[int, bool]:
    """(sample count, whether any sample is non-zero) for a headerless s16le body."""
    samples = np.frombuffer(body, dtype="<i2")
    return samples.size, bool(np.any(samples != 0))


def _wait_for_terminal_event(events: RecordingEvents) -> None:
    """Block (briefly) until `events` has recorded one of `TERMINAL_EVENTS`.

    See `EVENT_POLL_TIMEOUT_SECONDS`'s comment: the response has already reached the test
    client by the time `.post()` returns, but `SpeechResponse.__call__`'s cleanup -- which
    is what actually emits the outcome event -- can still be a few loop iterations behind.
    """
    deadline = time.monotonic() + EVENT_POLL_TIMEOUT_SECONDS
    while not any(name in TERMINAL_EVENTS for name, _ in events.calls):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"no {TERMINAL_EVENTS} event within {EVENT_POLL_TIMEOUT_SECONDS}s "
                f"(got: {events.calls})"
            )
        time.sleep(0.01)


def _assert_plausible_speech(response, events: RecordingEvents) -> None:
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-sample-rate"] == str(SAMPLE_RATE)
    body = response.content
    assert len(body) > 0
    assert len(body) % 2 == 0  # s16le: every sample is 2 bytes
    sample_count, has_signal = _pcm_stats(body)
    duration = sample_count / SAMPLE_RATE
    assert 0.5 <= duration <= 20.0, f"implausible duration {duration:.2f}s for a short sentence"
    assert has_signal, "audio is entirely silence"

    # Prove the stream actually finished (and, for the multi-piece test, that every piece
    # ran): exactly one `speech.completed`, never a `speech.aborted`/`speech.failed`
    # alongside or instead of it.
    _wait_for_terminal_event(events)
    completed = [fields for name, fields in events.calls if name == "speech.completed"]
    aborted = [fields for name, fields in events.calls if name == "speech.aborted"]
    failed = [fields for name, fields in events.calls if name == "speech.failed"]
    assert len(completed) == 1, f"expected exactly one speech.completed, got {events.calls}"
    assert not aborted and not failed, f"unexpected abort/failure events: {events.calls}"

    expected_seconds = len(body) / BYTES_PER_SAMPLE / SAMPLE_RATE
    tolerance_seconds = DEFAULT_CHUNK_MAX * CODEC_FRAME_SAMPLES / SAMPLE_RATE
    assert completed[0]["audio_seconds_sent"] == pytest.approx(
        expected_seconds, abs=tolerance_seconds
    )


def test_voice_design_returns_plausible_audio(speech_app) -> None:
    client, events = speech_app

    response = client.post(SPEECH_PATH, data={"text": "Hello there, this is a short test."})

    _assert_plausible_speech(response, events)


def test_inline_reference_returns_plausible_audio(speech_app) -> None:
    client, events = speech_app
    if VOICE_DIR is None:
        pytest.skip(
            "REFERENCE_VOICES_DIR is not set "
            "(set it to a directory containing eric/eric.wav and eric.txt)"
        )
    ref_wav = VOICE_DIR / "eric.wav"
    ref_txt = VOICE_DIR / "eric.txt"
    if not ref_wav.is_file() or not ref_txt.is_file():
        pytest.skip(
            f"reference voice sample not found under {VOICE_DIR} "
            "(set REFERENCE_VOICES_DIR to a directory containing eric/eric.wav and eric.txt)"
        )
    ref_text = ref_txt.read_text(encoding="utf-8").strip()

    with ref_wav.open("rb") as ref_audio:
        response = client.post(
            SPEECH_PATH,
            data={"text": "Hello there, this is a short test.", "ref_text": ref_text},
            files={"ref_audio": ("eric.wav", ref_audio, "audio/wav")},
        )

    _assert_plausible_speech(response, events)

    # T049: reference_audio.predicted_frames' formula (an approximation of the real
    # codec's own resample+frame arithmetic, reference_audio.py's own docstring) must
    # actually agree with the real bundled codec on a genuine reference clip -- this is
    # the one thing tests/test_reference_audio.py's unit tests, which never touch the
    # real codec, can't itself prove. A mismatch would still be reported as a warning,
    # not fail the request (routes_speech.py's _check_frame_prediction docstring), so
    # this is checked as its own assertion rather than folded into a busy status check.
    mismatches = [fields for name, fields in events.calls if name == "speech.frame_prediction_mismatch"]
    assert mismatches == [], f"predicted_frames disagreed with the real codec: {mismatches}"


def test_multi_piece_request_splits_and_emits_pieces(speech_app) -> None:
    client, events = speech_app
    text = (
        "The city was quiet this morning. "
        "A gentle breeze moved through the tall trees outside. "
        "Everyone welcomed the sunshine after last night's storm."
    )

    response = client.post(SPEECH_PATH, data={"text": text, "split_chars": "40"})

    accepted = [fields for name, fields in events.calls if name == "speech.accepted"]
    assert len(accepted) == 1
    assert accepted[0]["pieces"] >= 2

    # Also proves every piece actually ran to completion, not just that the route
    # accepted a multi-piece plan: exactly one speech.completed, matching the full body.
    _assert_plausible_speech(response, events)

    # T046 review, finding 7: the whole stream being non-empty only proves *some*
    # piece produced audio -- a silently empty later piece would still leave the
    # overall response non-empty. speech.piece_done (one per piece, emitted from
    # routes_speech.py's _iter_pieces) catches that: every piece the request accepted
    # must actually report frames > 0, in order.
    piece_events = [fields for name, fields in events.calls if name == "speech.piece_done"]
    assert len(piece_events) == accepted[0]["pieces"]
    assert [fields["piece_index"] for fields in piece_events] == list(range(len(piece_events)))
    assert all(fields["frames"] > 0 for fields in piece_events), piece_events


def test_first_piece_without_room_gives_400_text_too_long(gpu_env, speech_app) -> None:
    """BC-47's "first piece has no room" half: `split_chars=0` keeps the whole text as
    one piece, and it must be dense enough that its tokenized length alone -- before
    any CFG or graph-bucket padding -- already leaves no room in the 2,048-token
    context. Grown from the real tokenizer rather than a fixed string, so this stays
    reachable (or honestly reports that it isn't) regardless of exactly how many
    tokens per character this tokenizer produces.
    """
    client, _ = speech_app
    max_seq_len = gpu_env.runtime.config.max_seq_len
    unit = "這是一段用來測試上下文長度是否足夠的密集中文句子，內容不需要有意義。"

    text = ""
    token_count = 0
    while len(text) + len(unit) <= MAX_TEXT_CHARS:
        text += unit
        token_count = len(gpu_env.tokenizer(text)["input_ids"])
        if token_count > max_seq_len:
            break

    if token_count <= max_seq_len:
        pytest.skip(
            "even the full MAX_TEXT_CHARS (10,000) of dense CJK text tokenizes to only "
            f"{token_count} tokens, short of the {max_seq_len}-token context -- BC-47's "
            "first-piece no-room case isn't reachable within the text length limit with "
            "this tokenizer"
        )

    response = client.post(SPEECH_PATH, data={"text": text, "split_chars": "0"})

    assert response.status_code == 400
    assert response.json() == {"error": "text is too long", "code": "text_too_long"}
