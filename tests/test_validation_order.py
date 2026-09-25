"""FR-007 validation-order tests (specs/003-cpp-compatible-api/tasks.md T046).

contracts/http-api.md "Validation order" (BC-07):

    1. body size (413); 2. field syntax and ranges (400); 3. reference consistency
    (400); 4. unknown voice_id (404); 5. audio decode and limits (400); 6. busy (409).

Each test below either holds the `GpuGate` (so a request that reaches the busy check
would get `409`) or combines two stages' violations in one request, and asserts the
*earlier* stage's error wins -- proving the later stage's check never even ran.

Reuses `tests/test_routes_speech.py`'s app-wiring helpers/fixtures rather than
rebuilding them.
"""

# Every fixture imported below (`client`, `components`, `readiness`, `ready_client`) is
# reused, by pytest's own name-based discovery, as a same-named parameter on the tests
# in this file -- the standard way to share fixtures across modules without a
# `conftest.py`. Ruff's F811 ("redefinition") otherwise fires on every one of those
# parameters (and on the same-named local variables a couple of tests below also use),
# since it can't tell a fixture parameter/shadow from an accidental one.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from breeze_infer.gpu import GpuGate
from breeze_infer.limits import MAX_BODY_BYTES
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import _serve_speech
from models.fast_streaming import NoRoomError
from tests.fakes import FakeCodec, RecordingEvents
from tests.test_routes_speech import (  # noqa: F401 -- fixtures reused by name
    SPEECH_PATH,
    _build_components,
    _client_for,
    _fake_runtime,
    _wav_bytes,
    client,
    components,
    readiness,
    ready_client,
)

# --- BC-07: an invalid request while busy gets its validation error, not 409 -------


def test_bc_07_invalid_request_while_busy_gets_400_not_409(
    ready_client: TestClient, components
) -> None:
    """C++ checked "busy" before validating anything; a request that's invalid on its
    own must never see 409 just because generation happens to be running."""
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(
            SPEECH_PATH, data={"text": "hello", "cfg_scale": "banana"}
        )
    finally:
        lease.release()

    assert response.status_code == 400
    assert response.json()["code"] in ("invalid_field",)


# --- field syntax (stage 2) before unknown voice_id (stage 4) ----------------------


def test_field_error_before_unknown_voice_id(ready_client: TestClient) -> None:
    """A request that is both field-invalid and names an unknown voice must fail on
    the field, not the voice lookup -- proof field parsing runs first."""
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello", "voice_id": "no-such-voice", "cfg_scale": "banana"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# --- reference consistency (stage 3) before decode (stage 5) -----------------------


def test_reference_rule_error_before_decode(ready_client: TestClient) -> None:
    """`ref_audio` with no `ref_text` is a reference-consistency error (stage 3); it
    must fire even when the audio itself is undecodable garbage that would otherwise
    also fail at decode (stage 5) -- proof the reference rules run first, without ever
    touching the decoder."""
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello"},
        files={"ref_audio": ("ref.wav", b"not audio, just noise" * 50, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text is required with ref_audio",
        "code": "ref_text_required",
    }


# --- unknown voice_id (stage 4) before busy (stage 6) -------------------------------


def test_unknown_voice_before_busy(ready_client: TestClient, components) -> None:
    """An unknown `voice_id` needs no reference decode at all, so it should reach its
    404 even while the gate is held -- proof the voice lookup (stage 4) runs before
    busy (stage 6), skipping straight past decode (stage 5, nothing to decode)."""
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(
            SPEECH_PATH, data={"text": "hello", "voice_id": "no-such-voice"}
        )
    finally:
        lease.release()

    assert response.status_code == 404
    assert response.json() == {"error": "unknown voice_id", "code": "unknown_voice"}


# --- reference consistency (stage 3) before busy (stage 6) --------------------------


def test_reference_rule_error_before_busy(ready_client: TestClient, components) -> None:
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(
            SPEECH_PATH,
            data={"text": "hello"},
            files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    finally:
        lease.release()

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text is required with ref_audio",
        "code": "ref_text_required",
    }


# --- decode (stage 5) before busy (stage 6) -----------------------------------------


def test_decode_error_before_busy(ready_client: TestClient, components) -> None:
    """Undecodable audio, with a well-formed `ref_text` alongside it (so the reference
    rules at stage 3 already pass), must still fail at decode (stage 5) rather than
    busy (stage 6) -- held gate proves it."""
    lease = components.gate.try_acquire()
    assert lease is not None
    try:
        response = ready_client.post(
            SPEECH_PATH,
            data={"text": "hello", "ref_text": "a transcript"},
            files={"ref_audio": ("ref.wav", b"not audio, just noise" * 50, "audio/wav")},
        )
    finally:
        lease.release()

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_audio"


# --- BC-06: oversize multipart gets the 413 envelope --------------------------------


def test_bc_06_oversize_multipart_gets_413_envelope(ready_client: TestClient) -> None:
    """BC-06: the C++ server has no body-size cap at all. `body_limit.py`'s immediate-
    rejection path decides this from `Content-Length` alone, so a spoofed header is
    enough -- no need to actually upload 26+ MiB."""
    response = ready_client.post(
        SPEECH_PATH,
        data={"text": "hello"},
        headers={"content-length": str(MAX_BODY_BYTES + 1)},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }


# --- Coordinator review of the T045-T049 batch (routes_speech.py) ------------------
#
# The tests below cover the additional fixes requested alongside T045-T049: only
# NoRoomError maps to 400 text_too_long (finding 1); a lease is never leaked to a
# second cancellation during a slow GPU call (finding 2); X-Request-Id and a correct
# ttfa_ms reach every response, including ones a dependency failure answers before the
# route body ever runs (findings 5, 8); and DONE-with-no-audio still reports its
# speech.failed event regardless of what session.aclose() does next (finding 6).


class _RaisingRuntime:
    """Wraps a real `FakeRuntime` (`tests/fakes.py`), but `iter_audio_chunks` always
    raises `error` on its first `next()` -- lets these tests drive
    `routes_speech.py`'s own exception-mapping for `session.step()`'s first call
    directly, without needing `tests/fakes.py`'s own room-check path (out of this
    task's touch list -- see the T046 coordinator note) to ever reach that condition.
    """

    def __init__(self, inner: Any, error: BaseException) -> None:
        self._inner = inner
        self._error = error
        self.tokenizer = inner.tokenizer
        self.model = inner.model
        self.audio_tokenizer = inner.audio_tokenizer
        self.sample_rate = inner.sample_rate

    def max_new_tokens_room(self, *args: Any, **kwargs: Any) -> int:
        return self._inner.max_new_tokens_room(*args, **kwargs)

    def iter_audio_chunks(self, *_args: Any, **_kwargs: Any):
        def _gen():
            raise self._error
            yield  # pragma: no cover -- unreachable; makes this a generator function

        return _gen()


def test_no_room_error_during_priming_maps_to_400_text_too_long() -> None:
    """T046 review, finding 1: `models.fast_streaming.NoRoomError` (a `ValueError`
    subclass) raised once priming has moved past piece 0 within a single step must
    still become `400 text_too_long`, the same as the pre-check for piece 0 itself."""
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        runtime = _RaisingRuntime(
            _fake_runtime(), NoRoomError("prompt leaves no room to generate")
        )
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 400
        assert response.json() == {"error": "text is too long", "code": "text_too_long"}
    finally:
        components.gpu.shutdown()


def test_other_valueerror_during_priming_stays_a_500_not_400() -> None:
    """T046 review, finding 1: a bare `ValueError` that isn't `NoRoomError` (a server
    bug -- a bad override, a malformed `inputs` shape) must not be mistaken for "text
    too long". Before this fix, `except ValueError` caught *every* `ValueError` here,
    so this same request would have come back `400 text_too_long` instead of `500`."""
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        runtime = _RaisingRuntime(
            _fake_runtime(), ValueError("some unrelated invariant broke")
        )
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 500
        assert response.json() == {"error": "internal error", "code": "internal_error"}
        failed = [fields for name, fields in events.calls if name == "request.failed"]
        assert len(failed) == 1
        assert "some unrelated invariant broke" in failed[0]["error"]
    finally:
        components.gpu.shutdown()


# --- finding 5/8: a dependency failure (503) still carries X-Request-Id ------------


def test_503_gpu_unavailable_carries_x_request_id() -> None:
    """`GpuGate.poison()` makes `try_acquire`/`acquire` raise `GpuUnavailable`, answered
    by `errors.py`'s own handler *before* `routes_speech.py`'s route body ever runs
    (`require_ready` succeeds -- the model is ready -- but the gate itself is poisoned).
    T046 review, finding 8: this response must still carry `X-Request-Id`, sourced from
    the `_name_request` dependency that now runs ahead of `require_ready`, not from the
    route body (which never executes for this failure)."""
    readiness = Readiness()
    components = _build_components(readiness)
    try:
        readiness.mark_ready(_fake_runtime())
        components.gate.poison()
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 503
        assert response.json()["code"] == "gpu_unavailable"
        assert response.headers["x-request-id"]
    finally:
        components.gpu.shutdown()


@pytest.mark.xfail(
    reason=(
        "the 503 loading body is answered by routes_health.py's own ModelLoading "
        "handler (_loading_handler), not errors.py's ApiError/GpuUnavailable handlers "
        "this batch fixed -- routes_health.py is outside this batch's touch list, so "
        "this documents the gap rather than silently passing over it (T046 review, "
        "finding 8, remaining scope)."
    ),
    strict=True,
)
def test_503_loading_carries_x_request_id(client: TestClient) -> None:
    """The model isn't ready at all: `require_ready` itself raises before the route
    body runs. Same finding 8 -- an id must still reach the client."""
    response = client.post(SPEECH_PATH, data={"text": "hello there"})

    assert response.status_code == 503
    assert response.json()["code"] == "loading"
    assert response.headers["x-request-id"]


# --- finding 5: ttfa_ms is measured from receipt, not from generation start --------


def test_ttfa_ms_is_measured_from_receipt_not_generation_start() -> None:
    """`received_at` is read once, at the top of the route, before validation/decode;
    `started_at` (a separate, later reading) is when generation actually starts.
    `speech.first_audio`'s `ttfa_ms` must reflect the first gap, not the second --
    proven with a fake clock whose readings the two would disagree about if `ttfa_ms`
    were wrongly computed from `started_at` instead."""
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        readiness.mark_ready(_fake_runtime())

        # A clock that ticks 1.0s on every read after the first: received_at reads
        # "0.0", and by the time speech.first_audio's own clock() call happens (several
        # reads later, well after generation has started), a wrong "from started_at"
        # computation would report something close to 0 (started_at and that later
        # reading are close together), while the correct "from received_at" one
        # accumulates every tick in between -- large and clearly nonzero either way,
        # but the two would disagree sharply if the wrong origin were used.
        ticks = [0.0]

        def clock() -> float:
            value = ticks[0]
            ticks[0] += 1.0
            return value

        # `create_app` (api.py) hardcodes `clock=time.perf_counter`, with no injection
        # point -- api.py isn't in this batch's touch list -- so this builds the same
        # three pieces `create_app` does, by hand, with the fake clock wired to
        # `install_speech` directly (mirrors `test_body_limit.py`'s own small
        # hand-built apps for the same reason: a real dependency this test needs isn't
        # otherwise reachable).
        from fastapi import FastAPI

        from breeze_infer.errors import install_error_handlers
        from breeze_infer.routes_health import install_health
        from breeze_infer.routes_speech import install_speech

        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        install_error_handlers(app, events)
        install_health(app, readiness, components.ws_port)
        install_speech(app, components, clock=clock, new_request_id=lambda: "test-request")
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 200
        first_audio = next(
            fields for name, fields in events.calls if name == "speech.first_audio"
        )
        # At least several whole ticks elapsed between received_at (reading 0) and the
        # speech.first_audio clock() call: parsing, reference handling and priming each
        # read the clock at least once in between on this path.
        assert first_audio["ttfa_ms"] >= 2000.0
    finally:
        components.gpu.shutdown()


# --- finding 6: DONE-with-no-audio reports speech.failed regardless of the close ----


def test_done_with_no_audio_reports_speech_failed_reason_no_audio() -> None:
    readiness = Readiness()
    events = RecordingEvents()
    components = _build_components(readiness, events=events)
    try:
        runtime = _fake_runtime(chunks=0)  # DONE on the very first step
        readiness.mark_ready(runtime)
        client = _client_for(components)

        response = client.post(SPEECH_PATH, data={"text": "hello there"})

        assert response.status_code == 500
        failed = [fields for name, fields in events.calls if name == "speech.failed"]
        assert len(failed) == 1
        assert failed[0]["reason"] == "no_audio"
        # Emitted -- and so already decided -- before the close runs (T046 review,
        # finding 6): no gpu.close_failed for an ordinary close, so this is the only
        # outcome-shaped event this request produced.
        close_failed = [name for name, _ in events.calls if name == "gpu.close_failed"]
        assert close_failed == []
    finally:
        components.gpu.shutdown()


# --- finding 2: a lease is never leaked to a second cancellation -------------------


def _multipart_request(text: str, ref_text: str, ref_audio: bytes) -> tuple[bytes, str]:
    """A real, pre-encoded `multipart/form-data` body + `Content-Type`, built by
    `httpx` rather than by hand: this test needs a genuine `starlette.requests.Request`
    to drive `_serve_speech` directly (below its route wrapper), which needs real wire
    bytes to parse, not a `TestClient` call -- this scenario cancels the serving
    coroutine directly, which a synchronous `TestClient.post()` call has no way to do.
    """
    built = httpx.Request(
        "POST",
        "http://test/v1/audio/speech",
        data={"text": text, "ref_text": ref_text},
        files={"ref_audio": ("ref.wav", ref_audio, "audio/wav")},
    )
    built.read()
    return built.content, built.headers["content-type"]


def _asgi_request(body: bytes, content_type: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/speech",
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode("latin-1"))],
    }
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


class _SlowCodec(FakeCodec):
    """A `FakeCodec` whose `encode()` blocks on a `threading.Event` first -- lets a
    test hold `resolve_reference`'s GPU-thread call open for as long as it likes,
    simulating a slow reference encode."""

    def __init__(self, unblock: threading.Event) -> None:
        super().__init__()
        self._unblock = unblock

    def encode(self, wav: Any, sr: int, return_dict: bool = True) -> Any:
        self._unblock.wait(timeout=5.0)
        return super().encode(wav, sr, return_dict=return_dict)


def test_lease_survives_a_second_cancellation_during_a_slow_resolve_reference() -> None:
    """T046 review, finding 2: cancelling the request coroutine *twice* while it waits
    (shielded) for an in-flight GPU call used to be able to skip the lease release
    entirely (`with contextlib.suppress(Exception)` doesn't catch `CancelledError`, so
    a second cancellation escaped it, skipping the `lease.release()`/`session.aclose()`
    line that followed). The fix releases from a `gpu_task` done-callback instead, so
    the gate stays held only until the GPU call actually finishes, then frees --
    regardless of how many times this coroutine itself was cancelled meanwhile.
    """

    async def _gate_is_busy(gate: GpuGate) -> bool:
        # `try_acquire()` *takes* the gate when it succeeds (it's a real acquire, not a
        # peek), so a probe that finds it free must hand it straight back -- otherwise
        # this check would itself be the thing holding the gate.
        probe = gate.try_acquire()
        if probe is None:
            return True
        probe.release()
        return False

    async def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.005) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await predicate():
                return True
            await asyncio.sleep(interval)
        return False

    async def scenario() -> None:
        readiness = Readiness()
        components = _build_components(readiness)
        try:
            unblock = threading.Event()
            runtime = _fake_runtime()
            runtime.audio_tokenizer = _SlowCodec(unblock)
            readiness.mark_ready(runtime)

            body, content_type = _multipart_request("hello there", "a transcript", _wav_bytes())
            http_request = _asgi_request(body, content_type)

            task = asyncio.ensure_future(
                _serve_speech(
                    http_request,
                    runtime,
                    components,
                    request_id="test-request",
                    received_at=0.0,
                    clock=lambda: 0.0,
                )
            )
            # Let the coroutine run up to (and block inside) the slow encode: field
            # parsing and the real wav decode (asyncio.to_thread) both cross a real OS
            # thread boundary, which needs real wall-clock time to resolve, not just
            # event-loop yields -- so this polls with real (short) sleeps rather than
            # counting a fixed number of `asyncio.sleep(0)` turns.
            assert await _wait_until(lambda: _gate_is_busy(components.gate)), (
                "gate never became busy -- did _serve_speech fail before try_acquire?"
                f" task done={task.done()}"
                + (f" exception={task.exception()!r}" if task.done() and not task.cancelled() else "")
            )

            task.cancel()
            await asyncio.sleep(0)  # enters the except block, starts awaiting the shield
            task.cancel()  # a *second* cancellation while that await is still pending
            for _ in range(20):
                await asyncio.sleep(0)

            # The GPU call is still blocked (unblock isn't set yet): the gate must still
            # be held, whatever became of `task` itself.
            assert await _gate_is_busy(components.gate), "gate freed before the GPU call finished"

            unblock.set()  # let the blocked encode() finish on the GPU thread

            async def _gate_is_free() -> bool:
                probe = components.gate.try_acquire()
                if probe is None:
                    return False
                probe.release()
                return True

            freed = await _wait_until(_gate_is_free)
            assert freed, "gate never freed after the GPU call finished"
            lease = components.gate.try_acquire()
            assert lease is not None
            lease.release()

            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            components.gpu.shutdown()

    asyncio.run(scenario())
