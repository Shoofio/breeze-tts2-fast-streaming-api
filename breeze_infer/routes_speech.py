"""`POST /v1/audio/speech` (specs/003-cpp-compatible-api/tasks.md T041/T042;
contracts/http-api.md).

FR-007's order, as far as this phase goes (the full order -- body limit, then fields, then
reference rules, then unknown voice, then decode, then busy -- arrives with T049; body-limit
enforcement already lives in `body_limit.py`'s ASGI middleware, outside this route entirely):

1. `require_ready` (a FastAPI dependency, not a `yield` one -- `streaming.py`'s docstring
   explains why a `SpeechResponse` route can't use those: their exit code runs between
   building the response and calling it, so a failure there would never release the GPU
   through the normal path);
2. `read_fields` and `parse_speech` (`http_fields.py`) -- field syntax, ranges and the
   `reference_conflict`/`ref_text_required`/`reference_required` checks all happen inside
   `parse_speech` already;
3. a `VoiceRef` is `404 unknown_voice` for now -- voices arrive in Phase 7 (T049 names this
   the stub lookup), so every `voice_id` is unknown until then;
4. `split_text` and the "nothing to speak" check are CPU-only, so they run here too, before
   the reference decode and the busy check (review-agent pass 1, finding 1: FR-007 puts every
   `400`/`404` before `409 busy`, and this is a `400` this phase can already produce);
5. an `InlineRef`'s bytes are decoded (`reference_audio.decode`) on a worker thread
   (`asyncio.to_thread`), never the event loop -- libsndfile's decode is blocking CPU work;
6. `gate.try_acquire()`, else `409 busy` -- or `GpuUnavailable` if the gate is poisoned,
   which propagates past this route to `errors.py`'s own handler (`503 gpu_unavailable`);
7. reference resolution and piece 0's preparation run on the `GpuThread` (`synthesis.py`),
   each awaited through `asyncio.shield` (finding 7): a cancelled or failed request must not
   release the lease while that GPU-thread call is still actually running -- the executor is
   single-threaded, but releasing early would still tell the next request "free" while our
   own abandoned work is still really queued ahead of it;
8. piece 0's room (`FastBreezeStreamingRuntime.max_new_tokens_room`, checked once, on the
   `GpuThread`, right after its inputs are built) decides `400 text_too_long` before any
   generation starts -- the "first piece has no room" half of BC-47. A later piece's own room
   isn't checked this way yet (T052/T053, deferred): if priming happens to run past piece 0
   within one step (e.g. an all-pad piece 0), the runtime's own "no room" `ValueError` is
   still mapped to `400 text_too_long` here rather than an opaque `500`; the same failure
   *after* streaming has started already ends the stream abnormally through the ordinary
   exception path (streaming.py), which needs no change;
9. the first audio chunk is primed (`GpuSession.step()`) before the `200` is even chosen
   (R2/R3, BC-17): `DONE` on this first step means the whole request produced no audio. That
   is a `500`, but raised as a plain exception (not `ApiError`) so it takes the *unhandled*
   path in `errors.py` -- `ApiError(500, ...)` would be an ordinary, keep-alive JSON response,
   while the contract says every `500` closes the connection (finding 5), which only happens
   when `ServerErrorMiddleware` sees an exception it re-raises after answering.

`request_id` is set on `http_request.state` as the first thing this route does (finding 4), so
even a failure the catch-all `Exception` handler reports (`request.failed`) correlates with the
same id a client's own `X-Request-Id` header would show. `ApiError`s raised anywhere in here are
caught once, at the very end, purely to add that header to the response too -- `errors.py`'s own
`ApiError` handler has no request to read an id from otherwise.

`speech.accepted` is emitted once a request has cleared every check above and is about to start
generating (piece count, reference kind); `speech.first_audio` once priming actually produced
audio. Its `ttfa_ms` is timed from `received_at`, read at the very top of this route (finding 8:
research.md R17 measures "time to first audio" the same way `bench_api.py` does, from the
request onward -- not from `started_at`, a separate, later reading `SpeechResponse` uses for its
own `rtf`, which stays "when generation started", not "when the request arrived", so slow
validation or a slow reference decode doesn't skew a real-time factor that is meant to describe
the GPU alone). Neither `speech.accepted` nor `speech.first_audio` is an outcome event -- those
(`speech.completed`/`aborted`/`failed`) are `SpeechResponse`'s own, and only ever follow a real
`200`, except for `speech.failed` in the "no audio at all" case above, which this route emits
itself since no `SpeechResponse` is ever built for it.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the route's `Annotated[...]`,
# which refers to the local `components`, when the route is defined (routes_health.py does the
# same, for the same reason).
import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, Request
from starlette.responses import Response

from breeze_infer import reference_audio
from breeze_infer.errors import ApiError, api_error_response
from breeze_infer.events import Emitter
from breeze_infer.gpu import DONE, GpuGate, GpuSession, GpuThread
from breeze_infer.http_fields import (
    InlineRef,
    NoReference,
    ReferenceSpec,
    SpeechRequest,
    VoiceRef,
    parse_speech,
    read_fields,
)
from breeze_infer.routes_health import Readiness
from breeze_infer.settings import Settings
from breeze_infer.streaming import SpeechResponse
from breeze_infer.synthesis import (
    generate_piece,
    piece_seed,
    prepare_piece,
    resolve_reference,
)
from breeze_infer.text_split import split_text


class SpeechComponents(Protocol):
    """The subset of `breeze_infer.api.Components` this route needs, duck-typed rather than
    imported: `api.py` registers this router (mirroring `routes_health.install_health`), so
    importing `Components` from it here would make the import circular.
    """

    settings: Settings
    events: Emitter
    gate: GpuGate
    gpu: GpuThread
    readiness: Readiness


def _reference_kind(reference: ReferenceSpec) -> str:
    """`speech.accepted`'s `reference` field. A `VoiceRef` never reaches here: the route
    rejects it with `404 unknown_voice` long before a request gets this far."""
    if isinstance(reference, NoReference):
        return "none"
    if isinstance(reference, InlineRef):
        return "inline"
    raise TypeError(f"unexpected reference spec: {reference!r}")


def _prepare_first_piece(
    runtime: Any, reference: Any, text: str, request: SpeechRequest
) -> tuple[dict[str, Any], int]:
    """Piece 0's model inputs, and the frame room the runtime has for them. GPU-thread only:
    `prepare_piece` touches the tokenizer and model, `max_new_tokens_room` touches `inputs`'
    tensors.

    Returns the inputs alongside the room so `_iter_pieces` can reuse them instead of
    building piece 0 twice, and so the route can turn `room <= 0` into `400 text_too_long`
    before any generation starts -- without inspecting the runtime's own `ValueError`
    messages (`max_new_tokens_room` raises that only for a genuinely bad override, e.g. a
    non-finite `temperature`, which is a `500`, not this).
    """
    inputs = prepare_piece(
        runtime.tokenizer, runtime.model, reference, text, request.instruction, request.cfg_scale
    )
    room = runtime.max_new_tokens_room(request.max_new_tokens, inputs)
    return inputs, room


def _iter_pieces(
    runtime: Any,
    reference: Any,
    pieces: list[str],
    request: SpeechRequest,
    request_id: str,
    first_inputs: dict[str, Any],
    *,
    chunk_first: int,
    chunk_max: int,
) -> Iterator[bytes]:
    """One request's whole PCM stream: `generate_piece` chained over every piece.

    A plain sync generator, not `itertools.chain`: `GpuSession.step()` calls `next()` on
    this generator on the `GpuThread` (`gpu.py`), so building piece `i > 0`'s inputs *inside*
    this loop -- rather than all up front -- is what makes "each piece's inputs are prepared
    on the GPU thread lazily, before its first step" true for free, with no extra `gpu.run()`
    calls anywhere in this route. Piece 0 is the one exception: its inputs (and its room
    check) already ran, on the `GpuThread`, before this generator was even built, so they are
    passed in rather than rebuilt.
    """
    for index, text in enumerate(pieces):
        inputs = (
            first_inputs
            if index == 0
            else prepare_piece(
                runtime.tokenizer,
                runtime.model,
                reference,
                text,
                request.instruction,
                request.cfg_scale,
            )
        )
        yield from generate_piece(
            runtime,
            inputs,
            request_id=request_id,
            seed=piece_seed(request.seed, index),
            chunk_first=chunk_first,
            chunk_max=chunk_max,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_new_tokens,
        )


async def _rest_of_audio(session: GpuSession[bytes]) -> AsyncGenerator[bytes, None]:
    """The body `SpeechResponse` streams after the primed first chunk: step the session
    until it reports `DONE`. Mirrors `tests/test_speech_abort.py`'s own `rest_of_audio`."""
    while True:
        chunk = await session.step()
        if chunk is DONE:
            return
        yield chunk


async def _serve_speech(
    http_request: Request,
    runtime: Any,
    components: SpeechComponents,
    *,
    request_id: str,
    received_at: float,
    clock: Callable[[], float],
) -> Response:
    """Everything `install_speech`'s route does beyond naming the request and adding
    `X-Request-Id` to whatever this raises -- a separate function so that wrapping (module
    docstring, finding 4) has nothing else to do."""
    fields = await read_fields(http_request)
    request = parse_speech(fields, components.settings)

    if isinstance(request.reference, VoiceRef):
        raise ApiError(404, "unknown_voice", "unknown voice_id")

    # CPU-only, so it runs before the reference decode and the busy check (module docstring,
    # step 4): text.strip() is non-blank (BC-10, parse_speech), but split_text also drops
    # units with no letter or digit to speak (text_split.py's _speakable) -- text like "..."
    # clears text_required but leaves nothing to synthesize.
    pieces = split_text(request.text, budget=request.split_chars)
    if not pieces:
        raise ApiError(400, "text_required", "text is required")

    decoded_audio = None
    if isinstance(request.reference, InlineRef):
        # Off the event loop: libsndfile's decode is blocking CPU work and must not stall
        # every other request in flight while it runs.
        decoded_audio = await asyncio.to_thread(
            reference_audio.decode, request.reference.audio_bytes
        )

    # None means busy (409); a poisoned gate raises GpuUnavailable instead (gpu.py), which
    # propagates straight past this route to errors.py's own handler (503 gpu_unavailable).
    lease = components.gate.try_acquire()
    if lease is None:
        raise ApiError(409, "busy", "busy")

    session: GpuSession[bytes] | None = None
    gpu_task: asyncio.Task[Any] | None = None
    try:
        gpu_task = asyncio.ensure_future(
            resolve_reference(
                request.reference,
                decoded_audio=decoded_audio,
                audio_tokenizer=runtime.audio_tokenizer,
                gpu=components.gpu,
            )
        )
        reference = await asyncio.shield(gpu_task)
        gpu_task = None

        gpu_task = asyncio.ensure_future(
            components.gpu.run(_prepare_first_piece, runtime, reference, pieces[0], request)
        )
        first_inputs, room = await asyncio.shield(gpu_task)
        gpu_task = None
        if room <= 0:
            raise ApiError(400, "text_too_long", "text is too long")

        components.events.emit(
            "speech.accepted",
            request_id=request_id,
            pieces=len(pieces),
            reference=_reference_kind(request.reference),
        )

        gen = _iter_pieces(
            runtime,
            reference,
            pieces,
            request,
            request_id,
            first_inputs,
            chunk_first=components.settings.chunk_first,
            chunk_max=components.settings.chunk_max,
        )
        session = GpuSession(lease, components.gpu, gen)
        started_at = clock()  # generation start: SpeechResponse's own rtf origin
        try:
            first_chunk = await session.step()
        except ValueError as error:
            # T052/T053 (deferred): a later piece's own room check and clamp aren't built
            # yet, so this is reached only if priming ran past piece 0 within a single step
            # (e.g. an all-pad piece 0) and the next piece has no room. Still a text-too-long
            # failure, not an opaque 500, until that phase gives it a proper pre-check.
            raise ApiError(400, "text_too_long", "text is too long") from error
    except BaseException:
        # A cancellation (client disconnect) or any other failure while a GPU-thread call is
        # still in flight must not free the lease before that call has actually finished:
        # cancelling *this* await doesn't cancel work already submitted to the (single-
        # threaded) GPU executor (gpu.py), so releasing early would tell the next request
        # "free" while our own abandoned work is still really queued ahead of it there.
        if gpu_task is not None and not gpu_task.done():
            with contextlib.suppress(Exception):
                # Suppressed, not propagated: the exception this `except` is already
                # handling must be what reaches the caller, even if the abandoned GPU call
                # itself also fails, for some unrelated reason, while we wait for it here.
                await asyncio.shield(gpu_task)
        if session is not None:
            await session.aclose()
        else:
            lease.release()
        raise

    if first_chunk is DONE:
        await session.aclose()
        components.events.emit(
            "speech.failed",
            level="error",
            request_id=request_id,
            reason="no_audio",
            audio_seconds=0.0,
        )
        # Not ApiError: every 500 must close the connection (contracts/http-api.md), which
        # only happens on the genuinely unhandled path (errors.py's catch-all re-raises after
        # answering); ApiError(500, ...) would be an ordinary, keep-alive JSON response.
        raise RuntimeError("generation produced no audio")

    components.events.emit(
        "speech.first_audio",
        request_id=request_id,
        ttfa_ms=(clock() - received_at) * 1000,
    )

    return SpeechResponse(
        first_chunk=first_chunk,
        body=_rest_of_audio(session),
        session=session,
        events=components.events,
        request_id=request_id,
        sample_rate=int(runtime.sample_rate),
        clock=clock,
        started_at=started_at,
        headers={"X-Request-Id": request_id},
    )


def install_speech(
    app: FastAPI,
    components: SpeechComponents,
    *,
    clock: Callable[[], float],
    new_request_id: Callable[[], str],
) -> None:
    """Register `POST /v1/audio/speech`.

    `clock` and `new_request_id` are injected (Constitution III): `new_request_id` names each
    request for its `speech.*` events, `request.state.request_id` and its `X-Request-Id`
    response header; `clock` times `ttfa_ms` and feeds `SpeechResponse`'s own `rtf` (module
    docstring: two different readings of the same clock, not the same reading reused).
    """

    @app.post("/v1/audio/speech")
    async def speech(
        http_request: Request,
        runtime: Annotated[Any, Depends(components.readiness.require_ready)],
    ) -> Response:
        request_id = new_request_id()
        # Set before anything below can fail, so even a request that never gets past field
        # parsing still correlates with this id in the catch-all handler's request.failed
        # (errors.py reads request.state.request_id).
        http_request.state.request_id = request_id
        received_at = clock()
        try:
            return await _serve_speech(
                http_request,
                runtime,
                components,
                request_id=request_id,
                received_at=received_at,
                clock=clock,
            )
        except ApiError as exc:
            # errors.py's own ApiError handler has no request to read an id from; this adds
            # the same header a 200 gets, cheaply, since this route already has the id.
            return api_error_response(exc, headers={"X-Request-Id": request_id})
