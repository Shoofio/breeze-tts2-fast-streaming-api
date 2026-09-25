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
3. `split_text` and the "nothing to speak" check are CPU-only, so they run here too, before
   the voice lookup, the reference decode and the busy check (review-agent pass 1, finding 1:
   FR-007 puts every `400`/`404` before `409 busy`, and this is a `400` this phase can already
   produce; T046 review: it's still stage 2, "field syntax and ranges", a property of `text`
   alone -- so it belongs before stage 4's unknown-voice `404` too, not just before decode/busy);
4. a `VoiceRef` is `404 unknown_voice` for now -- voices arrive in Phase 7 (T049 names this
   the stub lookup);
5. an `InlineRef`'s bytes are decoded (`reference_audio.decode`) on a worker thread
   (`asyncio.to_thread`), never the event loop -- libsndfile's decode is blocking CPU work;
5a. piece 0's room is checked on the CPU, also on a worker thread (`_check_first_piece_room`:
   its tokenized text plus the reference's *predicted* frames), so "no room" is a `400
   text_too_long` even while the GPU is busy -- the "first piece has no room" half of BC-47;
6. `gate.try_acquire()`, else `409 busy` -- or `GpuUnavailable` if the gate is poisoned,
   which propagates past this route to `errors.py`'s own handler (`503 gpu_unavailable`);
7. reference resolution and piece 0's preparation run on the `GpuThread` (`synthesis.py`),
   each awaited through `asyncio.shield` (finding 7): a cancelled or failed request must not
   release the lease while that GPU-thread call is still actually running -- the executor is
   single-threaded, but releasing early would still tell the next request "free" while our
   own abandoned work is still really queued ahead of it;
8. piece 0's room is checked again on the `GpuThread`, on its real inputs, right after they are
   built (`FastBreezeStreamingRuntime.max_new_tokens_room`): the backstop for a codec whose
   frame count differs from 5a's prediction. Every later piece is prepared and sized on the
   `GpuThread` inside the body (`_iter_pieces`); one with no room raises `NoRoomError`, which
   ends a started stream abnormally through the ordinary exception path (streaming.py) -- the
   "later piece" half of BC-47 -- or, if priming ran past piece 0 within one step (piece 0
   yielded no audio), is still mapped to `400 text_too_long` here;
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
import traceback
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
from breeze_infer.limits import ANCHOR_CHARS
from breeze_infer.routes_health import Readiness
from breeze_infer.settings import Settings
from breeze_infer.streaming import SpeechResponse
from breeze_infer.synthesis import (
    CodesRef,
    NoRef,
    PieceRoom,
    Reference,
    anchor_codes,
    codec_samples_per_frame,
    generate_piece,
    piece_frame_limit,
    piece_room,
    piece_seed,
    predicted_room,
    prepare_piece,
    resolve_reference,
    stand_in_reference,
)
from breeze_infer.text_split import split_text
from models.fast_streaming import NoRoomError


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


def _check_frame_prediction(
    decoded_audio: reference_audio.DecodedAudio | None,
    reference: Any,
    events: Emitter,
    *,
    request_id: str,
) -> None:
    """T049: `reference_audio.predicted_frames` is only ever a prediction -- the real
    codec is what actually decides a reference's frame count once it encodes it
    (`resolve_reference`, just above this call). A mismatch here means the formula
    (`reference_audio.py`'s own docstring has the arithmetic) has drifted from what the
    bundled codec really does, which is worth knowing about without failing a request
    over it: the encoded codes are already in hand and just as usable either way, so
    this only emits `speech.frame_prediction_mismatch` (a warning, not an error) and
    lets synthesis continue on the actual codes.
    """
    if decoded_audio is None or not isinstance(reference, CodesRef):
        return
    actual_frames = reference.codes.shape[0]
    if actual_frames == decoded_audio.predicted_frames:
        return
    events.emit(
        "speech.frame_prediction_mismatch",
        level="warning",
        request_id=request_id,
        predicted_frames=decoded_audio.predicted_frames,
        actual_frames=actual_frames,
    )


async def _check_first_piece_room(
    runtime: Any,
    request: SpeechRequest,
    text: str,
    decoded_audio: reference_audio.DecodedAudio | None,
) -> None:
    """`400 text_too_long` if piece 0 has no room, decided before the GPU gate is taken.

    FR-007 puts every `400` before `409 busy`, so this can't wait for the codec: an inline
    reference is sized by its *predicted* frame count (`stand_in_reference`), and the inputs
    are built on the CPU (`predicted_room`). It runs on a worker thread, not the event loop,
    because tokenizing a long piece is blocking CPU work. The real inputs are checked again
    on the GPU thread once the reference is encoded (`_prepare_first_piece`), which covers a
    codec whose frame count differs from the prediction.
    """
    stand_in = stand_in_reference(
        request.reference,
        None if decoded_audio is None else decoded_audio.predicted_frames,
        int(runtime.model.config.num_codebooks),
    )
    room = await asyncio.to_thread(
        predicted_room,
        runtime,
        stand_in,
        text,
        request.instruction,
        request.cfg_scale,
        request.max_new_tokens,
    )
    if room.room <= 0:
        raise ApiError(400, "text_too_long", "text is too long")


def _prepare_first_piece(
    runtime: Any, reference: Reference, text: str, request: SpeechRequest
) -> tuple[dict[str, Any], PieceRoom]:
    """Piece 0's model inputs, and the frame room the runtime has for them. GPU-thread only:
    `prepare_piece` touches the tokenizer and model, `max_new_tokens_room` touches `inputs`'
    tensors.

    Returns the inputs alongside the room so `_iter_pieces` can reuse both instead of
    building piece 0 twice, and so the route can turn no room into `400 text_too_long`
    before any generation starts -- without inspecting the runtime's own `ValueError`
    messages (`max_new_tokens_room` raises that only for a genuinely bad override, e.g. a
    non-finite `temperature`, which is a `500`, not this).
    """
    inputs = prepare_piece(
        runtime.tokenizer, runtime.model, reference, text, request.instruction, request.cfg_scale
    )
    return inputs, piece_room(runtime, inputs, request.max_new_tokens)



# `generate_piece`/`ramp_pcm` (synthesis.py) discard everything but each chunk's PCM
# bytes; `_iter_pieces` below counts those bytes back into a frame count for
# `speech.piece_done` (T046 review, finding 7) rather than reach into `synthesis.py` for
# a second per-piece signal, using the same `samples_per_frame` (`codec_samples_per_frame`)
# the route already computed once for `generate_piece` itself.
_PCM_BYTES_PER_SAMPLE = 2  # s16le


def _iter_pieces(
    runtime: Any,
    reference: Reference,
    pieces: list[str],
    request: SpeechRequest,
    request_id: str,
    first_inputs: dict[str, Any],
    first_room: PieceRoom,
    events: Emitter,
    *,
    chunk_first: int,
    chunk_max: int,
    samples_per_frame: int,
) -> Iterator[bytes]:
    """One request's whole PCM stream: `generate_piece` chained over every piece.

    A plain sync generator, not `itertools.chain`: `GpuSession.step()` calls `next()` on
    this generator on the `GpuThread` (`gpu.py`), so building piece `i > 0`'s inputs *inside*
    this loop -- rather than all up front -- is what makes "each piece's inputs are prepared
    on the GPU thread lazily, before its first step" true for free, with no extra `gpu.run()`
    calls anywhere in this route. Piece 0 is the one exception: its inputs (and its room
    check) already ran, on the `GpuThread`, before this generator was even built, so they are
    passed in rather than rebuilt.

    `events.emit` is called from here on the `GpuThread`, not the event loop -- `Emitter`
    is documented safe for that (`events.py`) -- once each piece's own chunks are
    exhausted, with `speech.piece_done{piece_index, frames}` (T046 review, finding 7): a
    multi-piece GPU test can otherwise only see the whole request succeeded, not that
    *every* piece actually produced audio, since a silently empty later piece would still
    leave the overall stream non-empty.

    Each piece is sized against its own room (`piece_frame_limit`): a partial room clamps
    it, and no room raises `NoRoomError`, which aborts the stream once the `200` is out
    (BC-47). With no reference, piece 0's frames are collected through the runtime's
    `token_observer` and, once piece 0 has finished, become every later piece's reference
    together with its text (`anchor_codes`). Only piece 0 can anchor (data-model.md
    "Reference"): if it produced no non-pad frame, the later pieces stay voice design. A
    cancelled or failed piece 0 never reaches the anchoring step at all.
    """
    anchoring = isinstance(reference, NoRef) and len(pieces) > 1
    pad_id = int(runtime.model.config.codebook_pad_token_id)
    for index, text in enumerate(pieces):
        if index == 0:
            inputs, room = first_inputs, first_room
        else:
            inputs = prepare_piece(
                runtime.tokenizer,
                runtime.model,
                reference,
                text,
                request.instruction,
                request.cfg_scale,
            )
            room = piece_room(runtime, inputs, request.max_new_tokens)
        max_new_tokens = piece_frame_limit(
            room, events, request_id=request_id, piece_index=index
        )
        frames: list[Any] | None = [] if anchoring and index == 0 else None
        piece_bytes = 0
        for chunk in generate_piece(
            runtime,
            inputs,
            request_id=request_id,
            seed=piece_seed(request.seed, index),
            chunk_first=chunk_first,
            chunk_max=chunk_max,
            samples_per_frame=samples_per_frame,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=max_new_tokens,
            token_observer=None if frames is None else frames.append,
        ):
            piece_bytes += len(chunk)
            yield chunk
        events.emit(
            "speech.piece_done",
            request_id=request_id,
            piece_index=index,
            frames=piece_bytes // _PCM_BYTES_PER_SAMPLE // samples_per_frame,
        )
        if frames is not None:
            codes = anchor_codes(frames, pad_id)
            if codes is not None:
                reference = CodesRef(codes=codes, ref_text=text)


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

    # CPU-only, so it runs before the voice lookup, the reference decode and the busy
    # check (FR-007 review: this is still stage 2, "field syntax and ranges" -- it's a
    # property of `text` alone parse_speech's own field checks already validated, not a
    # new stage of its own -- so it must come before stage 4's unknown-voice `404` too,
    # not just before decode/busy): text.strip() is non-blank (BC-10, parse_speech), but
    # split_text also drops units with no letter or digit to speak (text_split.py's
    # _speakable) -- text like "..." clears text_required but leaves nothing to
    # synthesize.
    #
    # With no reference, piece 0 is packed against the soft opening budget (US3 scenario 1):
    # it becomes the anchor for every later piece, so it should be short enough for a quick
    # first audio. A reference already fixes the voice, so there is no opening piece.
    first_budget = ANCHOR_CHARS if isinstance(request.reference, NoReference) else 0
    pieces = split_text(request.text, budget=request.split_chars, first_budget=first_budget)
    if not pieces:
        raise ApiError(400, "text_required", "text is required")

    if isinstance(request.reference, VoiceRef):
        raise ApiError(404, "unknown_voice", "unknown voice_id")

    decoded_audio = None
    if isinstance(request.reference, InlineRef):
        # Off the event loop: libsndfile's decode is blocking CPU work and must not stall
        # every other request in flight while it runs.
        decoded_audio = await asyncio.to_thread(
            reference_audio.decode, request.reference.audio_bytes
        )

    await _check_first_piece_room(runtime, request, pieces[0], decoded_audio)

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
        _check_frame_prediction(
            decoded_audio, reference, components.events, request_id=request_id
        )

        gpu_task = asyncio.ensure_future(
            components.gpu.run(_prepare_first_piece, runtime, reference, pieces[0], request)
        )
        first_inputs, first_room = await asyncio.shield(gpu_task)
        gpu_task = None
        if first_room.room <= 0:
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
            first_room,
            components.events,
            chunk_first=components.settings.chunk_first,
            chunk_max=components.settings.chunk_max,
            samples_per_frame=codec_samples_per_frame(runtime),
        )
        session = GpuSession(lease, components.gpu, gen)
        started_at = clock()  # generation start: SpeechResponse's own rtf origin
        try:
            first_chunk = await session.step()
        except NoRoomError as error:
            # Piece 0's room was checked above, so this is a later piece with no room
            # (`piece_frame_limit`) reached while priming, i.e. before the `200`: priming runs
            # past piece 0 within one step only if piece 0 yielded no audio at all. With no
            # response started yet, it is still a `400 text_too_long`, not an aborted stream.
            #
            # Only `NoRoomError` maps to this `400` (T046 review, finding 1): the runtime's
            # `iter_audio_chunks` also raises a plain `ValueError` for a genuinely invalid
            # override or a malformed `inputs` shape -- a server bug, not "text too long" --
            # and that must still surface as an unhandled `500` (below, uncaught) with a
            # `request.failed` event, not be mistaken for this.
            raise ApiError(400, "text_too_long", "text is too long") from error
    except BaseException:
        # A cancellation (client disconnect) or any other failure while a GPU-thread call is
        # still in flight must not free the lease before that call has actually finished:
        # cancelling *this* await doesn't cancel work already submitted to the (single-
        # threaded) GPU executor (gpu.py), so releasing early would tell the next request
        # "free" while our own abandoned work is still really queued ahead of it there.
        if gpu_task is not None and not gpu_task.done():
            # Released from a done-callback, not from code below that only runs if this
            # `except` block itself runs to completion (T046 review, finding 2 -- a lease
            # leak): a *second* cancellation of this coroutine while the `await` just below
            # is waiting can itself raise out of the `with suppress(...)` (it catches
            # `BaseException`, but only around that one `await` -- nothing stops a `raise`
            # reaching this frame from further out, e.g. from `finally` unwinding above).
            # `add_done_callback` runs exactly once, whenever the GPU thread actually
            # finishes this call, regardless of how many times *this* coroutine gets
            # interrupted waiting for it -- the same pattern `GpuSession._release` already
            # uses for the analogous case just below (`session.aclose()`).
            gpu_task.add_done_callback(lambda _task: lease.release())
            with contextlib.suppress(BaseException):
                # Suppressed, not propagated: the exception this `except` is already
                # handling must be what reaches the caller, even if the abandoned GPU call
                # itself also fails, for some unrelated reason, while we wait for it here.
                await asyncio.shield(gpu_task)
        elif session is not None:
            try:
                await session.aclose()
            except Exception as close_error:  # noqa: BLE001 -- reported, never raised
                # A close failure here must never replace the exception this `except`
                # block is already handling (T046 review, finding 6): a `400` followed by
                # a `GpuCloseTimeout` must still reach the client as the `400`, not an
                # opaque close error. Reported as its own event instead (mirrors
                # streaming.py's `_close_and_report`, which the same rule already governs
                # for a response that streamed past its first chunk) -- the same
                # `gpu.close_failed` name `api.py`'s own `_report_close_failed` uses for an
                # abandoned close's error (data-model.md's event list), just with a
                # `request_id` here since this one has a request to attribute it to.
                components.events.emit(
                    "gpu.close_failed",
                    level="error",
                    request_id=request_id,
                    error=repr(close_error),
                    traceback="".join(traceback.format_exception(close_error)),
                )
        else:
            lease.release()
        raise

    if first_chunk is DONE:
        # Emitted before the close (T046 review, finding 6): the outcome -- "no audio at
        # all" -- is already decided the moment `session.step()` reports `DONE`, so it must
        # not wait on (or be lost to) whatever `session.aclose()` does next.
        components.events.emit(
            "speech.failed",
            level="error",
            request_id=request_id,
            reason="no_audio",
            audio_seconds=0.0,
        )
        try:
            await session.aclose()
        except Exception as close_error:  # noqa: BLE001 -- reported, never raised
            # Same rule as the `except BaseException` cleanup above: a close failure must
            # not replace "no_audio" as the reason this request failed.
            components.events.emit(
                "gpu.close_failed",
                level="error",
                request_id=request_id,
                error=repr(close_error),
                traceback="".join(traceback.format_exception(close_error)),
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

    async def _name_request(http_request: Request) -> str:
        """A `Depends()` of its own, declared before `require_ready` below (T046 review,
        finding 8): FastAPI resolves same-level dependencies in declaration order, so this
        runs -- and sets `request.state.request_id` -- even when `require_ready` itself
        then raises `503 loading`/`gpu_unavailable`, which happens *before* the route
        body (and so before this id would otherwise ever be set) gets to run at all.
        Without this, that response would have no id for `errors.py`'s handlers to attach
        as `X-Request-Id`, and no correlation for the client to report back.
        """
        request_id = new_request_id()
        http_request.state.request_id = request_id
        return request_id

    @app.post("/v1/audio/speech")
    async def speech(
        http_request: Request,
        request_id: Annotated[str, Depends(_name_request)],
        runtime: Annotated[Any, Depends(components.readiness.require_ready)],
    ) -> Response:
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
