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
4. an `InlineRef`'s bytes are decoded (`reference_audio.decode`) on a worker thread
   (`asyncio.to_thread`), never the event loop -- libsndfile's decode is blocking CPU work;
5. `gate.try_acquire()`, else `409 busy`;
6. reference resolution, splitting and piece 0's preparation all run on the `GpuThread`
   (`synthesis.py`); a failure here releases the lease before this function returns, since
   no `SpeechResponse` exists yet to own that release;
7. piece 0's room (`FastBreezeStreamingRuntime.max_new_tokens_room`, checked once, on the
   `GpuThread`, right after its inputs are built) decides `400 text_too_long` before any
   generation starts -- the "first piece has no room" half of BC-47. A later piece's room
   isn't checked this way (T053 aborts the stream instead, mid-generation, the same as any
   other post-`200` failure);
8. the first audio chunk is primed (`GpuSession.step()`) before the `200` is even chosen
   (R2/R3, BC-17): `DONE` on this first step means the whole request produced no audio, which
   is a failure before streaming, not an empty stream, so it is `500 internal_error`, not `200`
   with an empty body.

`speech.accepted` is emitted once a request has cleared every check above and is about to
start generating (piece count, reference kind); `speech.first_audio` once priming actually
produced audio (`ttfa_ms`, timed from the same `started_at` `SpeechResponse` uses for its own
`rtf`). Neither is an outcome event -- those (`speech.completed`/`aborted`/`failed`) are
`SpeechResponse`'s own, and only ever follow a real `200`.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the route's `Annotated[...]`,
# which refers to the local `components`, when the route is defined (routes_health.py does the
# same, for the same reason).
import asyncio
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, Request
from starlette.responses import Response

from breeze_infer import reference_audio
from breeze_infer.errors import ApiError
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


def install_speech(
    app: FastAPI,
    components: SpeechComponents,
    *,
    clock: Callable[[], float],
    new_request_id: Callable[[], str],
) -> None:
    """Register `POST /v1/audio/speech`.

    `clock` and `new_request_id` are injected (Constitution III): `new_request_id` names
    each request for its `speech.*` events and its `X-Request-Id` response header; `clock`
    times `ttfa_ms` here and feeds `SpeechResponse`'s own `rtf` (the same `started_at`
    reading serves both, so they share one origin: the moment generation -- not validation,
    not reference preparation -- actually starts).
    """

    @app.post("/v1/audio/speech")
    async def speech(
        http_request: Request,
        runtime: Annotated[Any, Depends(components.readiness.require_ready)],
    ) -> Response:
        request_id = new_request_id()
        fields = await read_fields(http_request)
        request = parse_speech(fields, components.settings)

        if isinstance(request.reference, VoiceRef):
            raise ApiError(404, "unknown_voice", "unknown voice_id")

        decoded_audio = None
        if isinstance(request.reference, InlineRef):
            # Off the event loop: libsndfile's decode is blocking CPU work and must not
            # stall every other request in flight while it runs.
            decoded_audio = await asyncio.to_thread(
                reference_audio.decode, request.reference.audio_bytes
            )

        lease = components.gate.try_acquire()
        if lease is None:
            raise ApiError(409, "busy", "busy")

        session: GpuSession[bytes] | None = None
        try:
            reference = await resolve_reference(
                request.reference,
                decoded_audio=decoded_audio,
                audio_tokenizer=runtime.audio_tokenizer,
                gpu=components.gpu,
            )

            pieces = split_text(request.text, budget=request.split_chars)
            if not pieces:
                # text.strip() is non-blank (BC-10, parse_speech), but split_text also
                # drops units with no letter or digit to speak (text_split.py's
                # _speakable) -- text like "..." clears text_required but leaves nothing
                # to synthesize.
                raise ApiError(400, "text_required", "text is required")

            first_inputs, room = await components.gpu.run(
                _prepare_first_piece, runtime, reference, pieces[0], request
            )
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
            started_at = clock()
            first_chunk = await session.step()
        except BaseException:
            # The lease has a session to own its release only once `session` exists; before
            # that, this function is the only owner (mirrors tests/test_speech_abort.py's
            # own `speech` handler).
            if session is not None:
                await session.aclose()
            else:
                lease.release()
            raise

        if first_chunk is DONE:
            await session.aclose()
            raise ApiError(500, "internal_error", "generation produced no audio")

        components.events.emit(
            "speech.first_audio",
            request_id=request_id,
            ttfa_ms=(clock() - started_at) * 1000,
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
