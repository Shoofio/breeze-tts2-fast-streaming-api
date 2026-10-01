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
3. `split_text` and the "nothing to speak" safety net are CPU-only, so they run here too, before
   the voice lookup, the reference decode and the busy check (review-agent pass 1, finding 1:
   FR-007 puts every `400`/`404` before `409 busy`, and this is a `400` this phase can already
   produce; T046 review: it's still stage 2, "field syntax and ranges", a property of `text`
   alone -- so it belongs before stage 4's unknown-voice `404` too, not just before decode/busy);
4. a `VoiceRef` is looked up in the voice registry (`VoiceRegistry.lookup`, exact id), else
   `404 unknown_voice`. The prefix cache's token is read in that same event-loop step, with no
   `await` between them (`VoicePrefixCache.token`): a `DELETE` that lands after it, however long
   this request then takes to reach the gate, keeps this request's prefix out of the cache
   (T066). The resolved voice (its codes, transcript, prefix length and prefix cache key, held in
   memory since it was registered) is what the request uses from here on, even if the voice is
   deleted meanwhile;
5. an `InlineRef`'s bytes are decoded (`reference_audio.decode`) on a worker thread
   (`asyncio.to_thread`), never the event loop -- libsndfile's decode is blocking CPU work;
5a. piece 0's room is checked on the CPU, on the CPU tokenizer's own thread (`_size_first_piece`:
   its tokenized text plus the reference's *predicted* frames), so "no room" is a `400
   text_too_long` even while the GPU is busy -- the "first piece has no room" half of BC-47. A
   voice is sized on the path it will take: its prefix without an override (a stored prefix the
   runtime would refuse to build has no room at all), its codes with one. The codes path of a
   voice without an override is the out-of-memory fallback's, sized only when that fallback is
   needed (step 7);
6. `gate.try_acquire()`, else `409 busy` -- or `GpuUnavailable` if the gate is poisoned,
   which propagates past this route to `errors.py`'s own handler (`503 gpu_unavailable`);
6a. with no reference and more than one piece, every later piece's prompt length is queued
   on the CPU tokenizer's sizing worker right after the lease (`_start_anchor_sizing`), not before
   the busy check (review #2 on 2d9070a): it runs on the CPU while piece 0 itself prepares and
   generates on the `GpuThread` below, and `_anchor_for_later_pieces` only blocks on the result
   once piece 0 has actually finished, by which point it has usually already. That worker is
   its own, not the one the pre-gate checks queue on (review 34 finding 5). Every exit that
   doesn't need the result abandons it (review 33 on 10f0c29), and the anchor is skipped
   rather than the stream aborted if it fails or is still unfinished after
   `ANCHOR_SIZING_TIMEOUT_SECONDS`;
7. reference resolution and piece 0's preparation run on the `GpuThread` (`synthesis.py`),
   each awaited through `asyncio.shield` (finding 7): a cancelled or failed request must not
   release the lease while that GPU-thread call is still actually running -- the executor is
   single-threaded, but releasing early would still tell the next request "free" while our
   own abandoned work is still really queued ahead of it. A voice is resolved by
   `_voice_reference` instead: with a `ref_text` override it is its codes and that text (the
   codes path; nothing to run), otherwise its cached KV prefix (the prefix path), built under
   this lease on a miss (`VoicePrefixCache.get_or_build`). A cancel during that build hands the
   lease to the build, which releases it when the GPU work ends, so the route's own release is
   then a no-op. A build that runs out of GPU memory evicts every cached prefix and falls back
   to the codes path for this request (`speech.prefix_fallback`), never a `500`: if piece 0 has
   room on that path, measured then on the CPU tokenizer's worker; if not, `503
   gpu_out_of_memory`;
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

The request's id comes from `request_id.RequestIdMiddleware` (`request.state.request_id`), which
also stamps it on every response as `X-Request-Id`, errors included, so this route's `speech.*`
events, a `request.failed` from the catch-all handler and the client's header all agree.

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
import math
import struct
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypeVar

from fastapi import Depends, FastAPI, Request
from starlette.responses import Response

from breeze_infer import reference_audio
from breeze_infer.errors import ApiError, error_fields, report_unhandled
from breeze_infer.events import Emitter
from breeze_infer.executors import SHUTDOWN_MESSAGE, run_until_shutdown
from breeze_infer.gpu import (
    DONE,
    GpuCloseTimeout,
    GpuGate,
    GpuSession,
    GpuThread,
    GpuUnavailable,
    gpu_call_under_lease,
    report_close_failed,
)
from breeze_infer.http_fields import (
    InlineRef,
    NoReference,
    ReferenceSpec,
    SpeechRequest,
    VoiceRef,
    parse_speech,
    read_fields,
)
from breeze_infer.limits import (
    ANCHOR_CHARS,
    ANCHOR_SIZING_TIMEOUT_SECONDS,
    WAV_SEND_TIMEOUT_SECONDS,
)
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_voices import VoiceSlot
from breeze_infer.settings import Settings
from breeze_infer.streaming import SpeechResponse
from breeze_infer.synthesis import (
    AnchorSizing,
    CodesRef,
    NoRef,
    PieceRoom,
    PrefixBuildOutOfMemory,
    PrefixRef,
    Reference,
    UnbuiltPrefix,
    anchor_codes,
    anchor_sizing,
    build_voice_prefix,
    codec_samples_per_frame,
    generate_piece,
    piece_frame_limit,
    piece_room,
    piece_seed,
    predicted_room,
    prefix_of,
    prepare_piece,
    release_cached_gpu_memory,
    resolve_reference,
    stand_in_reference,
    voice_reference,
)
from breeze_infer.text_split import split_text
from breeze_infer.voice_prefix import VoicePrefixCache
from breeze_infer.voice_registry import ResolvedVoice
from models.fast_streaming import NoRoomError

_T = TypeVar("_T")


class CpuTokenizer:
    """The tokenizers the CPU-side sizing uses -- copies of the runtime's, never the same
    object -- and the threads that use them.

    `_size_first_piece` tokenizes before the gate, and `_start_anchor_sizing` right after it,
    while the GPU thread may be using the runtime's tokenizer for another request's piece. A
    Hugging Face fast tokenizer is one Rust object behind a borrow checker, and using it from
    two threads at once can fail with "Already borrowed", so the sizing gets its own instances.
    The model load makes both copies on the GPU thread (`LoadedModel.from_runtime`), and
    `api.Components.mark_ready` installs them together with the runtime, so neither a request
    nor the event loop ever waits for a copy.

    Two kinds of work use them, each on its own single-worker executor with its own copy:
    - `run`: piece 0's room check (`_size_first_piece`), before the busy check. Any number of
      requests can be there at once, including ones bound for `409`, so their checks queue on
      this one worker, with no lock.
    - `submit`: the lease holder's anchor sizing (`_start_anchor_sizing`). Only one lease
      exists at a time, so this worker has only that request's sizing to run, or rarely also a
      previous holder's that was already running when abandoned. On the pre-gate worker it
      would queue behind every check in a burst of requests, time out, and the speaker would
      change mid-stream (review 34 finding 5).
    The workers run at the same time, so they can't share a copy. A lock around one shared copy
    would serialise them again, and a Python lock is not fair: the pre-gate worker, releasing it
    and taking it again for its next queued check, could keep the sizing waiting. So the
    sizing worker gets a second copy.

    Its own executors rather than asyncio's default one, because that pool is shared with the
    reference decode, form parsing and `gpu.shutdown`: a burst of requests blocked on one
    tokenizer would fill it. `api._drain_gpu` shuts them down at server stop, after cancelling
    every request it already knows about -- but a request that arrived just as the drain
    started might not be one of them (review #3 on 2d9070a); `run` and `submit` answer that race
    with `GpuUnavailable` (the same `503 gpu_unavailable` a poisoned gate answers with) rather
    than a `500` or a silent cancellation.
    """

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._sizing_tokenizer: Any = None
        # Every worker thread either executor starts, for `join`. Recorded by each thread
        # itself as it starts (`initializer`), so this needs nothing private from the executor.
        self._workers: list[threading.Thread] = []
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="breeze-cpu-tokenizer",
            initializer=self._record_worker,
        )
        self._sizing_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="breeze-anchor-sizing",
            initializer=self._record_worker,
        )

    def install(self, tokenizer: Any, sizing_tokenizer: Any) -> None:
        """Use `tokenizer` for the pre-gate checks and `sizing_tokenizer` for the anchor
        sizing: two separate copies, both already made by the model load. Nothing is copied
        here: this runs on the event loop (`api.Components.mark_ready`), where a real
        tokenizer's ~700 ms deep copy would stall every request.
        """
        if tokenizer is None or sizing_tokenizer is None:
            raise ValueError("no CPU tokenizer to install: the model load makes the copies")
        if tokenizer is sizing_tokenizer:
            raise ValueError("one CPU tokenizer for both workers: they run at the same time")
        self._tokenizer = tokenizer
        self._sizing_tokenizer = sizing_tokenizer

    async def run(self, fn: Callable[..., _T], *args: Any) -> _T:
        """`fn(tokenizer, *args)` on the pre-gate worker, the only thread that uses its copy.
        After `shutdown()`, or for a call it cancelled while still queued, `GpuUnavailable`
        (`executors.run_until_shutdown`).

        WebSocket pieces' anchor checks queue here too (ws_server), since they also run before
        the gate: at most one per session (16 sessions), each at most two small prompts, so
        they can delay a piece-0 check."""
        return await run_until_shutdown(
            self._executor, self._call, fn, args, what="the CPU tokenizer"
        )

    def submit(self, fn: Callable[..., _T], *args: Any) -> Future[_T]:
        """`fn(sizing tokenizer, *args)`, queued on the sizing worker, for a caller on another
        thread (the GPU thread) to block on with `.result()` -- `run`'s `await` needs the
        event loop, which the GPU thread doesn't have.

        Raises `GpuUnavailable` exactly as `run` does, for a call arriving after `shutdown()`.
        A call `shutdown()` cancelled while still queued is the GPU-thread caller's own problem
        (`.result()` raises `concurrent.futures.CancelledError`): there is no task here whose
        cancel count could tell that apart from anything else, since nothing here is awaited.
        """
        try:
            return self._sizing_executor.submit(self._call_sizing, fn, args)
        except RuntimeError as error:
            if SHUTDOWN_MESSAGE not in str(error):
                raise
            raise GpuUnavailable("the CPU tokenizer is shutting down") from error

    def _call(self, fn: Callable[..., _T], args: tuple[Any, ...]) -> _T:
        return fn(_installed(self._tokenizer), *args)

    def _call_sizing(self, fn: Callable[..., _T], args: tuple[Any, ...]) -> _T:
        return fn(_installed(self._sizing_tokenizer), *args)

    def _record_worker(self) -> None:
        self._workers.append(threading.current_thread())

    def shutdown(self) -> None:
        """Refuse new work and cancel queued calls on both workers. A call already running
        finishes on its own, and its thread ends after it."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._sizing_executor.shutdown(wait=False, cancel_futures=True)

    def join(self, timeout: float) -> bool:
        """After `shutdown()`, wait up to `timeout` seconds in all for both workers' threads to
        end, and return whether they have.

        For a test teardown: the GPU tests share one tokenizer copy across tests, each with its
        own `CpuTokenizer`, so a call still running must finish before the next test's workers
        use the copy. The wait is bounded so that a wedged call fails the teardown instead of
        hanging the whole session (review 34 finding 6).
        """
        deadline = time.monotonic() + timeout
        for worker in self._workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in self._workers)


def _installed(tokenizer: Any) -> Any:
    if tokenizer is None:
        raise RuntimeError("no CPU tokenizer: the model load installs one before ready")
    return tokenizer


@dataclass(frozen=True)
class _AnchorSizingJob:
    """The lease holder's queued anchor sizing (`_start_anchor_sizing`), and whether the route
    itself gave up on it.

    `future.result()` raises the same `CancelledError` whoever cancelled it. Only a
    `CpuTokenizer.shutdown()` is worth reporting (`speech.anchor_skipped`, `shutdown`); when the
    route cancels it, the request no longer needs it or is ending anyway. So the route never
    cancels `future` directly, only through `abandon()`, which records that first
    (review 34 finding 3).
    """

    future: Future[AnchorSizing]
    abandoned: threading.Event = field(default_factory=threading.Event)

    def abandon(self) -> None:
        """Cancel the sizing if it is still queued; one already running finishes on its own."""
        # Set before cancelling, so a GPU thread woken by the cancel already sees it.
        self.abandoned.set()
        self.future.cancel()


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
    cpu_tokenizer: CpuTokenizer
    voices: VoiceSlot


def _reference_fields(reference: ReferenceSpec) -> dict[str, Any]:
    """`speech.accepted`'s `reference` field for a request with no voice. A voice's fields come
    from `_voice_reference`, which knows which path it took."""
    if isinstance(reference, NoReference):
        return {"reference": "none"}
    if isinstance(reference, InlineRef):
        return {"reference": "inline"}
    raise TypeError(f"unexpected reference spec: {reference!r}")


@dataclass(frozen=True)
class _VoiceLookup:
    """A `VoiceRef` resolved before the gate (`_look_up_voice`): the voice, the prefix cache it
    was resolved against, and that cache's token as read in the same event-loop step."""

    spec: VoiceRef
    voice: ResolvedVoice
    prefix_cache: VoicePrefixCache
    resolved_token: int


def _look_up_voice(components: SpeechComponents, spec: VoiceRef) -> _VoiceLookup:
    """The registry lookup and the token read together, with no `await` (module docstring,
    step 4); `404 unknown_voice` if there is no such voice."""
    services = components.voices.get()
    voice = services.registry.lookup(spec.voice_id)
    resolved_token = services.prefix_cache.token()
    if voice is None:
        raise ApiError(404, "unknown_voice", "unknown voice_id")
    return _VoiceLookup(spec, voice, services.prefix_cache, resolved_token)


async def _voice_reference(
    lookup: _VoiceLookup,
    runtime: Any,
    components: SpeechComponents,
    request: SpeechRequest,
    first_text: str,
    *,
    lease: Any,
    request_id: str,
) -> tuple[Reference, dict[str, Any]]:
    """The reference for a resolved voice, and its `speech.accepted` fields (`reference`:
    `voice_prefix`, with `warm` for a cache hit, or `voice_codes`). Needs `lease` held.

    `synthesis.voice_reference` decides the path, as it did for the pre-gate stand-in: with a
    `ref_text` override, the codes path (the stored codes with the given text). Otherwise the
    prefix path: the voice's KV prefix from the cache, under the key the registry computed when
    the voice was registered, built on a miss on the GPU thread under `lease`, with the token
    read when the voice was resolved. Not shielded: if this request is cancelled during the
    build, `get_or_build` hands `lease` to the build (module docstring, step 7).

    A build that runs out of GPU memory is not cached (`VoicePrefixCache`), and has already
    freed what it held (`PrefixBuildOutOfMemory`); `_fall_back_to_codes` then decides what this
    request gets (`first_text` is piece 0's text). Any other build error propagates (a `500`).
    """
    voice = lookup.voice
    shape = voice_reference(voice, lookup.spec.ref_text_override)
    if isinstance(shape, CodesRef):
        return shape, {"reference": "voice_codes"}
    built = await voice_prefix(
        shape,
        voice.prefix_key,
        lookup.prefix_cache,
        runtime,
        components.gpu,
        lease=lease,
        resolved_token=lookup.resolved_token,
        request_id=request_id,
    )
    if isinstance(built, PrefixOutOfMemory):
        return await _fall_back_to_codes(
            lookup,
            shape,
            runtime,
            components,
            request,
            first_text,
            request_id=request_id,
            error=built.error,
        )
    reference, warm = built
    return reference, {"reference": "voice_prefix", "warm": warm}


@dataclass(frozen=True)
class PrefixOutOfMemory:
    """`voice_prefix`'s answer when the build ran out of GPU memory. Every cached prefix has
    been evicted and PyTorch's cache emptied by then, so the voice's codes path, which needs
    more memory than the failed build, gets the most room there is. `error` is the build's
    message, for `speech.prefix_fallback`."""

    error: str


async def voice_prefix(
    shape: UnbuiltPrefix,
    prefix_key: tuple[str, str],
    prefix_cache: VoicePrefixCache,
    runtime: Any,
    gpu: GpuThread,
    *,
    lease: Any,
    resolved_token: int,
    request_id: str,
) -> tuple[PrefixRef, bool] | PrefixOutOfMemory:
    """A voice's prefix reference and whether it was already cached (`warm`), built on the GPU
    thread under `lease` on a miss, with the prefix cache's `resolved_token` as read when the
    voice was resolved. Shared by the speech route and the WebSocket (ws_server), which each
    decide what an out-of-memory build means for them (`PrefixOutOfMemory`).

    Not shielded: if the caller is cancelled during the build, `get_or_build` hands `lease` to
    the build (module docstring, step 7). A build that runs out of GPU memory is not cached
    (`VoicePrefixCache`) and has already freed what it held (`PrefixBuildOutOfMemory`). Any other
    build error propagates."""

    def build() -> Any:
        return gpu.run(build_voice_prefix, runtime, shape.codes, shape.ref_text)

    try:
        prefix, warm = await prefix_cache.get_or_build(
            prefix_key, build, lease=lease, resolved_token=resolved_token, request_id=request_id
        )
    except PrefixBuildOutOfMemory as error:
        prefix_cache.evict_all(reason="oom", request_id=request_id)
        await gpu_call_under_lease(gpu, lease, release_cached_gpu_memory)
        return PrefixOutOfMemory(str(error))
    return PrefixRef(prefix=prefix, ref_text=shape.ref_text), warm


def report_prefix_fallback(
    events: Any, *, request_id: str, voice_id: str, reason: str, error: str
) -> None:
    """`speech.prefix_fallback` (data-model.md "Events"): a voice's prefix build ran out of GPU
    memory, and the request (or WebSocket session: its id is the `request_id`) either took the
    codes path (`out_of_memory`) or was refused for lack of room there (`no_room`)."""
    events.emit(
        "speech.prefix_fallback",
        level="warning",
        request_id=request_id,
        voice_id=voice_id,
        reason=reason,
        error=error,
    )


async def _fall_back_to_codes(
    lookup: _VoiceLookup,
    shape: UnbuiltPrefix,
    runtime: Any,
    components: SpeechComponents,
    request: SpeechRequest,
    first_text: str,
    *,
    request_id: str,
    error: str,
) -> tuple[Reference, dict[str, Any]]:
    """After a prefix build ran out of GPU memory: the voice's codes path with its stored text,
    if piece 0 has room on it, else `503 gpu_out_of_memory` (review 43 #3, #4).

    The codes path needs more memory than the build that failed, so `voice_prefix` has already
    evicted every cached prefix (`voice.prefix_evicted`, reason `oom`; up to
    `VOICE_PREFIX_CACHE_BYTES` of other voices' KV) and emptied PyTorch's cache on the GPU
    thread.

    The pre-gate check sized this voice on its prefix path only, so piece 0's room on the codes
    path is measured here, on the CPU tokenizer's worker (off the loop, with its own copy). No
    room means the request can't be served right now, which is not the text's fault: a `503`
    with its own code, before the `200`, rather than `400 text_too_long` after the gate.
    `speech.prefix_fallback` records either outcome (`reason`: `out_of_memory` for the fallback,
    `no_room` for the refusal). Only piece 0 is sized here; a later piece is sized when it runs,
    and on the codes path can still find no room after the `200` (data-model.md "Reference").
    """
    codes_path = shape.codes_path()
    room = await components.cpu_tokenizer.run(
        _measure_first_piece, runtime, request, first_text, codes_path
    )
    fits = room.room > 0
    report_prefix_fallback(
        components.events,
        request_id=request_id,
        voice_id=lookup.voice.id,
        reason="out_of_memory" if fits else "no_room",
        error=error,
    )
    if not fits:
        raise ApiError(503, "gpu_out_of_memory", "not enough GPU memory for this voice right now")
    return codes_path, {"reference": "voice_codes"}


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


def _opening_budget(request: SpeechRequest) -> int:
    """`split_text`'s `first_budget`: the soft opening budget with no reference, but never
    more than `split_chars`, so piece 0 is never the largest piece and this agrees with the
    streaming `segment()` path, which drops the opening budget when it isn't smaller
    (review 26b #3). `0` (no opening piece) with a reference or with no length splitting."""
    if not isinstance(request.reference, NoReference) or request.split_chars <= 0:
        return 0
    return min(ANCHOR_CHARS, request.split_chars)


async def _size_first_piece(
    runtime: Any,
    cpu_tokenizer: CpuTokenizer,
    request: SpeechRequest,
    pieces: list[str],
    decoded_audio: reference_audio.DecodedAudio | None,
    voice: ResolvedVoice | None,
) -> None:
    """`400 text_too_long` if piece 0 has no room, decided before the GPU gate is taken.

    FR-007 puts every `400` before `409 busy`, so this can't wait for the codec: an inline
    reference is sized by its *predicted* frame count (`stand_in_reference`), a voice by its
    resolved codes and transcript (as the prefix it will build, or with an override as codes),
    and the inputs are built on the CPU (`predicted_room`). It runs on the CPU tokenizer's own
    thread, not the event loop, because tokenizing is blocking CPU work, and with that thread's copy
    (`CpuTokenizer`), never the one the GPU thread uses. The real inputs are checked again
    on the GPU thread once the reference is encoded (`_prepare_first_piece`), which covers a
    codec whose frame count differs from the prediction.

    The later pieces' own lengths, needed for the anchor decision, are measured after the
    lease instead (`_start_anchor_sizing`, review #2 on 2d9070a), on a worker of their own
    (review 34 finding 5): a request about to get a `409` must never size its whole text, and
    the lease holder's sizing must never queue behind this pre-gate worker's checks.
    """
    stand_in = stand_in_reference(
        request.reference,
        None if decoded_audio is None else decoded_audio.predicted_frames,
        int(runtime.model.config.num_codebooks),
        voice=voice,
    )
    room = await cpu_tokenizer.run(_measure_first_piece, runtime, request, pieces[0], stand_in)
    if room.room <= 0:
        raise ApiError(400, "text_too_long", "text is too long")


def _measure_first_piece(
    tokenizer: Any,
    runtime: Any,
    request: SpeechRequest,
    text: str,
    reference: Reference | UnbuiltPrefix,
) -> PieceRoom:
    """Piece 0's room (its text `text`) with `reference`, run by `CpuTokenizer.run` with its
    copy: `_size_first_piece`'s blocking part before the gate, and the out-of-memory fallback's
    codes-path check after it (`_fall_back_to_codes`). Only piece 0 is measured; a later piece is
    sized when it runs, on whichever path the request took (BC-47)."""
    return predicted_room(
        runtime,
        tokenizer,
        reference,
        text,
        request.instruction,
        request.cfg_scale,
        request.max_new_tokens,
    )


def _start_anchor_sizing(
    runtime: Any, cpu_tokenizer: CpuTokenizer, request: SpeechRequest, pieces: list[str]
) -> _AnchorSizingJob | None:
    """Queues what the anchor decision after piece 0 needs (`_anchor_for_later_pieces`) on the
    CPU tokenizer's sizing worker, right after the lease is taken (review #2 on 2d9070a) -- `None`
    with a reference or a single piece: only piece 0 can ever anchor (data-model.md
    "Reference"), and a single piece has no later ones to measure.

    Queuing rather than awaiting here is the point: this runs on the CPU while piece 0 itself
    prepares and generates on the `GpuThread` below, instead of serially before either. The
    caller hands the returned job to `_iter_pieces`, which blocks on it (`.result()`, on
    the GPU thread, once piece 0 has actually finished) only when it is actually needed.
    """
    if not isinstance(request.reference, NoReference) or len(pieces) < 2:
        return None
    return _AnchorSizingJob(cpu_tokenizer.submit(_size_later_pieces, runtime, request, pieces))


def _size_later_pieces(
    tokenizer: Any, runtime: Any, request: SpeechRequest, pieces: list[str]
) -> AnchorSizing:
    """`_start_anchor_sizing`'s queued part, run by `CpuTokenizer.submit` with its copy."""
    return anchor_sizing(
        runtime, tokenizer, pieces[0], pieces[1:], request.instruction, request.cfg_scale
    )


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
    return inputs, piece_room(
        runtime, inputs, request.max_new_tokens, prefix=prefix_of(reference)
    )



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
    sizing_job: _AnchorSizingJob | None,
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
    together with its text, unless `_anchor_for_later_pieces` decides against it, from the
    later pieces' lengths `sizing_job` is still measuring, queued right after the lease
    (`None`: no anchoring, as with a reference or a single piece); their inputs are still built
    here, one at a time. Only piece 0 can anchor (data-model.md "Reference"); without an anchor
    the later pieces stay voice design. A cancelled or failed piece 0 never reaches the
    anchoring step.

    However this generator ends -- exhausted, failed, or closed by a disconnect -- the sizing
    is abandoned if still queued (review 33 on 10f0c29), so nothing is left on the sizing
    worker for the next lease holder's own sizing to wait behind. The anchor decision abandons
    it earlier whenever it doesn't need it (review 34 finding 1). One already running or done
    is left alone.
    """
    try:
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
                room = piece_room(
                    runtime, inputs, request.max_new_tokens, prefix=prefix_of(reference)
                )
            max_new_tokens = piece_frame_limit(
                room,
                events,
                request_id=request_id,
                piece_index=index,
                requested=request.max_new_tokens,
            )
            frames: list[Any] | None = [] if sizing_job is not None and index == 0 else None
            piece_bytes = 0
            for chunk in generate_piece(
                runtime,
                inputs,
                request_id=request_id,
                seed=piece_seed(request.seed, index),
                chunk_first=chunk_first,
                chunk_max=chunk_max,
                samples_per_frame=samples_per_frame,
                prefix=prefix_of(reference),
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
                reference = _anchor_for_later_pieces(
                    runtime,
                    frames,
                    text,
                    sizing_job,
                    request,
                    events,
                    request_id=request_id,
                    frame_limit=max_new_tokens,
                )
    finally:
        if sizing_job is not None:
            sizing_job.abandon()


def _anchor_for_later_pieces(
    runtime: Any,
    frames: list[Any],
    text: str,
    sizing_job: _AnchorSizingJob,
    request: SpeechRequest,
    events: Emitter,
    *,
    request_id: str,
    frame_limit: int,
) -> Reference:
    """The reference for the pieces after piece 0: piece 0's own audio and text, or `NoRef()`.

    `frames` are every frame piece 0 generated (pad frames included), `frame_limit` the
    `max_new_tokens` it ran with. The anchor is skipped, with `speech.anchor_skipped`:
    - `piece_truncated`: piece 0 used its whole limit, so it stopped at its cap or room
      rather than at EOS, and its audio may end mid-word -- not a clean reference;
    - `no_room`: the anchor would leave some later piece a smaller effective frame limit than
      it would have without it (`_anchor_costs_frames`). The anchor is never trimmed to fit
      instead: its codes must stay paired with its text.
    Or the later pieces' lengths never arrived (review 33 on 10f0c29). The `200` is already out
    by then, so skipping the anchor is the only safe answer left, not aborting the stream:
    - `shutdown`: `CpuTokenizer.shutdown()` cancelled the sizing while it was still queued
      (review #3 on 2d9070a);
    - `sizing_timeout`: it was still unfinished after `ANCHOR_SIZING_TIMEOUT_SECONDS`. The
      sizing has a worker of its own, so this is only a backstop against a wedged or starved
      CPU. This GPU thread must not wait on it for long: a disconnect's `gen.close()` would
      queue behind this wait and, past `GPU_CLOSE_TIMEOUT_SECONDS`, poison the gate;
    - `sizing_failed`: it raised (a template or tokenizer error, or no CPU tokenizer). That is
      still a server bug, so it is also reported, as `speech.anchor_sizing_failed` with its
      traceback -- not `request.failed`, since the request itself goes on to succeed (review
      34 finding 4).
    Zero non-pad frames means there is nothing to anchor on; that needs no event.

    The route can also cancel the sizing while this waits (`_serve_speech`'s cleanup, when the
    request is cancelled or fails while this step is still in flight on the GPU thread). That
    is the request ending, not a shutdown, and nothing more will stream, so it emits no event
    at all (`_AnchorSizingJob.abandon`, review 34 finding 3).

    Whenever the answer doesn't need the sizing -- piece 0 truncated, nothing to anchor on, or
    the wait timed out -- it is abandoned right here (review 34 finding 1), not left queued
    while every later piece streams.

    This runs on the GPU thread between piece 0 and piece 1, so it tokenizes nothing: every
    later piece's length was already queued right after the lease (`sizing_job`,
    `_start_anchor_sizing`), running on the CPU while piece 0 itself generated here, and the
    anchor's frame count is all that was missing -- `.result()` only blocks for whatever, if
    anything, is left of that by now, and for at most `ANCHOR_SIZING_TIMEOUT_SECONDS`.
    """
    if len(frames) >= frame_limit:
        sizing_job.abandon()
        _skip_anchor(events, request_id, "piece_truncated")
        return NoRef()
    codes = anchor_codes(frames, int(runtime.model.config.codebook_pad_token_id))
    if codes is None:
        sizing_job.abandon()
        return NoRef()
    try:
        sizing = sizing_job.future.result(timeout=ANCHOR_SIZING_TIMEOUT_SECONDS)
    except FutureCancelledError:
        # Abandoned by the route means the request is ending: nothing more will stream, so
        # there is nothing to report. Otherwise only a shutdown cancels it.
        if not sizing_job.abandoned.is_set():
            _skip_anchor(events, request_id, "shutdown")
        return NoRef()
    except TimeoutError:
        sizing_job.abandon()
        _skip_anchor(events, request_id, "sizing_timeout")
        return NoRef()
    except Exception as error:  # noqa: BLE001 - reported; piece 0 is already out
        events.emit(
            "speech.anchor_sizing_failed",
            level="error",
            request_id=request_id,
            **error_fields(error),
        )
        _skip_anchor(events, request_id, "sizing_failed")
        return NoRef()
    if _anchor_costs_frames(runtime, sizing, int(codes.shape[0]), request.max_new_tokens):
        _skip_anchor(events, request_id, "no_room")
        return NoRef()
    return CodesRef(codes=codes, ref_text=text)


def _anchor_costs_frames(
    runtime: Any, sizing: AnchorSizing, anchor_frames: int, requested: int | None
) -> bool:
    """The `no_room` rule (decided with the user, 2026-09-25): some later piece would get a
    smaller effective frame limit -- `min(cap, room)` -- with the anchor than without it.
    This also catches a piece already clamped below its cap *without* the anchor, whose room
    the anchor can shrink further, even to zero (BC-47) -- the case the old floor missed by
    only ever comparing an anchored room to the cap, never to the room the same piece would
    have gotten without the anchor.

    `room_for_length` is already the effective limit (`min(cap, context room)` --
    `models/fast_streaming.py`'s own rule), so the comparison needs nothing else from `cap`
    itself; CFG rows and graph-bucket padding count in it exactly as they will when the piece
    runs.
    """
    for length in sizing.later_lengths:
        limit_without = runtime.room_for_length(requested, length)
        anchored = sizing.anchored(length, anchor_frames)
        limit_with = runtime.room_for_length(requested, anchored)
        if limit_with < limit_without:
            return True
    return False


def _skip_anchor(events: Emitter, request_id: str, reason: str) -> None:
    events.emit("speech.anchor_skipped", request_id=request_id, piece_index=0, reason=reason)


async def _outlast(gpu_task: asyncio.Future[Any]) -> None:
    """Wait for an abandoned GPU-thread call, during cleanup for the exception already in
    flight -- a cancellation, since only an interruption leaves the call running -- without
    letting anything but a process exit replace that exception.

    - The call's own failure is dropped: the caller's exception is what must be reported.
    - Another cancellation while waiting stops the wait (the lease is released by the
      call's done-callback, not here) and is undone (`_undo_cancel`), so the caller re-raises
      the original with the task's cancellation count as it was.
    - `KeyboardInterrupt`/`SystemExit` are not caught: the process is going down.
    """
    try:
        await asyncio.shield(gpu_task)
    except asyncio.CancelledError:
        _undo_cancel()
    except Exception:  # noqa: BLE001, S110 -- the caller's own exception takes precedence
        pass


def _undo_cancel() -> None:
    """Withdraw one cancellation request from the current task: one that arrived while a
    cancellation was already being handled, and that the re-raised original stands for."""
    task = asyncio.current_task()
    if task is not None:
        task.uncancel()


async def _close_quietly(
    session: GpuSession[bytes], events: Emitter, request_id: str, original: BaseException
) -> None:
    """Close `session` during cleanup for `original`, a failure that must still reach the
    client (T046 review, finding 6), and that the caller raises once this returns.

    A close error is reported (`gpu.report_close_failed`, which leaves a close timeout to the
    gate's own `gpu.close_timeout`), never raised. `aclose()` has released or poisoned the gate
    by the time it returns or raises. A cancellation while closing -- raised by `aclose()`, or
    the cause of a `GpuCloseTimeout` (`GpuThread.wait_closed` raises that one `from` the
    cancellation) -- follows one rule:
    - `original` is an `Exception`: the cancellation wins, as `wait_closed` requires: timeouts
      and task groups depend on seeing it. It is raised here, chained `from original`, which
      is first reported as errors.py's catch-all handler would have (`report_unhandled`),
      since that handler will now never see it;
    - otherwise `original` is a cancellation, or a `KeyboardInterrupt`/`SystemExit` taking the
      process down: the new cancellation is undone, as in `_outlast`, and the caller re-raises
      `original` unchanged.
    """
    try:
        await session.aclose()
    except asyncio.CancelledError as cancelled:
        _settle_cancel_while_closing(cancelled, original, events, request_id)
    except Exception as close_error:  # noqa: BLE001 -- the close error is reported, not raised
        report_close_failed(events, close_error, request_id=request_id)
        cancelled = close_error.__cause__
        if isinstance(close_error, GpuCloseTimeout) and isinstance(
            cancelled, asyncio.CancelledError
        ):
            _settle_cancel_while_closing(cancelled, original, events, request_id)


def _settle_cancel_while_closing(
    cancelled: asyncio.CancelledError,
    original: BaseException,
    events: Emitter,
    request_id: str,
) -> None:
    """`_close_quietly`'s rule for a cancellation that arrived while closing."""
    if isinstance(original, Exception):
        report_unhandled(events, request_id, original)
        raise cancelled from original
    _undo_cancel()


async def _rest_of_audio(session: GpuSession[bytes]) -> AsyncGenerator[bytes, None]:
    """The body `SpeechResponse` streams after the primed first chunk: step the session
    until it reports `DONE`. Mirrors `tests/test_speech_abort.py`'s own `rest_of_audio`."""
    while True:
        chunk = await session.step()
        if chunk is DONE:
            return
        yield chunk


async def _buffered(
    session: GpuSession[bytes],
    events: Emitter,
    request_id: str,
    *,
    sample_rate: int,
    primed_bytes: int,
) -> AsyncGenerator[bytes, None]:
    """The GET route's body (research R4): the same chunks as `_rest_of_audio`, but generated
    at full speed into a buffer rather than one chunk per send, so the GPU is released when
    generation ends, not when the client has read everything. A browser that pauses, or
    plays slowly, must not hold the one GPU while it drains (FR-013, FR-014); it costs only
    the buffer's memory. That memory is not capped: a client that keeps reading, however
    slowly, keeps it (a deliberate choice; spec Assumptions, research R4).

    A producer task, started on the first iteration (after the primed chunk is sent), steps
    the session into an unbounded queue:
    - a chunk is queued as it comes;
    - a failure from `step()` closes the session, then is queued and re-raised here, so the
      stream ends as `speech.failed`, as on the POST route;
    - at `DONE` it closes the session, which releases the gate, then emits
      `speech.generated` and queues the end marker (`None`). A failed close is queued
      instead, so the request is reported as failed.

    Closing here is safe although `SpeechResponse` closes the session again when the response
    ends: `GpuSession.aclose()` is idempotent, and every call waits for the same close.
    `primed_bytes` is the PCM already generated before the `200`, so `audio_seconds` counts
    the whole request's audio.

    However this generator ends, its `finally` cancels the producer: after a disconnect
    mid-generation that cancels the `session.step()` in flight, as cancelling
    `_rest_of_audio` does on the POST route, so the GPU is freed within one chunk (FR-015).
    """
    queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

    async def produce() -> None:
        generated = primed_bytes
        try:
            while (chunk := await session.step()) is not DONE:
                generated += len(chunk)
                queue.put_nowait(chunk)
        except Exception as error:  # noqa: BLE001 -- re-raised by the body, below
            # Close first: queued behind every buffered chunk, the error would otherwise keep
            # the GPU held until the client had read them all (FR-014). A close failure here is
            # suppressed: the response's final close raises it again and reports it, as on the
            # POST route. Once step() has raised, the generator is finished, so this close has
            # nothing left to do and is not expected to fail.
            with contextlib.suppress(Exception):
                await session.aclose()
            queue.put_nowait(error)
            return
        try:
            await session.aclose()
        except Exception as error:  # noqa: BLE001 -- re-raised by the body, below
            # Reported through the stream, not left to the response's final close: after a
            # close timeout that close waits afresh and may succeed, so the request would be
            # reported as completed on a poisoned gate.
            queue.put_nowait(error)
            return
        events.emit(
            "speech.generated",
            request_id=request_id,
            audio_seconds=generated / _PCM_BYTES_PER_SAMPLE / sample_rate,
            format="wav",
        )
        queue.put_nowait(None)

    producer = asyncio.create_task(produce())
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        producer.cancel()
        # `wait`, not `await producer`: it neither raises the producer's own cancellation
        # nor swallows one aimed at this task (a suppressed `await` would lose it).
        await asyncio.wait([producer])


def wav_header(sample_rate: int) -> bytes:
    """The 44-byte header of a mono s16le WAV stream (contracts/http-wav-stream.md).

    The RIFF and `data` sizes are `0xFFFFFFFF`, the usual "unknown length" value: the header
    goes out before the audio is generated, so the body's final size isn't known yet.
    """
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        0xFFFFFFFF,
    )


async def _serve_speech(
    http_request: Request,
    runtime: Any,
    components: SpeechComponents,
    *,
    request_id: str,
    received_at: float,
    clock: Callable[[], float],
    wav: bool = False,
) -> Response:
    """Everything `install_speech`'s routes do once they have the runtime and the request's id;
    separate so tests can drive (and cancel) it directly. `wav` selects the GET route's
    response: the same PCM behind a WAV header, as `audio/wav`, delivered from a buffer
    (`_buffered`)."""
    fields = await read_fields(http_request)
    request = parse_speech(fields, components.settings)

    # CPU-only, so it runs before the voice lookup, the reference decode and the busy
    # check (FR-007 stage 2, "field syntax and ranges"). `parse_speech` already rejects text
    # with nothing speakable (`text_required`, by `text_split.speakable`, the rule split_text
    # drops units by),
    # so an empty split is not expected here: the check below is only a safety net, should
    # the two rules ever drift apart, so such text still gets the same `400` rather than an
    # IndexError on `pieces[0]`.
    #
    # With no reference, piece 0 is packed against the soft opening budget (US3 scenario 1):
    # it becomes the anchor for every later piece, so it should be short enough for a quick
    # first audio. A reference already fixes the voice, so there is no opening piece.
    first_budget = _opening_budget(request)
    pieces = split_text(request.text, budget=request.split_chars, first_budget=first_budget)
    if not pieces:
        raise ApiError(400, "text_required", "text is required")

    voice_lookup = None
    if isinstance(request.reference, VoiceRef):
        voice_lookup = _look_up_voice(components, request.reference)

    decoded_audio = None
    if isinstance(request.reference, InlineRef):
        # Off the event loop: libsndfile's decode is blocking CPU work and must not stall
        # every other request in flight while it runs.
        decoded_audio = await asyncio.to_thread(
            reference_audio.decode, request.reference.audio_bytes
        )

    await _size_first_piece(
        runtime,
        components.cpu_tokenizer,
        request,
        pieces,
        decoded_audio,
        None if voice_lookup is None else voice_lookup.voice,
    )

    # None means busy (409); a poisoned gate raises GpuUnavailable instead (gpu.py), which
    # propagates straight past this route to errors.py's own handler (503 gpu_unavailable).
    lease = components.gate.try_acquire()
    if lease is None:
        raise ApiError(409, "busy", "busy")

    session: GpuSession[bytes] | None = None
    gpu_task: asyncio.Task[Any] | None = None
    sizing_job: _AnchorSizingJob | None = None
    try:
        # Queued right after the lease (review #2 on 2d9070a), not before the busy check: a
        # request that gets 409 above must never run this at all. `None` with a reference or a
        # single piece (`_start_anchor_sizing`).
        sizing_job = _start_anchor_sizing(runtime, components.cpu_tokenizer, request, pieces)
        if voice_lookup is not None:
            reference, accepted_fields = await _voice_reference(
                voice_lookup,
                runtime,
                components,
                request,
                pieces[0],
                lease=lease,
                request_id=request_id,
            )
        else:
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
            accepted_fields = _reference_fields(request.reference)
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
            **accepted_fields,
        )

        gen = _iter_pieces(
            runtime,
            reference,
            pieces,
            request,
            request_id,
            first_inputs,
            first_room,
            sizing_job,
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
    except BaseException as error:
        # A cancellation (client disconnect) or any other failure while a GPU-thread call is
        # still in flight must not free the lease before that call has actually finished:
        # cancelling *this* await doesn't cancel work already submitted to the (single-
        # threaded) GPU executor (gpu.py), so releasing early would tell the next request
        # "free" while our own abandoned work is still really queued ahead of it there.
        #
        # The anchor sizing is abandoned first (review 33 on 10f0c29), so a failed request
        # leaves nothing queued ahead of the next lease holder's own sizing. Only the anchor
        # decision after piece 0 waits on it, and before the `200` the stream has not got that
        # far unless piece 0 produced no audio at all. Then this can race that decision's wait
        # on the GPU thread: `abandon()`, not a bare cancel, tells it the request is ending, so
        # it ends early without reporting a shutdown (review 34 finding 3). If the generator
        # started, closing it abandons the sizing too (`_iter_pieces`); this covers every exit
        # before that.
        if sizing_job is not None:
            sizing_job.abandon()
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
            await _outlast(gpu_task)
        elif session is not None:
            await _close_quietly(session, components.events, request_id, error)
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
        # Not ApiError: every 500 must close the connection (contracts/http-api.md), which
        # only happens on the genuinely unhandled path (errors.py's catch-all re-raises after
        # answering); ApiError(500, ...) would be an ordinary, keep-alive JSON response.
        no_audio = RuntimeError("generation produced no audio")
        # Same rule as the `except BaseException` cleanup above: a close failure must not
        # replace "no_audio" as the reason this request failed (a cancel still wins).
        await _close_quietly(session, components.events, request_id, no_audio)
        raise no_audio

    components.events.emit(
        "speech.first_audio",
        request_id=request_id,
        ttfa_ms=(clock() - received_at) * 1000,
    )

    sample_rate = int(runtime.sample_rate)
    wav_fields: dict[str, Any] = {}
    if wav:
        body = _buffered(
            session,
            components.events,
            request_id,
            sample_rate=sample_rate,
            primed_bytes=len(first_chunk),
        )
        # The header rides in the first chunk, so its 44 bytes count toward
        # `audio_seconds_sent` (under 1 ms of audio; research R4 accepts that).
        first_chunk = wav_header(sample_rate) + first_chunk
        wav_fields = {
            "headers": {"Accept-Ranges": "none"},
            "media_type": "audio/wav",
            "event_fields": {"format": "wav"},
            # Delivery comes from `_buffered`'s buffer and no longer holds the GPU, so only
            # a reader stalled for 10 minutes is cut off, and no minimum rate applies: an
            # infinite grace keeps `_send_audio`'s budget infinite (FR-016).
            "send_timeout": WAV_SEND_TIMEOUT_SECONDS,
            "min_rate_grace": math.inf,
        }
    else:
        body = _rest_of_audio(session)
    return SpeechResponse(
        first_chunk=first_chunk,
        body=body,
        session=session,
        events=components.events,
        request_id=request_id,
        sample_rate=sample_rate,
        clock=clock,
        started_at=started_at,
        **wav_fields,
    )


def install_speech(
    app: FastAPI,
    components: SpeechComponents,
    *,
    clock: Callable[[], float],
) -> None:
    """Register `POST /v1/audio/speech` (raw PCM) and `GET /v1/audio/speech.wav` (the same
    fields in the query string, answered as a WAV stream). Both are served by `_serve_speech`.

    `clock` is injected (Constitution III): it times `ttfa_ms` and feeds `SpeechResponse`'s own
    `rtf` (module docstring: two different readings of the same clock, not the same reading
    reused). The request's id is `request_id.RequestIdMiddleware`'s, read from its state.
    """
    @app.post("/v1/audio/speech")
    async def speech(
        http_request: Request,
        runtime: Annotated[Any, Depends(components.readiness.require_ready)],
    ) -> Response:
        return await _serve_speech(
            http_request,
            runtime,
            components,
            request_id=http_request.state.request_id,
            received_at=clock(),
            clock=clock,
        )

    @app.get("/v1/audio/speech.wav")
    async def speech_wav(
        http_request: Request,
        runtime: Annotated[Any, Depends(components.readiness.require_ready)],
    ) -> Response:
        return await _serve_speech(
            http_request,
            runtime,
            components,
            request_id=http_request.state.request_id,
            received_at=clock(),
            clock=clock,
            wav=True,
        )
