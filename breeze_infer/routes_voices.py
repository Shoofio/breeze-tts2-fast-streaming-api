"""`POST /v1/voices`, `GET /v1/voices` and `DELETE /v1/voices/{id}` (specs/003-cpp-compatible-api/
tasks.md T065; contracts/http-api.md; data-model.md "Voice").

`POST` runs the contract's order of checks:
1. body size (`body_limit.py`, before the route runs);
2. fields (`http_fields.parse_voice_create`);
3. a named voice whose name is taken, ignoring case (a skipped file's name included):
   `409 voice_exists`;
4. an unnamed voice already registered: `200` with the existing entry, without the gate;
5. decode and limits (`reference_audio.decode`, on a worker thread), then the prefix the voice
   would build, measured on the CPU tokenizer's worker from its predicted frames: one the runtime
   would refuse to build is `400 voice_too_long` (`synthesis.prefix_fits`). An unnamed voice is
   then looked up once more, since an identical `POST` may have registered it meanwhile;
6. busy (`gate.try_acquire`): `409 busy`;
7. the encode, on the `GpuThread` through `audio.encode_prompt_waveform`. A frame count other
   than the predicted one is reported (`voice.frame_prediction_mismatch`) and measured again.
   Codes the startup scan would refuse (`voice_file.checked_codes`: frames, codebooks, code range,
   `encode_ms`) are a `500 internal_error`, before anything is written or registered;
8. a named voice's file is written, and the name checked again at commit (`VoiceStore.create`):
   a clash is `409 voice_exists`, a write failure `500 voice_write_failed`.

Every store and registry change runs on the voice services' one worker thread
(`VoiceServices.change`), never the event loop (the store holds its lock across a rename and an
fsync) and never asyncio's default pool, which the speech route's decode and form parsing share:
a burst of changes on a slow disk waits on its own thread. One thread also means one change at a
time, so a `DELETE` can't land between a `POST`'s file commit and its registration, and an
unnamed id can't be registered between a `POST`'s last lookup and its own registration. Registry
reads (`name_taken`, `find_unnamed`, `lookup`) stay on the event loop, under the registry's lock.

What follows a change on the event loop (dropping a cached prefix, which has no locks, and the
`voice.*` events) is scheduled by that worker itself, right after the change: it runs even if
the request was cancelled while the change ran, and before the request resumes.

`DELETE` matches the id exactly as `GET` lists it (or a skipped file's exact stem): the store
and the registry both do. It never checks busy, and it drops the voice's cached prefix with the
DELETE's own `request_id`.

The two storage `500`s carry their own codes, so they are answered here rather than by the
catch-all handler; they still close the connection (`Connection: close`), like every `500`
(contracts/http-api.md "Errors"), and are reported as `request.failed`.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the routes' `Annotated[...]`,
# which refers to the local `components`, when the routes are defined (routes_speech.py does
# the same, for the same reason).
import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, Protocol, TypeVar

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from breeze_infer import reference_audio, voice_file
from breeze_infer.audio import encode_prompt_waveform
from breeze_infer.errors import ApiError, api_error_response, report_unhandled
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuLease, GpuThread
from breeze_infer.http_fields import parse_voice_create, read_fields
from breeze_infer.routes_health import Readiness
from breeze_infer.synthesis import (
    codec_samples_per_frame,
    measure_voice_prefix,
    prefix_fits,
)
from breeze_infer.voice_prefix import VoicePrefixCache
from breeze_infer.voice_registry import (
    NameTaken,
    RemovedVoice,
    VoiceEntry,
    VoiceRegistry,
    api_record,
    unnamed_id,
)
from breeze_infer.voice_store import VoiceExists, VoiceStore

_T = TypeVar("_T")


def _voice_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-voices")


@dataclass(frozen=True)
class VoiceServices:
    """What the voice routes work on, built by the startup scan (`api.open_voices`), and the one
    worker thread every store and registry change runs on (module docstring). `api._drain_gpu`
    shuts that thread down with the server, as it does the CPU tokenizer's."""

    store: VoiceStore
    registry: VoiceRegistry
    prefix_cache: VoicePrefixCache
    executor: ThreadPoolExecutor = field(default_factory=_voice_thread)

    async def change(self, change: Callable[[], _T], then: Callable[[_T], None]) -> _T:
        """Run `change()` on the voice thread and return its result; `then(result)` runs on the
        event loop once it has succeeded.

        The voice thread schedules `then` itself, as soon as `change` returns, rather than the
        caller running it after its `await`: a request cancelled while its change runs (a client
        disconnect) can't stop the change, so it must not skip what goes with it either. `then`
        is on the loop because the prefix cache has no locks; it runs before the caller resumes.
        """
        loop = asyncio.get_running_loop()

        def run() -> _T:
            result = change()
            loop.call_soon_threadsafe(then, result)
            return result

        return await loop.run_in_executor(self.executor, run)

    def shutdown(self) -> None:
        """Refuse new changes and drop queued ones; one already running finishes on its own."""
        self.executor.shutdown(wait=False, cancel_futures=True)


class VoiceSlot:
    """Empty until the startup scan's `VoiceServices` are installed (`api.Components.mark_ready`,
    before the server reports ready). Read and written on the event loop only."""

    def __init__(self) -> None:
        self._services: VoiceServices | None = None

    def install(self, services: VoiceServices) -> None:
        self._services = services

    def get(self) -> VoiceServices:
        if self._services is None:
            raise RuntimeError("no voice store: the startup scan installs it before ready")
        return self._services

    def shutdown(self) -> None:
        """Stop the installed services' voice thread; nothing to stop before startup ends."""
        if self._services is not None:
            self._services.shutdown()


class VoiceComponents(Protocol):
    """The subset of `api.Components` these routes need, duck-typed for the same reason as
    `routes_speech.SpeechComponents` (importing `Components` here would be circular)."""

    events: Emitter
    gate: GpuGate
    gpu: GpuThread
    readiness: Readiness
    voices: VoiceSlot
    # `routes_speech.CpuTokenizer`, which imports this module: `Any` rather than a cycle.
    cpu_tokenizer: Any


def _record(entry: VoiceEntry, runtime: Any) -> dict[str, Any]:
    return api_record(
        voice_id=entry.id,
        ref_text=entry.ref_text,
        frames=entry.frames,
        encode_ms=entry.encode_ms,
        saved=entry.saved,
        sample_rate=int(runtime.sample_rate),
        samples_per_frame=codec_samples_per_frame(runtime),
    )


def _storage_failure(
    events: Emitter, request_id: str, error: OSError, code: str, message: str
) -> JSONResponse:
    report_unhandled(events, request_id, error)
    return api_error_response(ApiError(500, code, message), headers={"Connection": "close"})


async def _measured_prefix(runtime: Any, cpu_tokenizer: Any, ref_text: str, frames: int) -> int:
    """The length of the prefix a voice of `frames` frames and `ref_text` builds to, measured on
    the CPU tokenizer's worker (`synthesis.measure_voice_prefix`); `400 voice_too_long` if the
    runtime would refuse to build it. The length depends only on the frame count, so zero codes
    stand in for codes not encoded yet."""
    codes = np.zeros((frames, int(runtime.model.config.num_codebooks)), dtype=np.int16)
    prefix_len = await cpu_tokenizer.run(
        lambda tokenizer: measure_voice_prefix(runtime, tokenizer, codes, ref_text)
    )
    if not prefix_fits(runtime, prefix_len):
        raise ApiError(400, "voice_too_long", "the reference is too long for the model's context")
    return prefix_len


def _encode(
    audio_tokenizer: Any, decoded: reference_audio.DecodedAudio, clock: Callable[[], float]
) -> tuple[np.ndarray, int]:
    """GPU-thread only: the reference's codes and how long the encode took, in whole ms."""
    started = clock()
    codes = encode_prompt_waveform(audio_tokenizer, decoded.samples, decoded.sample_rate)
    return codes.numpy(), round((clock() - started) * 1000)


async def _encode_under_lease(
    lease: GpuLease, gpu: GpuThread, audio_tokenizer: Any, decoded: Any, clock: Any
) -> tuple[np.ndarray, int]:
    """Run `_encode` on the GPU thread and release `lease` when that call has really finished.

    The release is a done-callback, not a `finally`: if this request is cancelled (a client
    disconnect) while the encode runs, the call still runs to the end on the GPU thread, and
    the gate must not tell the next request it is free before then (routes_speech.py does the
    same for its own GPU calls). The callback also retrieves an abandoned call's exception, so
    it isn't logged as never retrieved.
    """

    def release(task: asyncio.Future[Any]) -> None:
        lease.release()
        if not task.cancelled():
            task.exception()

    encode = asyncio.ensure_future(gpu.run(_encode, audio_tokenizer, decoded, clock))
    encode.add_done_callback(release)
    return await asyncio.shield(encode)


def _checked_codes(raw: np.ndarray, encode_ms: int, store: VoiceStore) -> np.ndarray:
    """The encode's codes as the voice will keep them (int16), if the startup scan would accept
    them. Otherwise a plain exception, so the catch-all answers `500 internal_error`, closes the
    connection and reports `request.failed`: the codec produced something no voice can hold,
    which is the server's fault, not the request's."""
    try:
        return voice_file.checked_codes(
            raw, encode_ms, codebooks=store.codebooks, codebook_size=store.codebook_size
        )
    except voice_file.VoiceFileError as error:
        raise RuntimeError(f"the encode produced codes no voice can hold: {error}") from error


@dataclass(frozen=True)
class _Registered:
    """A registration's outcome: the entry to answer with, whether this request created it, and
    the unnamed voice it evicted, if any."""

    entry: VoiceEntry
    created: bool
    evicted: str | None = None


def _commit_saved(services: VoiceServices, voice: voice_file.VoiceFile, prefix_len: int) -> _Registered:
    """Voice thread: write the file (the store re-checks the name under its own lock, and its
    commit refuses a file already on disk), then register it.

    The registry refusing a name the store had free means the two have drifted apart. The file
    is then taken back out before `NameTaken` propagates: a `409` must leave nothing on disk to
    reappear at the next restart. If that removal fails, its `OSError` propagates instead."""
    stored = services.store.create(voice)
    try:
        entry = services.registry.register_saved(stored, prefix_len=prefix_len)
    except NameTaken:
        services.store.remove(stored.id)
        raise
    return _Registered(entry, created=True)


def _register_unnamed(
    registry: VoiceRegistry,
    voice_id: str,
    ref_text: str,
    codes: np.ndarray,
    encode_ms: int,
    prefix_len: int,
) -> _Registered:
    """Voice thread: register an unnamed voice, unless an identical `POST` got there first (its
    encode can finish, and free the gate, before it registers). This thread makes every registry
    change, so nothing can register the id between the lookup and the registration."""
    existing = registry.find_unnamed(voice_id)
    if existing is not None:
        return _Registered(existing, created=False)
    entry, evicted = registry.register_unnamed(
        id=voice_id,
        ref_text=ref_text,
        codes=codes,
        frames=int(codes.shape[0]),
        encode_ms=encode_ms,
        prefix_len=prefix_len,
    )
    return _Registered(entry, created=True, evicted=evicted)


def _after_registering(
    services: VoiceServices, events: Emitter, request_id: str, registered: _Registered
) -> None:
    """Event loop: drop an evicted voice's cached prefix, and report a voice this request
    created (none for one another request had already registered)."""
    if registered.evicted is not None:
        services.prefix_cache.invalidate(registered.evicted, request_id=request_id)
    if registered.created:
        entry = registered.entry
        events.emit(
            "voice.created",
            request_id=request_id,
            voice_id=entry.id,
            saved=entry.saved,
            frames=entry.frames,
            encode_ms=entry.encode_ms,
        )


@dataclass(frozen=True)
class _Removal:
    """What a `DELETE` took away: the file (the store's answer) and the registry entry."""

    file_removed: bool
    removed: RemovedVoice | None

    @property
    def found(self) -> bool:
        return self.file_removed or self.removed is not None


def _remove(services: VoiceServices, voice_id: str) -> _Removal:
    """Voice thread: remove the file if there is one, then the registry entry. A store failure
    raises before the registry is touched, so the voice stays registered."""
    file_removed = services.store.remove(voice_id)
    return _Removal(file_removed, services.registry.remove(voice_id))


def _after_removing(
    services: VoiceServices, events: Emitter, request_id: str, voice_id: str, removal: _Removal
) -> None:
    """Event loop: drop the voice's cached prefix and report the delete.

    A saved voice or a skipped file's name has a file; an unnamed voice doesn't. When the store
    and the registry disagree about that (a file removed for an id the registry didn't hold, or
    the reverse), the delete still stands, and `voice.store_mismatch` records the drift."""
    if not removal.found:
        return
    services.prefix_cache.invalidate(voice_id, request_id=request_id)
    kind = None if removal.removed is None else removal.removed.kind
    if removal.file_removed != (kind in ("saved", "reserved")):
        events.emit(
            "voice.store_mismatch",
            level="warning",
            request_id=request_id,
            voice_id=voice_id,
            file_removed=removal.file_removed,
            kind=kind,
        )
    if kind is not None:
        events.emit("voice.deleted", request_id=request_id, voice_id=voice_id, kind=kind)


async def _create_voice(
    http_request: Request,
    runtime: Any,
    components: VoiceComponents,
    clock: Callable[[], float],
    request_id: str,
) -> JSONResponse:
    fields = await read_fields(http_request)
    request = parse_voice_create(fields)
    services = components.voices.get()
    registry = services.registry

    if request.name is not None:
        if registry.name_taken(request.name):
            raise ApiError(409, "voice_exists", "voice already exists")
        voice_id = request.name
    else:
        # Hashing up to 25 MiB is CPU work: off the event loop.
        voice_id = await asyncio.to_thread(unnamed_id, request.audio_bytes, request.ref_text)
        existing = registry.find_unnamed(voice_id)
        if existing is not None:
            return JSONResponse(_record(existing, runtime))

    decoded = await asyncio.to_thread(reference_audio.decode, request.audio_bytes)
    prefix_len = await _measured_prefix(
        runtime, components.cpu_tokenizer, request.ref_text, decoded.predicted_frames
    )
    if request.name is None:
        # Again, just before the gate: an identical POST may have registered the voice while
        # this one decoded, and answering with it saves an encode.
        existing = registry.find_unnamed(voice_id)
        if existing is not None:
            return JSONResponse(_record(existing, runtime))

    # None means busy; a poisoned gate raises GpuUnavailable (503), as for speech.
    lease = components.gate.try_acquire()
    if lease is None:
        raise ApiError(409, "busy", "busy")
    raw_codes, encode_ms = await _encode_under_lease(
        lease, components.gpu, runtime.audio_tokenizer, decoded, clock
    )

    predicted = decoded.predicted_frames
    mispredicted = int(raw_codes.shape[0]) != predicted
    if mispredicted:
        # As speech reports it (`routes_speech._check_frame_prediction`): the prediction formula
        # has drifted from the codec. The real codes are what the voice keeps.
        components.events.emit(
            "voice.frame_prediction_mismatch",
            level="warning",
            request_id=request_id,
            predicted_frames=predicted,
            actual_frames=int(raw_codes.shape[0]),
        )
    codes = _checked_codes(raw_codes, encode_ms, services.store)
    if mispredicted:
        prefix_len = await _measured_prefix(
            runtime, components.cpu_tokenizer, request.ref_text, int(codes.shape[0])
        )

    after = partial(_after_registering, services, components.events, request_id)
    if request.name is None:
        registered = await services.change(
            partial(_register_unnamed, registry, voice_id, request.ref_text, codes, encode_ms, prefix_len),
            after,
        )
        return JSONResponse(_record(registered.entry, runtime))

    voice = voice_file.VoiceFile(
        id=voice_id,
        ref_text=request.ref_text,
        frames=int(codes.shape[0]),
        codebooks=int(codes.shape[1]),
        codes=codes,
        codes_sha256=voice_file.codes_sha256(codes),
        codec_fingerprint=services.store.codec_fingerprint,
        encode_ms=encode_ms,
        created_at="",  # stamped by the store when it writes the file
    )
    try:
        registered = await services.change(partial(_commit_saved, services, voice, prefix_len), after)
    except (VoiceExists, NameTaken):
        raise ApiError(409, "voice_exists", "voice already exists") from None
    except OSError as error:
        return _storage_failure(
            components.events,
            request_id,
            error,
            "voice_write_failed",
            "could not write the voice file",
        )
    return JSONResponse(_record(registered.entry, runtime))


async def _delete_voice(
    voice_id: str, components: VoiceComponents, request_id: str
) -> JSONResponse:
    services = components.voices.get()
    try:
        removal = await services.change(
            partial(_remove, services, voice_id),
            partial(_after_removing, services, components.events, request_id, voice_id),
        )
    except OSError as error:
        # Including IsADirectoryError, for a skipped entry that is a directory: it stays
        # reserved until someone removes it by hand (data-model.md "Lifecycle").
        return _storage_failure(
            components.events,
            request_id,
            error,
            "voice_delete_failed",
            "could not delete the voice file",
        )
    if not removal.found:
        raise ApiError(404, "unknown_voice", "unknown voice_id")
    return JSONResponse({"deleted": voice_id, "file_kept": False})


def install_voices(app: FastAPI, components: VoiceComponents, *, clock: Callable[[], float]) -> None:
    """Register `POST`/`GET /v1/voices` and `DELETE /v1/voices/{id}`. All three answer
    `503 loading` until the model is loaded and the voice directory scanned. `clock` times each
    encode (`encode_ms`), injected as `install_speech`'s is."""
    require_ready = components.readiness.require_ready

    @app.post("/v1/voices")
    async def create_voice(
        http_request: Request, runtime: Annotated[Any, Depends(require_ready)]
    ) -> JSONResponse:
        return await _create_voice(
            http_request, runtime, components, clock, http_request.state.request_id
        )

    @app.get("/v1/voices")
    async def list_voices(runtime: Annotated[Any, Depends(require_ready)]) -> JSONResponse:
        records = components.voices.get().registry.list_records(
            sample_rate=int(runtime.sample_rate),
            samples_per_frame=codec_samples_per_frame(runtime),
        )
        return JSONResponse(records)

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(
        voice_id: str,
        http_request: Request,
        _runtime: Annotated[Any, Depends(require_ready)],
    ) -> JSONResponse:
        return await _delete_voice(voice_id, components, http_request.state.request_id)
