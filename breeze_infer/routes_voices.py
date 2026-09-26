"""`POST /v1/voices`, `GET /v1/voices` and `DELETE /v1/voices/{id}` (specs/003-cpp-compatible-api/
tasks.md T065; contracts/http-api.md; data-model.md "Voice").

`POST` runs the contract's order of checks:
1. body size (`body_limit.py`, before the route runs);
2. fields (`http_fields.parse_voice_create`);
3. a named voice whose name is taken, ignoring case (a skipped file's name included):
   `409 voice_exists`;
4. an unnamed voice already registered: `200` with the existing entry, without the gate;
5. decode and limits (`reference_audio.decode`, on a worker thread);
6. busy (`gate.try_acquire`): `409 busy`;
7. the encode, on the `GpuThread` through `audio.encode_prompt_waveform`;
8. a named voice's file is written, and the name checked again at commit, under the store's
   lock (`VoiceStore.create`): a clash is `409 voice_exists`, a write failure
   `500 voice_write_failed`.

Every call into `VoiceStore` runs on a worker thread, never the event loop: the store holds its
lock across a rename and an fsync. Each store change runs together with the registry update
that goes with it under `VoiceServices.write_lock`, so a `DELETE` can't land between a `POST`'s
file commit and its registration (leaving a registered voice with no file), nor a `POST` between
a `DELETE`'s file removal and its unregistration.

`DELETE` matches the id exactly as `GET` lists it (or a skipped file's exact stem): the store
and the registry both do. It never checks busy, and it drops the voice's cached prefix with the
DELETE's own `request_id` (`VoicePrefixCache.invalidate`, on the event loop: the cache has no
locks).

The two storage `500`s carry their own codes, so they are answered here rather than by the
catch-all handler; they still close the connection (`Connection: close`), like every `500`
(contracts/http-api.md "Errors"), and are reported as `request.failed`.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the routes' `Annotated[...]`,
# which refers to the local `components`, when the routes are defined (routes_speech.py does
# the same, for the same reason).
import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

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
from breeze_infer.synthesis import codec_samples_per_frame
from breeze_infer.voice_prefix import VoicePrefixCache
from breeze_infer.voice_registry import (
    RemovedVoice,
    VoiceEntry,
    VoiceRegistry,
    api_record,
    unnamed_id,
)
from breeze_infer.voice_store import VoiceExists, VoiceStore


@dataclass(frozen=True)
class VoiceServices:
    """What the voice routes work on, built by the startup scan (`api.open_voices`)."""

    store: VoiceStore
    registry: VoiceRegistry
    prefix_cache: VoicePrefixCache
    # Held on a worker thread across one store change and its registry update (module
    # docstring). Never on the event loop.
    write_lock: threading.Lock = field(default_factory=threading.Lock)


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


class VoiceComponents(Protocol):
    """The subset of `api.Components` these routes need, duck-typed for the same reason as
    `routes_speech.SpeechComponents` (importing `Components` here would be circular)."""

    events: Emitter
    gate: GpuGate
    gpu: GpuThread
    readiness: Readiness
    voices: VoiceSlot


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


def _commit_saved(services: VoiceServices, voice: voice_file.VoiceFile) -> VoiceEntry:
    """Worker thread: write the file (the store re-checks the name under its own lock, and its
    commit refuses a file already on disk), then register it, both under `write_lock`."""
    with services.write_lock:
        stored = services.store.create(voice)
        return services.registry.register_saved(stored)


def _remove(services: VoiceServices, voice_id: str) -> RemovedVoice | None:
    """Worker thread: remove the file if there is one, then the registry entry, both under
    `write_lock`. A store failure raises before the registry is touched, so the voice stays
    registered."""
    with services.write_lock:
        services.store.remove(voice_id)
        return services.registry.remove(voice_id)


async def _create_voice(
    http_request: Request, runtime: Any, components: VoiceComponents, request_id: str
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

    # None means busy; a poisoned gate raises GpuUnavailable (503), as for speech.
    lease = components.gate.try_acquire()
    if lease is None:
        raise ApiError(409, "busy", "busy")
    codes, encode_ms = await _encode_under_lease(
        lease, components.gpu, runtime.audio_tokenizer, decoded, registry.clock
    )

    if request.name is not None:
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
            entry = await asyncio.to_thread(_commit_saved, services, voice)
        except VoiceExists:
            raise ApiError(409, "voice_exists", "voice already exists") from None
        except OSError as error:
            return _storage_failure(
                components.events,
                request_id,
                error,
                "voice_write_failed",
                "could not write the voice file",
            )
    else:
        entry, evicted = registry.register_unnamed(
            id=voice_id,
            ref_text=request.ref_text,
            codes=codes,
            frames=int(codes.shape[0]),
            encode_ms=encode_ms,
        )
        if evicted is not None:
            services.prefix_cache.invalidate(evicted, request_id=request_id)

    components.events.emit(
        "voice.created",
        request_id=request_id,
        voice_id=entry.id,
        saved=entry.saved,
        frames=entry.frames,
        encode_ms=entry.encode_ms,
    )
    return JSONResponse(_record(entry, runtime))


async def _delete_voice(
    voice_id: str, components: VoiceComponents, request_id: str
) -> JSONResponse:
    services = components.voices.get()
    try:
        removed = await asyncio.to_thread(_remove, services, voice_id)
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
    if removed is None:
        raise ApiError(404, "unknown_voice", "unknown voice_id")
    services.prefix_cache.invalidate(voice_id, request_id=request_id)
    components.events.emit(
        "voice.deleted", request_id=request_id, voice_id=voice_id, kind=removed.kind
    )
    return JSONResponse({"deleted": voice_id, "file_kept": False})


def install_voices(app: FastAPI, components: VoiceComponents) -> None:
    """Register `POST`/`GET /v1/voices` and `DELETE /v1/voices/{id}`. All three answer
    `503 loading` until the model is loaded and the voice directory scanned."""
    require_ready = components.readiness.require_ready

    @app.post("/v1/voices")
    async def create_voice(
        http_request: Request, runtime: Annotated[Any, Depends(require_ready)]
    ) -> JSONResponse:
        return await _create_voice(
            http_request, runtime, components, http_request.state.request_id
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
