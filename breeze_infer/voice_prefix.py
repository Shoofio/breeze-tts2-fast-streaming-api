"""Byte-bounded LRU of cached-KV reference prefixes, keyed by `voice_file.prefix_key`.

data-model.md "Reference": the "prefix" variant is a `ReferencePrefix` (cached KV) plus
the stored `ref_text`, for a saved or unnamed voice with no override. This module owns
only the cache half -- when to build, reuse and evict -- not the tensors or how they're
built: `get_or_build` awaits a caller-supplied `build` on a miss. The speech route binds
`build` to ``gpu.run(synthesis.build_voice_prefix, runtime, codes, ref_text)``, which calls
`runtime.build_reference_prefix` on the GPU thread (T066). The cache reads one
attribute of what `build` returns, `prefix_len`, and never imports torch, so it's testable
on the CPU with a fake `build`, including one that raises `torch.cuda.OutOfMemoryError`.

Ported from `A:breeze_infer/voice_prefix.py` (research.md R13), with these changes:
- The key is `(voice_id, content_hash)` from `voice_file.prefix_key`, a hash of the
  `ref_text` and the codes, since the KV depends on both.
- Memory is bounded by bytes (`limits.VOICE_PREFIX_CACHE_BYTES`), not a count: an entry
  costs `prefix_len * bytes_per_token`, and `kv_bytes_per_token` computes the latter from
  the model config. A prefix bigger than the whole budget is returned but not cached.
- `get_or_build` is async, since the real build is a `GpuThread.run(...)` coroutine.
- `build()`'s exceptions, CUDA out-of-memory included, propagate unchanged with nothing
  cached; T066 catches out-of-memory, drops every entry (`evict_all`) and falls back to the
  codes path.
- `A:`'s startup `warm()` is dropped: the spec doesn't ask for one.

Concurrency. Everything here runs on the event loop, and the cache has no locks:
- The caller of `get_or_build` must hold the `GpuGate` lease (it passes the lease, and a
  lease that no longer holds the gate is refused). Every miss therefore happens under the
  lease, so builds are already serialised by the gate, and no two requests can be waiting
  on builds at once.
- A build runs in a task of its own, and the caller awaits it through `asyncio.shield`. A
  caller cancelled mid-build still sees `CancelledError`, but the GPU work it started keeps
  going (the `GpuThread` can't abandon it), so the build takes over the caller's lease
  (`GpuLease.hand_over`) and releases it when it finishes, as `GpuSession` releases after
  its close: the next request's GPU work can't start queued behind work it didn't ask for.
  The caller's own `release()` of a handed-over lease is a no-op, so its usual `finally`
  stays correct. The result is cached when the build finishes; if the build raises instead,
  there is no caller left to receive the error, so it becomes a `voice.prefix_build_failed`
  event rather than asyncio's "Task exception was never retrieved".
- Deletes and builds are ordered by one monotonic counter. The caller reads `token()` in
  the same event-loop step that resolves the voice from the registry -- before it waits for
  the gate, with no `await` in between -- and passes it to `get_or_build` as
  `resolved_token`;
  `invalidate(voice_id)` (on delete) records the counter's next value for that voice. A
  build whose voice was deleted after its `resolved_token` is returned to its caller but
  not cached. So a request that resolved X, then waited for the gate while X was deleted
  (and perhaps re-registered with the same audio and text, so the same key), never caches
  the deleted voice's prefix.
- Events are emitted only once every change to the cache is made, so an `on_event` handler
  can read or use the cache. If it raises, the error goes to the event loop's exception
  handler, which logs it with its traceback, and the operation still completes: this
  module's decision, so a telemetry fault never fails a request or a delete whose work is
  already done (and, in `get_or_build`, never throws away a finished GPU build).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from breeze_infer.gpu import GpuLease
from breeze_infer.limits import VOICE_PREFIX_CACHE_BYTES

# `(voice_id, content_hash)`, always from `voice_file.prefix_key`.
VoiceKey = tuple[str, str]

# Runs the GPU work and returns the `ReferencePrefix` (or, in tests, anything with a
# `prefix_len`). Zero-arg, so the caller binds the runtime, inputs and `GpuThread` itself.
Build = Callable[[], Awaitable[Any]]

OnEvent = Callable[..., Any]

_Events = list[tuple[str, dict[str, Any]]]


def kv_bytes_per_token(*, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """GPU bytes one token of a `ReferencePrefix` holds: a key and a value per layer, each
    `num_kv_heads x head_dim` elements (the `kv` tensor is `[layers, 2, kv_heads,
    prefix_len, head_dim]`)."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


# How many deletes the cache remembers for `get_or_build`'s staleness check. Remembering
# every delete would grow without bound over the process's life. Pruning to "deletes newer
# than the oldest build in flight" isn't sound, because a request can hold a token for any
# time before it reaches `get_or_build` (it waits for the gate first) and the cache can't
# see it. So the oldest delete is forgotten instead, and any token older than a forgotten
# delete is treated as stale: its build is still returned, just not cached. A delete is a
# user action; 1,024 is far more than can land during one request's wait.
DELETES_REMEMBERED = 1024


class VoicePrefixCache:
    """An LRU of built prefixes holding at most `budget_bytes` of estimated KV."""

    def __init__(
        self,
        *,
        budget_bytes: int = VOICE_PREFIX_CACHE_BYTES,
        bytes_per_token: int,
        on_event: OnEvent,
    ) -> None:
        if budget_bytes < 1 or bytes_per_token < 1:
            raise ValueError("budget_bytes and bytes_per_token must be at least 1")
        self._budget_bytes = int(budget_bytes)
        self._bytes_per_token = int(bytes_per_token)
        self._on_event = on_event
        self._entries: OrderedDict[VoiceKey, tuple[Any, int]] = OrderedDict()  # prefix, bytes
        self._used_bytes = 0
        self._counter = 0
        self._deleted_at: OrderedDict[str, int] = OrderedDict()  # voice_id -> counter, oldest first
        self._forgotten_up_to = 0  # the newest counter value among forgotten deletes

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def token(self) -> int:
        """The current delete counter. Read it when resolving the voice, and pass it to
        `get_or_build` as `resolved_token`."""
        return self._counter

    def get(self, key: VoiceKey) -> Any | None:
        """A cache hit only; never builds. Marks `key` most-recently-used on a hit."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    async def get_or_build(
        self, key: VoiceKey, build: Build, *, lease: GpuLease, resolved_token: int, request_id: str
    ) -> tuple[Any, bool]:
        """Return `(prefix, warm)`; `warm` is True on a cache hit.

        `lease` must hold the GPU gate (`RuntimeError` if not; nothing is built).
        `resolved_token` is `token()` as read when the caller resolved the voice. On a
        miss, awaits `build()` and caches the result, evicting least recently used entries
        until it fits. A result bigger than the whole budget, or one whose voice was
        deleted after `resolved_token`, is returned but not cached. `build()`'s exception
        propagates unchanged with nothing cached. If this call is cancelled while the build
        runs, the build takes over `lease` and releases it when it finishes.
        """
        if not lease.held:
            raise RuntimeError("get_or_build needs a GPU lease that holds the gate")
        hit = self.get(key)
        if hit is not None:
            return hit, True
        task = asyncio.ensure_future(self._build(key, build, resolved_token, request_id))
        abandoned = False  # the caller was cancelled, so nobody will receive the result
        orphan_lease: GpuLease | None = None  # the caller's hold, handed to the build

        def report_failure(error: BaseException) -> None:
            self._emit([("voice.prefix_build_failed", {
                "voice_id": key[0], "error": f"{type(error).__name__}: {error}", "request_id": request_id,
            })])

        def finished(done: asyncio.Task[Any]) -> None:
            # Added before `shield` adds its own callback, so the gate is free again before
            # anyone awaiting the build resumes.
            if orphan_lease is not None:
                orphan_lease.release()
            if done.cancelled():
                return
            error = done.exception()  # always retrieved, even with nobody awaiting it
            if error is not None and abandoned:
                report_failure(error)

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task), False
        except asyncio.CancelledError:
            abandoned = True
            if not task.done():
                if lease.held:
                    orphan_lease = lease.hand_over()
            elif not task.cancelled() and task.exception() is not None:
                # The build failed just before the cancel reached this caller: `finished`
                # has already run, so report the error here instead.
                report_failure(task.exception())
            raise

    async def _build(self, key: VoiceKey, build: Build, resolved_token: int, request_id: str) -> Any:
        voice_id = key[0]
        prefix = await build()
        tokens = int(prefix.prefix_len)
        size = tokens * self._bytes_per_token
        events: _Events = [
            ("voice.prefix_built", {"voice_id": voice_id, "tokens": tokens, "bytes": size, "request_id": request_id})
        ]
        if self._deleted_since(voice_id, resolved_token):
            events.append(_evicted(voice_id, "deleted", request_id))
        elif size > self._budget_bytes:
            events.append(_evicted(voice_id, "too_large", request_id))
        else:
            self._discard(key)
            while self._used_bytes + size > self._budget_bytes:
                oldest = next(iter(self._entries))
                self._discard(oldest)
                events.append(_evicted(oldest[0], "budget", request_id))
            self._entries[key] = (prefix, size)
            self._used_bytes += size
        self._emit(events)
        return prefix

    def invalidate(self, voice_id: str, *, request_id: str | None = None) -> bool:
        """Drop every entry for `voice_id`, and make any build of it by a request that
        resolved it before now not cache its result. Called on delete (data-model.md
        "Voice" lifecycle) with the DELETE's `request_id`. Returns whether a cached entry
        was removed."""
        self._counter += 1
        self._deleted_at.pop(voice_id, None)
        self._deleted_at[voice_id] = self._counter
        if len(self._deleted_at) > DELETES_REMEMBERED:
            _forgotten, self._forgotten_up_to = self._deleted_at.popitem(last=False)
        removed = [key for key in self._entries if key[0] == voice_id]
        for key in removed:
            self._discard(key)
        self._emit([_evicted(voice_id, "deleted", request_id) for _key in removed])
        return bool(removed)

    def evict_all(self, *, reason: str, request_id: str | None) -> int:
        """Drop every entry, with one `voice.prefix_evicted` (`reason`) each, and return how many
        there were. The speech route's out-of-memory fallback (`reason="oom"`): the codes path
        needs more memory than the build that failed, so every other voice's KV goes too. Not a
        delete, so nothing here affects `get_or_build`'s staleness check."""
        evicted = list(self._entries)
        self._entries.clear()
        self._used_bytes = 0
        self._emit([_evicted(voice_id, reason, request_id) for voice_id, _hash in evicted])
        return len(evicted)

    def _deleted_since(self, voice_id: str, resolved_token: int) -> bool:
        return (
            resolved_token < self._forgotten_up_to
            or self._deleted_at.get(voice_id, 0) > resolved_token
        )

    def _discard(self, key: VoiceKey) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._used_bytes -= entry[1]

    def _emit(self, events: _Events) -> None:
        """The one place events leave the cache, always after its state is final. An
        `on_event` error is reported to the loop's exception handler, not raised: see the
        module docstring."""
        for event, fields in events:
            try:
                self._on_event(event, **fields)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                asyncio.get_running_loop().call_exception_handler(
                    {"message": f"voice prefix cache: on_event({event!r}) raised", "exception": error}
                )


def _evicted(voice_id: str, reason: str, request_id: str | None) -> tuple[str, dict[str, Any]]:
    fields: dict[str, Any] = {"voice_id": voice_id, "reason": reason}
    if request_id is not None:
        fields["request_id"] = request_id
    return "voice.prefix_evicted", fields
