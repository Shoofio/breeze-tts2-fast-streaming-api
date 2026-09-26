"""Byte-bounded LRU of cached-KV reference prefixes, keyed by `voice_file.prefix_key`.

data-model.md "Reference": the "prefix" variant is a `ReferencePrefix` (cached KV) plus
the stored `ref_text`, for a saved or unnamed voice with no override. This module owns
only the cache half -- when to build, reuse and evict -- not the tensors or how they're
built: `get_or_build` awaits a caller-supplied `build` on a miss. T066 wires `build` to
``lambda: gpu.run(runtime.build_reference_prefix, prefix_inputs)``. The cache reads one
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
  cached; T066 catches out-of-memory and falls back to the codes path.
- `A:`'s startup `warm()` is dropped: the spec doesn't ask for one.

Concurrency. Everything here runs on the event loop, and the cache has no locks:
- The caller of `get_or_build` must hold the `GpuGate` lease (it passes the lease, and a
  lease that no longer holds the gate is refused). Every miss therefore happens under the
  lease, so builds are already serialised by the gate.
- A build runs in a task of its own, and callers await it through `asyncio.shield`. A
  caller that is cancelled mid-build still sees `CancelledError`, but the GPU work it
  started keeps going (the `GpuThread` can't abandon it anyway), and its result is cached
  when it finishes. Until then, a miss on the same key joins that build instead of queueing
  a second one. That build may outlive the cancelled caller's lease; the `GpuThread` still
  runs it before any later GPU work, so nothing overlaps on the device.
- `invalidate(voice_id)` (on delete) bumps a per-voice generation. A build that started
  under an older generation is still returned to the caller that asked for it, but isn't
  cached, and a later miss doesn't join it: a delete during a build followed by a
  re-register of the same audio and text builds afresh.
- Events are emitted only once every change to the cache is made, so an `on_event` handler
  can read or use the cache. If it raises inside `get_or_build`, the build's result is
  already cached and the caller still gets it; the error goes to the event loop's
  exception handler, which logs it with its traceback. Telemetry must never fail a request
  (Constitution VII), and failing this one would throw away finished GPU work. In
  `invalidate`, which has nothing to lose, the error propagates once the entries are gone.
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
        self._generations: dict[str, int] = {}
        self._building: dict[VoiceKey, asyncio.Task[Any]] = {}

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: VoiceKey) -> Any | None:
        """A cache hit only; never builds. Marks `key` most-recently-used on a hit."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    async def get_or_build(self, key: VoiceKey, build: Build, *, lease: GpuLease) -> tuple[Any, bool]:
        """Return `(prefix, warm)`; `warm` is True on a cache hit.

        `lease` must hold the GPU gate (`RuntimeError` if not; nothing is built). On a
        miss, awaits `build()` -- or joins a build of the same key already running -- and
        caches the result, evicting least recently used entries until it fits. A result
        bigger than the whole budget, or one whose voice was invalidated meanwhile, is
        returned but not cached. `build()`'s exception propagates unchanged with nothing
        cached.
        """
        if not lease.held:
            raise RuntimeError("get_or_build needs a GPU lease that holds the gate")
        hit = self.get(key)
        if hit is not None:
            return hit, True
        task = self._building.get(key)
        if task is None:
            # The generation is read now, not when the task first runs, so a delete landing
            # in between still counts as "during the build".
            generation = self._generations.get(key[0], 0)
            task = asyncio.ensure_future(self._build(key, build, generation))
            self._building[key] = task
        return await asyncio.shield(task), False

    async def _build(self, key: VoiceKey, build: Build, generation: int) -> Any:
        voice_id = key[0]
        try:
            prefix = await build()
        finally:
            if self._building.get(key) is asyncio.current_task():
                del self._building[key]
        tokens = int(prefix.prefix_len)
        size = tokens * self._bytes_per_token
        events: _Events = [("voice.prefix_built", {"voice_id": voice_id, "tokens": tokens, "bytes": size})]
        if self._generations.get(voice_id, 0) != generation:
            events.append(_evicted(voice_id, "deleted"))
        elif size > self._budget_bytes:
            events.append(_evicted(voice_id, "too_large"))
        else:
            self._discard(key)
            while self._used_bytes + size > self._budget_bytes:
                oldest = next(iter(self._entries))
                self._discard(oldest)
                events.append(_evicted(oldest[0], "budget"))
            self._entries[key] = (prefix, size)
            self._used_bytes += size
        self._emit_reporting_errors(events)
        return prefix

    def invalidate(self, voice_id: str) -> bool:
        """Drop every entry for `voice_id`, and make any build of it still running not
        cache its result. Called on delete (data-model.md "Voice" lifecycle). Returns
        whether a cached entry was removed."""
        self._generations[voice_id] = self._generations.get(voice_id, 0) + 1
        for key in [key for key in self._building if key[0] == voice_id]:
            del self._building[key]
        removed = [key for key in self._entries if key[0] == voice_id]
        for key in removed:
            self._discard(key)
        for _key in removed:
            event, fields = _evicted(voice_id, "deleted")
            self._on_event(event, **fields)
        return bool(removed)

    def _discard(self, key: VoiceKey) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._used_bytes -= entry[1]

    def _emit_reporting_errors(self, events: _Events) -> None:
        for event, fields in events:
            try:
                self._on_event(event, **fields)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed; see module docstring
                asyncio.get_running_loop().call_exception_handler(
                    {"message": f"voice prefix cache: on_event({event!r}) raised", "exception": error}
                )


def _evicted(voice_id: str, reason: str) -> tuple[str, dict[str, Any]]:
    return "voice.prefix_evicted", {"voice_id": voice_id, "reason": reason}
