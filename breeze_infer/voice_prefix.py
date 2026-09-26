"""Bounded LRU of cached-KV reference prefixes, keyed by `(voice_id, codes_sha256)`.

data-model.md "Reference": the "prefix" variant is a `ReferencePrefix` (cached KV) plus
the stored `ref_text`, for a saved or unnamed voice with no override. This module owns
only the cache half -- deciding when to build, reuse and evict -- not the tensors
themselves or how they're built: `get_or_build` takes a `build` callable and awaits it
on a miss. It knows nothing about `torch`, `models.fast_streaming.ReferencePrefix`, the
`GpuThread` or the `GpuGate` (research.md R14); the real caller (T065/T066's
`routes_voices.py`/`synthesis.py`) wires `build` to run
`models.fast_streaming.FastBreezeStreamingRuntime.build_reference_prefix` on the
`GpuThread` while it holds the `GpuGate`, e.g. ``lambda: gpu.run(runtime
.build_reference_prefix, prefix_inputs)``. That keeps this module CPU-only and testable
with a plain fake `build`, including one that raises `torch.cuda.OutOfMemoryError` --
this module never imports `torch`, so it exists and runs the same with or without a
CUDA device.

Ported from `A:breeze_infer/voice_prefix.py` (research.md R13: "Port, re-keyed by
`(voice_id, codes_sha256)` so a delete-then-re-register can't serve a stale KV"), with
two changes beyond the re-key:
- `get_or_build` is `async` and awaits `build()`, since the real build is a `GpuThread
  .run(...)` coroutine, not a plain synchronous call.
- `build()`'s exception (including a CUDA out-of-memory error) is left to propagate
  completely unhandled, undocumented in `A:` because it had no such case: this cache
  does not itself fall back to the codes path on out-of-memory. It only guarantees
  that a failed build leaves no trace (nothing inserted, the build lock released) so a
  caller can catch the error and build the codes-path reference instead (T066).
`warm()` (`A:`'s best-effort startup warm-up of saved voices) is dropped: nothing
before T065/T066 wires voices in at all, so there is nothing to warm yet, and adding it
now would be an abstraction with no caller (Constitution/CLAUDE.md: no layer "for
future flexibility").

Threading -- read this before touching the locks:
- `get`, `get_or_build`, `invalidate`, `__len__` and `__contains__` all run on the
  event loop, called from request handlers (T065's `routes_voices.py` DELETE, and
  T066's `synthesis.py` reference resolution).
- `_lock` (`threading.Lock`, not `asyncio.Lock`) guards only `_entries`, the LRU dict
  itself, and is held for a handful of dict operations at a time -- never across
  `build()` (a GPU call) or `_on_event` (I/O in general). A `threading.Lock` is enough
  even though the caller is single-threaded (the event loop): it matches the style of
  the rest of this codebase's shared, non-async state (e.g. `VoiceRegistry`'s own
  lock) and stays correct even if a future caller reaches this cache from a worker
  thread too.
- `_build_lock` (`asyncio.Lock`) serialises the *building* half -- awaiting `build()`
  and inserting its result -- across the whole cache, one lock for every key, not one
  per key. It has to be `asyncio.Lock`, not `threading.Lock`: it is held across
  `await build()`, and a `threading.Lock` held there would block the entire event
  loop (every other request, not just a second one for this same voice) for as long
  as the GPU build takes. One lock for the whole cache, not one per key, is enough for
  the same reason `A:`'s single `threading.Lock` was: the server already serialises
  all GPU access through one `GpuGate` (research.md R14), so two builds are never
  really concurrent in production -- this lock only protects against two *cache*
  callers racing to build the same key before that gate-based serialisation would
  otherwise settle it, and it costs nothing to keep global.
- The actual GPU work happens inside the caller-supplied `build()`, on the
  `GpuThread`, while the caller holds the `GpuGate`. Neither is this module's
  concern; it only awaits whatever `build()` returns.
"""

from __future__ import annotations

import threading
from asyncio import Lock as AsyncLock
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

# `(voice_id, codes_sha256)`. A tuple, not a dataclass: it's a pure lookup key with no
# behaviour of its own, and every caller already has both parts to hand (T066:
# `SavedVoice`/`MemoryVoice`'s id, and `codes_sha256` from the file/inline encode).
VoiceKey = tuple[str, str]

# Runs the actual GPU work and returns the `ReferencePrefix` (or a stand-in with the
# same shape, in tests) for one key. Zero-arg so a caller can bind everything the real
# build needs (the runtime, the prefix inputs, the `GpuThread`) with a `lambda` or
# `functools.partial` at the call site; opaque here on purpose (see module docstring).
Build = Callable[[], Awaitable[Any]]

OnEvent = Callable[..., Any]


class VoicePrefixCache:
    """An LRU of at most `capacity` built prefixes, keyed by `(voice_id,
    codes_sha256)`.

    `capacity` has no default: nothing in data-model.md or limits.py gives this cache
    a size the way `UNNAMED_VOICE_CAP` bounds the voice registry, so inventing one here
    would be a guess this module has no basis for -- the caller (T065's composition
    root) must supply it explicitly, same as `A:`'s cache required.
    """

    def __init__(self, capacity: int, *, on_event: OnEvent) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = int(capacity)
        self._on_event = on_event
        self._entries: OrderedDict[VoiceKey, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._build_lock = AsyncLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def get(self, key: VoiceKey) -> Any | None:
        """A cache hit only; never builds. Marks `key` most-recently-used on a hit."""
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
            return hit

    async def get_or_build(self, key: VoiceKey, build: Build) -> tuple[Any, bool]:
        """Return `(prefix, warm)`; `warm` is True on a cache hit.

        On a miss, awaits `build()` and inserts its result, evicting the least
        recently used entry first if the cache is already at `capacity`. `build()`'s
        exception propagates completely unchanged -- including
        `torch.cuda.OutOfMemoryError` -- and nothing is inserted: see the module
        docstring for what that contract means for T066.
        """
        hit = self.get(key)
        if hit is not None:
            return hit, True
        async with self._build_lock:
            # Re-check: another caller may have built this key while we waited for
            # the lock.
            hit = self.get(key)
            if hit is not None:
                return hit, True
            prefix = await build()
            with self._lock:
                self._insert_locked(key, prefix)
        self._on_event("voice.prefix_built", voice_id=key[0])
        return prefix, False

    def _insert_locked(self, key: VoiceKey, prefix: Any) -> None:
        # Caller already holds `_lock`.
        evicted: list[VoiceKey] = []
        self._entries.pop(key, None)  # re-inserting an existing key must not double-count it
        while len(self._entries) >= self._capacity:
            old_key, _ = self._entries.popitem(last=False)
            evicted.append(old_key)
        self._entries[key] = prefix
        for old_key in evicted:
            self._on_event("voice.prefix_evicted", voice_id=old_key[0], reason="lru")

    def invalidate(self, voice_id: str) -> bool:
        """Drop every cached entry for `voice_id`, regardless of `codes_sha256`.

        Called on delete (data-model.md "Voice" lifecycle: both the saved and unnamed
        delete paths end "prefix cache entry dropped"). The `(voice_id, codes_sha256)`
        key already means a stale entry can never be *returned* after a
        delete-and-re-register -- the new registration gets a new key -- but this
        still frees it immediately rather than waiting for the LRU to age it out, and
        is the only way to drop it at all for a delete with no re-register.

        Returns whether anything was removed.
        """
        removed: list[VoiceKey] = []
        with self._lock:
            for key in list(self._entries):
                if key[0] == voice_id:
                    del self._entries[key]
                    removed.append(key)
        for key in removed:
            self._on_event("voice.prefix_evicted", voice_id=key[0], reason="deleted")
        return bool(removed)
