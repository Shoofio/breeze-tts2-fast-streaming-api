"""tests/test_voice_prefix_cache.py -- breeze_infer/voice_prefix.py (T058, T064).

Pure and CPU-only: no filesystem, no GPU, no real `ReferencePrefix`. Ported from
`A:tests/test_voice_prefix_cache.py`, re-keyed by `(voice_id, codes_sha256)`
(research.md R13: "so a delete-then-re-register can't serve a stale KV") instead of
`A:`'s bare `voice_id`, and async (`get_or_build` awaits `build()`, which the real
caller wires to `await gpu.run(...)` -- research.md R14's `GpuThread`).

The cache never imports `torch` or `models.fast_streaming`: it treats whatever
`build()` returns (or raises) as opaque, so it exists and runs the same whether or
not a CUDA device -- or even a CUDA build of torch -- is present. This suite runs
under `CUDA_VISIBLE_DEVICES=` and never constructs a real `ReferencePrefix`; the one
"OOM" test constructs a real `torch.cuda.OutOfMemoryError` only to prove that class is
just an exception type, not something that needs a live GPU.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from breeze_infer import voice_prefix
from breeze_infer.voice_prefix import VoicePrefixCache


class _Prefix:
    """Stand-in for `models.fast_streaming.ReferencePrefix`: the cache never looks
    inside its values, so a plain marker object proves that (identity is enough)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"_Prefix({self.label!r})"


class _Builder:
    """A fake `build` callable: records every call, and can be told to fail (with any
    exception, e.g. `torch.cuda.OutOfMemoryError`) for specific keys."""

    def __init__(self, fail_for: dict[tuple[str, str], BaseException] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail_for = fail_for or {}

    def for_key(self, key: tuple[str, str]):
        async def build() -> _Prefix:
            self.calls.append(key)
            error = self._fail_for.get(key)
            if error is not None:
                raise error
            return _Prefix(f"{key[0]}:{key[1]}")

        return build


def _events() -> tuple[list[dict], object]:
    calls: list[dict] = []

    def on_event(event: str, **fields: object) -> None:
        calls.append({"event": event, **fields})

    return calls, on_event


def _cache(capacity: int = 2) -> tuple[VoicePrefixCache, list[dict]]:
    events, on_event = _events()
    cache = VoicePrefixCache(capacity, on_event=on_event)
    return cache, events


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        VoicePrefixCache(0, on_event=lambda *a, **k: None)


def test_module_never_imports_torch() -> None:
    """The cache is opaque to what `build()` returns/raises, so it must not need torch
    (or a CUDA device) just to exist -- unlike `models.fast_streaming`, which the
    codebase already treats as expensive/GPU-flavoured to import (see `tests/fakes.py`
    on `FakeRuntime`). Confirms the class in this module is usable with no CUDA."""
    assert not hasattr(voice_prefix, "torch")


def test_get_is_a_peek_that_never_builds() -> None:
    cache, _ = _cache()
    builder = _Builder()

    assert cache.get(("voice_a", "sha1")) is None
    assert builder.calls == []
    assert len(cache) == 0
    assert ("voice_a", "sha1") not in cache


def test_miss_builds_and_hit_reuses_without_building() -> None:
    cache, events = _cache()
    builder = _Builder()
    key = ("voice_a", "sha1")

    async def main() -> None:
        first, warm_first = await cache.get_or_build(key, builder.for_key(key))
        second, warm_second = await cache.get_or_build(key, builder.for_key(key))
        assert warm_first is False
        assert warm_second is True
        assert first is second
        assert cache.get(key) is first

    asyncio.run(main())

    assert builder.calls == [key]
    assert [e["event"] for e in events] == ["voice.prefix_built"]
    assert events[0]["voice_id"] == "voice_a"


def test_capacity_evicts_least_recently_used() -> None:
    cache, events = _cache(capacity=2)
    builder = _Builder()
    a, b, c = ("voice_a", "sha_a"), ("voice_b", "sha_b"), ("voice_c", "sha_c")

    async def main() -> None:
        await cache.get_or_build(a, builder.for_key(a))
        await cache.get_or_build(b, builder.for_key(b))
        await cache.get_or_build(a, builder.for_key(a))  # a is now most recently used
        await cache.get_or_build(c, builder.for_key(c))

    asyncio.run(main())

    assert b not in cache
    assert a in cache and c in cache
    assert len(cache) == 2
    evictions = [e for e in events if e["event"] == "voice.prefix_evicted"]
    assert evictions == [{"event": "voice.prefix_evicted", "voice_id": "voice_b", "reason": "lru"}]


def test_invalidate_removes_and_a_later_get_or_build_rebuilds() -> None:
    cache, events = _cache()
    builder = _Builder()
    key = ("voice_a", "sha1")

    async def main() -> None:
        await cache.get_or_build(key, builder.for_key(key))
        assert cache.invalidate("voice_a") is True
        assert cache.invalidate("voice_a") is False
        _, warm = await cache.get_or_build(key, builder.for_key(key))
        assert warm is False

    asyncio.run(main())

    assert builder.calls == [key, key]
    assert {"event": "voice.prefix_evicted", "voice_id": "voice_a", "reason": "deleted"} in events


def test_delete_and_re_register_never_serves_the_stale_kv() -> None:
    """The whole point of keying on `(voice_id, codes_sha256)` rather than bare
    `voice_id` (research.md R13): even with no explicit `invalidate`, a re-register
    under the same id but different content is a new key, so the old KV is never
    handed back -- `build` runs again rather than reusing the stale entry."""
    cache, _ = _cache()
    builder = _Builder()
    old_key = ("voice_a", "sha_old")
    new_key = ("voice_a", "sha_new")

    async def main() -> None:
        old_prefix, _ = await cache.get_or_build(old_key, builder.for_key(old_key))
        # No explicit invalidate() here -- simulating a caller that forgot, or a race.
        new_prefix, warm = await cache.get_or_build(new_key, builder.for_key(new_key))
        assert warm is False
        assert new_prefix is not old_prefix
        assert cache.get(old_key) is old_prefix  # still there until evicted or invalidated

    asyncio.run(main())

    assert builder.calls == [old_key, new_key]


def test_delete_then_re_register_drops_the_stale_entry_via_invalidate() -> None:
    """The real lifecycle (data-model.md "Voice" -- delete drops the prefix cache
    entry): a caller invalidates by `voice_id` on delete, before any re-register can
    even happen, so the stale key is gone rather than merely unreachable."""
    cache, _ = _cache()
    builder = _Builder()
    old_key = ("voice_a", "sha_old")
    new_key = ("voice_a", "sha_new")

    async def main() -> None:
        await cache.get_or_build(old_key, builder.for_key(old_key))
        assert cache.invalidate("voice_a") is True
        assert cache.get(old_key) is None
        await cache.get_or_build(new_key, builder.for_key(new_key))
        assert cache.get(old_key) is None
        assert len(cache) == 1

    asyncio.run(main())


def test_invalidate_only_drops_entries_for_that_voice_id() -> None:
    cache, _ = _cache(capacity=4)
    builder = _Builder()
    a, b = ("voice_a", "sha_a"), ("voice_b", "sha_b")

    async def main() -> None:
        await cache.get_or_build(a, builder.for_key(a))
        await cache.get_or_build(b, builder.for_key(b))
        assert cache.invalidate("voice_a") is True

    asyncio.run(main())

    assert a not in cache
    assert b in cache


def test_builder_exception_propagates_without_inserting() -> None:
    key = ("voice_bad", "sha1")
    cache, events = _cache()
    builder = _Builder(fail_for={key: RuntimeError("cannot build voice_bad")})

    async def main() -> None:
        with pytest.raises(RuntimeError):
            await cache.get_or_build(key, builder.for_key(key))

    asyncio.run(main())

    assert key not in cache
    assert len(cache) == 0
    assert events == []


def test_out_of_memory_propagates_unchanged_and_leaves_no_trace() -> None:
    """T064's contract for T066: `get_or_build` never catches
    `torch.cuda.OutOfMemoryError` (or wraps it) -- it propagates exactly as `build()`
    raised it, with nothing inserted and the build lock released, so the caller can
    catch it and fall back to the codes path, and a later call for the same key can
    still try again (not left permanently stuck)."""
    key = ("voice_a", "sha1")
    cache, events = _cache()
    oom = torch.cuda.OutOfMemoryError("simulated CUDA OOM")
    builder = _Builder(fail_for={key: oom})

    async def main() -> None:
        with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
            await cache.get_or_build(key, builder.for_key(key))
        assert excinfo.value is oom

        # Nothing was inserted, and a later attempt for the same key can still
        # succeed -- the failed build must not wedge the per-cache build lock.
        assert key not in cache
        assert len(cache) == 0

        ok_builder = _Builder()
        _prefix, warm = await cache.get_or_build(key, ok_builder.for_key(key))
        assert warm is False
        assert key in cache

    asyncio.run(main())

    assert events == [{"event": "voice.prefix_built", "voice_id": "voice_a"}]


def test_concurrent_misses_of_the_same_key_build_once() -> None:
    cache, _ = _cache()
    calls: list[tuple[str, str]] = []
    started = asyncio.Event()
    release = asyncio.Event()
    key = ("voice_a", "sha1")

    async def blocking_build() -> _Prefix:
        calls.append(key)
        started.set()
        await release.wait()
        return _Prefix("voice_a:sha1")

    async def main() -> tuple[tuple, tuple]:
        first = asyncio.create_task(cache.get_or_build(key, blocking_build))
        await started.wait()
        second = asyncio.create_task(cache.get_or_build(key, blocking_build))
        # Give `second` a turn to reach (and block on) the build lock before we
        # release the first build.
        await asyncio.sleep(0)
        release.set()
        return await first, await second

    result_first, result_second = asyncio.run(main())

    assert calls == [key]
    assert result_first[0] is result_second[0]
    assert sorted(warm for _, warm in [result_first, result_second]) == [False, True]
