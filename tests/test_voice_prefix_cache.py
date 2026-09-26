"""tests/test_voice_prefix_cache.py -- breeze_infer/voice_prefix.py (T058, T064).

Pure and CPU-only: no filesystem, no GPU, no real `ReferencePrefix`. Ported from
`A:tests/test_voice_prefix_cache.py`, re-keyed by `voice_file.prefix_key`'s
`(voice_id, content_hash)` instead of `A:`'s bare `voice_id`, bounded by bytes instead
of a count, and async (`get_or_build` awaits `build()`, which the real caller wires to
`await gpu.run(...)`). pytest-asyncio isn't installed, so each test drives its own
event loop with `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from breeze_infer.gpu import GpuGate, GpuLease
from breeze_infer.voice_file import prefix_key
from breeze_infer.voice_prefix import (
    DELETES_REMEMBERED,
    VoicePrefixCache,
    kv_bytes_per_token,
)

REQUEST_ID = "req-1"

REPO_ROOT = Path(__file__).resolve().parents[1]

# A lost wake-up shows up as a hang; fail instead of hanging the suite.
TIMEOUT = 5.0


def _run(main: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(asyncio.wait_for(main, TIMEOUT))


async def _settle() -> None:
    # Let every ready task run until it blocks again.
    for _ in range(5):
        await asyncio.sleep(0)


class _Prefix:
    """Stand-in for `models.fast_streaming.ReferencePrefix`: the cache reads only
    `prefix_len`, and identity is enough to tell two builds apart."""

    def __init__(self, label: str, prefix_len: int = 10) -> None:
        self.label = label
        self.prefix_len = prefix_len

    def __repr__(self) -> str:
        return f"_Prefix({self.label!r}, {self.prefix_len})"


class _Builder:
    """A fake `build` callable: records every call, and can be told to fail (with any
    exception, e.g. `torch.cuda.OutOfMemoryError`) for specific keys."""

    def __init__(self, fail_for: dict[tuple[str, str], BaseException] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail_for = fail_for or {}

    def for_key(self, key: tuple[str, str], prefix_len: int = 10):
        async def build() -> _Prefix:
            self.calls.append(key)
            error = self._fail_for.get(key)
            if error is not None:
                raise error
            return _Prefix(f"{key[0]}:{key[1]}", prefix_len)

        return build


def _lease() -> GpuLease:
    lease = GpuGate().try_acquire()
    assert lease is not None
    return lease


def _get(cache: VoicePrefixCache, key, build, lease: GpuLease, *, token: int | None = None):
    """`get_or_build` as T066 calls it: `token` is what the caller read when it resolved
    the voice (by default, just now)."""
    return cache.get_or_build(
        key, build, lease=lease, resolved_token=cache.token() if token is None else token, request_id=REQUEST_ID
    )


def _cache(budget_bytes: int = 100, on_event=None) -> tuple[VoicePrefixCache, list[dict]]:
    """A cache with 1 byte per token, so a prefix's cost is just its `prefix_len`."""
    events: list[dict] = []

    def record(event: str, **fields: object) -> None:
        events.append({"event": event, **fields})

    cache = VoicePrefixCache(
        budget_bytes=budget_bytes, bytes_per_token=1, on_event=on_event or record
    )
    return cache, events


def test_kv_bytes_per_token_matches_the_checkpoint() -> None:
    # 28 layers, 8 KV heads, head_dim 128, bf16: 2 x 28 x 8 x 128 x 2.
    assert kv_bytes_per_token(num_layers=28, num_kv_heads=8, head_dim=128, dtype_bytes=2) == 114_688


@pytest.mark.parametrize(("budget", "per_token"), [(0, 1), (1, 0)])
def test_budget_and_bytes_per_token_must_be_positive(budget: int, per_token: int) -> None:
    with pytest.raises(ValueError):
        VoicePrefixCache(budget_bytes=budget, bytes_per_token=per_token, on_event=lambda *a, **k: None)


def test_module_never_imports_torch() -> None:
    """The cache is opaque to what `build()` returns or raises, so importing it must not
    pull torch in. Run in a fresh interpreter: this test process already has torch."""
    subprocess.run(
        [sys.executable, "-c", "import breeze_infer.voice_prefix, sys; assert 'torch' not in sys.modules"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_get_is_a_peek_that_never_builds() -> None:
    cache, _ = _cache()
    assert cache.get(("voice_a", "h1")) is None
    assert len(cache) == 0
    assert ("voice_a", "h1") not in cache


def test_get_or_build_refuses_a_lease_that_does_not_hold_the_gate() -> None:
    """The precondition that replaces the cache's own locks: every miss runs under the
    GPU lease, so builds are already serialised by the gate."""
    cache, _ = _cache()
    builder = _Builder()
    key = ("voice_a", "h1")
    lease = _lease()
    lease.release()

    with pytest.raises(RuntimeError, match="lease"):
        _run(_get(cache, key, builder.for_key(key), lease))
    assert builder.calls == []


def test_miss_builds_and_hit_reuses_without_building() -> None:
    cache, events = _cache()
    builder = _Builder()
    key = ("voice_a", "h1")
    lease = _lease()

    async def main() -> None:
        first, warm_first = await _get(cache, key, builder.for_key(key), lease)
        second, warm_second = await _get(cache, key, builder.for_key(key), lease)
        assert (warm_first, warm_second) == (False, True)
        assert first is second
        assert cache.get(key) is first

    _run(main())

    assert builder.calls == [key]
    assert events == [
        {"event": "voice.prefix_built", "voice_id": "voice_a", "tokens": 10, "bytes": 10, "request_id": REQUEST_ID}
    ]
    assert cache.used_bytes == 10


def test_budget_evicts_least_recently_used_until_the_new_entry_fits() -> None:
    cache, events = _cache(budget_bytes=30)
    builder = _Builder()
    a, b, c, d = ("a", "ha"), ("b", "hb"), ("c", "hc"), ("d", "hd")
    lease = _lease()

    async def main() -> None:
        for key in (a, b, c):
            await _get(cache, key, builder.for_key(key, 10), lease)
        await _get(cache, a, builder.for_key(a), lease)  # a is now most recent
        await _get(cache, d, builder.for_key(d, 15), lease)  # needs b and c gone

    _run(main())

    assert a in cache and d in cache
    assert b not in cache and c not in cache
    assert cache.used_bytes == 25
    evictions = [e for e in events if e["event"] == "voice.prefix_evicted"]
    assert evictions == [
        {"event": "voice.prefix_evicted", "voice_id": "b", "reason": "budget", "request_id": REQUEST_ID},
        {"event": "voice.prefix_evicted", "voice_id": "c", "reason": "budget", "request_id": REQUEST_ID},
    ]


def test_a_prefix_larger_than_the_whole_budget_is_returned_but_not_cached() -> None:
    cache, events = _cache(budget_bytes=30)
    builder = _Builder()
    small, huge = ("small", "h1"), ("huge", "h2")
    lease = _lease()

    async def main() -> None:
        await _get(cache, small, builder.for_key(small, 10), lease)
        prefix, warm = await _get(cache, huge, builder.for_key(huge, 31), lease)
        assert warm is False
        assert prefix.label == "huge:h2"

    _run(main())

    assert huge not in cache
    assert small in cache  # nothing evicted to make room for an entry that can't fit
    assert cache.used_bytes == 10
    assert events[-1] == {
        "event": "voice.prefix_evicted", "voice_id": "huge", "reason": "too_large", "request_id": REQUEST_ID
    }


def test_invalidate_removes_and_a_later_get_or_build_rebuilds() -> None:
    cache, events = _cache()
    builder = _Builder()
    key = ("voice_a", "h1")
    lease = _lease()

    async def main() -> None:
        await _get(cache, key, builder.for_key(key), lease)
        assert cache.invalidate("voice_a", request_id="req-delete") is True
        assert cache.invalidate("voice_a") is False
        assert cache.used_bytes == 0
        _, warm = await _get(cache, key, builder.for_key(key), lease)
        assert warm is False

    _run(main())

    assert builder.calls == [key, key]
    assert {
        "event": "voice.prefix_evicted", "voice_id": "voice_a", "reason": "deleted", "request_id": "req-delete"
    } in events


def test_invalidate_only_drops_entries_for_that_voice_id() -> None:
    cache, _ = _cache()
    builder = _Builder()
    a, b = ("voice_a", "ha"), ("voice_b", "hb")
    lease = _lease()

    async def main() -> None:
        await _get(cache, a, builder.for_key(a), lease)
        await _get(cache, b, builder.for_key(b), lease)
        assert cache.invalidate("voice_a") is True

    _run(main())

    assert a not in cache and b in cache


def test_a_delete_during_the_build_then_the_same_re_register_is_a_miss() -> None:
    """Review 39 #1: a DELETE lands while the voice's prefix is building, and the voice is
    re-registered with the same audio and text, so the same key. The build that started
    before the delete must not be cached; the re-registered voice builds afresh."""
    cache, events = _cache()
    key = ("voice_a", "h1")
    lease = _lease()
    release = asyncio.Event()
    calls: list[str] = []

    async def stale_build() -> _Prefix:
        calls.append("stale")
        await release.wait()
        return _Prefix("stale")

    async def fresh_build() -> _Prefix:
        calls.append("fresh")
        return _Prefix("fresh")

    async def main() -> None:
        first = asyncio.create_task(_get(cache, key, stale_build, lease))
        await _settle()
        cache.invalidate("voice_a")
        release.set()
        stale, _ = await first
        assert stale.label == "stale"  # the request that started it still gets it
        assert key not in cache

        fresh, warm = await _get(cache, key, fresh_build, lease)
        assert (fresh.label, warm) == ("fresh", False)

    _run(main())

    assert calls == ["stale", "fresh"]
    assert {
        "event": "voice.prefix_evicted", "voice_id": "voice_a", "reason": "deleted", "request_id": REQUEST_ID
    } in events


def test_a_stale_build_finishing_last_does_not_replace_the_fresh_entry() -> None:
    """Review 39 #1 with the other order: the re-registered voice's build finishes before
    the stale one. The stale result must not overwrite it."""
    cache, _ = _cache()
    key = ("voice_a", "h1")
    lease = _lease()
    release = asyncio.Event()
    calls: list[str] = []

    async def stale_build() -> _Prefix:
        calls.append("stale")
        await release.wait()
        return _Prefix("stale")

    async def fresh_build() -> _Prefix:
        calls.append("fresh")
        return _Prefix("fresh")

    async def main() -> None:
        first = asyncio.create_task(_get(cache, key, stale_build, lease))
        await _settle()
        cache.invalidate("voice_a")
        fresh, warm = await _get(cache, key, fresh_build, lease)
        assert (fresh.label, warm) == ("fresh", False)
        release.set()
        await first
        assert cache.get(key) is fresh  # the stale build finished last but didn't replace it

    _run(main())

    assert calls == ["stale", "fresh"]


def test_prefix_key_covers_ref_text_as_well_as_codes() -> None:
    """Review 39 #2: the prefix's KV depends on the transcript too, so the same codes with a
    different `ref_text` must be a different key."""
    codes = np.arange(12, dtype=np.int16).reshape(6, 2)

    assert prefix_key("voice_a", "hello", codes) == prefix_key("voice_a", "hello", codes.copy())
    assert prefix_key("voice_a", "hello", codes) != prefix_key("voice_a", "hello there", codes)
    assert prefix_key("voice_a", "hello", codes)[0] == "voice_a"
    # Same bytes, other shape: still a different reference.
    assert prefix_key("voice_a", "hello", codes) != prefix_key("voice_a", "hello", codes.reshape(2, 6))


def test_an_on_event_that_reads_the_cache_during_an_eviction_sees_final_state() -> None:
    """Review 39 #3: events are emitted only once every change is made, so a handler that
    reads the cache (or touches the LRU order via `get`) sees consistent state and
    can't deadlock or corrupt an eviction in progress."""
    seen: list[tuple[str, int, int, bool]] = []
    a, b = ("a", "ha"), ("b", "hb")

    def on_event(event: str, **fields: object) -> None:
        seen.append((event, len(cache), cache.used_bytes, cache.get(b) is not None))
        cache.get(a)  # a hit on an evicted key must not resurrect or reorder anything

    cache = VoicePrefixCache(budget_bytes=10, bytes_per_token=1, on_event=on_event)
    builder = _Builder()
    lease = _lease()

    async def main() -> None:
        await _get(cache, a, builder.for_key(a, 10), lease)
        await _get(cache, b, builder.for_key(b, 10), lease)

    _run(main())

    assert seen[-2:] == [("voice.prefix_built", 1, 10, True), ("voice.prefix_evicted", 1, 10, True)]
    assert a not in cache and b in cache


def test_a_raising_on_event_leaves_the_cache_consistent_and_still_returns_the_prefix() -> None:
    """Review 39 #3: a raising `on_event` in `get_or_build` can't undo the build. The cache
    is already final, the caller gets the prefix, and the error goes to the event loop's
    exception handler (logged with its traceback) rather than failing the request."""
    a, b = ("a", "ha"), ("b", "hb")
    reported: list[BaseException] = []

    def on_event(event: str, **fields: object) -> None:
        if event == "voice.prefix_evicted":
            raise RuntimeError("sink broke")

    cache = VoicePrefixCache(budget_bytes=10, bytes_per_token=1, on_event=on_event)
    builder = _Builder()
    lease = _lease()

    async def main() -> _Prefix:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: reported.append(context["exception"])
        )
        await _get(cache, a, builder.for_key(a, 10), lease)
        prefix, warm = await _get(cache, b, builder.for_key(b, 10), lease)
        assert warm is False
        return prefix

    prefix = _run(main())

    assert prefix.label == "b:hb"
    assert a not in cache and cache.get(b) is prefix
    assert (len(cache), cache.used_bytes) == (1, 10)
    assert [str(error) for error in reported] == ["sink broke"]


def test_a_raising_on_event_in_invalidate_is_reported_and_the_delete_still_completes() -> None:
    """Review 40 #5/#6: one policy for `on_event` errors. `invalidate` returns normally with
    the entries gone, the error reaches the loop's exception handler, and the cache keeps
    emitting events for the voice afterwards."""
    key = ("voice_a", "h1")
    fail = False
    events: list[str] = []
    reported: list[str] = []

    def on_event(event: str, **fields: object) -> None:
        if fail:
            raise RuntimeError("sink broke")
        events.append(event)

    cache = VoicePrefixCache(budget_bytes=100, bytes_per_token=1, on_event=on_event)
    lease = _lease()

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: reported.append(str(context["exception"]))
        )
        nonlocal fail
        await _get(cache, key, _Builder().for_key(key), lease)
        fail = True
        assert cache.invalidate("voice_a") is True
        assert key not in cache
        assert cache.used_bytes == 0
        fail = False
        await _get(cache, key, _Builder().for_key(key), lease)

    _run(main())

    assert reported == ["sink broke"]
    assert events == ["voice.prefix_built", "voice.prefix_built"]


def test_a_voice_deleted_after_its_request_resolved_it_is_not_cached() -> None:
    """Review 40 #3/#4: a request resolves voice X (reading the token), X is deleted while
    the request waits for the gate, then the request builds X's prefix. The build is
    returned to that request but never cached; a request resolving X afresh caches it."""
    cache, events = _cache()
    key = ("voice_x", "h1")
    lease = _lease()
    builder = _Builder()

    async def main() -> None:
        resolved = cache.token()
        cache.invalidate("voice_x", request_id="req-delete")
        prefix, warm = await _get(cache, key, builder.for_key(key), lease, token=resolved)
        assert (prefix.label, warm) == ("voice_x:h1", False)
        assert key not in cache

        await _get(cache, key, builder.for_key(key), lease)
        assert key in cache

    _run(main())

    assert {
        "event": "voice.prefix_evicted", "voice_id": "voice_x", "reason": "deleted", "request_id": REQUEST_ID
    } in events


def test_forgotten_deletes_never_let_a_stale_build_be_cached() -> None:
    """Only `DELETES_REMEMBERED` deletes are kept. A token older than a forgotten delete
    can't tell whether its voice was the one deleted, so its build isn't cached."""
    cache, _ = _cache()
    key = ("voice_x", "h1")
    lease = _lease()
    builder = _Builder()

    async def main() -> None:
        resolved = cache.token()
        cache.invalidate("voice_x")
        for index in range(DELETES_REMEMBERED):
            cache.invalidate(f"other_{index}")
        await _get(cache, key, builder.for_key(key), lease, token=resolved)
        assert key not in cache
        await _get(cache, key, builder.for_key(key), lease)
        assert key in cache

    _run(main())


def test_a_cancelled_caller_hands_its_lease_to_the_build_until_it_finishes() -> None:
    """Review 40 #2 (and 39 #6): a cancelled caller's build keeps running on the GPU, so
    it keeps the gate: the next holder can't start GPU work queued behind it. The build
    releases the gate exactly once when it finishes, its result is cached, and the next
    request is a warm hit with no second build."""
    gate = GpuGate()
    cache, _ = _cache()
    key = ("voice_a", "h1")
    release = asyncio.Event()
    calls: list[str] = []
    reported: list[str] = []

    async def build() -> _Prefix:
        calls.append("build")
        await release.wait()
        return _Prefix("built")

    async def request() -> tuple[_Prefix, bool]:
        # The caller's usual shape (T066): acquire, get_or_build, release in `finally`.
        lease = await gate.acquire()
        try:
            return await _get(cache, key, build, lease)
        finally:
            lease.release()

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: reported.append(context["message"])
        )
        first = asyncio.create_task(request())
        await _settle()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert gate.try_acquire() is None  # the orphaned build holds the gate

        second = asyncio.create_task(request())
        await _settle()
        assert not second.done()  # queued behind the build, not running beside it
        release.set()
        prefix, warm = await second
        assert (prefix.label, warm) == ("built", True)
        assert gate.try_acquire() is not None  # both leases released, each once

    _run(main())

    assert calls == ["build"]
    assert reported == []  # a second release would have raised in the done-callback


def test_a_build_that_fails_after_its_caller_was_cancelled_is_reported_once() -> None:
    """Review 40 #1: with no caller left to receive a build's exception, it becomes one
    `voice.prefix_build_failed` event, not asyncio's "Task exception was never
    retrieved"."""
    gate = GpuGate()
    lease = gate.try_acquire()
    assert lease is not None
    cache, events = _cache()
    key = ("voice_a", "h1")
    release = asyncio.Event()
    reported: list[str] = []

    async def build() -> _Prefix:
        await release.wait()
        raise RuntimeError("OOM")

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: reported.append(context["message"])
        )
        first = asyncio.create_task(_get(cache, key, build, lease))
        await _settle()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        await _settle()
        gc.collect()  # "never retrieved" is reported when the task is collected
        await _settle()
        assert gate.try_acquire() is not None  # the handed-over lease was released

    _run(main())

    assert events == [
        {"event": "voice.prefix_build_failed", "voice_id": "voice_a", "error": "RuntimeError: OOM",
         "request_id": REQUEST_ID}
    ]
    assert reported == []
    assert key not in cache


def test_builder_exception_propagates_without_inserting() -> None:
    key = ("voice_bad", "h1")
    cache, events = _cache()
    builder = _Builder(fail_for={key: RuntimeError("cannot build voice_bad")})

    with pytest.raises(RuntimeError, match="cannot build"):
        _run(_get(cache, key, builder.for_key(key), _lease()))

    assert key not in cache
    assert (len(cache), cache.used_bytes) == (0, 0)
    assert events == []


def test_out_of_memory_propagates_unchanged_and_leaves_no_trace() -> None:
    """The contract for T066: `get_or_build` never catches `torch.cuda.OutOfMemoryError`
    (or wraps it). It propagates exactly as `build()` raised it, with nothing inserted, so
    the caller can fall back to the codes path, and a later call can still try again.
    Constructing the exception needs no GPU."""
    key = ("voice_a", "h1")
    cache, events = _cache()
    oom = torch.cuda.OutOfMemoryError("simulated CUDA OOM")
    lease = _lease()

    async def main() -> None:
        with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
            await _get(cache, key, _Builder(fail_for={key: oom}).for_key(key), lease)
        assert excinfo.value is oom
        assert key not in cache

        _prefix, warm = await _get(cache, key, _Builder().for_key(key), lease)
        assert warm is False
        assert key in cache

    _run(main())

    assert [e["event"] for e in events] == ["voice.prefix_built"]


def test_a_build_failure_that_races_the_cancel_is_still_reported() -> None:
    """The build fails, then the caller is cancelled before it resumes to receive the
    error: the error is still reported once, not dropped."""
    cache, events = _cache()
    key = ("voice_a", "h1")
    lease = _lease()
    failed = asyncio.Event()

    async def build() -> _Prefix:
        failed.set()
        raise RuntimeError("OOM")

    async def main() -> None:
        first = asyncio.create_task(_get(cache, key, build, lease))
        await failed.wait()
        await asyncio.sleep(0)  # the build task finishes; the caller hasn't resumed yet
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    _run(main())

    assert [e["event"] for e in events] == ["voice.prefix_build_failed"]
