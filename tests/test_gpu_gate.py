"""GpuGate (research.md R14). pytest-asyncio isn't installed, so each test drives its own
event loop with `asyncio.run`."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from breeze_infer.gpu import GpuGate, GpuLease, GpuUnavailable

# A broken handoff shows up as a waiter that never wakes; fail instead of hanging the suite.
TIMEOUT = 5.0


def _run(main: Coroutine[Any, Any, None]) -> None:
    asyncio.run(asyncio.wait_for(main, TIMEOUT))


async def _settle() -> None:
    # Let every ready task run until it blocks again. A few rounds, because waking a waiter
    # and running its continuation take separate loop iterations.
    for _ in range(5):
        await asyncio.sleep(0)


def _hold(gate: GpuGate) -> GpuLease:
    lease = gate.try_acquire()
    assert lease is not None
    return lease


def test_try_acquire_takes_a_free_gate_and_fails_while_held() -> None:
    gate = GpuGate()
    lease = _hold(gate)
    assert gate.try_acquire() is None
    lease.release()
    assert gate.try_acquire() is not None


def test_acquire_on_a_free_gate_does_not_signal_queued() -> None:
    async def main() -> None:
        gate = GpuGate()
        waits: list[str] = []
        lease = await gate.acquire(on_wait=lambda: waits.append("queued"))
        assert waits == []
        assert gate.try_acquire() is None
        lease.release()

    _run(main())


def test_acquire_on_a_held_gate_signals_queued_then_waits() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        waits: list[str] = []
        task = asyncio.create_task(gate.acquire(on_wait=lambda: waits.append("queued")))
        await _settle()
        assert waits == ["queued"]
        assert not task.done()
        holder.release()
        (await task).release()

    _run(main())


def test_release_hands_over_in_fifo_order() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        order: list[int] = []

        async def piece(n: int) -> None:
            lease = await gate.acquire()
            order.append(n)
            await asyncio.sleep(0)  # hold across a suspension, like a real piece
            lease.release()

        tasks = [asyncio.create_task(piece(n)) for n in range(4)]
        await _settle()
        holder.release()
        await asyncio.gather(*tasks)
        assert order == [0, 1, 2, 3]
        assert gate.try_acquire() is not None  # the last release freed it

    _run(main())


def test_http_try_acquire_fails_while_a_websocket_waiter_is_queued() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        assert gate.try_acquire() is None

        # The race asyncio.Lock loses: right after release, before the woken waiter has run,
        # the gate must already belong to the waiter.
        holder.release()
        assert not waiter.done()
        assert gate.try_acquire() is None

        lease = await waiter
        assert gate.try_acquire() is None
        lease.release()
        assert gate.try_acquire() is not None

    _run(main())


def test_a_waiter_cancelled_while_queued_is_skipped() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        cancelled = asyncio.create_task(gate.acquire())
        survivor = asyncio.create_task(gate.acquire())
        await _settle()

        cancelled.cancel()
        holder.release()  # before the cancelled task has run its cleanup
        lease = await survivor
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        lease.release()
        assert gate.try_acquire() is not None

    _run(main())


def test_the_only_waiter_cancelled_leaves_the_gate_free_after_release() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert gate.try_acquire() is None  # still held by the original holder
        holder.release()
        assert gate.try_acquire() is not None

    _run(main())


def test_a_waiter_cancelled_after_handoff_passes_the_gate_on() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        unlucky = asyncio.create_task(gate.acquire())
        next_in_line = asyncio.create_task(gate.acquire())
        await _settle()

        holder.release()  # hands the gate to `unlucky`...
        unlucky.cancel()  # ...which is cancelled before it gets to run
        with pytest.raises(asyncio.CancelledError):
            await unlucky
        (await next_in_line).release()
        assert gate.try_acquire() is not None

    _run(main())


def test_a_waiter_cancelled_after_handoff_frees_the_gate_if_alone() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        unlucky = asyncio.create_task(gate.acquire())
        await _settle()

        holder.release()
        unlucky.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unlucky
        assert gate.try_acquire() is not None

    _run(main())


def test_a_waiter_closed_after_handoff_passes_the_gate_on() -> None:
    # GeneratorExit rather than CancelledError: what an abandoned coroutine gets on close().
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        abandoned = gate.acquire()
        abandoned.send(None)  # runs up to the wait, now queued
        next_in_line = asyncio.create_task(gate.acquire())
        await _settle()

        holder.release()  # hands the gate to `abandoned`...
        abandoned.close()  # ...which is thrown GeneratorExit instead of resuming
        (await next_in_line).release()
        assert gate.try_acquire() is not None

    _run(main())


def test_a_second_release_of_the_same_lease_is_rejected() -> None:
    gate = GpuGate()
    lease = _hold(gate)
    lease.release()
    with pytest.raises(RuntimeError):
        lease.release()
    assert gate.try_acquire() is not None


def test_a_stale_lease_cannot_release_the_current_holder() -> None:
    async def main() -> None:
        gate = GpuGate()
        first = _hold(gate)
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        first.release()
        second = await waiter

        with pytest.raises(RuntimeError):
            first.release()  # a late double release must not free `second`'s gate
        assert gate.try_acquire() is None
        second.release()

    _run(main())


# --- poisoned: a gen.close() never finished, so the GPU can't be trusted again -------------


def test_try_acquire_raises_gpu_unavailable_once_poisoned_even_if_free() -> None:
    """Busy (`None`) and poisoned (`GpuUnavailable`) are different failures for an HTTP
    caller (`409` vs `503`), so a poisoned-but-unowned gate must not look "free" to
    `try_acquire` just because nobody holds it."""
    gate = GpuGate()
    lease = _hold(gate)
    lease.poison()
    lease.release()

    with pytest.raises(GpuUnavailable):
        gate.try_acquire()


def test_a_poisoned_gate_refuses_new_holders_and_fails_queued_waiters() -> None:
    async def main() -> None:
        poisoned: list[str] = []
        gate = GpuGate(on_poisoned=lambda: poisoned.append("poisoned"))
        holder = _hold(gate)
        waiters = [asyncio.create_task(gate.acquire()) for _ in range(2)]
        await _settle()

        holder.poison()
        holder.poison()  # idempotent: reported once
        for waiter in waiters:
            with pytest.raises(GpuUnavailable):
                await waiter
        # Poisoned is a different failure from busy (503 gpu_unavailable, not 409 busy), so
        # try_acquire raises here instead of returning None.
        with pytest.raises(GpuUnavailable):
            gate.try_acquire()
        with pytest.raises(GpuUnavailable):
            await gate.acquire()
        assert poisoned == ["poisoned"]

        holder.release()  # the stuck close finishing later doesn't reopen the gate
        with pytest.raises(GpuUnavailable):
            gate.try_acquire()
        with pytest.raises(GpuUnavailable):
            await gate.acquire()

    _run(main())


def test_a_waiter_failed_by_poisoning_then_cancelled_does_not_raise_from_cleanup() -> None:
    async def main() -> None:
        gate = GpuGate()
        holder = _hold(gate)
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        holder.poison()
        waiter.cancel()  # races the failure; either outcome, but never a stray error
        with pytest.raises((GpuUnavailable, asyncio.CancelledError)):
            await waiter

    _run(main())
