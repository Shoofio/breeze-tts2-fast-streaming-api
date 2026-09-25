"""GpuGate (research.md R14). pytest-asyncio isn't installed, so each test drives its own
event loop with `asyncio.run`."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from breeze_infer.gpu import GpuGate

# A broken handoff shows up as a waiter that never wakes; fail instead of hanging the suite.
TIMEOUT = 5.0


def _run(main: Coroutine[Any, Any, None]) -> None:
    asyncio.run(asyncio.wait_for(main, TIMEOUT))


async def _settle() -> None:
    # Let every ready task run until it blocks again. A few rounds, because waking a waiter
    # and running its continuation take separate loop iterations.
    for _ in range(5):
        await asyncio.sleep(0)


def test_try_acquire_takes_a_free_gate_and_fails_while_held() -> None:
    gate = GpuGate()
    assert gate.try_acquire()
    assert not gate.try_acquire()
    gate.release()
    assert gate.try_acquire()


def test_acquire_on_a_free_gate_does_not_wait() -> None:
    async def main() -> None:
        gate = GpuGate()
        waits: list[str] = []
        assert await gate.acquire(on_wait=lambda: waits.append("queued")) is False
        assert waits == []
        assert not gate.try_acquire()

    _run(main())


def test_acquire_on_a_held_gate_signals_queued_then_waits() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        waits: list[str] = []
        task = asyncio.create_task(gate.acquire(on_wait=lambda: waits.append("queued")))
        await _settle()
        assert waits == ["queued"]
        assert not task.done()
        gate.release()
        assert await task is True

    _run(main())


def test_release_hands_over_in_fifo_order() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        order: list[int] = []

        async def piece(n: int) -> None:
            await gate.acquire()
            order.append(n)
            await asyncio.sleep(0)  # hold across a suspension, like a real piece
            gate.release()

        tasks = [asyncio.create_task(piece(n)) for n in range(4)]
        await _settle()
        gate.release()
        await asyncio.gather(*tasks)
        assert order == [0, 1, 2, 3]
        assert gate.try_acquire()  # the last release freed it

    _run(main())


def test_http_try_acquire_fails_while_a_websocket_waiter_is_queued() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        assert not gate.try_acquire()

        # The race asyncio.Lock loses: right after release, before the woken waiter has run,
        # the gate must already belong to the waiter.
        gate.release()
        assert not waiter.done()
        assert not gate.try_acquire()

        assert await waiter is True
        assert not gate.try_acquire()
        gate.release()
        assert gate.try_acquire()

    _run(main())


def test_a_waiter_cancelled_while_queued_is_skipped() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        cancelled = asyncio.create_task(gate.acquire())
        survivor = asyncio.create_task(gate.acquire())
        await _settle()

        cancelled.cancel()
        gate.release()  # before the cancelled task has run its cleanup
        assert await survivor is True
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        gate.release()
        assert gate.try_acquire()

    _run(main())


def test_the_only_waiter_cancelled_leaves_the_gate_free_after_release() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        waiter = asyncio.create_task(gate.acquire())
        await _settle()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert not gate.try_acquire()  # still held by the original holder
        gate.release()
        assert gate.try_acquire()

    _run(main())


def test_a_waiter_cancelled_after_handoff_passes_the_gate_on() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        unlucky = asyncio.create_task(gate.acquire())
        next_in_line = asyncio.create_task(gate.acquire())
        await _settle()

        gate.release()  # hands the gate to `unlucky`...
        unlucky.cancel()  # ...which is cancelled before it gets to run
        with pytest.raises(asyncio.CancelledError):
            await unlucky
        assert await next_in_line is True

        gate.release()
        assert gate.try_acquire()

    _run(main())


def test_a_waiter_cancelled_after_handoff_frees_the_gate_if_alone() -> None:
    async def main() -> None:
        gate = GpuGate()
        gate.try_acquire()
        unlucky = asyncio.create_task(gate.acquire())
        await _settle()

        gate.release()
        unlucky.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unlucky
        assert gate.try_acquire()

    _run(main())


def test_releasing_a_free_gate_is_an_error() -> None:
    gate = GpuGate()
    with pytest.raises(RuntimeError):
        gate.release()
    gate.try_acquire()
    gate.release()
    with pytest.raises(RuntimeError):
        gate.release()
