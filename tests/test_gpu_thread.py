"""GpuThread (research.md R14), with a stub `set_device` so no GPU is needed."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Generator

import pytest

from breeze_infer.gpu import DONE, GpuThread

# Long enough never to fire on a healthy run; only there so a bug hangs a test, not the suite.
TIMEOUT = 5.0


class _DeviceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def __call__(self, device: object) -> None:
        self.calls.append((device, threading.get_ident()))


def _counting(n: int, seen: list[int]) -> Generator[int, None, None]:
    for i in range(n):
        seen.append(threading.get_ident())
        yield i


def test_all_work_runs_on_one_thread_that_set_device_ran_on_once() -> None:
    stub = _DeviceStub()
    gpu = GpuThread("cuda:1", stub)
    seen: list[int] = []

    async def main() -> None:
        for _ in range(3):
            seen.append(await gpu.run(threading.get_ident))
        gen = _counting(2, seen)
        assert await gpu.step(gen) == 0
        assert await gpu.step(gen) == 1
        await gpu.close(gen)

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()

    assert len(stub.calls) == 1
    device, set_device_thread = stub.calls[0]
    assert device == "cuda:1"
    assert set(seen) == {set_device_thread}
    assert set_device_thread != threading.get_ident()


def test_run_passes_arguments_and_returns_the_result() -> None:
    gpu = GpuThread(0, _DeviceStub())
    try:
        assert asyncio.run(gpu.run(divmod, 7, 2)) == (3, 1)
    finally:
        gpu.shutdown()


def test_step_returns_done_when_the_generator_is_exhausted() -> None:
    gpu = GpuThread(0, _DeviceStub())

    async def main() -> None:
        gen = _counting(1, [])
        assert await gpu.step(gen) == 0
        assert await gpu.step(gen) is DONE
        assert await gpu.step(gen) is DONE  # stays exhausted

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()


def test_exceptions_propagate_from_run_and_step() -> None:
    gpu = GpuThread(0, _DeviceStub())

    def boom() -> None:
        raise ValueError("from run")

    def failing() -> Generator[int, None, None]:
        yield 1
        raise KeyError("from step")

    async def main() -> None:
        with pytest.raises(ValueError, match="from run"):
            await gpu.run(boom)
        gen = failing()
        assert await gpu.step(gen) == 1
        with pytest.raises(KeyError, match="from step"):
            await gpu.step(gen)
        assert await gpu.run(lambda: "still usable") == "still usable"

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()


def _held_generator(
    started: threading.Event, proceed: threading.Event, log: list[str]
) -> Generator[int, None, None]:
    try:
        started.set()
        assert proceed.wait(TIMEOUT)
        log.append("step finished")
        yield 1
        yield 2
    finally:
        log.append("closed")


def test_close_queues_behind_an_in_flight_step() -> None:
    gpu = GpuThread(0, _DeviceStub())
    started, proceed = threading.Event(), threading.Event()
    log: list[str] = []
    gen = _held_generator(started, proceed, log)

    async def main() -> None:
        step = asyncio.create_task(gpu.step(gen))
        assert await asyncio.to_thread(started.wait, TIMEOUT)
        # Were close() not queued it would raise "generator already executing".
        close = asyncio.create_task(gpu.close(gen))
        await asyncio.sleep(0.05)
        assert not close.done()
        proceed.set()
        assert await step == 1
        await close

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()
    assert log == ["step finished", "closed"]


def test_a_cancelled_step_still_finishes_on_the_thread_and_can_then_be_closed() -> None:
    gpu = GpuThread(0, _DeviceStub())
    started, proceed = threading.Event(), threading.Event()
    log: list[str] = []
    gen = _held_generator(started, proceed, log)

    async def main() -> None:
        step = asyncio.create_task(gpu.step(gen))
        assert await asyncio.to_thread(started.wait, TIMEOUT)
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step
        assert log == []  # the awaiter gave up, but the step is still running
        close = asyncio.create_task(gpu.close(gen))
        proceed.set()
        await close

    try:
        asyncio.run(main())
    finally:
        gpu.shutdown()
    assert log == ["step finished", "closed"]


def test_shutdown_waits_for_queued_work_then_refuses_more() -> None:
    gpu = GpuThread(0, _DeviceStub())
    started, proceed = threading.Event(), threading.Event()
    log: list[str] = []

    def hold() -> None:
        started.set()
        assert proceed.wait(TIMEOUT)

    async def main() -> None:
        running = asyncio.create_task(gpu.run(hold))
        queued = asyncio.create_task(gpu.run(log.append, "queued"))
        assert await asyncio.to_thread(started.wait, TIMEOUT)

        # shutdown() blocks, so call it off the loop and watch it wait.
        stopping = asyncio.create_task(asyncio.to_thread(gpu.shutdown))
        await asyncio.sleep(0.05)
        assert not stopping.done()
        proceed.set()
        await stopping
        await running
        await queued
        assert log == ["queued"]

        with pytest.raises(RuntimeError):
            await gpu.run(log.append, "too late")

    asyncio.run(main())
