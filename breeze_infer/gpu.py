"""The GPU gate and the GPU thread (specs/003-cpp-compatible-api/research.md R14).

Two concrete classes, one job each:

- `GpuGate` decides *who* may use the GPU. It lives on the event loop only. HTTP asks with
  `try_acquire()` and answers `409 busy` on failure; a WebSocket piece queues with
  `await acquire()`. Release hands the gate straight to the next waiter.
- `GpuThread` decides *where* GPU work runs: every CUDA call (model load, prepare, each
  `next(gen)`, each `gen.close()`) goes through one thread, so device selection and RNG state
  stay on that thread and a `close()` can never overlap a `next()`.

Neither imports torch: production injects `torch.cuda.set_device`, tests inject a stub.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


class GpuGate:
    """A FIFO gate with direct handoff, for use on the event loop only.

    Not `asyncio.Lock`: its `locked()` reports free while a woken waiter has not yet run, so an
    HTTP `try_acquire` could steal the GPU from a queued WebSocket piece. Here `release()` keeps
    `_held` set and passes ownership to the next waiter itself, which gives the invariant:
    whenever the gate is not held, no live waiter exists. So "held" alone answers `try_acquire`,
    and it already covers "a waiter is pending".
    """

    def __init__(self) -> None:
        self._held = False
        # Each future is resolved by `release()` to hand over ownership. A cancelled one may
        # linger here until its task runs and removes it; `release()` skips it meanwhile.
        self._waiters: deque[asyncio.Future[None]] = deque()

    def try_acquire(self) -> bool:
        """Take the gate if it is free right now. Never waits."""
        if self._held:
            return False
        self._held = True
        return True

    async def acquire(self, on_wait: Callable[[], None] | None = None) -> bool:
        """Take the gate, waiting in FIFO order. Returns True if it had to wait.

        `on_wait` is called synchronously just before the first suspension, and only if this
        call is going to wait. The WebSocket worker uses it to enqueue `queued` ahead of
        blocking. A callback rather than a `would_wait()` query because nothing can change
        between the check and joining the queue; with a separate query, any `await` the caller
        made in between (e.g. sending `queued`) could make the answer stale. It must not await.
        """
        if self.try_acquire():
            return False
        if on_wait is not None:
            on_wait()
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            # Cancelled after `release()` already handed us the gate: we own it but nobody will
            # ever release it, so pass it on before re-raising.
            if waiter.done() and not waiter.cancelled():
                self.release()
            raise
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
        return True

    def release(self) -> None:
        """Hand the gate to the next live waiter, or free it if there is none.

        Raises `RuntimeError` when the gate is not held: that is a double release, a bug in
        the caller, and ignoring it would hide a second holder running on the GPU.
        """
        if not self._held:
            raise RuntimeError("GpuGate.release() called while the gate is not held")
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():  # skip waiters cancelled while queued
                waiter.set_result(None)
                return  # `_held` stays True: ownership moved without a free window
        self._held = False


class _Done:
    def __repr__(self) -> str:
        return "DONE"


DONE: Any = _Done()
"""What `GpuThread.step()` returns when the generator is exhausted.

`StopIteration` can't be raised through a Future (Python turns it into `RuntimeError` inside
coroutines), so exhaustion is reported as this value instead."""


def _next_or_done(gen: Generator[T, None, None]) -> T:
    try:
        return next(gen)
    except StopIteration:
        return DONE


class GpuThread:
    """Runs all GPU work on one dedicated thread, awaited from the event loop.

    Cancellation: cancelling a coroutine awaiting `run`/`step`/`close` stops a call that is
    still queued, but a call already running on the thread can't be interrupted and runs to
    completion (its result is dropped). A caller cancelled during `step()` must therefore
    still `close()` the generator; that close queues behind the running step, so it is safe.
    """

    def __init__(self, device: Any, set_device: Callable[[Any], None]) -> None:
        # max_workers=1 gives exactly one long-lived thread, and FIFO order for everything
        # submitted to it. The initializer runs once, on that thread, before any work. If it
        # raises, the executor is broken and every call fails with `BrokenThreadPool` (the
        # original error is logged by concurrent.futures).
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="breeze-gpu",
            initializer=set_device,
            initargs=(device,),
        )

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(*args)` on the GPU thread. Its exception, if any, is raised here."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def step(self, gen: Generator[T, None, None]) -> T:
        """Advance `gen` once on the GPU thread; return its next item, or `DONE`."""
        return await self.run(_next_or_done, gen)

    async def close(self, gen: Generator[Any, None, None]) -> None:
        """Close `gen` on the GPU thread, after any step already queued or running."""
        await self.run(gen.close)

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work. Queued work always still runs, because it may include the
        `gen.close()` calls that release GPU state; `wait` only chooses whether to block the
        calling thread until it has. Blocking, so don't call it with `wait=True` on the event
        loop while GPU work may still be queued.
        """
        self._executor.shutdown(wait=wait)
