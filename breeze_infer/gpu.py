"""GPU gate, GPU thread and generation session (specs/003-cpp-compatible-api/research.md R14).

Concrete classes, one job each:

- `GpuGate` decides *who* may use the GPU. It lives on the event loop only. HTTP asks with
  `try_acquire()` and answers `409 busy` on failure; a WebSocket piece queues with
  `await acquire()`. Both hand back a `GpuLease`; releasing it passes the gate straight to the
  next waiter.
- `GpuThread` decides *where* GPU work runs: every CUDA call (model load, prepare, each
  `next(gen)`, each `gen.close()`) goes through one thread, so device selection and RNG state
  stay on that thread and a `close()` can never overlap a `next()`.
- `GpuSession` ties the two together for one generation, so HTTP and WebSocket share a single
  rule for the end of a generation: close the generator on the GPU thread, then release.

None imports torch: production injects `torch.cuda.set_device`, tests inject a stub.
"""

from __future__ import annotations

import asyncio
import enum
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class GpuLease:
    """Proof of holding the gate. Only the current lease can release it, and only once."""

    def __init__(self, gate: GpuGate) -> None:
        self._gate = gate

    def release(self) -> None:
        """Hand the gate on. Raises `RuntimeError` if this lease no longer holds it: a double
        release is a caller bug, and honouring it would free a gate someone else now holds."""
        self._gate._release(self)


class GpuGate:
    """A FIFO gate with direct handoff, for use on the event loop only.

    Not `asyncio.Lock`: its `locked()` reports free while a woken waiter has not yet run, so an
    HTTP `try_acquire` could steal the GPU from a queued WebSocket piece. Here a release sets
    the next waiter's lease as the owner itself, which gives the invariant: whenever there is no
    owner, no live waiter exists. So "owned" alone answers `try_acquire`, and it already covers
    "a waiter is pending".
    """

    def __init__(self) -> None:
        self._owner: GpuLease | None = None
        # Each future is resolved with a lease to hand over ownership. A cancelled one may
        # linger here until its task runs and removes it; `_release` skips it meanwhile.
        self._waiters: deque[asyncio.Future[GpuLease]] = deque()

    def try_acquire(self) -> GpuLease | None:
        """Take the gate if it is free right now, else None. Never waits."""
        if self._owner is not None:
            return None
        self._owner = GpuLease(self)
        return self._owner

    async def acquire(self, on_wait: Callable[[], None] | None = None) -> GpuLease:
        """Take the gate, waiting in FIFO order.

        `on_wait` is called synchronously just before the first suspension, and only if this
        call is going to wait. The WebSocket worker uses it to enqueue `queued` ahead of
        blocking. A callback rather than a `would_wait()` query because nothing can change
        between the check and joining the queue; with a separate query, any `await` the caller
        made in between (e.g. sending `queued`) could make the answer stale. It must not await.
        """
        lease = self.try_acquire()
        if lease is not None:
            return lease
        if on_wait is not None:
            on_wait()
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            return await waiter
        except BaseException:
            # Interrupted after the gate was already handed to us (CancelledError, or
            # GeneratorExit if the coroutine is closed): we own it but will never release it,
            # so pass it on before re-raising.
            if waiter.done() and not waiter.cancelled():
                waiter.result().release()
            raise
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    def _release(self, lease: GpuLease) -> None:
        if lease is not self._owner:
            raise RuntimeError(
                "GpuLease.release() called by a lease that doesn't hold the gate"
            )
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():  # skip waiters cancelled while queued
                self._owner = GpuLease(self)
                waiter.set_result(self._owner)
                return  # ownership moved without a free window
        self._owner = None


class Done(enum.Enum):
    """The type of `DONE`. An enum member, so `item is DONE` narrows `T | Done` to `T`."""

    DONE = enum.auto()


DONE = Done.DONE
"""What `GpuThread.step()` returns when the generator is exhausted.

`StopIteration` can't be raised through a Future (Python turns it into `RuntimeError` inside
coroutines), so exhaustion is reported as this value instead."""


def _next_or_done(gen: Generator[T, None, None]) -> T | Done:
    try:
        return next(gen)
    except StopIteration:
        return DONE


class GpuThread:
    """Runs all GPU work on one dedicated thread, awaited from the event loop.

    Cancellation: cancelling a coroutine awaiting `run()` or `step()` drops a call that is
    still queued, and a call already running finishes with its result discarded (a thread
    can't be interrupted). Neither leaves anything behind: a caller that stepped a generator
    must `close()` it anyway, whatever happened to the step. `close()` itself is the exception:
    it always runs, because a skipped close would leave the generator to be closed by the
    garbage collector on whatever thread happens to drop it.
    """

    def __init__(self, device: Any, set_device: Callable[[Any], None]) -> None:
        # max_workers=1 gives exactly one long-lived thread, and FIFO order for everything
        # submitted to it.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="breeze-gpu"
        )
        # Submitted first, so it runs before any other work. Not the executor's `initializer`:
        # a failing initializer only surfaces as `BrokenThreadPool`, hiding the real error.
        self._device_set = self._executor.submit(set_device, device)

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        # FIFO means set_device has already finished; this re-raises its error, if any.
        self._device_set.result()
        return fn(*args)

    def _submit(self, fn: Callable[..., T], *args: Any) -> Future[T]:
        return self._executor.submit(self._call, fn, *args)

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(*args)` on the GPU thread. Its exception, if any, is raised here."""
        return await asyncio.wrap_future(self._submit(fn, *args))

    async def step(self, gen: Generator[T, None, None]) -> T | Done:
        """Advance `gen` once on the GPU thread; return its next item, or `DONE`."""
        return await self.run(_next_or_done, gen)

    async def close(self, gen: Generator[Any, None, None]) -> None:
        """Close `gen` on the GPU thread, after any step already queued or running.

        The close is never skipped. If the caller is cancelled meanwhile, this still waits for
        the close to finish and only then re-raises the cancellation.
        """
        done = asyncio.wrap_future(self._submit(gen.close))
        cancelled: asyncio.CancelledError | None = None
        while not done.done():
            try:
                # Unlike awaiting `done` directly, `wait` doesn't cancel it when we are.
                await asyncio.wait([done])
            except asyncio.CancelledError as error:
                cancelled = error
        if cancelled is not None:
            # Retrieve any close error so it isn't lost as "never retrieved", but let the
            # cancellation win: timeouts and task groups depend on seeing it.
            raise cancelled from done.exception()
        done.result()

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work. Queued work always still runs, because it may include the
        `gen.close()` calls that release GPU state; `wait` only chooses whether to block the
        calling thread until it has. Blocking, so don't call it with `wait=True` on the event
        loop while GPU work may still be queued.
        """
        self._executor.shutdown(wait=wait)


class GpuSession(Generic[T]):
    """One generation on the GPU: holds the lease for the generator's whole life.

    The one rule both HTTP (streaming.py) and WebSocket (ws_server.py) need at the end of a
    generation, whatever ended it: close the generator on the GPU thread, and only then release
    the gate. Releasing earlier would let the next request start while a cancelled step is
    still running on the thread. Use `async with`, or call `aclose()` from a `finally` when the
    session outlives the function that opened it (a streaming response).
    """

    def __init__(
        self, lease: GpuLease, gpu: GpuThread, gen: Generator[T, None, None]
    ) -> None:
        self._lease = lease
        self._gpu = gpu
        self._gen = gen
        self._closed = False

    async def step(self) -> T | Done:
        """The next item, or `DONE`. Raises `RuntimeError` after `aclose()`: the gate is gone."""
        if self._closed:
            raise RuntimeError("GpuSession.step() after aclose()")
        return await self._gpu.step(self._gen)

    async def aclose(self) -> None:
        """Close the generator on the GPU thread, then release the gate. Safe to call twice.

        If the caller is cancelled, this still finishes both before re-raising (see
        `GpuThread.close`). If the close raises, the gate is released anyway.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await self._gpu.close(self._gen)
        finally:
            self._lease.release()

    async def __aenter__(self) -> GpuSession[T]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
