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
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
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


class GpuCloseTimeout(Exception):
    """`gen.close()` did not finish within `GPU_CLOSE_TIMEOUT_SECONDS`.

    The close is still queued or running on the GPU thread. A `GpuSession` keeps holding the
    gate until it finishes (for good, if the GPU is hung), so later requests answer `busy`
    instead of running on a GPU that is still occupied. The caller logs it.
    """


# How long a caller waits for `gen.close()`. A close normally waits behind at most one
# in-flight step and then takes milliseconds; 30 s is far past that, so past it the GPU is
# presumably hung. Waiting for ever would hold a request task, and with it shutdown, hostage.
GPU_CLOSE_TIMEOUT_SECONDS = 30.0


class GpuThread:
    """Runs all GPU work on one dedicated thread, awaited from the event loop.

    Cancellation: cancelling a coroutine awaiting `run()` or `step()` drops a call that is
    still queued, and a call already running finishes with its result discarded (a thread
    can't be interrupted). Neither leaves anything behind: a caller that stepped a generator
    must `close()` it anyway, whatever happened to the step. `close()` itself is the exception:
    it always runs, because a skipped close would leave the generator to be closed by the
    garbage collector on whatever thread happens to drop it.

    `on_close_error` receives a `gen.close()` error that no caller will see: the caller was
    cancelled, closed, or timed out before the close finished. It is called on the event loop.
    """

    def __init__(
        self,
        device: Any,
        set_device: Callable[[Any], None],
        on_close_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        # max_workers=1 gives exactly one long-lived thread, and FIFO order for everything
        # submitted to it.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="breeze-gpu"
        )
        # Submitted first, so it runs before any other work. Not the executor's `initializer`:
        # a failing initializer only surfaces as `BrokenThreadPool`, hiding the real error.
        self._device_set = self._executor.submit(set_device, device)
        self._on_close_error = on_close_error
        # `shutdown()` stops new run/step calls at once but keeps taking closes until the
        # queue is empty, so a request cancelled during shutdown still gets its close. The lock
        # makes "queue empty, stop taking closes" one step against a close being submitted.
        self._lock = threading.Lock()
        self._accepting = True  # run() and step() allowed
        self._drained = False  # nothing allowed, not even close()
        self._pending: set[Future[Any]] = set()

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        # FIFO means set_device has already finished. A fresh error each time, so callers
        # don't share (and keep extending the traceback of) one exception object.
        error = self._device_set.exception()
        if error is not None:
            raise RuntimeError("set_device failed") from error
        return fn(*args)

    def _submit(
        self, fn: Callable[..., T], *args: Any, is_close: bool = False
    ) -> Future[T]:
        with self._lock:
            if self._drained or not (self._accepting or is_close):
                raise RuntimeError("GpuThread is shut down")
            future = self._executor.submit(fn, *args)
            self._pending.add(future)
        # Outside the lock: a future that is already done runs the callback right here.
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future: Future[Any]) -> None:
        with self._lock:
            self._pending.discard(future)

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(*args)` on the GPU thread. Its exception, if any, is raised here."""
        return await asyncio.wrap_future(self._submit(self._call, fn, *args))

    async def step(self, gen: Generator[T, None, None]) -> T | Done:
        """Advance `gen` once on the GPU thread; return its next item, or `DONE`."""
        return await self.run(_next_or_done, gen)

    def submit_close(self, gen: Generator[Any, None, None]) -> asyncio.Future[None]:
        """Queue `gen.close()` on the GPU thread, after any step already queued or running.
        The returned future completes when the close has run. Call on the event loop.

        Runs even if `set_device` failed (a close touches no new device state) and even after
        `shutdown()` has started, until the thread has drained. After that, the future fails
        with `RuntimeError` instead.
        """
        try:
            future = self._submit(gen.close, is_close=True)
        except RuntimeError as error:
            future = Future()
            future.set_exception(error)
        return asyncio.wrap_future(future)

    async def wait_closed(self, closing: asyncio.Future[None]) -> None:
        """Wait for a close from `submit_close`, for at most `GPU_CLOSE_TIMEOUT_SECONDS`.

        - Cancelled meanwhile: keeps waiting, and re-raises the cancellation once the close
          has finished. A close error then goes to `on_close_error`, not to the caller: the
          cancellation must win, because timeouts and task groups depend on seeing it.
        - Interrupted by anything else (`GeneratorExit` when this coroutine is closed, which
          forbids waiting any longer): re-raises at once; the close keeps running.
        - Timed out: raises `GpuCloseTimeout`, even if also cancelled, so the caller learns
          the GPU is still busy. The close keeps running.

        When the caller stops waiting early, a later close error goes to `on_close_error`.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GPU_CLOSE_TIMEOUT_SECONDS
        cancelled: asyncio.CancelledError | None = None
        while not closing.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                closing.add_done_callback(self._report_close_error)
                raise GpuCloseTimeout(
                    f"gen.close() still running after {GPU_CLOSE_TIMEOUT_SECONDS:g} s"
                ) from cancelled
            try:
                # Unlike awaiting `closing` directly, `wait` doesn't cancel it when we are.
                await asyncio.wait([closing], timeout=remaining)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                closing.add_done_callback(self._report_close_error)
                raise
        if cancelled is not None:
            self._report_close_error(closing)
            raise cancelled
        closing.result()

    async def close(self, gen: Generator[Any, None, None]) -> None:
        """Close `gen` on the GPU thread: `submit_close` then `wait_closed`."""
        await self.wait_closed(self.submit_close(gen))

    def _report_close_error(self, closing: asyncio.Future[None]) -> None:
        # Reading the exception also marks it retrieved, so asyncio doesn't log it as lost.
        error = None if closing.cancelled() else closing.exception()
        if error is not None and self._on_close_error is not None:
            self._on_close_error(error)

    def shutdown(self, timeout: float | None = None) -> bool:
        """Stop the GPU thread once everything queued has run. Returns whether it did within
        `timeout` seconds (None: no limit).

        New `run()`/`step()` calls are refused at once. `close()` is still accepted until the
        queue is empty, so a generation cancelled during shutdown still gets closed on this
        thread. Blocking: don't call it on the event loop while GPU work may be queued. After
        a `False`, the thread is still busy and a later call can wait again.
        """
        with self._lock:
            self._accepting = False
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = {future for future in self._pending if not future.done()}
                if not pending:
                    self._drained = True
                    break
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            wait_futures(pending, timeout=remaining)
        self._executor.shutdown(wait=True)
        return True


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
        self._closing: asyncio.Future[None] | None = None

    async def step(self) -> T | Done:
        """The next item, or `DONE`. Raises `RuntimeError` after `aclose()`: the gate is gone."""
        if self._closing is not None:
            raise RuntimeError("GpuSession.step() after aclose()")
        return await self._gpu.step(self._gen)

    async def aclose(self) -> None:
        """Close the generator on the GPU thread, then release the gate.

        The release is a callback on the close itself, not a step of this coroutine, so it
        happens exactly when `gen.close()` has finished, however the caller ended: cancelled,
        closed (`GeneratorExit`) or timed out. Repeated and concurrent calls all wait for the
        same close. Raises what the close raised (the gate is released anyway), and
        `GpuCloseTimeout` if it hasn't finished in time; the gate then stays held until it does.
        Cancellation behaves as in `GpuThread.wait_closed`.
        """
        if self._closing is None:
            self._closing = self._gpu.submit_close(self._gen)
            # Added before anyone waits, so it runs before any waiter resumes: the gate is
            # already free when `aclose()` returns.
            self._closing.add_done_callback(self._release)
        await self._gpu.wait_closed(self._closing)

    def _release(self, _closing: asyncio.Future[None]) -> None:
        self._lease.release()

    async def __aenter__(self) -> GpuSession[T]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
