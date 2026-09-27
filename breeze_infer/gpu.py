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
import traceback
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
        self._handed_over = False

    def release(self) -> None:
        """Hand the gate on. Raises `RuntimeError` if this lease no longer holds it: a double
        release is a caller bug, and honouring it would free a gate someone else now holds.
        A no-op after `hand_over()`: the successor releases instead."""
        if self._handed_over:
            return
        self._gate._release(self)

    def hand_over(self) -> GpuLease:
        """Move this lease's hold on the gate to a new lease and return it, with no free
        window in between. For GPU work that outlives its holder (a cancelled caller's
        prefix build, voice_prefix.py): the work releases the successor when it finishes,
        and this lease's own `release()` becomes a no-op, so the holder's usual `finally:
        lease.release()` stays correct. Raises `RuntimeError` if this lease isn't held."""
        if not self.held:
            raise RuntimeError("GpuLease.hand_over() called by a lease that doesn't hold the gate")
        successor = GpuLease(self._gate)
        self._gate._owner = successor
        self._handed_over = True
        return successor

    @property
    def held(self) -> bool:
        """Whether this lease still holds the gate: code that must only run under the lease
        (voice_prefix.VoicePrefixCache.get_or_build) checks it."""
        return self._gate._owner is self

    def poison(self) -> None:
        """Mark the GPU unusable for the rest of the process: this lease's `gen.close()` never
        finished (`GpuCloseTimeout`), so the GPU may still be busy. See `GpuGate.poison`."""
        self._gate.poison()


class GpuUnavailable(Exception):
    """`GpuGate.acquire()` on a poisoned gate: a `gen.close()` never finished, so the GPU may
    still be busy and nothing new is allowed on it until the process restarts."""


class GpuGate:
    """A FIFO gate with direct handoff, for use on the event loop only.

    Not `asyncio.Lock`: its `locked()` reports free while a woken waiter has not yet run, so an
    HTTP `try_acquire` could steal the GPU from a queued WebSocket piece. Here a release sets
    the next waiter's lease as the owner itself, which gives the invariant: whenever there is no
    owner, no live waiter exists. So "owned" alone answers `try_acquire`, and it already covers
    "a waiter is pending".

    Poisoned (after a `GpuCloseTimeout`), it refuses everyone for good: `try_acquire` and
    `acquire` both raise `GpuUnavailable` (busy and poisoned are different failures for an
    HTTP caller -- `409` vs `503` -- so `None` alone can't mean both any more), and queued
    waiters fail with it too. Recovery is a restart: `on_poisoned` (injected; production
    marks `/health` unhealthy) is called once.
    """

    def __init__(self, on_poisoned: Callable[[], None] | None = None) -> None:
        self._on_poisoned = on_poisoned
        self._poisoned = False
        self._owner: GpuLease | None = None
        # Each future is resolved with a lease to hand over ownership. A cancelled one may
        # linger here until its task runs and removes it; `_release` skips it meanwhile.
        self._waiters: deque[asyncio.Future[GpuLease]] = deque()

    def try_acquire(self) -> GpuLease | None:
        """Take the gate if it is free right now, else `None` (busy). Raises
        `GpuUnavailable` if the gate is poisoned: that is a different failure from busy (a
        caller answers it `503 gpu_unavailable`, not `409 busy`), so `None` can't mean both.
        Never waits.
        """
        if self._poisoned:
            raise GpuUnavailable("the GPU stopped responding")
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

        Raises `GpuUnavailable` if the gate is poisoned, now or while waiting -- the "now" case
        comes straight from `try_acquire`, which raises the same way.
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
            if (
                waiter.done()
                and not waiter.cancelled()
                and waiter.exception() is None  # not failed by `poison()`
            ):
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
        if self._poisoned:
            self._owner = None  # nobody to hand it to: `poison()` failed the waiters
            return
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():  # skip waiters cancelled while queued
                self._owner = GpuLease(self)
                waiter.set_result(self._owner)
                return  # ownership moved without a free window
        self._owner = None

    def poison(self) -> None:
        """Refuse every current and future holder; idempotent. Called by `GpuSession` when a
        close times out, so no request runs on a GPU that is still busy."""
        if self._poisoned:
            return
        self._poisoned = True
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_exception(GpuUnavailable("the GPU stopped responding"))
        if self._on_poisoned is not None:
            self._on_poisoned()


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

    The close is still queued or running on the GPU thread. A `GpuSession` then poisons the
    gate (`GpuGate.poison`), so no later request runs on a GPU that may still be occupied, and
    the server reports itself unhealthy until restarted. The caller logs it.
    """


def report_close_failed(events: Any, error: BaseException, *, request_id: str | None = None) -> None:
    """Emit `gpu.close_failed` for a `gen.close()` that raised; the one reporter for it.

    Used for a close no request saw (`GpuThread`'s `on_close_error`, wired in `api.main`) and
    for a close the speech route had to swallow so its own failure reaches the client
    (`routes_speech.py`), which passes its `request_id`. A `GpuCloseTimeout` is skipped: the
    close hasn't failed, it is still running, and the gate's `on_poisoned` callback already
    reports that as `gpu.close_timeout`. `events` is duck-typed (`events.Emitter.emit`).
    """
    if isinstance(error, GpuCloseTimeout):
        return
    fields = {} if request_id is None else {"request_id": request_id}
    events.emit(
        "gpu.close_failed",
        level="error",
        error=repr(error),
        traceback="".join(traceback.format_exception(error)),
        **fields,
    )


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

    `on_close_error` receives a `gen.close()` error that a caller won't see because it was
    cancelled, closed, or timed out before the close finished. Once per close, however many
    callers gave up; on the event loop.
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
        # Counted as pending, so `shutdown(timeout)` also waits for (and honours its timeout
        # against) a set_device that is still running.
        self._pending.add(self._device_set)
        self._device_set.add_done_callback(self._forget)
        # Close futures some caller stopped waiting for: their error goes to `on_close_error`.
        self._abandoned: set[asyncio.Future[None]] = set()

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
        closing = asyncio.wrap_future(future)
        # One callback per close, so an abandoned close's error is reported exactly once.
        closing.add_done_callback(self._report_if_abandoned)
        return closing

    async def wait_closed(self, closing: asyncio.Future[None]) -> None:
        """Wait for a close from `submit_close`, for at most `GPU_CLOSE_TIMEOUT_SECONDS`.

        - Cancelled meanwhile: keeps waiting, and re-raises the cancellation once the close
          has finished. A close error then goes to `on_close_error`, not to this caller: the
          cancellation must win, because timeouts and task groups depend on seeing it.
        - Interrupted by anything else (`GeneratorExit` when this coroutine is closed, which
          forbids waiting any longer): re-raises at once; the close keeps running.
        - Timed out: raises `GpuCloseTimeout`, even if also cancelled, so the caller learns
          the GPU is still busy. The close keeps running.

        In every case but the first, a later close error goes to `on_close_error`.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GPU_CLOSE_TIMEOUT_SECONDS
        cancelled: asyncio.CancelledError | None = None
        while not closing.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                self._abandoned.add(closing)
                raise GpuCloseTimeout(
                    f"gen.close() still running after {GPU_CLOSE_TIMEOUT_SECONDS:g} s"
                ) from cancelled
            try:
                # Unlike awaiting `closing` directly, `wait` doesn't cancel it when we are.
                await asyncio.wait([closing], timeout=remaining)
            except asyncio.CancelledError as error:
                # Marked now, while the close is still running, so its done-callback (which
                # runs before this waiter resumes) knows to report the error.
                self._abandoned.add(closing)
                cancelled = error
            except BaseException:
                self._abandoned.add(closing)
                raise
        if cancelled is not None:
            raise cancelled
        closing.result()

    async def close(self, gen: Generator[Any, None, None]) -> None:
        """Close `gen` on the GPU thread: `submit_close` then `wait_closed`."""
        await self.wait_closed(self.submit_close(gen))

    def _report_if_abandoned(self, closing: asyncio.Future[None]) -> None:
        if closing not in self._abandoned:
            return
        self._abandoned.discard(closing)
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
        `GpuCloseTimeout` if it hasn't finished in time: the gate is then poisoned, since the
        GPU may still be busy. Cancellation behaves as in `GpuThread.wait_closed`.
        """
        if self._closing is None:
            self._closing = self._gpu.submit_close(self._gen)
            # Added before anyone waits, so it runs before any waiter resumes: the gate is
            # already free when `aclose()` returns.
            self._closing.add_done_callback(self._release)
        try:
            await self._gpu.wait_closed(self._closing)
        except GpuCloseTimeout:
            self._lease.poison()
            raise

    def _release(self, _closing: asyncio.Future[None]) -> None:
        self._lease.release()

    async def __aenter__(self) -> GpuSession[T]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


async def gpu_call_under_lease(gpu: GpuThread, lease: GpuLease, fn: Callable[..., T], *args: Any) -> T:
    """`gpu.run(fn, *args)` for a holder of `lease` (the speech route, a WebSocket piece).
    If the caller is cancelled while the call runs, the call keeps the gate
    (`GpuLease.hand_over`) until it has really finished, as a cancelled prefix build does
    (`VoicePrefixCache.get_or_build`); the caller's own `release()` is then a no-op."""
    call = asyncio.ensure_future(gpu.run(fn, *args))
    try:
        return await asyncio.shield(call)
    except asyncio.CancelledError:
        if not call.done() and lease.held:
            successor = lease.hand_over()

            def finished(done: asyncio.Future[Any]) -> None:
                successor.release()
                if not done.cancelled():
                    done.exception()  # retrieved: nobody is left to receive it

            call.add_done_callback(finished)
        raise
