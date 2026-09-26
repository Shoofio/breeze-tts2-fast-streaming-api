"""Awaiting work on one of the server's own single-worker executors, and what a server stop does
to it.

`routes_speech.CpuTokenizer.run` (the pre-gate room checks) and `routes_voices.VoiceServices.change`
(every store and registry change) each run blocking work on an executor of their own, which
`api._drain_gpu` shuts down at server stop, cancelling calls still queued. It cancels every
request it already knows about first, but a request that arrived just as the drain started might
not be one of them (review #3 on 2d9070a, review 43 #2). That race is answered with
`GpuUnavailable`, the same `503 gpu_unavailable` a poisoned gate answers with, rather than a
`500` or a silent cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any, TypeVar

from breeze_infer.gpu import GpuUnavailable

_T = TypeVar("_T")

# `concurrent.futures.ThreadPoolExecutor.submit`'s message once it has been shut down.
SHUTDOWN_MESSAGE = "cannot schedule new futures after shutdown"


async def run_until_shutdown(
    executor: Executor, fn: Callable[..., _T], *args: Any, what: str
) -> _T:
    """`fn(*args)` on `executor`, awaited from the event loop; `GpuUnavailable` (naming `what`)
    if the executor has been shut down, or shuts down while the call is still queued.

    A call still queued when `shutdown()` cancels it surfaces here as an ordinary
    `asyncio.CancelledError`: by type alone, the same as this coroutine's own task being
    cancelled for a real reason (a client disconnect, or `_drain_gpu`'s own `task.cancel()` on a
    request it *did* know about). Told apart by the task's own cancel count (`Task.cancelling()`):
    `shutdown()` cancelling a queued future out from under an otherwise-uncancelled task can only
    be that race, never a real cancellation of *this* task, so only that case is remapped; a
    genuine one is re-raised unchanged.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    cancelling_before = 0 if task is None else task.cancelling()
    try:
        return await loop.run_in_executor(executor, fn, *args)
    except RuntimeError as error:
        if SHUTDOWN_MESSAGE not in str(error):
            raise
        raise GpuUnavailable(f"{what} is shutting down") from error
    except asyncio.CancelledError:
        if task is not None and task.cancelling() > cancelling_before:
            raise
        raise GpuUnavailable(f"{what} is shutting down") from None
