"""Composition root: settings, then components, then the HTTP server (Constitution III).

This is the only module that reads the process environment, the clock or stdout, and the only
one that knows about signals and sockets. Everything else gets what it needs passed in.

Run with `python -m breeze_infer.api <model> [flags]`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import signal
import socket
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from starlette.types import ASGIApp

from breeze_infer import __version__
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.cors import CorsMiddleware, CorsPolicy
from breeze_infer.errors import install_error_handlers
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import TCP_USER_TIMEOUT_MS
from breeze_infer.model_loading import LoadedModel, load_model
from breeze_infer.routes_health import Readiness, install_health
from breeze_infer.runtime import get_dist_info
from breeze_infer.settings import Settings, settings_from_args
from breeze_infer.version_header import VersionHeaderMiddleware


@dataclass(frozen=True)
class Components:
    """Everything the routes (and, later, the WebSocket server) are built from."""

    settings: Settings
    events: Emitter
    gate: GpuGate
    gpu: GpuThread
    readiness: Readiness
    ws_port: Callable[[], int]


def create_app(components: Components) -> ASGIApp:
    """Build the FastAPI app and wrap it in the pure-ASGI middleware, outermost first:
    version header, CORS, body limit, app.
    """
    # No /docs, /redoc or /openapi.json: FR-001 allows exactly the contract's routes.
    app = FastAPI(title="Breeze TTS", docs_url=None, redoc_url=None, openapi_url=None)
    install_error_handlers(app, components.events)
    install_health(app, components.readiness, components.ws_port)

    policy = CorsPolicy(origins=components.settings.cors)
    inner: ASGIApp = CorsMiddleware(BodyLimitMiddleware(app), policy, app.router)
    # CORS sits outside the body limit and the app, so even its 413s and 500s carry CORS
    # headers; the version header stays outermost, so CORS's own preflight and 403 responses
    # carry X-Breeze-Version too.
    return VersionHeaderMiddleware(inner, version=__version__)


# The first SIGINT/SIGTERM stops accepting and lets open responses finish for this long; then
# uvicorn cancels them (a streaming client sees an aborted stream, R2). Long enough for a short
# reply already streaming to finish, short enough that a restart isn't held up by a long one.
GRACEFUL_SHUTDOWN_SECONDS = 10.0

# After uvicorn has stopped: how long the cancelled requests' `gen.close()` calls and the GPU
# thread get to drain. A close normally takes milliseconds, so past this the GPU is presumably
# stuck, and a hard exit beats hanging in the interpreter's join of the GPU thread.
GPU_DRAIN_SECONDS = 10.0


def bind_http_sockets(host: str, port: int) -> list[socket.socket]:
    """Bind and listen on every address `host` resolves to, before uvicorn starts, so a taken
    port fails fast and the kernel options below are in place for every accepted connection
    (research.md R5).

    Mirrors `asyncio.create_server`: one socket per address, IPv6 sockets IPv6-only so they
    don't clash with the IPv4 bind on the same port, and an address whose family the host has
    disabled (EADDRNOTAVAIL) is skipped. Any other failure closes what was bound and raises.
    """
    infos = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    sockets: list[socket.socket] = []
    try:
        # dict.fromkeys: getaddrinfo can repeat an address; keep the first, in order.
        for family, kind, proto, _, address in dict.fromkeys(infos):
            sock = socket.socket(family, kind, proto)
            sockets.append(sock)
            # Only on POSIX, where it just allows rebinding over TIME_WAIT. On Windows it would
            # let us bind a port another server is already using (asyncio makes the same choice).
            if os.name == "posix":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6"):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            # Accepted sockets inherit this: the kernel drops a connection whose peer has
            # stopped acknowledging data, which no application timeout can do (R5). Linux only.
            user_timeout = getattr(socket, "TCP_USER_TIMEOUT", None)
            if user_timeout is not None:
                sock.setsockopt(socket.IPPROTO_TCP, user_timeout, TCP_USER_TIMEOUT_MS)
            try:
                sock.bind(address)
            except OSError as exc:
                if exc.errno != errno.EADDRNOTAVAIL:
                    raise
                sockets.pop().close()  # the family is disabled here (bpo-30945)
                continue
            sock.listen()
        if not sockets:
            raise OSError(f"could not bind on any address of {host!r}")
    except BaseException:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def _address(sock: socket.socket) -> str:
    host, port = sock.getsockname()[:2]
    return f"[{host}]:{port}" if sock.family == socket.AF_INET6 else f"{host}:{port}"


class _Server(uvicorn.Server):
    """uvicorn's server without its own signal capture; `_install_signal_handlers` routes the
    signals instead.

    uvicorn 0.52.4 wraps `serve()` in `capture_signals()`, which replaces the SIGINT/SIGTERM
    handlers with `signal.signal` and, on exit, re-raises the captured signal against whatever
    handler was there before. That re-raise would kill the process before `main()` has shut the
    GPU thread down, and two handlers for one signal is confusing, so it is always off.
    """

    def capture_signals(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def request_exit(server: uvicorn.Server) -> None:
    """First signal: stop gracefully, letting open responses finish for up to
    `GRACEFUL_SHUTDOWN_SECONDS`. Second: stop now, dropping every open connection.
    """
    if server.should_exit:
        server.force_exit = True
        # uvicorn 0.52.4 still awaits `server.wait_closed()` on a forced exit, and since
        # Python 3.12.1 that waits for every open connection to close, so a streaming response
        # would hold the "forced" exit until it ended. Aborting the transports ends them now;
        # `serve()` then cancels the request tasks, which close their generations.
        for connection in list(server.server_state.connections):
            connection.transport.abort()
    server.should_exit = True


def _exit_signals() -> list[signal.Signals]:
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):  # Windows' Ctrl+Break
        signals.append(signal.SIGBREAK)
    return signals


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, server: uvicorn.Server
) -> Callable[[], None]:
    """Route SIGINT/SIGTERM (and SIGBREAK on Windows) to `request_exit`. Returns the undo.

    `loop.add_signal_handler` where the loop has it. Windows loops don't (NotImplementedError),
    and it refuses off the main thread (RuntimeError); then `signal.signal`, handing over to
    the loop thread-safely. If that is refused too (not the main thread), no handlers: whoever
    runs this loop off the main thread owns stopping it.
    """
    signals = _exit_signals()
    added: list[signal.Signals] = []
    try:
        for sig in signals:
            loop.add_signal_handler(sig, request_exit, server)
            added.append(sig)
    except (NotImplementedError, RuntimeError):
        for sig in added:
            loop.remove_signal_handler(sig)
    else:

        def remove() -> None:
            for sig in signals:
                loop.remove_signal_handler(sig)

        return remove

    def handle(_signum: int, _frame: object) -> None:
        loop.call_soon_threadsafe(request_exit, server)

    previous: dict[signal.Signals, Any] = {}
    try:
        for sig in signals:
            previous[sig] = signal.signal(sig, handle)
    except ValueError:  # "signal only works in main thread"
        pass

    def restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    return restore


async def load_in_background(
    components: Components, load: Callable[[], LoadedModel], server: uvicorn.Server
) -> bool:
    """Load the model on the GPU thread, then mark the server ready.

    Any failure, including one marking it ready or reporting it, is fatal: the server is told
    to exit (and `serve` returns non-zero) rather than answering `503 loading` forever.
    """
    try:
        loaded = await components.gpu.run(load)
        components.readiness.mark_ready(loaded.runtime)
        components.events.emit(
            "model.loaded", sample_rate=int(loaded.runtime.sample_rate), **loaded.report
        )
    except asyncio.CancelledError:
        raise  # serve() is stopping; not a load failure
    except BaseException as exc:  # noqa: BLE001 - even SystemExit from a loader ends the load
        components.events.emit(
            "model.load_failed",
            level="error",
            error=repr(exc),
            traceback="".join(traceback.format_exception(exc)),
        )
        server.should_exit = True
        return False
    return True


@dataclass(frozen=True)
class ServeOutcome:
    """How `serve()` ended. `hard_exit` means the GPU thread is still busy (a model load, or a
    close past `GPU_DRAIN_SECONDS`): a normal exit would join it and hang for as long as that
    takes, so `main()` ends the process with `os._exit(exit_code)` instead.
    """

    exit_code: int
    hard_exit: bool = False


async def serve(
    components: Components,
    app: ASGIApp,
    sockets: list[socket.socket],
    load: Callable[[], LoadedModel],
) -> ServeOutcome:
    """Serve HTTP on `sockets` while the model loads in the background, until a signal or a
    failed load stops it; then drain the GPU thread (see `_stop_gpu`).
    """
    server = _Server(
        uvicorn.Config(
            app,
            lifespan="off",
            # h11 explicitly: research R2/R3 measured the abort-without-terminator and
            # disconnect behaviour on it.
            http="h11",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
    )
    restore_signals = _install_signal_handlers(asyncio.get_running_loop(), server)
    try:
        loading = asyncio.create_task(load_in_background(components, load, server))
        components.events.emit(
            "server.started",
            host=components.settings.host,
            port=sockets[0].getsockname()[1],
            addresses=[_address(sock) for sock in sockets],
        )
        try:
            await server.serve(sockets=sockets)
        finally:
            outcome = await _stop_gpu(components, server, loading)
    finally:
        restore_signals()
    return outcome


async def _stop_gpu(
    components: Components, server: uvicorn.Server, loading: asyncio.Task[bool]
) -> ServeOutcome:
    """Wind the GPU down once uvicorn has stopped: every generation closed on the GPU thread,
    then the thread stopped. Bounded by `GPU_DRAIN_SECONDS`; past it, or with a model load
    still running (it can't be interrupted), the outcome asks `main()` for a hard exit.
    """
    load_failed = loading.done() and not loading.cancelled() and not loading.result()
    exit_code = 1 if load_failed else 0
    if not loading.done():
        loading.cancel()
        components.events.emit(
            "server.stopping", level="warning", reason="load in progress"
        )
        return ServeOutcome(exit_code, hard_exit=True)

    # Requests uvicorn left running: it doesn't cancel them on a forced exit, and cancels
    # without waiting when the graceful timeout expires. Each one's GpuSession queues its
    # gen.close() as it unwinds, and the GPU thread must still be there to run it.
    requests = list(server.server_state.tasks)
    for task in requests:
        task.cancel()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GPU_DRAIN_SECONDS
    if requests:
        await asyncio.wait(requests, timeout=GPU_DRAIN_SECONDS)
    remaining = max(0.0, deadline - loop.time())
    if not await asyncio.to_thread(components.gpu.shutdown, remaining):
        components.events.emit(
            "server.stopping", level="warning", reason="gpu drain timed out"
        )
        return ServeOutcome(exit_code, hard_exit=True)
    return ServeOutcome(exit_code)


def _cuda_device(environ: Mapping[str, str]) -> str:
    """The device for this process: `cuda:<local rank>` by `runtime.get_dist_info`'s rule, or
    `cpu` without CUDA.
    """
    if not torch.cuda.is_available():
        return "cpu"
    _, _, local_rank = get_dist_info(environ)
    return f"cuda:{local_rank}"


def _report_close_failed(events: Emitter, error: BaseException) -> None:
    """A `gen.close()` error no request saw (its caller was cancelled or gave up waiting)."""
    events.emit(
        "gpu.close_failed",
        level="error",
        error=repr(error),
        traceback="".join(traceback.format_exception(error)),
    )


def _select_no_device(_: str) -> None:
    """`set_device` stand-in without CUDA; the load then fails with a clear runtime error."""


def main(argv: Sequence[str] | None = None) -> None:
    settings = settings_from_args(argv)
    events = Emitter(sys.stdout, time.time)

    try:
        sockets = bind_http_sockets(settings.host, settings.port)
    except OSError as exc:
        events.emit(
            "server.bind_failed",
            level="error",
            host=settings.host,
            port=settings.port,
            error=str(exc),
        )
        raise SystemExit(
            f"breeze: cannot listen on {settings.host}:{settings.port}: {exc}"
        ) from None

    # The one read of the process environment (Constitution III): the device choice, and the
    # mapping the compile-cache setup exports TORCHINDUCTOR_CACHE_DIR into.
    environ = os.environ
    device = _cuda_device(environ)
    set_device = torch.cuda.set_device if device.startswith("cuda") else _select_no_device
    components = Components(
        settings=settings,
        events=events,
        gate=GpuGate(),
        gpu=GpuThread(device, set_device, partial(_report_close_failed, events)),
        readiness=Readiness(),
        ws_port=lambda: 0,  # the WebSocket server arrives in Phase 8 (T077)
    )
    app = create_app(components)
    load = partial(load_model, settings, device, environ)
    with asyncio.Runner() as runner:
        outcome = runner.run(serve(components, app, sockets, load))
        if outcome.hard_exit:
            # Inside the runner on purpose: its cleanup, and then the interpreter's exit, both
            # wait for the GPU thread, which is still busy. The only process-exit call in the
            # package, and only here at the composition root.
            sys.stdout.flush()
            os._exit(outcome.exit_code)
    if outcome.exit_code:
        raise SystemExit(outcome.exit_code)


if __name__ == "__main__":
    main()
