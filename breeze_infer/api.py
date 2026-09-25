"""Composition root: settings, then components, then the HTTP server (Constitution III).

This is the only module that reads the process environment, the clock or stdout, and the only
one that knows about signals and sockets. Everything else gets what it needs passed in.

Run with `python -m breeze_infer.api <model> [flags]`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import sys
import time
import traceback
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch
import uvicorn
from fastapi import FastAPI
from starlette.types import ASGIApp

from breeze_infer import __version__
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.errors import install_error_handlers
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import TCP_USER_TIMEOUT_MS
from breeze_infer.model_loading import LoadedModel, load_model
from breeze_infer.routes_health import Readiness, install_health
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
    version header, (CORS, T028), body limit, app.
    """
    # No /docs, /redoc or /openapi.json: FR-001 allows exactly the contract's routes.
    app = FastAPI(title="Breeze TTS", docs_url=None, redoc_url=None, openapi_url=None)
    install_error_handlers(app, components.events)
    install_health(app, components.readiness, components.ws_port)

    inner: ASGIApp = BodyLimitMiddleware(app)
    # T028: CorsMiddleware wraps `inner` here, inside the version header, so that CORS's own
    # preflight and 403 responses carry X-Breeze-Version too.
    return VersionHeaderMiddleware(inner, version=__version__)


def bind_http_socket(host: str, port: int) -> socket.socket:
    """Bind and listen before uvicorn starts, so a taken port fails fast and the kernel
    options below are in place for every accepted connection (research.md R5).
    """
    family, kind, proto, _, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.socket(family, kind, proto)
    try:
        # Only on POSIX, where it just allows rebinding over TIME_WAIT. On Windows it would let
        # us bind a port another server is already using (asyncio makes the same choice).
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Accepted sockets inherit this: the kernel drops a connection whose peer has stopped
        # acknowledging data, which no application timeout can do (R5). Linux only.
        user_timeout = getattr(socket, "TCP_USER_TIMEOUT", None)
        if user_timeout is not None:
            sock.setsockopt(socket.IPPROTO_TCP, user_timeout, TCP_USER_TIMEOUT_MS)
        sock.bind(address)
        sock.listen()
    except BaseException:
        sock.close()
        raise
    return sock


class _Server(uvicorn.Server):
    """uvicorn's server, minus its own signal capture when ours is installed.

    uvicorn 0.52.4 wraps `serve()` in `capture_signals()`, which replaces the SIGINT/SIGTERM
    handlers with `signal.signal` and, on exit, re-raises the captured signal against whatever
    handler was there before. Two handlers for one signal is confusing, and the re-raise would
    kill the process before `main()` has shut the GPU thread down, so it is switched off
    whenever `_install_signal_handlers` succeeded.
    """

    uses_own_signal_handlers = False

    def capture_signals(self) -> contextlib.AbstractContextManager[None]:
        if self.uses_own_signal_handlers:
            return contextlib.nullcontext()
        return super().capture_signals()


def request_exit(server: uvicorn.Server) -> None:
    """First signal: stop gracefully, letting open responses finish. Second: stop now."""
    if server.should_exit:
        server.force_exit = True
    server.should_exit = True


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, server: _Server) -> None:
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, request_exit, server)
    except NotImplementedError:
        # Windows event loops have no add_signal_handler; uvicorn's own capture stays on.
        return
    server.uses_own_signal_handlers = True


async def load_in_background(
    components: Components, load: Callable[[], LoadedModel], server: uvicorn.Server
) -> bool:
    """Load the model on the GPU thread, then mark the server ready.

    A failed load is fatal: the server is told to exit (and `serve` returns non-zero) rather
    than answering `503 loading` forever.
    """
    try:
        loaded = await components.gpu.run(load)
    except Exception as exc:  # noqa: BLE001 - any load failure ends the process
        components.events.emit(
            "model.load_failed",
            level="error",
            error=repr(exc),
            traceback="".join(traceback.format_exception(exc)),
        )
        server.should_exit = True
        return False
    components.readiness.mark_ready(loaded.runtime)
    components.events.emit(
        "model.loaded", sample_rate=int(loaded.runtime.sample_rate), **loaded.report
    )
    return True


async def serve(
    components: Components,
    app: ASGIApp,
    sock: socket.socket,
    load: Callable[[], LoadedModel],
) -> int:
    """Serve HTTP on `sock` while the model loads in the background. Returns the exit code."""
    server = _Server(
        uvicorn.Config(
            app,
            lifespan="off",
            # h11 explicitly: research R2/R3 measured the abort-without-terminator and
            # disconnect behaviour on it.
            http="h11",
            log_config=None,
            access_log=False,
        )
    )
    _install_signal_handlers(asyncio.get_running_loop(), server)
    loading = asyncio.create_task(load_in_background(components, load, server))
    components.events.emit(
        "server.started", host=components.settings.host, port=sock.getsockname()[1]
    )
    try:
        await server.serve(sockets=[sock])
    finally:
        # Stops waiting for a load still in progress. The GPU call itself can't be interrupted:
        # the interpreter joins the GPU thread at exit, after that call and any queued
        # `gen.close()` have run. wait=False keeps the event loop free meanwhile.
        loading.cancel()
        components.gpu.shutdown(wait=False)
    load_failed = loading.done() and not loading.cancelled() and not loading.result()
    return 1 if load_failed else 0


def _cuda_device(environ: MutableMapping[str, str]) -> str:
    """The device for this process: `cuda:<LOCAL_RANK>`, falling back to RANK, then 0
    (the same rule as `runtime.get_dist_info`), or `cpu` without CUDA.
    """
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{int(environ.get('LOCAL_RANK', environ.get('RANK', '0')))}"


def _select_no_device(_: str) -> None:
    """`set_device` stand-in without CUDA; the load then fails with a clear runtime error."""


def main(argv: Sequence[str] | None = None) -> None:
    settings = settings_from_args(argv)
    events = Emitter(sys.stdout, time.time)

    try:
        sock = bind_http_socket(settings.host, settings.port)
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
        gpu=GpuThread(device, set_device),
        readiness=Readiness(),
        ws_port=lambda: 0,  # the WebSocket server arrives in Phase 8 (T077)
    )
    app = create_app(components)
    load = partial(load_model, settings, device, environ)
    exit_code = asyncio.run(serve(components, app, sock, load))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
