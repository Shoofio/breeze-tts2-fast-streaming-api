"""`GET`/`HEAD /health` and the readiness check every other route uses (FR-036, BC-24).

Until the model is loaded, `/health` and every route that depends on `Readiness.require_ready`
answer `503 {"status":"loading","error":"model is loading","code":"loading"}`. That body is not
the plain `{"error","code"}` envelope (it also carries `status`), so it has its own exception
and handler here instead of going through `ApiError`.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the route's `Annotated[...]`,
# which refers to the local `readiness`, when the route is defined.
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

_LOADING_BODY = {"status": "loading", "error": "model is loading", "code": "loading"}


class ModelLoading(Exception):
    """Raised by `Readiness.require_ready` while the model is still loading."""


class Readiness:
    """Holds the loaded runtime; `None` until the background load finishes.

    Read and written on the event loop only: the loader awaits the `GpuThread` and then calls
    `mark_ready` from the loop, so no lock is needed.
    """

    def __init__(self) -> None:
        self._runtime: Any = None

    @property
    def runtime(self) -> Any:
        return self._runtime

    def mark_ready(self, runtime: Any) -> None:
        self._runtime = runtime

    async def require_ready(self) -> Any:
        """FastAPI dependency: the loaded runtime, or the `503 loading` response."""
        if self._runtime is None:
            raise ModelLoading
        return self._runtime


async def _loading_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(dict(_LOADING_BODY), status_code=503)


def install_health(
    app: FastAPI, readiness: Readiness, ws_port: Callable[[], int]
) -> None:
    """Register `/health` and the `503 loading` handler that `require_ready` relies on.

    `ws_port` returns the WebSocket port that is actually listening, or 0 when there is none
    (BC-24); it is a callable because that is only known once the WebSocket server has bound.
    """
    app.add_exception_handler(ModelLoading, _loading_handler)

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health(
        runtime: Annotated[Any, Depends(readiness.require_ready)],
    ) -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "sample_rate": int(runtime.sample_rate), "ws_port": ws_port()}
        )
