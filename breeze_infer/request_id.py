"""Names every HTTP request and stamps the name on its response as `X-Request-Id`.

One place for it, so every response carries the header however it was produced: a route, an
error handler, a dependency that failed before any route ran (`503 loading`/`gpu_unavailable`),
or the body-limit and CORS middleware inside this one. The id is also put in the request's
state (`request.state.request_id`), where the routes and error handlers read it for their
events, so a client's `X-Request-Id` matches the server's log.

Pure ASGI, like `version_header.py`, for the same reason: `BaseHTTPMiddleware` buffers the
response and would end an aborted stream cleanly (research.md R1). Wrapping `send` adds no
buffering and lets a mid-stream failure propagate as before.
"""

from __future__ import annotations

from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER_NAME = b"x-request-id"


class RequestIdMiddleware:
    """Gives each `http` request an id from `new_request_id` (injected: Constitution III).

    Non-`http` scopes pass through untouched. An `X-Request-Id` an inner layer already set is
    replaced, not duplicated, so the header always matches the id in the request's state.
    """

    def __init__(self, app: ASGIApp, *, new_request_id: Callable[[], str]) -> None:
        self.app = app
        self._new_request_id = new_request_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._new_request_id()
        # A new state dict rather than writing into the server's: the scope's state may be a
        # copy the server shares out per request, but nothing here depends on that.
        scope = {**scope, "state": {**scope.get("state", {}), "request_id": request_id}}
        header_value = request_id.encode("latin-1")

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != HEADER_NAME
                ]
                headers.append((HEADER_NAME, header_value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_id)
