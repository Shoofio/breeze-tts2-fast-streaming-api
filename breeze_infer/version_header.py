"""Stamps every HTTP response with the server version (FR-037a).

This is pure ASGI rather than Starlette's ``BaseHTTPMiddleware``, which is banned
project-wide: it buffers the response through an internal stream and turns a
mid-stream exception into a clean chunked terminator instead of letting it
propagate (see specs/003-cpp-compatible-api/research.md R1) -- exactly the C++
server defect (BC-17) this project fixes elsewhere. Wrapping ``send`` directly
avoids that failure mode and adds no buffering.

This only sees exceptions that reach it as a normal ``http.response.start`` message --
i.e. ones an inner exception handler already turned into a response (see errors.py).
An exception that escapes a layer *outside* this middleware, or outside Starlette's
``ServerErrorMiddleware`` generally, before any response has started, propagates to
uvicorn as a bare connection failure: uvicorn sends its own minimal 500 with no
handler and no header. That's inherent to being outside ``send``'s reach, not a bug
here; it's why the CORS and body-limit layers (research.md R6, R8) must not raise
before sending a response themselves.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER_NAME = b"x-breeze-version"


class VersionHeaderMiddleware:
    """Adds ``X-Breeze-Version: <version>`` to every ``http.response.start``.

    Non-``http`` scopes (``lifespan``, ``websocket``) pass through untouched --
    the WebSocket handshake response carries its own version header, added
    where that handshake is built.
    """

    def __init__(self, app: ASGIApp, version: str) -> None:
        self.app = app
        self._version = version.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_version(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Drop any existing header with this name rather than appending a second one --
                # a client reading headers by name typically only sees the first match, so a
                # stale or duplicate value from an inner layer must not sit ahead of ours.
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != HEADER_NAME
                ]
                headers.append((HEADER_NAME, self._version))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_version)
