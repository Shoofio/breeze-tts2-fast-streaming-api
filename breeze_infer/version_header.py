"""Stamps every HTTP response with the server version (FR-037a).

This is pure ASGI rather than Starlette's ``BaseHTTPMiddleware``, which is banned
project-wide: it buffers the response through an internal stream and turns a
mid-stream exception into a clean chunked terminator instead of letting it
propagate (see specs/003-cpp-compatible-api/research.md R1) -- exactly the C++
server defect (BC-17) this project fixes elsewhere. Wrapping ``send`` directly
avoids that failure mode and adds no buffering.
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

    def __init__(self, app: ASGIApp, version: str = "") -> None:
        self.app = app
        self._version = version.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_version(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((HEADER_NAME, self._version))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_version)
