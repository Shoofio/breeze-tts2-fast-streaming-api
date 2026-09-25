"""Stamps every HTTP response with the server version (FR-037a).

This is pure ASGI rather than Starlette's ``BaseHTTPMiddleware``, which is banned
project-wide: it buffers the response through an internal stream and turns a
mid-stream exception into a clean chunked terminator instead of letting it
propagate (see specs/003-cpp-compatible-api/research.md R1) -- exactly the C++
server defect (BC-17) this project fixes elsewhere. Wrapping ``send`` directly
avoids that failure mode and adds no buffering.

As the outermost layer, this is also the last chance to attach the version header at
all, so it doubles as a safety net: if something below it raises *before* any
``http.response.start`` has been sent (a bug in a layer outside ``errors.py``'s
handlers, e.g. body-limit or CORS middleware), this catches it, sends a plain
``500 internal_error`` envelope with the version header, and does not re-raise --
otherwise that exception would reach uvicorn as a bare connection failure with no
handler and no header at all. Once a response has started, this stops being a safety
net: a failure after that point re-raises uncaught (research.md R2 requires the
connection to close without a clean chunked terminator, not a fabricated response),
which is exactly what happens today for a mid-stream generation failure.
"""

from __future__ import annotations

import json
import re

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER_NAME = b"x-breeze-version"

# Printable ASCII, no space, no control characters (so also no CR/LF -- header injection).
_VALID_VERSION = re.compile(r"^[\x21-\x7e]+$")


class VersionHeaderMiddleware:
    """Adds ``X-Breeze-Version: <version>`` to every ``http.response.start``.

    Non-``http`` scopes (``lifespan``, ``websocket``) pass through untouched --
    the WebSocket handshake response carries its own version header, added
    where that handshake is built.
    """

    def __init__(self, app: ASGIApp, version: str) -> None:
        if not _VALID_VERSION.fullmatch(version):
            raise ValueError(f"invalid version: {version!r}")
        self.app = app
        self._version = version.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_with_version(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
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

        try:
            await self.app(scope, receive, send_with_version)
        except Exception:
            # See module docstring: this is the outermost layer and the last chance to attach
            # the header, so any exception from below is a candidate for the safety-net response
            # -- but only if nothing has been sent yet.
            if response_started:
                raise
            body = json.dumps({"error": "internal error", "code": "internal_error"}).encode(
                "utf-8"
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (HEADER_NAME, self._version),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
