"""Pure-ASGI request-body size cap (specs/003-cpp-compatible-api/research.md R6).

`BaseHTTPMiddleware` is banned (R1), so this is a plain ASGI callable wrapping the app
directly. There are two different rejection paths, because there are two different
points where an oversized body can be discovered:

- A `Content-Length` over the limit is known before the app runs at all, so nothing
  has installed `errors.install_error_handlers` yet to catch anything -- this
  middleware has to build and send the JSON envelope itself.
- Without a (trustworthy) `Content-Length` -- a chunked body, or one that lies -- the
  only way to find out is to count bytes as they arrive. That happens while some
  endpoint is `await`ing `request.form()` or `request.body()`, i.e. deep inside the
  wrapped FastAPI app's own call stack, so raising `ApiError` there reaches the
  installed exception handlers exactly the way any other in-request failure does
  (see research.md R2: a failure before the response starts is just a JSON 4xx).
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from breeze_infer.errors import ApiError
from breeze_infer.limits import MAX_BODY_BYTES

_STATUS = 413
_CODE = "payload_too_large"
_MESSAGE = "request body is too large"


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # malformed header: fall back to counting the body
    return None


async def _reject_immediately(send: Send) -> None:
    """Send the envelope directly: this runs before the wrapped app does."""
    body = json.dumps({"error": _MESSAGE, "code": _CODE}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": _STATUS,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    """Rejects HTTP request bodies larger than `limit` bytes."""

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.limit:
            await _reject_immediately(send)
            return

        seen = 0

        async def counting_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise ApiError(_STATUS, _CODE, _MESSAGE)
            return message

        await self.app(scope, counting_receive, send)
