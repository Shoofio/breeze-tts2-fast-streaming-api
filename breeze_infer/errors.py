"""Error envelope and exception handlers (specs/003-cpp-compatible-api/research.md R7).

Every error response -- however it originates, from a raised `ApiError`, a framework
`HTTPException` (no matching route, wrong method), a malformed multipart body, or an
unhandled bug -- gets the same body shape, `{"error": "<message>", "code": "<code>"}`,
with `Content-Type: application/json`. A client only ever needs one parser.

Ported from `api-alignment` (`api.py:539-599`), which had `{"error"}` alone; this adds
the `code` field and a dedicated handler for multipart parser errors, per
specs/003-cpp-compatible-api/contracts/http-api.md.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from python_multipart.exceptions import FormParserError
from starlette.exceptions import HTTPException as StarletteHTTPException

from breeze_infer.events import Emitter


class ApiError(StarletteHTTPException):
    """An error whose body is the standard envelope: `{"error", "code"}`.

    Subclasses Starlette's `HTTPException` (research.md R1) rather than plain `Exception`.
    FastAPI's own body-parsing code (`fastapi/routing.py`, around the `request.form()`/
    `request.json()` call it makes for `Form(...)`-typed parameters) catches `HTTPException`
    and re-raises it unchanged, but wraps any *other* exception into a generic
    `400 "There was an error parsing the body"`. Without this base class, an `ApiError` raised
    from `counting_receive` (body_limit.py) while FastAPI parsed a `Form(...)` route's body got
    silently rewrapped, turning a `413` into a `400`.

    This doesn't hand the response to the generic `StarletteHTTPException` handler below:
    Starlette looks up exception handlers by walking `type(exc).__mro__` and using the first
    match (`starlette/_exception_handler.py:_lookup_exception_handler`), and `ApiError` itself
    is registered first in that MRO, so the handler registered for `ApiError` still wins.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(status_code=status, detail=message)
        self.status = status
        self.code = code
        self.message = message


def _envelope(message: str, code: str) -> dict[str, str]:
    return {"error": message, "code": code}


# A fixed status -> (code, message) table for bare `StarletteHTTPException`s -- i.e. ones this
# codebase never raises itself (`ApiError` is used instead): no matching route, wrong method,
# and Starlette's own multipart limits (`starlette/requests.py`, raised as
# `HTTPException(400, detail=...)` for too many files/fields, an oversized part, or a missing
# boundary). `exc.detail` is intentionally never surfaced for a mapped status: it can be
# Starlette's own internal wording, which isn't this API's documented message, and for the
# generic-phrase case (no detail given) there's nothing informative in it anyway. Anything not
# in the table is a status this contract doesn't otherwise produce; it gets a generic 500-style
# envelope rather than guessing at a message from `exc.detail`.
_HTTP_EXCEPTION_RESPONSES: dict[int, tuple[str, str]] = {
    400: ("invalid_field", "could not parse the request body"),
    404: ("not_found", "not found"),
    405: ("method_not_allowed", "method not allowed"),
    413: ("payload_too_large", "request body is too large"),
}
_FALLBACK_CODE_AND_MESSAGE = ("internal_error", "internal error")


def install_error_handlers(app: FastAPI, events: Emitter) -> None:
    """Register every exception handler the contract needs (contracts/http-api.md)."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # No route matched, wrong method, and Starlette's own multipart limits all raise this
        # (never `ApiError`), each carrying whatever `headers` the raiser set -- e.g. a 405's
        # `Allow` header -- which must reach the client. `exc.detail` is not: see
        # `_HTTP_EXCEPTION_RESPONSES` above.
        code, message = _HTTP_EXCEPTION_RESPONSES.get(exc.status_code, _FALLBACK_CODE_AND_MESSAGE)
        return JSONResponse(
            _envelope(message, code), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        # R8: a 413 closes the connection, matching body_limit.py's immediate-rejection path
        # (the client is mid-upload; the server isn't going to read and discard the rest).
        headers = {"Connection": "close"} if exc.status == 413 else None
        return JSONResponse(
            _envelope(exc.message, exc.code), status_code=exc.status, headers=headers
        )

    @app.exception_handler(FormParserError)
    async def _form_parser_error_handler(
        _: Request, exc: FormParserError
    ) -> JSONResponse:
        # python-multipart raises this (or a subclass, e.g. MultipartParseError) for
        # a malformed body; without this handler it would fall through to the bare
        # Exception handler below and come back as an opaque 500. The parser's own
        # text describes its internals, so the client gets a fixed message.
        del exc
        return JSONResponse(
            _envelope("could not parse the request body", "invalid_field"), status_code=400
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Reachable only via routes not yet ported to raw request/form parsing (see
        # contracts/http-api.md); once ported, malformed input is coerced or rejected
        # by the fields module instead, and this handler stops firing for them.
        del exc
        return JSONResponse(_envelope("invalid request", "invalid_field"), status_code=400)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never leak str(exc) to the client -- it can carry internal paths or other
        # detail that isn't the API's contract to expose -- but keep the real error
        # in the structured event log via repr(). Starlette's ServerErrorMiddleware
        # re-raises after this response is sent, so uvicorn also prints a traceback;
        # that duplicate is intended.
        request_id = getattr(request.state, "request_id", None)
        if not request_id:
            request_id = f"api-{uuid.uuid4().hex}"
        events.emit("request.failed", level="error", request_id=request_id, error=repr(exc))
        return JSONResponse(_envelope("internal error", "internal_error"), status_code=500)
