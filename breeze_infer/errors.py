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


class ApiError(Exception):
    """An error whose body is the standard envelope: `{"error", "code"}`.

    A plain `Exception`, not a Starlette `HTTPException`: this codebase never uses `Form(...)`
    parameters (research.md R1 -- endpoints take `Request` and parse forms themselves), so there
    is no FastAPI body-parsing code that would rewrap an exception raised mid-`request.form()`.
    Subclassing `HTTPException` was tried and reverted: it makes `isinstance(exc, HTTPException)`
    true, which routes an `ApiError` through Starlette's status-code handler lookup
    (`status_handlers`, keyed by int, from `@app.exception_handler(<int>)`) *before* the
    class-based lookup that finds this module's own `ApiError` handler -- a handler some other
    part of the app registers for a bare status code could silently steal the response. Plain
    `Exception` avoids that, and also keeps `str`/`repr` showing the message, not
    `HTTPException`'s own `"<status>: <detail>"` form.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _envelope(message: str, code: str) -> dict[str, str]:
    return {"error": message, "code": code}


def api_error_response(exc: ApiError) -> JSONResponse:
    """Build the one response shape every `ApiError` produces.

    Shared by the `ApiError` exception handler below and `body_limit.py`'s immediate-rejection
    path (which runs before the app does, so no exception handler is installed yet to catch
    anything raised there) -- both 413s the body-limit middleware can produce must be
    byte-for-byte the same response, and building them from one place is how that's kept true
    rather than merely asserted.
    """
    return JSONResponse(_envelope(exc.message, exc.code), status_code=exc.status)


# A fixed status -> (code, message) table for bare `StarletteHTTPException`s -- i.e. ones this
# codebase never raises itself (`ApiError` is used instead): no matching route, wrong method,
# and Starlette's own multipart limits (`starlette/requests.py`, raised as
# `HTTPException(400, detail=...)` for too many files/fields, an oversized part, or a missing
# boundary). `exc.detail` is intentionally never surfaced for a mapped status: it can be
# Starlette's own internal wording, which isn't this API's documented message, and for the
# generic-phrase case (no detail given) there's nothing informative in it anyway. A status not
# in the table is one this contract doesn't otherwise produce; it falls back to a genuine
# `500 internal_error` rather than pairing the unmapped status with a guessed message.
_HTTP_EXCEPTION_RESPONSES: dict[int, tuple[str, str]] = {
    400: ("invalid_field", "could not parse the request body"),
    404: ("not_found", "not found"),
    405: ("method_not_allowed", "method not allowed"),
    413: ("payload_too_large", "request body is too large"),
}


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
        mapped = _HTTP_EXCEPTION_RESPONSES.get(exc.status_code)
        if mapped is None:
            return JSONResponse(
                _envelope("internal error", "internal_error"), status_code=500, headers=exc.headers
            )
        code, message = mapped
        return JSONResponse(
            _envelope(message, code), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return api_error_response(exc)

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
