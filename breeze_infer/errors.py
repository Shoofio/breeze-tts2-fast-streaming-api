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

import http
import uuid
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from python_multipart.exceptions import FormParserError
from starlette.exceptions import HTTPException as StarletteHTTPException


class EventEmitter(Protocol):
    """What `install_error_handlers` needs from the composition root's event sink.

    Structural typing rather than importing `breeze_infer.events` directly: the only
    contract this module relies on is "has an `emit(name, **fields)` method", and
    tests can satisfy it with a small recording fake instead of the real emitter.
    """

    def emit(self, name: str, **fields: Any) -> Any: ...


class ApiError(Exception):
    """An error whose body is the standard envelope: `{"error", "code"}`."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _envelope(message: str, code: str) -> dict[str, str]:
    return {"error": message, "code": code}


# Statuses the framework itself can produce before any route body runs (no matching
# route, wrong method) never reach `ApiError`; give them the same envelope so a client
# only ever has one error shape to parse. Any other bare `StarletteHTTPException`
# (nothing in this codebase raises one directly; `ApiError` is used instead) falls
# back to a generic code, since the contract's code catalog only documents these two.
_DEFAULT_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}
_GENERIC_HTTP_CODE = "http_error"


def install_error_handlers(app: FastAPI, events: EventEmitter) -> None:
    """Register every exception handler the contract needs (contracts/http-api.md)."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # No route matched, wrong method, and multipart parse failures all raise
        # this (never `ApiError`), each carrying whatever `headers` and `detail` the
        # raiser set -- e.g. a 405's `Allow` header -- which must reach the client.
        try:
            phrase = http.HTTPStatus(exc.status_code).phrase
        except ValueError:
            phrase = None  # a status code with no matching http.HTTPStatus member
        # Starlette's own default `detail` (when the raiser didn't pass one) is just
        # the status's phrase (e.g. "Not Found") -- that's "no specific detail", so
        # substitute our own message; anything else is surfaced to the client as-is.
        generic_detail = phrase is not None and exc.detail == phrase
        if isinstance(exc.detail, str) and not generic_detail:
            message = exc.detail
        else:
            message = phrase.lower() if phrase is not None else "error"
        code = _DEFAULT_HTTP_CODES.get(exc.status_code, _GENERIC_HTTP_CODE)
        return JSONResponse(
            _envelope(message, code), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(_envelope(exc.message, exc.code), status_code=exc.status)

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
