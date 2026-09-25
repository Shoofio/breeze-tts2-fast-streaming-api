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
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from python_multipart.exceptions import FormParserError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Match, Router
from starlette.types import Scope

from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuUnavailable


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


class StreamAborted(Exception):
    """A streamed response ended abnormally after its headers were sent (streaming.py).

    Raised only so uvicorn drops the connection without the chunked terminator (research.md
    R2); the original error is its `__cause__`. The response has already emitted its outcome
    event, and the headers are out, so no error response can be sent either: the catch-all
    handler below must neither answer nor report it a second time.
    """


def _envelope(message: str, code: str) -> dict[str, str]:
    return {"error": message, "code": code}


def _request_id_of(request: Request | None) -> str | None:
    """`request.state.request_id`, set by `request_id.RequestIdMiddleware` (which also
    stamps it on every response as `X-Request-Id`, so no handler here adds that header), or
    `None` when nothing set it: an app built without that middleware, or no `request` at
    all (`test_api_errors.py` calls a couple of these handlers directly with
    `request=None`).
    """
    if request is None:
        return None
    return getattr(request.state, "request_id", None)


def api_error_response(
    exc: ApiError, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Build the one response shape every `ApiError` produces.

    Shared by the `ApiError` exception handler below and `body_limit.py`'s immediate-rejection
    path (which runs before the app does, so no exception handler is installed yet to catch
    anything raised there) -- both 413s the body-limit middleware can produce must be
    byte-for-byte the same response, and building them from one place is how that's kept true
    rather than merely asserted.

    `headers` is passed straight to `JSONResponse` so a caller with extra headers to add (e.g.
    `cors.py`'s `_reject`, which needs its own CORS headers alongside a synthesized `403`/`404`/
    `405`) can hand them over here instead of building the response and then mutating `.headers`
    on it afterward (review-agent second-to-last pass, issue 8).
    """
    return JSONResponse(_envelope(exc.message, exc.code), status_code=exc.status, headers=headers)


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

# contracts/http-api.md "Busy and loading": the same body `routes_health.py`'s own
# `_GPU_UNAVAILABLE_BODY` answers `/health` with once `Readiness.mark_unhealthy()` has run.
# `GpuUnavailable` (`gpu.py`) is the other way a request meets a poisoned GPU: `GpuGate`
# refuses it directly (`try_acquire`/`acquire`) rather than through `require_ready`, so it
# needs its own handler here -- kept byte-for-byte the same body, not imported from
# routes_health.py, since that module's constant is private to it.
_GPU_UNAVAILABLE_BODY = {
    "status": "error",
    "error": "gpu is not responding",
    "code": "gpu_unavailable",
}


def http_status_error(status: int) -> tuple[str, str]:
    """The fixed ``(code, message)`` pair this codebase uses for a bare HTTP ``status`` such as
    ``404``/``405`` -- the same wording ``_http_exception_handler`` below gives Starlette's own
    "no route"/"wrong method" responses.

    Public so another module that itself needs to build one of these exact responses -- today,
    ``cors.py``'s preflight handling, which answers its own ``404``/``405`` when a path doesn't
    exist or a route doesn't support the requested method -- reuses this wording instead of
    copying the literal strings (review-agent final pass, issue 8). Raises ``KeyError`` for a
    status with no fixed wording here; unlike the handler below, there is no sensible "internal
    error" fallback for a caller that asked for a status this table was never meant to cover.
    """
    return _HTTP_EXCEPTION_RESPONSES[status]


def route_methods_for_path(router: Router, scope: Scope) -> tuple[bool, set[str], bool]:
    """The union of the declared methods of every *plain* route matching ``scope``'s path, plus
    whether any matched route can't be enumerated at all.

    Returns ``(matched, methods, any_method)``. ``matched`` is whether at least one route matches
    the path at all. ``methods`` is the union of every matched route's own ``.methods`` that
    *does* enumerate one -- whatever that actually contains, with no assumption here about what it
    is (it is *not* safe to assume ``HEAD`` comes along with ``GET``: base Starlette
    ``Route.__init__`` adds it automatically, but FastAPI's own ``APIRoute`` -- what
    ``@app.get()`` etc. actually build -- does not). FastAPI registers ``@app.get(path)``/
    ``@app.post(path)`` on the same path as two separate ``Route`` objects, not one route with two
    methods, so every matching route must contribute -- both here and in
    ``_http_exception_handler``'s own ``405`` below, which otherwise only sees Starlette's own
    ``Router.app`` keeping just the *first* ``Match.PARTIAL`` route it finds (review-agent final
    pass, issue 4).

    ``Route.matches`` returns ``Match.NONE`` only when the *path* doesn't match; a path match with
    the "wrong" method comes back as ``Match.PARTIAL`` -- either counts as a match here. Not every
    matched entry is a plain ``Route`` with a ``.methods`` set: a ``Mount`` has none at all, and
    neither does a ``Route`` registered with ``methods=None`` (a raw ASGI sub-app, meaning "the
    sub-app decides"). Such an entry is skipped rather than crashing on a bare ``route.methods``
    access (review-agent second-to-last pass, issue 5), and sets ``any_method`` -- but it does
    *not* discard whatever plain routes matching the same path *did* enumerate (review-agent
    second-to-last pass, issue 4; the previous pass folded that into an all-or-nothing ``methods
    == set()``, silently losing known-good methods whenever a Mount happened to overlap). Callers
    must treat ``any_method`` as "something here might also accept a method not in ``methods``",
    not assume ``methods`` alone is the complete allowlist, whenever it is ``True``.
    """
    matched = False
    any_method = False
    methods: set[str] = set()
    for route in router.routes:
        match, _ = route.matches(scope)
        if match is Match.NONE:
            continue
        matched = True
        route_methods = getattr(route, "methods", None)
        if route_methods is None:
            any_method = True
            continue
        methods |= set(route_methods)
    return matched, methods, any_method


def install_error_handlers(app: FastAPI, events: Emitter) -> None:
    """Register every exception handler the contract needs (contracts/http-api.md)."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # No route matched, wrong method, and Starlette's own multipart limits all raise this
        # (never `ApiError`), each carrying whatever `headers` the raiser set -- e.g. a 405's
        # `Allow` header -- which must reach the client. `exc.detail` is not: see
        # `_HTTP_EXCEPTION_RESPONSES` above.
        mapped = _HTTP_EXCEPTION_RESPONSES.get(exc.status_code)
        if mapped is None:
            return JSONResponse(
                _envelope("internal error", "internal_error"),
                status_code=500,
                headers=exc.headers,
            )
        code, message = mapped
        headers = exc.headers
        # Starlette's own `Route.handle` set `exc.headers["Allow"]` from just the first matching
        # route it found (review issue 4) -- recomputed here as the union across every route that
        # matches this path, the same way `cors.py`'s preflight does. Only when `root_path` is
        # still empty, though: once routing has passed through a `Mount`, `Mount.matches` has
        # already extended `scope["root_path"]` by its own matched prefix, so re-matching against
        # `request.app.router` (the *top-level* router) would check whatever text is left after
        # stripping that prefix, not the original path -- liable to pick up an unrelated top-level
        # route that happens to share that remainder, replacing a Mount-internal 405's own
        # (correct) `Allow` with a wrong one (review-agent second-to-last pass, issue 3). There is
        # no general way to find "the router that actually handled this path" from here, so this
        # skips the recompute entirely in that case rather than risk a wrong answer.
        if exc.status_code == 405 and not request.scope.get("root_path"):
            matched, methods, _any_method = route_methods_for_path(
                request.app.router, request.scope
            )
            if matched and methods:
                headers = {**(headers or {}), "Allow": ", ".join(sorted(methods))}
        return JSONResponse(
            _envelope(message, code),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return api_error_response(exc)

    @app.exception_handler(GpuUnavailable)
    async def _gpu_unavailable_handler(_: Request, __: GpuUnavailable) -> JSONResponse:
        # A poisoned GpuGate (gpu.py): try_acquire/acquire raise this directly, rather than
        # going through Readiness -- distinct from busy (409), and from routes_health.py's
        # own GpuUnresponsive (raised by require_ready once Readiness.mark_unhealthy() has
        # run), but the two are meant to look identical to a client.
        return JSONResponse(dict(_GPU_UNAVAILABLE_BODY), status_code=503)

    @app.exception_handler(FormParserError)
    async def _form_parser_error_handler(_: Request, exc: FormParserError) -> JSONResponse:
        # python-multipart raises this (or a subclass, e.g. MultipartParseError) for
        # a malformed body; without this handler it would fall through to the bare
        # Exception handler below and come back as an opaque 500. The parser's own
        # text describes its internals, so the client gets a fixed message.
        del exc
        return JSONResponse(
            _envelope("could not parse the request body", "invalid_field"),
            status_code=400,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Reachable only via routes not yet ported to raw request/form parsing (see
        # contracts/http-api.md); once ported, malformed input is coerced or rejected
        # by the fields module instead, and this handler stops firing for them.
        del exc
        return JSONResponse(
            _envelope("invalid request", "invalid_field"),
            status_code=400,
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never leak str(exc) to the client -- it can carry internal paths or other
        # detail that isn't the API's contract to expose -- but keep the real error
        # in the structured event log via repr(). Starlette's ServerErrorMiddleware
        # re-raises after this response is sent, so uvicorn also prints a traceback;
        # that duplicate is intended.
        if isinstance(exc, StreamAborted):
            # ServerErrorMiddleware only sends this if the response hasn't started, and a
            # StreamAborted always comes after it has: this response is never sent.
            return JSONResponse(_envelope("internal error", "internal_error"), status_code=500)
        # Without the request-id middleware (a bare test app) there is no id to reuse, but
        # the event still needs one to be traceable.
        request_id = _request_id_of(request) or f"api-{uuid.uuid4().hex}"
        events.emit("request.failed", level="error", request_id=request_id, error=repr(exc))
        return JSONResponse(_envelope("internal error", "internal_error"), status_code=500)
