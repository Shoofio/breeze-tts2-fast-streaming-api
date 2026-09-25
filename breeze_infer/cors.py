"""Pure-ASGI CORS and cross-origin protection (FR-034, FR-035; research.md R8).

`BaseHTTPMiddleware` is banned project-wide (research.md R1): it buffers the response and turns a
mid-stream failure into a clean chunked terminator instead of propagating it. This module wraps
``send`` directly instead -- the same technique as ``body_limit.py`` and ``version_header.py`` --
so its headers reach every response, including a ``413``, a ``500`` and a streamed one.

Starlette's own ``CORSMiddleware`` was rejected (research.md R8): it never trims the allowlist,
sends no ``Vary`` for a disallowed origin, adds no headers to a ``500``, answers every preflight
with ``200`` regardless of whether the path exists, and lets a ``POST`` from a disallowed origin
reach the endpoint and run. This is a from-scratch replacement covering exactly the behaviour in
``specs/003-cpp-compatible-api/contracts/http-api.md`` "CORS".

Parsing and validating ``--cors`` (trimming, ``*`` mixed with other entries, deduplication) lives
in ``settings.py``. Canonicalizing one origin (``canonical_origin``) lives in the dependency-free
``origins.py``, imported by both ``settings.py`` (to validate and normalize each allowlist entry
at startup) and this module (to canonicalize the incoming ``Origin`` header before comparing), so
neither of those two needs to import the other. A request whose origin differs from an allowlist
entry only by a default port or letter case still matches; a malformed ``Origin`` is simply
treated as not allowed, never raised.
"""

from __future__ import annotations

from dataclasses import dataclass

from starlette.responses import Response
from starlette.routing import Match, Router
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from breeze_infer.errors import ApiError, api_error_response
from breeze_infer.origins import canonical_origin

# Every method a disallowed cross-origin request is *not* blocked for: GET/HEAD can't write or
# spend GPU, and OPTIONS is how a preflight itself arrives (review issue 6 -- every other method,
# not just POST/DELETE, can have side effects, so all of them are blocked).
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MAX_AGE = b"86400"
_EXPOSE_HEADERS = b"X-Sample-Rate, X-Sample-Format, X-Breeze-Version"

# Fixed (status, code, message) triples this middleware itself can answer with. Built through
# `errors.api_error_response` (review issue 7) rather than importing errors.py's private
# status-code table, so a client sees a response built exactly the same way whether Starlette's
# own routing or this middleware produced it -- without reaching into that module's internals.
_NOT_FOUND = (404, "not_found", "not found")
_METHOD_NOT_ALLOWED = (405, "method_not_allowed", "method not allowed")
_ORIGIN_NOT_ALLOWED = (403, "origin_not_allowed", "origin not allowed")


@dataclass(frozen=True)
class CorsPolicy:
    """CORS launch configuration, straight from ``Settings.cors`` (data-model.md "Settings").

    ``origins`` is ``()`` when CORS is off, ``("*",)`` when any origin is allowed, or a tuple of
    normalized (lowercase scheme/host, trimmed, deduplicated) allowed origins -- exactly the shape
    ``settings.py`` already validates and builds.
    """

    origins: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.origins)

    @property
    def wildcard(self) -> bool:
        return self.origins == ("*",)


def origin_allowed(policy: CorsPolicy, origin: str | None) -> bool:
    """Whether ``origin`` (an ``Origin`` header value, or ``None`` if the header is absent) may
    be served.

    A well-behaved browser already sends a lowercase scheme and host with no default port, which
    is exactly the form ``settings.py`` normalizes the allowlist to -- checked first, as a fast
    path (review issue c8) that avoids canonicalizing on every request in the common case. Only
    when that exact match fails is ``origin`` run through ``canonical_origin`` too, so a redundant
    default port or a stray uppercase letter still matches. A malformed ``origin`` (never sent by
    a real browser) canonicalizes to nothing and is simply treated as not allowed, not raised.
    """
    if origin is None:
        return False
    if policy.wildcard:
        return True
    if origin in policy.origins:
        return True
    try:
        canonical = canonical_origin(origin)
    except ValueError:
        return False
    return canonical in policy.origins


def response_headers(
    policy: CorsPolicy, origin: str | None, allowed: bool
) -> list[tuple[bytes, bytes]]:
    """The CORS headers due on any response, preflight or not.

    ``allowed`` is ``origin_allowed(policy, origin)``, computed once per request by the caller
    (review issue c8) rather than recomputed here -- ``CorsMiddleware`` already needs it to decide
    whether to reject the request at all, before it knows what response headers to add.

    ``Vary: Origin`` goes on every response while in allowlist mode (BC-20) -- independent of
    ``allowed``, because a cache keyed only on the allowed response would otherwise serve it back
    for a different, disallowed origin. Wildcard mode needs no ``Vary``: the response never
    depends on the request's origin.
    """
    if not policy.enabled:
        return []
    headers: list[tuple[bytes, bytes]] = []
    if not policy.wildcard:
        headers.append((b"vary", b"Origin"))
    if allowed:
        # Echoes the raw `origin` (not a re-canonicalized form) by design: browsers require
        # Access-Control-Allow-Origin to echo their own serialized Origin value exactly, and
        # `_reject_unsafe_characters` (origins.py) already keeps anything but a browser-plausible
        # raw value from ever reaching here (review issue c5).
        allow_origin = b"*" if policy.wildcard else origin.encode("latin-1")
        headers.append((b"access-control-allow-origin", allow_origin))
        headers.append((b"access-control-expose-headers", _EXPOSE_HEADERS))
    return headers


def preflight_headers(
    route_methods: set[str], requested_headers: str | None
) -> list[tuple[bytes, bytes]]:
    """The extra headers a successful (``204``) preflight adds, on top of ``response_headers``."""
    methods = ", ".join(sorted(route_methods | {"OPTIONS"}))
    headers = [
        (b"access-control-allow-methods", methods.encode("latin-1")),
        (b"access-control-max-age", _MAX_AGE),
    ]
    if requested_headers is not None:
        # Echoed verbatim (BC-21): the client already told us exactly which headers its real
        # request will carry, so there is nothing to compute here.
        headers.append((b"access-control-allow-headers", requested_headers.encode("latin-1")))
    return headers


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope["headers"]:
        if key == name:
            return value.decode("latin-1")
    return None


def _route_methods(router: Router, scope: Scope) -> set[str] | None:
    """The union of the declared methods of every route matching ``scope``'s path -- whatever
    ``route.methods`` actually contains, with no assumption here about what that is. (It is
    *not* safe to assume ``HEAD`` comes along with ``GET``: base Starlette ``Route.__init__``
    adds it automatically, but FastAPI's own ``APIRoute`` -- what ``@app.get()`` etc. actually
    build -- does not; ``/health`` has both only because ``routes_health.py`` lists them both
    explicitly.) Returns ``None`` if no route matches the path at all.

    FastAPI registers ``@app.get(path)``/``@app.post(path)`` on the same path as two separate
    ``Route`` objects, not one route with two methods (review issue HIGH-1), so every matching
    route contributes to the union -- returning only the first one found silently hid the other
    routes' methods from a preflight. ``Route.matches`` returns ``Match.NONE`` only when the
    *path* doesn't match; a path match with the "wrong" method (here, every real route, since none
    of them declare ``OPTIONS`` themselves) comes back as ``Match.PARTIAL`` -- either is a match
    for our purposes. Not every matched entry is a plain ``Route`` with a ``.methods`` set (e.g. a
    ``Mount`` has none at all); such an entry is skipped rather than crashing on a bare
    ``route.methods`` access, and contributes nothing to the union (review issue 5).
    """
    methods: set[str] = set()
    found = False
    for route in router.routes:
        match, _ = route.matches(scope)
        if match is Match.NONE:
            continue
        route_methods = getattr(route, "methods", None)
        if route_methods is None:
            continue
        found = True
        methods |= set(route_methods)
    return methods if found else None


def _error_response(
    status: int, code: str, message: str, headers: list[tuple[bytes, bytes]]
) -> Response:
    response = api_error_response(ApiError(status, code, message))
    for name, value in headers:
        response.headers[name.decode("latin-1")] = value.decode("latin-1")
    return response


class CorsMiddleware:
    """Pure-ASGI CORS: preflight, ``403`` for a disallowed cross-origin write, and response
    headers on everything else, including error and streamed responses.

    Sits between the version header (outermost, so this layer's own ``204``/``403``/``404``/
    ``405`` still carry ``X-Breeze-Version``) and the body limit / app. ``router`` is the FastAPI
    app's own ``.router``, passed in separately from ``app`` because by the time this wraps
    ``app``, ``app`` is typically already wrapped by ``BodyLimitMiddleware`` -- ``router`` is used
    only to look up a path's declared methods for preflight, never to dispatch the request itself.

    With CORS off (``policy.enabled`` is ``False``), every ``OPTIONS`` request falls straight
    through to ``app`` (which answers with its own ``405``, since no route declares ``OPTIONS``);
    ``origin_allowed`` always returns ``False``, so any method other than ``GET``/``HEAD``/
    ``OPTIONS`` carrying an ``Origin`` still gets rejected (BC-23) even though CORS itself is off.
    """

    def __init__(self, app: ASGIApp, policy: CorsPolicy, router: Router) -> None:
        self.app = app
        self.policy = policy
        self.router = router

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        origin = _header(scope, b"origin")
        method = scope["method"].upper()
        requested_method = _header(scope, b"access-control-request-method")
        allowed = origin_allowed(self.policy, origin)

        # A real preflight always carries both headers -- Origin included, since no browser ever
        # omits it here (review issue 4). Without Origin, an OPTIONS + Access-Control-Request-
        # Method falls through to `app` below like any other request; `app` has no OPTIONS route,
        # so Starlette's own routing answers 405.
        if (
            self.policy.enabled
            and method == "OPTIONS"
            and requested_method is not None
            and origin is not None
        ):
            await self._preflight(scope, receive, send, origin, allowed, requested_method)
            return

        if method not in _SAFE_METHODS and origin is not None and not allowed:
            await self._reject(scope, receive, send, origin, allowed, *_ORIGIN_NOT_ALLOWED)
            return

        await self.app(scope, receive, self._wrap_send(send, origin, allowed))

    async def _preflight(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        origin: str,
        allowed: bool,
        requested_method: str,
    ) -> None:
        if not allowed:
            await self._reject(scope, receive, send, origin, allowed, *_ORIGIN_NOT_ALLOWED)
            return

        route_methods = _route_methods(self.router, scope)
        if route_methods is None:
            await self._reject(scope, receive, send, origin, allowed, *_NOT_FOUND)
            return
        if requested_method.upper() not in route_methods:
            headers = response_headers(self.policy, origin, allowed)
            # The matched route's own methods, same as a real wrong-method 405 would carry
            # (review issue 3) -- not "+ OPTIONS": Allow describes what the *resource* supports,
            # and OPTIONS is how you ask, not a method the resource itself implements.
            headers.append((b"allow", ", ".join(sorted(route_methods)).encode("latin-1")))
            status, code, message = _METHOD_NOT_ALLOWED
            await _error_response(status, code, message, headers)(scope, receive, send)
            return

        requested_headers = _header(scope, b"access-control-request-headers")
        headers = response_headers(self.policy, origin, allowed) + preflight_headers(
            route_methods, requested_headers
        )
        response = Response(status_code=204, headers=_as_str_headers(headers))
        await response(scope, receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        origin: str | None,
        allowed: bool,
        status: int,
        code: str,
        message: str,
    ) -> None:
        headers = response_headers(self.policy, origin, allowed)
        await _error_response(status, code, message, headers)(scope, receive, send)

    def _wrap_send(self, send: Send, origin: str | None, allowed: bool) -> Send:
        extra = response_headers(self.policy, origin, allowed)
        if not extra:
            return send

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        return send_with_cors


def _as_str_headers(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    return {name.decode("latin-1"): value.decode("latin-1") for name, value in headers}
