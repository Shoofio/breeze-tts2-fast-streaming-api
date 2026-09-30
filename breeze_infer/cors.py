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

import re
from collections.abc import Sequence
from dataclasses import dataclass

from starlette.responses import Response
from starlette.routing import Router
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from breeze_infer.errors import (
    ApiError,
    api_error_response,
    http_status_error,
    route_methods_for_path,
)
from breeze_infer.origins import canonical_origin

# Every method a disallowed cross-origin request is *not* blocked for: GET/HEAD can't write, and
# OPTIONS is how a preflight itself arrives (review issue 6 -- every other method, not just
# POST/DELETE, can have side effects, so all of them are blocked). One GET does spend GPU:
# `GET /v1/audio/speech.wav` synthesizes. GET skips the origin check, so any page can trigger
# synthesis there, with or without an Origin (a plain <audio> sends none). That is accepted on
# purpose (specs/004-browser-wav-stream, Clarifications).
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MAX_AGE = b"86400"
_EXPOSE_HEADERS = b"X-Sample-Rate, X-Sample-Format, X-Breeze-Version"
# RFC 7230 tchar: a bare HTTP method name, never a list, never anything with a space or comma in
# it. `Access-Control-Request-Method` must be exactly this before it's ever trusted as a method
# name to echo into a response header (review-agent second-to-last pass, issue 2) -- e.g.
# `"µ".upper()` is `"Μ"` (U+039C GREEK CAPITAL LETTER MU), which isn't Latin-1 and would crash
# encoding a header with it; a comma- or space-separated list must never be echoed as if it were
# one method either.
_TOKEN_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")

# Fixed (status, code, message) triples this middleware itself can answer with. 404/405 are built
# from `errors.http_status_error` (review-agent final pass, issue 8) rather than copied as string
# literals, so a client sees exactly the same wording whether Starlette's own routing or this
# middleware's own preflight handling produced the response. 403 has no entry there: it isn't a
# generic "no route"/"wrong method" status, so `errors.py`'s table doesn't cover it -- CORS is the
# only place `origin_not_allowed` means anything.
_NOT_FOUND = (404, *http_status_error(404))
_METHOD_NOT_ALLOWED = (405, *http_status_error(405))
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

    ``policy.enabled`` is checked first (review-agent final pass, issue 7): with CORS off there
    is no origin any request could match, so there's nothing to gain by even looking at
    ``origin``, let alone canonicalizing it.

    A well-behaved browser already sends a lowercase scheme and host with no default port, which
    is exactly the form ``settings.py`` normalizes the allowlist to -- checked next, as a fast
    path (review issue c8) that avoids canonicalizing on every request in the common case. Only
    when that exact match fails is ``origin`` run through ``canonical_origin`` too, so a redundant
    default port or a stray uppercase letter still matches. A malformed ``origin`` (never sent by
    a real browser) canonicalizes to nothing and is simply treated as not allowed, not raised.
    """
    if not policy.enabled:
        return False
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


def _allow_header(methods: set[str]) -> list[tuple[bytes, bytes]]:
    if not methods:
        return []
    return [(b"allow", ", ".join(sorted(methods)).encode("latin-1"))]


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

        matched, route_methods, any_method = route_methods_for_path(self.router, scope)
        if not matched:
            await self._reject(scope, receive, send, origin, allowed, *_NOT_FOUND)
            return

        requested = requested_method.upper()
        effective_methods = route_methods
        if requested not in route_methods:
            # `any_method` means a Mount, or a Route registered with `methods=None`, also matched
            # -- its sub-app decides its own methods, which this function can't enumerate, so a
            # method not already in `route_methods` isn't rejected outright (review-agent final
            # pass, issue 6) -- but only when it's a legitimate single HTTP method token; nothing
            # sane to echo back otherwise (review-agent second-to-last pass, issue 2), so that
            # case is rejected exactly like the plain "not a method this path supports" one below.
            if not (any_method and _TOKEN_RE.fullmatch(requested)):
                # The matched routes' own known methods, same as a real wrong-method 405 would
                # carry (review issue 3) -- not "+ OPTIONS": Allow describes what the *resource*
                # supports, and OPTIONS is how you ask, not a method the resource itself
                # implements. Empty when only an any-method route matched (nothing known to list).
                await self._reject(
                    scope, receive, send, origin, allowed, *_METHOD_NOT_ALLOWED,
                    extra_headers=_allow_header(route_methods),
                )
                return
            effective_methods = route_methods | {requested}

        requested_headers = _header(scope, b"access-control-request-headers")
        headers = response_headers(self.policy, origin, allowed) + preflight_headers(
            effective_methods, requested_headers
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
        extra_headers: Sequence[tuple[bytes, bytes]] = (),
    ) -> None:
        # Built once here (review-agent final pass, issue 9) and handed straight to
        # `api_error_response` (review-agent second-to-last pass, issue 8) instead of
        # constructing the response and then mutating `.headers` on it afterward -- so every
        # rejection this middleware sends (a disallowed origin's 403, preflight's own 404/405)
        # goes through the same header-assembly path instead of each caller repeating it.
        headers = _as_str_headers(response_headers(self.policy, origin, allowed) + list(extra_headers))
        response = api_error_response(ApiError(status, code, message), headers)
        await response(scope, receive, send)

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
