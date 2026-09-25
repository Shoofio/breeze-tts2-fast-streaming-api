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

Parsing and validating ``--cors`` (trimming, ``*`` mixed with other entries, deduplication, the
lowercase-scheme-and-host normalization) lives in ``settings.py``: this module only consumes the
already-normalized ``Settings.cors`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Match, Router
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from breeze_infer.errors import _HTTP_EXCEPTION_RESPONSES

_UNSAFE_METHODS = frozenset({"POST", "DELETE"})
_MAX_AGE = b"86400"
_EXPOSE_HEADERS = b"X-Sample-Rate, X-Sample-Format, X-Breeze-Version"

# Reuse the fixed wording the plain HTTPException handler already uses for these two statuses
# (errors.py), so a client sees exactly the same envelope whether Starlette's own routing or
# this middleware's own preflight handling produced the 404/405.
_NOT_FOUND_CODE, _NOT_FOUND_MESSAGE = _HTTP_EXCEPTION_RESPONSES[404]
_METHOD_NOT_ALLOWED_CODE, _METHOD_NOT_ALLOWED_MESSAGE = _HTTP_EXCEPTION_RESPONSES[405]
_ORIGIN_NOT_ALLOWED_CODE = "origin_not_allowed"
_ORIGIN_NOT_ALLOWED_MESSAGE = "origin not allowed"


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

    Comparison is exact, not case-insensitive: a browser's ``Origin`` header already carries a
    lowercase scheme and host, and ``settings.py`` normalizes the allowlist the same way, so no
    further folding happens here (an ``Origin`` header never carries a path to normalize).
    """
    if origin is None:
        return False
    return policy.wildcard or origin in policy.origins


def response_headers(policy: CorsPolicy, origin: str | None) -> list[tuple[bytes, bytes]]:
    """The CORS headers due on any response, preflight or not.

    ``Vary: Origin`` goes on every response while in allowlist mode (BC-20) -- independent of
    whether this particular ``origin`` is allowed, because a cache keyed only on the allowed
    response would otherwise serve it back for a different, disallowed origin. Wildcard mode
    needs no ``Vary``: the response never depends on the request's origin.
    """
    if not policy.enabled:
        return []
    headers: list[tuple[bytes, bytes]] = []
    if not policy.wildcard:
        headers.append((b"vary", b"Origin"))
    if origin_allowed(policy, origin):
        allow_origin = b"*" if policy.wildcard else origin.encode("latin-1")
        headers.append((b"access-control-allow-origin", allow_origin))
        headers.append((b"access-control-expose-headers", _EXPOSE_HEADERS))
    return headers


def preflight_headers(
    policy: CorsPolicy, route_methods: set[str], requested_headers: str | None
) -> list[tuple[bytes, bytes]]:
    """The extra headers a successful (``204``) preflight adds, on top of ``response_headers``.

    ``policy`` is accepted for symmetry with the other pure functions here (and in case a future
    contract change ties ``Max-Age`` or the allowed headers to the CORS mode); nothing in today's
    contract varies preflight headers by ``policy``.
    """
    del policy
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
    """The declared methods (including ``HEAD`` where ``GET`` is present) of the route matching
    ``scope``'s path, or ``None`` if no route matches the path at all.

    ``Route.matches`` returns ``Match.NONE`` only when the *path* doesn't match; a path match with
    the "wrong" method (here, every real route, since none of them declare ``OPTIONS`` themselves)
    comes back as ``Match.PARTIAL`` -- either way the route and its methods are what we want.
    """
    for route in router.routes:
        match, _ = route.matches(scope)
        if match is not Match.NONE:
            return set(route.methods or ())
    return None


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
    ``origin_allowed`` always returns ``False``, so a ``POST``/``DELETE`` carrying an ``Origin``
    still gets rejected (BC-23) even though CORS itself is off.
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

        if self.policy.enabled and method == "OPTIONS" and requested_method is not None:
            await self._preflight(scope, receive, send, origin, requested_method)
            return

        if method in _UNSAFE_METHODS and origin is not None and not origin_allowed(
            self.policy, origin
        ):
            await self._reject(
                scope, receive, send, 403, _ORIGIN_NOT_ALLOWED_CODE, _ORIGIN_NOT_ALLOWED_MESSAGE,
                origin,
            )
            return

        await self.app(scope, receive, self._wrap_send(send, origin))

    async def _preflight(
        self, scope: Scope, receive: Receive, send: Send, origin: str | None,
        requested_method: str,
    ) -> None:
        if origin is not None and not origin_allowed(self.policy, origin):
            await self._reject(
                scope, receive, send, 403, _ORIGIN_NOT_ALLOWED_CODE, _ORIGIN_NOT_ALLOWED_MESSAGE,
                origin,
            )
            return

        route_methods = _route_methods(self.router, scope)
        if route_methods is None:
            await self._reject(
                scope, receive, send, 404, _NOT_FOUND_CODE, _NOT_FOUND_MESSAGE, origin
            )
            return
        if requested_method.upper() not in route_methods:
            await self._reject(
                scope, receive, send, 405, _METHOD_NOT_ALLOWED_CODE, _METHOD_NOT_ALLOWED_MESSAGE,
                origin,
            )
            return

        requested_headers = _header(scope, b"access-control-request-headers")
        headers = response_headers(self.policy, origin) + preflight_headers(
            self.policy, route_methods, requested_headers
        )
        response = Response(status_code=204, headers=_as_str_headers(headers))
        await response(scope, receive, send)

    async def _reject(
        self, scope: Scope, receive: Receive, send: Send, status: int, code: str, message: str,
        origin: str | None,
    ) -> None:
        headers = _as_str_headers(response_headers(self.policy, origin))
        response = JSONResponse({"error": message, "code": code}, status_code=status, headers=headers)
        await response(scope, receive, send)

    def _wrap_send(self, send: Send, origin: str | None) -> Send:
        extra = response_headers(self.policy, origin)
        if not extra:
            return send

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        return send_with_cors


def _as_str_headers(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    return {name.decode("latin-1"): value.decode("latin-1") for name, value in headers}
