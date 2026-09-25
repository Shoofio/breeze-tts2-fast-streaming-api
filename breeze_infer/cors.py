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
in ``settings.py``, which imports ``canonical_origin`` from here to validate and normalize each
allowlist entry at startup; this module must never import ``settings`` back. This module's own
``origin_allowed`` runs the same ``canonical_origin`` over the incoming ``Origin`` header before
comparing, so a request whose origin differs from an allowlist entry only by a default port or
letter case still matches (a malformed ``Origin`` is simply treated as not allowed, never raised).
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Match, Router
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from breeze_infer.errors import _HTTP_EXCEPTION_RESPONSES

_UNSAFE_METHODS = frozenset({"POST", "DELETE"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
# What's left of a hostname once it's lowercase and, if it was an IDN, already turned to
# punycode -- ASCII letters, digits, '.' and '-' are the only characters a real DNS label (or a
# bracketed IP literal, handled separately below) can contain.
_HOST_CHARS_RE = re.compile(r"[a-z0-9.-]+")


def canonical_origin(value: str) -> str:
    """Canonicalize one ``scheme://host[:port]`` origin to the exact form a browser's ``Origin``
    header carries, so an allowlist entry and an incoming header that only differ cosmetically
    (case, a redundant default port, an expanded IPv6 literal, ...) compare equal.

    Used both by ``settings.py`` (to validate and normalize each ``--cors`` allowlist entry at
    startup) and by this module's own ``origin_allowed`` (to canonicalize the incoming ``Origin``
    header before comparing). Raises ``ValueError`` naming ``value`` on anything that isn't a bare
    origin; callers decide what that means -- a startup error in ``settings.py``, but never raised
    in ``origin_allowed``, which treats a malformed incoming ``Origin`` as simply not allowed.

    - Scheme must be ``http`` or ``https`` (lowercased).
    - Host is lowercased; an IDN is turned to punycode; an IP literal (IPv4 dotted-quad, or IPv6
      in brackets) is canonicalized with ``ipaddress`` -- so ``127.1`` (which browsers never send;
      ``ipaddress`` requires all four octets) and an IPv6 zone id are both rejected, and
      ``[0:0:0:0:0:0:0:1]`` canonicalizes to ``[::1]``. A hostname of ``*`` or containing any
      character outside ``[a-z0-9.-]`` after punycode conversion is rejected -- CORS has no
      wildcard-host concept; each origin must be listed.
    - Port must be plain digits, 1-65535 (browsers never send port ``0``); the scheme's own
      default port (80 for http, 443 for https) is dropped, since browsers omit it.
    - Userinfo, a path (including a lone trailing ``/``), a query or a fragment are all rejected,
      including an empty ``?`` or ``#`` -- none of those can ever appear in an ``Origin`` header.
    """
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"--cors origin {value!r} is invalid: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"--cors origin {value!r} must start with http:// or https://")

    if "@" in parsed.netloc:
        raise ValueError(
            f"--cors origin {value!r} must be a bare scheme://host[:port] origin (no userinfo)"
        )

    # `parsed.query`/`parsed.fragment` are empty strings both when the header is absent and when
    # it's present but empty (a bare trailing '?' or '#'), so the raw value is checked instead.
    has_query_or_fragment = "?" in value or "#" in value
    if parsed.path == "/" and not has_query_or_fragment:
        without_slash = value[: value.rindex("/")]
        raise ValueError(
            f"--cors origin {value!r} must not have a trailing slash; use {without_slash!r}"
        )
    if parsed.path or has_query_or_fragment:
        raise ValueError(
            f"--cors origin {value!r} must be a bare scheme://host[:port] origin, "
            "with no path, query or fragment"
        )

    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"--cors origin {value!r} has an invalid port") from exc

    if not host:
        raise ValueError(f"--cors origin {value!r} must be a bare scheme://host[:port] origin")
    if port == 0:
        raise ValueError(f"--cors origin {value!r} port must be between 1 and 65535")

    host = _canonical_host(value, host)

    port_suffix = "" if port is None or port == _DEFAULT_PORTS[scheme] else f":{port}"
    return f"{scheme}://{host}{port_suffix}"


def _canonical_host(value: str, host: str) -> str:
    """``host`` is ``urlsplit(value).hostname``: already lowercased, brackets stripped, and, for
    an IPv6 literal, hex digits already lowercased too."""
    if "%" in host:
        # A percent means an IPv6 zone id (`fe80::1%eth0`, or the URL-escaped `%25eth0`) --
        # meaningful only on the machine that assigned the zone, so it can never appear in a
        # browser's Origin header.
        raise ValueError(f"--cors origin {value!r} must not include an IPv6 zone id")

    if ":" in host:
        # Only reachable via a bracketed IPv6 literal -- urlsplit treats an unbracketed ':' as
        # the host/port separator -- so this must parse as IPv6 or the entry is malformed.
        try:
            return f"[{ipaddress.IPv6Address(host).compressed}]"
        except ValueError as exc:
            raise ValueError(f"--cors origin {value!r} has an invalid IPv6 host: {exc}") from exc

    if re.fullmatch(r"[0-9.]+", host):
        # All-digits-and-dots: this was clearly meant as an IPv4 literal, so a failure here (e.g.
        # '127.1', which ipaddress rejects since it isn't all four dotted-quad octets) is reported
        # as a bad IPv4 address rather than falling through to the hostname-syntax checks below.
        try:
            return str(ipaddress.IPv4Address(host))
        except ValueError as exc:
            raise ValueError(
                f"--cors origin {value!r} host must be dotted-quad IPv4 (got {host!r}): {exc}"
            ) from exc

    try:
        canonical = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"--cors origin {value!r} has an invalid IDN host: {exc}") from exc

    if "*" in canonical:
        raise ValueError(
            f"--cors origin {value!r}: wildcard hosts aren't supported; list each origin"
        )
    if not _HOST_CHARS_RE.fullmatch(canonical):
        raise ValueError(
            f"--cors origin {value!r} host must contain only letters, digits, '.' or '-' "
            f"(got {host!r})"
        )
    return canonical
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

    A well-behaved browser already sends a lowercase scheme and host with no default port, which
    is exactly the form ``settings.py`` normalizes the allowlist to -- but ``origin`` is run
    through the same ``canonical_origin`` anyway, so a redundant default port or a stray uppercase
    letter still matches. A malformed ``origin`` (never sent by a real browser) canonicalizes to
    nothing and is simply treated as not allowed, not raised.
    """
    if origin is None:
        return False
    if policy.wildcard:
        return True
    try:
        canonical = canonical_origin(origin)
    except ValueError:
        return False
    return canonical in policy.origins


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
