"""CORS wiring and cross-origin protection (T026; FR-034, FR-035; BC-19..23;
contracts/http-api.md "CORS").

Parsing and validating `--cors` (trimming, `*` mixed with other entries, deduplication, the
lowercase-scheme-and-host normalization) lives in `settings.py` and is already covered by
`tests/test_settings.py`; this file does not re-port those cases (research.md R8's `A:` reference
`_parse_cors_origins` tests are entirely superseded by that module). It covers only the runtime
behavior of `breeze_infer.cors.CorsMiddleware`, wired through `api.create_app` the same way the
real server builds it -- plus, per T026, `test_bc_19` and `test_bc_22` as settings-level checks
and BC-23's counting endpoint (the voice routes don't exist yet).
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route

from breeze_infer import __version__
from breeze_infer.api import Components, create_app
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.cors import (
    CorsMiddleware,
    CorsPolicy,
    origin_allowed,
    preflight_headers,
)
from breeze_infer.errors import http_status_error, route_methods_for_path
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import MAX_BODY_BYTES
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.settings import settings_from_args
from breeze_infer.version_header import VersionHeaderMiddleware
from tests.fakes import FakeRuntime, open_no_voices

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it
GOOD_ORIGIN = "https://good.example"
EVIL_ORIGIN = "https://evil.example"
NOT_FOUND = {"error": "not found", "code": "not_found"}
METHOD_NOT_ALLOWED = {"error": "method not allowed", "code": "method_not_allowed"}
ORIGIN_NOT_ALLOWED = {"error": "origin not allowed", "code": "origin_not_allowed"}


class _ExplodingRuntime:
    """A `Readiness`-ready runtime whose `sample_rate` raises, so `/health` answers `500`."""

    @property
    def sample_rate(self) -> int:
        raise RuntimeError("boom")


@pytest.fixture()
def make_client() -> Iterator[Callable[[Sequence[str]], TestClient]]:
    """Builds real apps through `api.create_app`, each ready immediately (`FakeRuntime`)."""
    made: list[Components] = []

    def factory(argv: Sequence[str] = (), *, runtime: object | None = None) -> TestClient:
        readiness = Readiness()
        readiness.mark_ready(runtime if runtime is not None else FakeRuntime())
        components = Components(
            settings=settings_from_args([MODEL_DIR, *argv]),
            events=Emitter(io.StringIO(), lambda: 0.0),
            gate=GpuGate(),
            gpu=GpuThread("cpu", lambda _device: None),
            readiness=readiness,
            ws_port=lambda: 8081,
            cpu_tokenizer=CpuTokenizer(),
            open_voices=open_no_voices,
        )
        made.append(components)
        # raise_server_exceptions=False: a 500 is a legitimate response to assert on here (its
        # own CORS/version headers), not a test failure -- see tests/test_api_errors.py.
        return TestClient(create_app(components), raise_server_exceptions=False)

    yield factory
    for components in made:
        components.gpu.shutdown()


def _wrapped(app: FastAPI, policy: CorsPolicy) -> TestClient:
    """The same middleware composition as `api.create_app`, around a small ad hoc app -- used
    only where the contract needs a route shape `create_app`'s real app doesn't have yet (a
    unsafe-method endpoint to count calls on, or a streaming one).
    """
    wrapped = VersionHeaderMiddleware(
        CorsMiddleware(BodyLimitMiddleware(app), policy, app.router), version=__version__
    )
    return TestClient(wrapped)


def _counting_app(policy: CorsPolicy) -> tuple[TestClient, list[str]]:
    app = FastAPI()
    calls: list[str] = []

    @app.post("/v1/voices")
    async def create_voice() -> dict:
        calls.append("post")
        return {"ok": True}

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str) -> dict:
        del voice_id
        calls.append("delete")
        return {"ok": True}

    return _wrapped(app, policy), calls


# ------------------------------------------------------------------- BC-19, BC-22 (settings)


def test_bc_19_allowlist_entries_are_trimmed() -> None:
    """BC-19: the C++ server never trimmed entries, so `"a, b"` could never match `"b"`."""
    settings = settings_from_args(
        [MODEL_DIR, "--cors", " https://a.example , https://b.example "]
    )

    assert settings.cors == ("https://a.example", "https://b.example")


def test_bc_22_star_mixed_with_origins_fails_at_startup() -> None:
    """BC-22: the C++ server treated a stray `"*"` in an allowlist as a dead, never-matching
    entry rather than rejecting the (almost certainly mistaken) configuration.
    """
    with pytest.raises(SystemExit):
        settings_from_args([MODEL_DIR, "--cors", "https://a.example,*"])


# --------------------------------------------------------------------------------- CORS off


def test_cors_off_adds_no_cors_headers(make_client: Callable[..., TestClient]) -> None:
    client = make_client()

    response = client.get("/health", headers={"Origin": EVIL_ORIGIN})

    assert "access-control-allow-origin" not in response.headers
    assert "access-control-expose-headers" not in response.headers
    assert "vary" not in response.headers


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_bc_23_disallowed_origin_rejected_before_the_endpoint_runs_with_cors_off(
    method: str,
) -> None:
    """BC-23: the C++ server let a cross-origin POST/DELETE run (and even write voice files)
    regardless of the CORS setting; with CORS off, no origin is allowed at all.
    """
    client, calls = _counting_app(CorsPolicy(()))
    path = "/v1/voices" if method == "POST" else "/v1/voices/x"

    response = client.request(method, path, headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED
    assert calls == []


# ------------------------------------------------------------------------------- BC-23 (on)


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_bc_23_disallowed_origin_rejected_before_the_endpoint_runs_with_an_allowlist(
    method: str,
) -> None:
    client, calls = _counting_app(CorsPolicy((GOOD_ORIGIN,)))
    path = "/v1/voices" if method == "POST" else "/v1/voices/x"

    response = client.request(method, path, headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED
    assert calls == []


def test_no_origin_requests_pass_regardless_of_cors_setting() -> None:
    """Requests without an `Origin` header are unaffected by CORS (FR-035)."""
    for policy in (CorsPolicy(()), CorsPolicy((GOOD_ORIGIN,))):
        client, calls = _counting_app(policy)

        response = client.post("/v1/voices")

        assert response.status_code == 200
        assert calls == ["post"]


def test_allowed_origin_unsafe_method_reaches_the_endpoint() -> None:
    client, calls = _counting_app(CorsPolicy((GOOD_ORIGIN,)))

    response = client.post("/v1/voices", headers={"Origin": GOOD_ORIGIN})

    assert response.status_code == 200
    assert calls == ["post"]


# --------------------------------------------------------------------------------- BC-20


def test_bc_20_vary_origin_on_every_response_in_allowlist_mode(
    make_client: Callable[..., TestClient],
) -> None:
    """BC-20: the C++ server only sent `Vary: Origin` on a response to a matching origin; a
    shared cache mixing that with a response to a *non*-matching origin was then wrong for
    someone else. It must be sent on every response while in allowlist mode.
    """
    client = make_client(["--cors", GOOD_ORIGIN])

    no_origin = client.get("/health")
    disallowed = client.get("/health", headers={"Origin": EVIL_ORIGIN})
    not_found = client.get("/nope", headers={"Origin": EVIL_ORIGIN})
    # GET (not POST/DELETE/PUT/...), so CORS's own now-broadened (review issue 6)
    # not-GET/HEAD/OPTIONS-and-disallowed check doesn't pre-empt this: the real
    # `Content-Length` alone must trip BodyLimitMiddleware's immediate-rejection path.
    too_large = client.request(
        "GET", "/health", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Origin": EVIL_ORIGIN}
    )

    assert no_origin.headers["vary"] == "Origin"
    assert disallowed.status_code == 200
    assert disallowed.headers["vary"] == "Origin"
    assert not_found.status_code == 404
    assert not_found.headers["vary"] == "Origin"
    assert too_large.status_code == 413
    assert too_large.headers["vary"] == "Origin"


def test_bc_20_vary_origin_on_a_plain_wrong_method_405() -> None:
    """BC-20, continued: a wrong-*method* 405 (as opposed to CORS's own preflight-405) must
    still carry `Vary: Origin` in allowlist mode. `GET` is used against a route that only
    registers `POST` -- `GET` is exempt from the disallowed-origin check (review issue 6), so
    this reaches Starlette's routing and comes back as a plain wrong-method 405, not CORS's own
    403.
    """
    client, calls = _counting_app(CorsPolicy((GOOD_ORIGIN,)))

    response = client.get("/v1/voices", headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 405
    assert response.headers["vary"] == "Origin"
    assert calls == []


def test_bc_20_vary_origin_on_a_500(make_client: Callable[..., TestClient]) -> None:
    client = make_client(["--cors", GOOD_ORIGIN], runtime=_ExplodingRuntime())

    response = client.get("/health", headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 500
    assert response.headers["vary"] == "Origin"


# --------------------------------------------------------------------------------- BC-21


def test_bc_21_preflight_is_route_aware(make_client: Callable[..., TestClient]) -> None:
    """BC-21: the C++ server answered `204` to a preflight for *any* path, advertising every
    method the server has anywhere -- not just the ones the requested path actually supports.
    """
    client = make_client(["--cors", GOOD_ORIGIN])

    existing_path = client.options(
        "/health",
        headers={
            "Origin": GOOD_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type, x-custom",
        },
    )
    unknown_path = client.options(
        "/nope",
        headers={"Origin": GOOD_ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    unsupported_method = client.options(
        "/health",
        headers={"Origin": GOOD_ORIGIN, "Access-Control-Request-Method": "POST"},
    )

    assert existing_path.status_code == 204
    assert existing_path.content == b""
    assert set(existing_path.headers["access-control-allow-methods"].split(", ")) == {
        "GET",
        "HEAD",
        "OPTIONS",
    }
    assert existing_path.headers["access-control-allow-headers"] == "content-type, x-custom"
    assert existing_path.headers["access-control-max-age"] == "86400"

    assert unknown_path.status_code == 404
    assert unknown_path.json() == NOT_FOUND

    assert unsupported_method.status_code == 405
    assert unsupported_method.json() == METHOD_NOT_ALLOWED
    # review issue 3: the preflight's own 405 carries an Allow header, same as a real
    # wrong-method response would (test_health.py's test_bc_18_wrong_method_on_health_is_405).
    assert set(unsupported_method.headers["allow"].split(", ")) == {"GET", "HEAD"}


def test_disallowed_preflight_gets_403(make_client: Callable[..., TestClient]) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.options(
        "/health",
        headers={"Origin": EVIL_ORIGIN, "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED


def test_preflight_unions_methods_across_separate_routes_on_one_path() -> None:
    """review issue HIGH-1: FastAPI registers `@app.get()`/`@app.post()` on the same path as two
    separate `Route` objects, not one route with two methods. `_route_methods` must union every
    matching route's methods, not return only the first route it finds -- otherwise a preflight
    for the second route's method wrongly comes back `405`.

    Also incidentally locks in the HIGH-1 "stop claiming HEAD is added for GET" correction:
    FastAPI's `APIRoute` (unlike base Starlette `Route`) does *not* auto-add `HEAD` when `GET` is
    declared, so a plain `@app.get()` route's `.methods` really is just `{"GET"}`.
    """
    app = FastAPI()

    @app.get("/v1/voices")
    async def list_voices() -> dict:
        return {}

    @app.post("/v1/voices")
    async def create_voice() -> dict:
        return {}

    client = _wrapped(app, CorsPolicy((GOOD_ORIGIN,)))

    response = client.options(
        "/v1/voices",
        headers={"Origin": GOOD_ORIGIN, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 204
    assert set(response.headers["access-control-allow-methods"].split(", ")) == {
        "GET",
        "POST",
        "OPTIONS",
    }


def test_preflight_without_origin_is_not_a_preflight(
    make_client: Callable[..., TestClient],
) -> None:
    """review issue 4: a real browser always sends `Origin` on a preflight; an `OPTIONS` with
    `Access-Control-Request-Method` but no `Origin` isn't one, and must fall through to the
    app's own wrong-method `405` rather than being answered as CORS preflight.
    """
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.options("/health", headers={"Access-Control-Request-Method": "GET"})

    assert response.status_code == 405
    assert "access-control-allow-methods" not in response.headers
    assert "access-control-max-age" not in response.headers


def test_route_methods_for_path_does_not_crash_on_a_route_with_no_methods_attribute() -> None:
    """review issue 5: a `Mount` (e.g. for static files) has no `.methods` attribute at all;
    `route_methods_for_path` (shared with `errors.py`'s 405 handler, review issue 4) must use
    `getattr(route, "methods", None)` rather than crashing on a bare `route.methods` access.
    """

    async def _sub_app(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    router = FastAPI().router
    router.routes = [Mount("/static", app=_sub_app)]
    scope = {"type": "http", "method": "OPTIONS", "path": "/static/thing", "headers": []}

    matched, methods, any_method = route_methods_for_path(router, scope)

    assert matched is True
    assert methods == set()
    assert any_method is True


def test_route_methods_for_path_no_match_is_not_matched() -> None:
    router = FastAPI().router
    router.routes = [Route("/health", _noop_endpoint, methods=["GET"])]
    scope = {"type": "http", "method": "OPTIONS", "path": "/nope", "headers": []}

    assert route_methods_for_path(router, scope) == (False, set(), False)


def test_route_methods_for_path_unions_plain_routes_even_when_a_mount_also_matches() -> None:
    """review-agent second-to-last pass, issue 4: a `Mount` matching the same path as a plain
    `Route` must not discard that route's own known methods -- both the union of every plain
    route's methods *and* the any-method flag are reported, not one instead of the other."""

    async def _sub_app(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    router = FastAPI().router
    router.routes = [
        # A `Mount` only matches *under* its prefix ("/multi/..."), never the bare prefix itself,
        # so both routes have to match "/multi/thing" specifically for this to exercise anything.
        Mount("/multi", app=_sub_app),
        Route("/multi/thing", _noop_endpoint, methods=["GET"]),
    ]
    scope = {"type": "http", "method": "OPTIONS", "path": "/multi/thing", "headers": []}

    matched, methods, any_method = route_methods_for_path(router, scope)

    assert matched is True
    # Base Starlette `Route` (unlike FastAPI's `APIRoute`) auto-adds HEAD when GET is declared.
    assert methods == {"GET", "HEAD"}
    assert any_method is True


def test_preflight_to_a_mounted_path_allows_any_method_instead_of_404(
    make_client: Callable[..., TestClient],
) -> None:
    """final review issue 6: a `Mount`'s sub-app decides its own methods; a path only a `Mount`
    matches must not come back `404` from preflight, and the requested method must be let
    through (not `405`) since there's no enumerable method list to reject it against.
    """
    app = FastAPI()

    async def _sub_app(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    app.router.routes.append(Mount("/static", app=_sub_app))
    client = _wrapped(app, CorsPolicy((GOOD_ORIGIN,)))

    response = client.options(
        "/static/thing",
        headers={"Origin": GOOD_ORIGIN, "Access-Control-Request-Method": "PUT"},
    )

    assert response.status_code == 204
    assert "PUT" in response.headers["access-control-allow-methods"].split(", ")


async def _call_asgi(app: object, headers: list[tuple[bytes, bytes]], path: str = "/static/thing") -> list[dict]:
    """Drives an ASGI app directly with a hand-built scope, bypassing httpx/`TestClient` --
    needed for `Access-Control-Request-Method: \\xb5`, which httpx itself refuses to send at all
    (`UnicodeEncodeError` client-side, since it isn't valid ASCII)."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "OPTIONS",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)
    return sent


@pytest.mark.parametrize(
    "acrm", [b"\xb5", b"GET, POST", b"GET POST", b"GET,DELETE"], ids=["mu-byte", "list-space", "list", "list-comma"]
)
def test_any_method_preflight_with_a_non_token_acrm_is_405_not_500(acrm: bytes) -> None:
    """review-agent second-to-last pass, issue 2: in the any-method (`Mount`) preflight case,
    `Access-Control-Request-Method` must be validated as a single RFC 7230 token before being
    trusted as a method name to echo back. `b"\\xb5"` decodes (Latin-1) to "µ"
    (MICRO SIGN); `"µ".upper()` is "Μ" (GREEK CAPITAL LETTER MU), which isn't Latin-1 and
    previously crashed `.encode("latin-1")` when building the response header -- a 500, not a
    controlled rejection. A comma- or space-separated list must also never be echoed as if it
    were one method. The fix answers `405 method_not_allowed` in every case, never a `500`.
    """
    app = FastAPI()

    async def _sub_app(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    app.router.routes.append(Mount("/static", app=_sub_app))
    wrapped = VersionHeaderMiddleware(
        CorsMiddleware(BodyLimitMiddleware(app), CorsPolicy((GOOD_ORIGIN,)), app.router),
        version=__version__,
    )
    headers = [
        (b"origin", GOOD_ORIGIN.encode()),
        (b"access-control-request-method", acrm),
    ]

    sent = asyncio.run(_call_asgi(wrapped, headers))

    start = sent[0]
    assert start["type"] == "http.response.start"
    assert start["status"] == 405
    header_names = {name for name, _ in start["headers"]}
    assert b"access-control-allow-methods" not in header_names


def test_origin_allowed_returns_false_immediately_when_cors_is_off() -> None:
    """final review issue 7: checked before even looking at `origin` -- with CORS off there is
    nothing any origin could match, so there's no reason to canonicalize it first."""
    assert origin_allowed(CorsPolicy(()), "not a valid origin at all") is False
    assert origin_allowed(CorsPolicy(()), None) is False


def test_http_status_error_supplies_the_wording_cors_reuses() -> None:
    """final review issue 8: `cors.py`'s own `404`/`405` no longer copy `errors.py`'s wording as
    string literals -- they're built from this public function instead."""
    assert http_status_error(404) == ("not_found", "not found")
    assert http_status_error(405) == ("method_not_allowed", "method not allowed")


def _noop_endpoint(request: object) -> None:
    del request


@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_every_unsafe_method_not_just_post_delete_is_blocked_from_a_disallowed_origin(
    method: str,
) -> None:
    """review issue 6: only `POST`/`DELETE` were blocked before; any method other than
    `GET`/`HEAD`/`OPTIONS` can write or have side effects, so all of them must be."""
    app = FastAPI()
    calls: list[str] = []

    @app.api_route("/v1/voices/x", methods=["PUT", "PATCH"])
    async def update_voice() -> dict:
        calls.append(method)
        return {"ok": True}

    client = _wrapped(app, CorsPolicy((GOOD_ORIGIN,)))

    response = client.request(method, "/v1/voices/x", headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED
    assert calls == []


def test_preflight_headers_no_longer_takes_a_policy_argument() -> None:
    """review issue 9: nothing in `preflight_headers` varies by `policy`; the dead parameter is
    dropped."""
    headers = dict(preflight_headers({"GET", "HEAD"}, "content-type"))

    assert set(headers[b"access-control-allow-methods"].split(b", ")) == {
        b"GET",
        b"HEAD",
        b"OPTIONS",
    }
    assert headers[b"access-control-max-age"] == b"86400"
    assert headers[b"access-control-allow-headers"] == b"content-type"


# ------------------------------------------------------------------------- expose-headers


def test_x_breeze_version_is_listed_in_expose_headers(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.get("/health", headers={"Origin": GOOD_ORIGIN})

    exposed = {name.strip() for name in response.headers["access-control-expose-headers"].split(",")}
    assert exposed == {"X-Sample-Rate", "X-Sample-Format", "X-Breeze-Version"}


# --------------------------------------------------------- allowed-origin headers, every status


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("GET", "/health", 200), ("GET", "/nope", 404)],
)
def test_allowed_origin_gets_acao_and_expose_headers(
    make_client: Callable[..., TestClient], method: str, path: str, status: int
) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.request(method, path, headers={"Origin": GOOD_ORIGIN})

    assert response.status_code == status
    assert response.headers["access-control-allow-origin"] == GOOD_ORIGIN
    assert "access-control-expose-headers" in response.headers


def test_allowed_origin_gets_acao_on_a_413(make_client: Callable[..., TestClient]) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.request(
        "GET", "/health", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Origin": GOOD_ORIGIN}
    )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == GOOD_ORIGIN
    assert "access-control-expose-headers" in response.headers


def test_allowed_origin_gets_acao_on_a_500(make_client: Callable[..., TestClient]) -> None:
    client = make_client(["--cors", GOOD_ORIGIN], runtime=_ExplodingRuntime())

    response = client.get("/health", headers={"Origin": GOOD_ORIGIN})

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == GOOD_ORIGIN
    assert "access-control-expose-headers" in response.headers


def test_allowed_origin_gets_acao_on_a_streamed_response() -> None:
    app = FastAPI()

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body() -> Iterator[bytes]:
            yield b"a"
            yield b"b"

        return StreamingResponse(body(), media_type="application/octet-stream")

    client = _wrapped(app, CorsPolicy((GOOD_ORIGIN,)))

    response = client.get("/stream", headers={"Origin": GOOD_ORIGIN})

    assert response.status_code == 200
    assert response.content == b"ab"
    assert response.headers["access-control-allow-origin"] == GOOD_ORIGIN
    assert "access-control-expose-headers" in response.headers


# ------------------------------------------------------------------------------ wildcard mode


def test_star_mode_gets_literal_acao_and_no_vary(
    make_client: Callable[..., TestClient],
) -> None:
    """`*` mode: `Access-Control-Allow-Origin: *`, and no `Vary` -- the response never depends
    on the request's origin, so there is nothing to vary on (contracts/http-api.md "CORS").
    """
    client = make_client(["--cors"])

    response = client.get("/health", headers={"Origin": EVIL_ORIGIN})

    assert response.headers["access-control-allow-origin"] == "*"
    assert "vary" not in response.headers


# --------------------------------------------------------------------------- X-Breeze-Version


def test_x_breeze_version_present_on_preflight_204_and_on_403(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    preflight = client.options(
        "/health",
        headers={"Origin": GOOD_ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    forbidden = client.options(
        "/health",
        headers={"Origin": EVIL_ORIGIN, "Access-Control-Request-Method": "GET"},
    )

    assert preflight.status_code == 204
    assert preflight.headers["x-breeze-version"] == __version__
    assert forbidden.status_code == 403
    assert forbidden.headers["x-breeze-version"] == __version__


# ------------------------------------------------------------------- canonical_origin matching


@pytest.mark.parametrize(
    "incoming_origin",
    ["https://good.example:443", "HTTPS://GOOD.EXAMPLE"],
)
def test_incoming_origin_with_default_port_or_uppercase_matches_allowlist(
    incoming_origin: str,
) -> None:
    """The allowlist entry is canonicalized at startup (default port dropped, lowercased); the
    incoming ``Origin`` header must be canonicalized the same way before comparing, so a browser
    that sends a redundant default port or (never happens in practice, but the check must not
    depend on it) different case still matches."""
    client, calls = _counting_app(CorsPolicy((GOOD_ORIGIN,)))

    response = client.post("/v1/voices", headers={"Origin": incoming_origin})

    assert response.status_code == 200
    assert calls == ["post"]


def test_garbage_origin_header_is_treated_as_disallowed() -> None:
    """A malformed ``Origin`` header (never sent by a real browser) must never raise out of
    ``canonical_origin`` -- it's simply not allowed, same as any other disallowed origin."""
    client, calls = _counting_app(CorsPolicy((GOOD_ORIGIN,)))

    response = client.post("/v1/voices", headers={"Origin": "not a valid origin"})

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED
    assert calls == []
