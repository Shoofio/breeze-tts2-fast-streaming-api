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

import io
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from breeze_infer import __version__
from breeze_infer.api import Components, create_app
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.cors import CorsMiddleware, CorsPolicy
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import MAX_BODY_BYTES
from breeze_infer.routes_health import Readiness
from breeze_infer.settings import settings_from_args
from breeze_infer.version_header import VersionHeaderMiddleware
from tests.fakes import FakeRuntime

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
    # PUT, not POST/DELETE, so CORS's own unsafe-method check doesn't produce a 403 first --
    # this must reach Starlette's routing and come back as a plain wrong-method 405.
    wrong_method = client.put("/health", headers={"Origin": EVIL_ORIGIN})
    # GET (not POST/DELETE), so CORS's own unsafe-method 403 check doesn't pre-empt this: the
    # real `Content-Length` alone must trip BodyLimitMiddleware's immediate-rejection path.
    too_large = client.request(
        "GET", "/health", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Origin": EVIL_ORIGIN}
    )

    assert no_origin.headers["vary"] == "Origin"
    assert disallowed.status_code == 200
    assert disallowed.headers["vary"] == "Origin"
    assert not_found.status_code == 404
    assert not_found.headers["vary"] == "Origin"
    assert wrong_method.status_code == 405
    assert wrong_method.headers["vary"] == "Origin"
    assert too_large.status_code == 413
    assert too_large.headers["vary"] == "Origin"


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


def test_disallowed_preflight_gets_403(make_client: Callable[..., TestClient]) -> None:
    client = make_client(["--cors", GOOD_ORIGIN])

    response = client.options(
        "/health",
        headers={"Origin": EVIL_ORIGIN, "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 403
    assert response.json() == ORIGIN_NOT_ALLOWED


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
