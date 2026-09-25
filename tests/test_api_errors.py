"""Error-envelope tests (specs/003-cpp-compatible-api/contracts/http-api.md).

Each test's docstring names the C++ behavior it rejects: BC-18 is the C++ server
returning an empty, non-JSON body for a `404`/`405`/`500`, and inconsistent behavior
for a wrong-method request (some routes 400, some 404, depending on the framework
path that rejected them). Every status here comes back as
`{"error": "<message>", "code": "<code>"}` with `Content-Type: application/json`.

Uses a throwaway app with only `install_error_handlers` wired in, not the real
`breeze_infer.api` composition root: the behavior under test -- which handler catches
which exception, and what body it produces -- depends only on `errors.py`, not on any
other route or middleware. `raise_server_exceptions=False` lets a 500 response come
back as a response instead of re-raising the exception in the test process.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from breeze_infer.errors import ApiError, install_error_handlers
from tests.fakes import RecordingEvents


def _app(events: RecordingEvents | None = None) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app, events if events is not None else RecordingEvents())

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    @app.get("/api-error")
    async def api_error() -> None:
        raise ApiError(409, "busy", "busy")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _client(events: RecordingEvents | None = None) -> TestClient:
    return TestClient(_app(events), raise_server_exceptions=False)


def test_bc_18_unhandled_exception_returns_internal_error_envelope() -> None:
    """BC-18: C++ crashes leave the client with no body at all to parse."""
    response = _client().get("/boom")

    assert response.status_code == 500
    assert response.json() == {"error": "internal error", "code": "internal_error"}
    assert response.headers["content-type"].startswith("application/json")


def test_bc_18_unhandled_exception_does_not_leak_the_exception_message() -> None:
    """BC-18: the envelope must never surface internal detail (e.g. `str(exc)`)."""
    response = _client().get("/boom")

    assert "boom" not in response.text


def test_bc_18_unhandled_exception_emits_request_failed_with_a_request_id() -> None:
    """BC-18: unlike a C++ crash, a 500 here still has to leave an audit trail."""
    events = RecordingEvents()

    _client(events).get("/boom")

    assert len(events.calls) == 1
    name, fields = events.calls[0]
    assert name == "request.failed"
    assert fields["request_id"]  # generated, since /boom never sets one
    assert "boom" in fields["error"]  # repr(exc), kept out of the response only


def test_bc_18_api_error_carries_its_own_code() -> None:
    response = _client().get("/api-error")

    assert response.status_code == 409
    assert response.json() == {"error": "busy", "code": "busy"}


def test_bc_18_unknown_route_is_404_with_a_json_envelope() -> None:
    """BC-18: C++ returns an empty body for an unknown path; this always has JSON."""
    response = _client().get("/this-route-does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"error": "not found", "code": "not_found"}
    assert response.headers["content-type"].startswith("application/json")


def test_bc_18_wrong_method_is_405_with_allow_header_and_json_envelope() -> None:
    """BC-18: C++ returns 400 for a wrong method on some routes, 404 on others."""
    # DELETE isn't declared for /health (only GET is), so this exercises Starlette's
    # own 405 path, including the `Allow` header it attaches.
    response = _client().delete("/health")

    assert response.status_code == 405
    assert response.json() == {"error": "method not allowed", "code": "method_not_allowed"}
    assert "GET" in response.headers["allow"]


def test_http_exception_detail_is_ignored_for_mapped_statuses() -> None:
    # R2/R3: a bare `StarletteHTTPException` (never raised as such in this app for 400 --
    # Starlette's own multipart limits do, e.g. too many files) must use the fixed table's
    # message, not `exc.detail` -- even when the raiser set a specific, non-generic detail.
    app = _app()
    handler = app.exception_handlers[StarletteHTTPException]
    exc = StarletteHTTPException(status_code=400, detail="Too many files. Maximum number is 1.")

    response = asyncio.run(handler(None, exc))  # type: ignore[arg-type]

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_http_exception_with_unmapped_status_falls_back_to_a_500_internal_error() -> None:
    # R4: only 400/404/405/413 have entries in the fixed table; this app never itself raises a
    # bare `StarletteHTTPException` for any other status, but the handler still needs to do
    # something sane for one. That's a genuine 500 (not the original status paired with
    # `internal_error`) -- an uncovered status means something the contract doesn't otherwise
    # produce is going on, which is exactly what `internal_error` means elsewhere in this module.
    app = _app()
    handler = app.exception_handlers[StarletteHTTPException]
    exc = StarletteHTTPException(status_code=401, detail="should not leak")

    response = asyncio.run(handler(None, exc))  # type: ignore[arg-type]

    assert response.status_code == 500
    assert json.loads(response.body) == {"error": "internal error", "code": "internal_error"}


def test_bc_18_malformed_multipart_body_is_400_invalid_field() -> None:
    """BC-18: C++'s multipart parser has no error path at all for this input."""
    app = _app()

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, object]:
        form = await request.form()
        return {"keys": list(form.keys())}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/upload",
        content=b"not actually multipart--broken",
        headers={"content-type": "multipart/form-data; boundary=X"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_field"
    assert body["error"]  # python-multipart's own message, not paraphrased


def test_starlette_multipart_too_many_files_is_400_invalid_field() -> None:
    """R2/R3: Starlette's own `max_files` check raises a bare `HTTPException(400, detail=...)`
    (`starlette/requests.py`); the fixed table's message must win over Starlette's wording."""
    app = _app()

    @app.post("/upload-limited")
    async def upload_limited(request: Request) -> dict[str, object]:
        form = await request.form(max_files=1)
        return {"keys": list(form.keys())}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/upload-limited", files=[("f1", ("a.txt", b"aaa")), ("f2", ("b.txt", b"bbb"))]
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_starlette_multipart_oversized_part_is_400_invalid_field() -> None:
    """R2/R3: a part over `max_part_size` raises the same bare `HTTPException(400, ...)`."""
    app = _app()

    @app.post("/upload-limited")
    async def upload_limited(request: Request) -> dict[str, object]:
        form = await request.form(max_part_size=64 * 1024)
        return {"keys": list(form.keys())}

    client = TestClient(app, raise_server_exceptions=False)
    # `files={name: (None, value)}` forces multipart encoding for a plain (non-file) field, so
    # this exercises `max_part_size`, which doesn't apply to urlencoded bodies.
    response = client.post("/upload-limited", files={"field": (None, "x" * (70 * 1024))})

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_starlette_multipart_missing_boundary_is_400_invalid_field() -> None:
    """R2/R3: a `multipart/form-data` content-type with no boundary parameter raises the same
    bare `HTTPException(400, ...)`."""
    app = _app()

    @app.post("/upload-limited")
    async def upload_limited(request: Request) -> dict[str, object]:
        form = await request.form()
        return {"keys": list(form.keys())}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/upload-limited",
        content=b"anything",
        headers={"content-type": "multipart/form-data"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_bc_18_pydantic_validation_error_is_400_invalid_field() -> None:
    app = _app()

    @app.get("/typed")
    async def typed(n: int) -> dict[str, int]:
        return {"n": n}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/typed", params={"n": "not-an-int"})

    assert response.status_code == 400
    assert response.json() == {"error": "invalid request", "code": "invalid_field"}


def test_wrong_method_on_a_split_route_lists_the_union_of_every_matching_route_s_methods() -> None:
    """review issue 4 (CORS final review): FastAPI registers `@app.get(path)`/`@app.post(path)`
    on the same path as two separate `Route` objects, not one route with two methods. Starlette's
    own `Route.handle` only reports the *first* matching route's own `Allow` header
    (`starlette/routing.py`'s `Router.app` keeps just the first `Match.PARTIAL` route), silently
    hiding the other route's method; the handler must recompute `Allow` as the union across every
    route matching the path -- the same `route_methods_for_path` helper `cors.py`'s preflight
    handling uses (`breeze_infer/errors.py`), so both give the same answer for the same path.
    """
    app = _app()

    @app.get("/v1/voices")
    async def list_voices() -> dict:
        return {}

    @app.post("/v1/voices")
    async def create_voice() -> dict:
        return {}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.put("/v1/voices")

    assert response.status_code == 405
    assert response.json() == {"error": "method not allowed", "code": "method_not_allowed"}
    assert set(response.headers["allow"].split(", ")) == {"GET", "POST"}


def test_bc_18_every_error_response_is_application_json() -> None:
    for response in (
        _client().get("/boom"),
        _client().get("/this-route-does-not-exist"),
        _client().delete("/health"),
        _client().get("/api-error"),
    ):
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert set(body) == {"error", "code"}
