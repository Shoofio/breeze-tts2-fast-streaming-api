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
import http
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from breeze_infer.errors import ApiError, install_error_handlers


class _RecordingEvents:
    """A small fake for the `events` argument: records every emitted event."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def emit(self, name: str, **fields: object) -> None:
        self.calls.append((name, fields))


def _app(events: _RecordingEvents | None = None) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app, events if events is not None else _RecordingEvents())

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


def _client(events: _RecordingEvents | None = None) -> TestClient:
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
    events = _RecordingEvents()

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


def test_http_exception_with_generic_detail_falls_back_to_lowercased_phrase() -> None:
    # 413 has no hand-picked entry in `_DEFAULT_HTTP_CODES` (unlike 404/405), so a
    # generic-detail 413 -- raised with no explicit `detail`, the way a
    # framework-internal check would -- must fall back to computing a message from
    # `http.HTTPStatus`'s own phrase rather than a hard-coded string: the exact
    # phrase differs across Python versions (e.g. 413's phrase was renamed from
    # "Request Entity Too Large" to "Content Too Large"), so this asserts against
    # the phrase the *running* interpreter computes. In this app, 413 is actually
    # raised as `ApiError` by `body_limit.py`, never as a bare `HTTPException`; this
    # exercises the generic fallback path in isolation, using 413 only as a stand-in
    # status the `_DEFAULT_HTTP_CODES` map doesn't special-case.
    app = _app()
    handler = app.exception_handlers[StarletteHTTPException]
    phrase = http.HTTPStatus(413).phrase
    exc = StarletteHTTPException(status_code=413)
    assert exc.detail == phrase  # sanity: Starlette's own generic default

    response = asyncio.run(handler(None, exc))  # type: ignore[arg-type]

    assert response.status_code == 413
    assert json.loads(response.body) == {"error": phrase.lower(), "code": "http_error"}


def test_http_exception_with_specific_detail_is_passed_through() -> None:
    # A non-generic detail (e.g. a multipart parse error's own message) must reach
    # the client verbatim, not be replaced by the phrase fallback.
    app = _app()
    handler = app.exception_handlers[StarletteHTTPException]
    exc = StarletteHTTPException(status_code=400, detail="malformed multipart body")

    response = asyncio.run(handler(None, exc))  # type: ignore[arg-type]

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": "malformed multipart body",
        "code": "http_error",
    }


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


def test_bc_18_pydantic_validation_error_is_400_invalid_field() -> None:
    app = _app()

    @app.get("/typed")
    async def typed(n: int) -> dict[str, int]:
        return {"n": n}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/typed", params={"n": "not-an-int"})

    assert response.status_code == 400
    assert response.json() == {"error": "invalid request", "code": "invalid_field"}


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
