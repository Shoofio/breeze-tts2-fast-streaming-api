"""Tests for breeze_infer.version_header (FR-037a)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from breeze_infer.version_header import VersionHeaderMiddleware

VERSION = "2.0.0.dev1"


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ok")
    def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    @app.get("/stream")
    def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"chunk-1"
            yield b"chunk-2"

        return StreamingResponse(body(), media_type="application/octet-stream")

    return app


@pytest.fixture()
def client() -> TestClient:
    wrapped = VersionHeaderMiddleware(_build_app(), version=VERSION)
    # raise_server_exceptions=False: Starlette's ServerErrorMiddleware sends
    # the 500 response from *inside* the app before re-raising, so the client
    # must not turn that re-raise into a test failure.
    return TestClient(wrapped, raise_server_exceptions=False)


def test_header_on_200(client: TestClient) -> None:
    response = client.get("/ok")
    assert response.status_code == 200
    assert response.headers["x-breeze-version"] == VERSION


def test_header_on_404(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.headers["x-breeze-version"] == VERSION


def test_header_on_500(client: TestClient) -> None:
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.headers["x-breeze-version"] == VERSION


def test_header_on_streamed_response(client: TestClient) -> None:
    response = client.get("/stream")
    assert response.status_code == 200
    assert response.headers["x-breeze-version"] == VERSION
    assert response.content == b"chunk-1chunk-2"


async def _send_413(
    scope: MutableMapping[str, Any],
    receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
    send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
) -> None:
    # Stands in for breeze_infer.body_limit, which lands separately (T015):
    # a tiny ASGI app that sends 413 itself, the same shape that middleware
    # will produce.
    del scope, receive
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": (
                b'{"error": "request body is too large", '
                b'"code": "payload_too_large"}'
            ),
        }
    )


def test_header_on_413() -> None:
    wrapped = VersionHeaderMiddleware(_send_413, version=VERSION)
    client = TestClient(wrapped)
    response = client.get("/anything")
    assert response.status_code == 413
    assert response.headers["x-breeze-version"] == VERSION
