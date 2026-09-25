"""Tests for breeze_infer.version_header (FR-037a)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from breeze_infer import __version__ as VERSION
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.errors import install_error_handlers
from breeze_infer.version_header import VersionHeaderMiddleware


class _RecordingEvents:
    def emit(self, name: str, **fields: object) -> None:  # pragma: no cover - unused
        pass


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
    # A minimal hand-rolled ASGI app -- not breeze_infer.body_limit, which is exercised for real
    # by test_header_on_real_chunked_413_path below -- kept here to test the header logic on a
    # 413 in isolation from any particular producer of one.
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


def test_version_is_a_required_argument() -> None:
    """V4: there's no sane default version string, so a caller must supply one."""
    with pytest.raises(TypeError):
        VersionHeaderMiddleware(_build_app())  # type: ignore[call-arg]


def test_header_on_real_chunked_413_path() -> None:
    """V5: the header must reach the client on the actual `BodyLimitMiddleware` 413 path, raised
    as `ApiError` from `counting_receive` mid-body and turned into a response by
    `errors.install_error_handlers` -- not just on a hand-rolled stand-in."""
    app = FastAPI()
    install_error_handlers(app, _RecordingEvents())

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"received": len(body)}

    limited = BodyLimitMiddleware(app, limit=8)
    wrapped = VersionHeaderMiddleware(limited, version=VERSION)
    client = TestClient(wrapped, raise_server_exceptions=False)

    def chunks() -> Iterator[bytes]:
        for _ in range(4):
            yield b"y" * 8

    response = client.post("/upload", content=chunks())

    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413
    assert response.headers["x-breeze-version"] == VERSION
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }
