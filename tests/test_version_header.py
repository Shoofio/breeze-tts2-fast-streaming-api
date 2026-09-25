"""Tests for breeze_infer.version_header (FR-037a)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from breeze_infer import __version__ as VERSION
from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.errors import install_error_handlers
from breeze_infer.version_header import VersionHeaderMiddleware
from tests.fakes import RecordingEvents


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
    receive: Receive,
    send: Send,
) -> None:
    # A minimal hand-rolled ASGI app -- not breeze_infer.body_limit, which is exercised for real
    # by test_header_on_real_chunked_413_path/test_header_on_real_content_length_413_path below
    # -- kept here to test the header logic on a 413 in isolation from any particular producer.
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


def test_version_must_be_printable_ascii_without_control_characters() -> None:
    """V5: a version string is stamped straight into a header value, so it must be restricted to
    printable, non-space ASCII -- no CR/LF (header injection), no other control characters, and
    no empty string."""
    for bad in ("", "has space", "has\ttab", "has\r\ninjection", "has\x00nul"):
        with pytest.raises(ValueError):
            VersionHeaderMiddleware(_build_app(), version=bad)


def test_replaces_mixed_case_duplicate_version_headers_with_exactly_one() -> None:
    """V1/V7: an inner layer might send more than one existing header under different casing
    (ASGI header names are conventionally lowercase, but nothing enforces that) -- all of them
    must be replaced, leaving exactly one `x-breeze-version` header."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"X-Breeze-Version", b"stale-upper"),
                    (b"x-breeze-version", b"stale-lower"),
                    (b"content-type", b"text/plain"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    wrapped = VersionHeaderMiddleware(app, version=VERSION)
    client = TestClient(wrapped)

    response = client.get("/anything")

    values = [value for name, value in response.headers.raw if name.lower() == b"x-breeze-version"]
    assert values == [VERSION.encode("ascii")]


def test_wrap_preserves_other_response_headers() -> None:
    """V8: filtering out existing `x-breeze-version` entries must not disturb any other header
    the inner app set -- name, value, and the rest of the header list are untouched."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"x-custom-marker", b"keep-me"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b"hi"})

    wrapped = VersionHeaderMiddleware(app, version=VERSION)
    client = TestClient(wrapped)

    response = client.get("/anything")

    assert response.headers["content-type"] == "text/plain"
    assert response.headers["x-custom-marker"] == "keep-me"
    assert response.headers["x-breeze-version"] == VERSION


def test_pre_response_exception_gets_a_versioned_500_and_is_not_reraised() -> None:
    """V2: an exception from a layer below this middleware, but before any response has started,
    must not propagate as a bare connection failure -- this middleware is the outermost layer,
    so it's the last chance to attach the version header."""

    async def broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send
        raise RuntimeError("boom-before-response")

    wrapped = VersionHeaderMiddleware(broken_app, version=VERSION)
    client = TestClient(wrapped)  # nothing re-raises, so the default (raise) setting is fine

    response = client.get("/anything")

    assert response.status_code == 500
    assert response.json() == {"error": "internal error", "code": "internal_error"}
    assert response.headers["x-breeze-version"] == VERSION


def test_mid_stream_exception_after_response_started_still_propagates() -> None:
    """V2: once the response has started, a failure must still propagate uncaught (research.md
    R2) -- swallowing it here would turn a broken mid-stream transfer into a falsely-clean
    response. Driven at the raw ASGI level (not through TestClient) so the re-raise itself, not
    just its eventual effect on the connection, is directly observable."""

    async def broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom-mid-stream")

    wrapped = VersionHeaderMiddleware(broken_app, version=VERSION)
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    with pytest.raises(RuntimeError, match="boom-mid-stream"):
        asyncio.run(wrapped({"type": "http", "headers": []}, receive, send))

    # Only the original http.response.start went out -- no fabricated 500 sent after it.
    assert len(sent) == 1
    assert sent[0]["type"] == "http.response.start"


def test_header_on_real_chunked_413_path() -> None:
    """V4: the header must reach the client on the actual `BodyLimitMiddleware` 413 path, raised
    as `ApiError` from `counting_receive` mid-body and turned into a response by
    `errors.install_error_handlers` -- not just on a hand-rolled stand-in."""
    app = FastAPI()
    install_error_handlers(app, RecordingEvents())

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


def test_header_on_real_content_length_413_path() -> None:
    """V4: the header must also reach the client on `BodyLimitMiddleware`'s other 413 path --
    the immediate rejection when `Content-Length` is already known to be over the limit, before
    the wrapped app even runs."""
    app = FastAPI()
    install_error_handlers(app, RecordingEvents())

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"received": len(body)}

    limited = BodyLimitMiddleware(app, limit=8)
    wrapped = VersionHeaderMiddleware(limited, version=VERSION)
    client = TestClient(wrapped, raise_server_exceptions=False)

    response = client.post("/upload", content=b"x" * 9)

    assert response.status_code == 413
    assert response.headers["x-breeze-version"] == VERSION
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }
