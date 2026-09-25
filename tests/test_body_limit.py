"""`BodyLimitMiddleware` tests (specs/003-cpp-compatible-api/research.md R6).

Each test's docstring names the C++ behavior it rejects: BC-06 is the C++ server
having no body-size cap at all, so an oversized upload runs unbounded instead of
failing fast with a `413`.

A small limit (64 bytes) is injected so the tests don't need to move real megabytes.
The FastAPI app has `errors.install_error_handlers` wired in, since a body that grows
past the limit mid-stream is raised as `ApiError` from inside the wrapped app's own
call stack (see body_limit.py's module docstring) and only reaches that path, not the
middleware's own immediate-rejection path.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from breeze_infer.body_limit import BodyLimitMiddleware
from breeze_infer.errors import install_error_handlers

_LIMIT = 64


class _RecordingEvents:
    def emit(self, name: str, **fields: object) -> None:  # pragma: no cover - unused
        pass


def _client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app, _RecordingEvents())

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"received": len(body)}

    wrapped = BodyLimitMiddleware(app, limit=_LIMIT)
    return TestClient(wrapped, raise_server_exceptions=False)


def test_bc_06_content_length_over_the_limit_is_rejected_immediately() -> None:
    """BC-06: C++ has no cap, so this body would otherwise run to completion."""
    response = _client().post("/upload", content=b"x" * (_LIMIT + 1))

    assert response.status_code == 413
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }
    assert response.headers["content-type"].startswith("application/json")


def test_bc_06_chunked_body_over_the_limit_is_rejected_once_it_grows_past_it() -> None:
    """BC-06: a chunked body has no Content-Length, so the cap must count bytes."""

    def chunks() -> Iterator[bytes]:
        for _ in range((_LIMIT // 8) + 2):
            yield b"y" * 8

    response = _client().post("/upload", content=chunks())

    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }


def test_bc_06_body_under_the_limit_passes_through() -> None:
    body = b"z" * (_LIMIT - 1)

    response = _client().post("/upload", content=body)

    assert response.status_code == 200
    assert response.json() == {"received": len(body)}


def test_bc_06_body_exactly_at_the_limit_passes_through() -> None:
    body = b"w" * _LIMIT

    response = _client().post("/upload", content=body)

    assert response.status_code == 200
    assert response.json() == {"received": len(body)}
