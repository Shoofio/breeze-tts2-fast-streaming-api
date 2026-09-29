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
from tests.fakes import RecordingEvents

_LIMIT = 64


def _chunked(data: bytes, chunk_size: int = 8) -> Iterator[bytes]:
    """Split `data` into pieces so httpx sends it chunked (no `Content-Length` header)."""
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


def _oversize_multipart_body(field_value_len: int) -> tuple[bytes, str]:
    """A well-formed `multipart/form-data` body whose single field is large enough to push the
    request past `_LIMIT`, along with the matching `content-type` header value."""
    boundary = "xxxxBOUNDARYxxxx"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="field"\r\n\r\n'
        f"{'a' * field_value_len}"
        f"\r\n--{boundary}--\r\n"
    ).encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def _client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app, RecordingEvents())

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"received": len(body)}

    @app.post("/upload-form")
    async def upload_form(request: Request) -> dict[str, int]:
        form = await request.form()
        return {"fields": len(form)}

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
    """BC-06: a body under the upload limit is unaffected by the new cap -- this pins the
    boundary against the two rejection tests above.
    """
    body = b"z" * (_LIMIT - 1)

    response = _client().post("/upload", content=body)

    assert response.status_code == 200
    assert response.json() == {"received": len(body)}


def test_bc_06_body_exactly_at_the_limit_passes_through() -> None:
    """BC-06: a body exactly at the upload limit passes through -- only strictly over the
    limit (the C++ server's unbounded body) gets `413`.
    """
    body = b"w" * _LIMIT

    response = _client().post("/upload", content=body)

    assert response.status_code == 200
    assert response.json() == {"received": len(body)}


def test_r1_chunked_oversize_to_a_request_form_route_is_413() -> None:
    """R1: an `ApiError` raised from `counting_receive` while `request.form()` reads a chunked,
    oversized multipart body must still surface as `413`. This codebase never uses `Form(...)`
    parameters (research.md R1: endpoints take `Request` and parse forms themselves), so there's
    no FastAPI body-parsing code that could rewrap the exception -- a plain `request.form()` call
    inside a route is ordinary application code, and any exception it raises propagates to the
    installed handlers exactly like one raised anywhere else in the route."""
    body, content_type = _oversize_multipart_body(field_value_len=200)

    response = _client().post(
        "/upload-form", content=_chunked(body), headers={"content-type": content_type}
    )

    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }


def test_r3_both_413_paths_produce_identical_responses() -> None:
    """R8: `_reject_immediately` (the immediate-rejection path, `Content-Length` known upfront)
    and the `ApiError` handler (the counting-receive path, discovered mid-body) both build their
    response via `errors.api_error_response` from an identical `ApiError` -- so the two must come
    back status-for-status, byte-for-byte, header-for-header the same."""
    immediate = _client().post("/upload", content=b"x" * (_LIMIT + 1))
    counted = _client().post("/upload", content=_chunked(b"y" * (_LIMIT + 1)))

    assert immediate.status_code == counted.status_code == 413
    assert immediate.content == counted.content
    assert immediate.json() == counted.json()
    assert immediate.headers["content-type"] == counted.headers["content-type"]
    assert immediate.headers["content-length"] == counted.headers["content-length"]


def test_r6_malformed_content_length_falls_back_to_counting_the_body() -> None:
    """A `Content-Length` that isn't a valid integer must not be trusted at face value -- the
    body still has to be counted as it arrives, exactly like a chunked request, both under and
    over the limit."""
    under = b"z" * (_LIMIT - 1)
    response = _client().post(
        "/upload", content=under, headers={"content-length": "not-a-number"}
    )
    assert response.status_code == 200
    assert response.json() == {"received": len(under)}

    over = b"y" * (_LIMIT + 1)
    response = _client().post("/upload", content=over, headers={"content-length": "not-a-number"})
    assert response.status_code == 413
    assert response.json() == {
        "error": "request body is too large",
        "code": "payload_too_large",
    }
