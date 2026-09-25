"""Tests for breeze_infer.request_id: one id per request, on every response.

The middleware unit tests use a small FastAPI app; the wiring tests go through
`api.create_app`, so they also prove the middleware sits outside the body limit, CORS and
every error handler.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from breeze_infer.limits import MAX_BODY_BYTES
from breeze_infer.request_id import RequestIdMiddleware
from breeze_infer.routes_health import Readiness
from tests.fakes import RecordingEvents
from tests.test_routes_speech import (
    SPEECH_PATH,
    _build_components,
    _client_for,
    _fake_runtime,
)


def _counter_ids():
    counter = itertools.count(1)
    return lambda: f"req-{next(counter)}"


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request) -> dict[str, str]:
        return {"request_id": request.state.request_id}

    @app.get("/sets-its-own")
    async def sets_its_own() -> JSONResponse:
        return JSONResponse({}, headers={"X-Request-Id": "stale"})

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            yield b"b"

        return StreamingResponse(body())

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    return app


@pytest.fixture()
def client() -> TestClient:
    wrapped = RequestIdMiddleware(_app(), new_request_id=_counter_ids())
    return TestClient(wrapped, raise_server_exceptions=False)


def test_each_request_gets_its_own_id_in_state_and_header(client: TestClient) -> None:
    first = client.get("/echo")
    second = client.get("/echo")

    assert first.json() == {"request_id": "req-1"}
    assert first.headers["x-request-id"] == "req-1"
    assert second.headers["x-request-id"] == "req-2"


def test_an_inner_x_request_id_is_replaced_not_duplicated(client: TestClient) -> None:
    response = client.get("/sets-its-own")

    assert response.headers.get_list("x-request-id") == ["req-1"]


def test_streamed_and_failed_responses_carry_the_id(client: TestClient) -> None:
    assert client.get("/stream").headers["x-request-id"] == "req-1"
    boom = client.get("/boom")
    assert boom.status_code == 500
    assert boom.headers["x-request-id"] == "req-2"
    assert client.get("/nowhere").headers["x-request-id"] == "req-3"


# --- wired by api.create_app ------------------------------------------------------------------


@pytest.fixture()
def components():
    readiness = Readiness()
    events = RecordingEvents()
    comps = _build_components(readiness, events=events)
    yield comps
    comps.gpu.shutdown()


def test_every_kind_of_response_carries_x_request_id(components) -> None:
    components.readiness.mark_ready(_fake_runtime())
    client = _client_for(components)

    responses = [
        client.get("/health"),
        client.get("/no-such-route"),
        client.delete("/health"),
        client.post(SPEECH_PATH, data={"text": "hello", "cfg_scale": "banana"}),
        client.post(
            SPEECH_PATH,
            content=b"x" * (MAX_BODY_BYTES + 1),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
    ]

    assert [r.status_code for r in responses] == [200, 404, 405, 400, 413]
    ids = [r.headers.get_list("x-request-id") for r in responses]
    assert all(len(found) == 1 and found[0] for found in ids)
    assert len({found[0] for found in ids}) == len(ids)


def test_a_speech_response_and_its_events_share_one_id(components) -> None:
    components.readiness.mark_ready(_fake_runtime())

    response = _client_for(components).post(SPEECH_PATH, data={"text": "hello there"})

    assert response.status_code == 200
    assert response.headers.get_list("x-request-id") == [
        fields["request_id"]
        for name, fields in components.events.calls
        if name == "speech.accepted"
    ]
