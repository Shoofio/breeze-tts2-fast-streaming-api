"""`/health`, readiness and the routes that must not exist (T023; FR-001, FR-036, BC-18, BC-24).

The app is built through `api.create_app`, so these cover the real wiring: error handlers,
middleware order and the version header. No GPU: the `GpuThread` gets a stub `set_device` and is
never used, and the runtime is `FakeRuntime`.
"""

import io
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from breeze_infer import __version__
from breeze_infer.api import Components, create_app
from breeze_infer.events import Emitter
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.routes_health import Readiness, install_health
from breeze_infer.settings import settings_from_args
from tests.fakes import FakeRuntime

LOADING = {"status": "loading", "error": "model is loading", "code": "loading"}
WS_PORT = 8081


def _components(readiness: Readiness) -> Components:
    # The model directory is never opened: nothing loads in these tests.
    return Components(
        settings=settings_from_args([str(Path(__file__).parent)]),
        events=Emitter(io.StringIO(), lambda: 0.0),
        gate=GpuGate(),
        gpu=GpuThread("cpu", lambda _device: None),
        readiness=readiness,
        ws_port=lambda: WS_PORT,
    )


@pytest.fixture()
def readiness() -> Readiness:
    return Readiness()


@pytest.fixture()
def client(readiness: Readiness) -> Iterator[TestClient]:
    components = _components(readiness)
    yield TestClient(create_app(components))
    components.gpu.shutdown()


@pytest.fixture()
def ready_client(readiness: Readiness, client: TestClient) -> TestClient:
    readiness.mark_ready(FakeRuntime())
    return client


def test_health_is_503_loading_until_the_model_is_ready(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == LOADING
    assert response.headers["content-type"].startswith("application/json")


def test_health_when_ready_has_exactly_status_sample_rate_and_ws_port(
    ready_client: TestClient,
) -> None:
    response = ready_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "sample_rate": 24000, "ws_port": WS_PORT}


def test_bc_24_ws_port_comes_from_the_provider(readiness: Readiness) -> None:
    """BC-24: `ws_port` is whatever is actually listening; 0 when nothing is."""
    readiness.mark_ready(FakeRuntime())
    components = replace(_components(readiness), ws_port=lambda: 0)
    try:
        body = TestClient(create_app(components)).get("/health").json()
    finally:
        components.gpu.shutdown()

    assert body["ws_port"] == 0


def test_head_health_is_supported_while_loading_and_when_ready(
    readiness: Readiness, client: TestClient
) -> None:
    loading = client.head("/health")
    readiness.mark_ready(FakeRuntime())
    ready = client.head("/health")

    assert loading.status_code == 503
    assert ready.status_code == 200
    assert ready.content == b""


def test_bc_18_unknown_path_is_404_envelope(ready_client: TestClient) -> None:
    response = ready_client.get("/nope")

    assert response.status_code == 404
    assert response.json() == {"error": "not found", "code": "not_found"}


@pytest.mark.parametrize("method", ["POST", "DELETE", "PUT"])
def test_bc_18_wrong_method_on_health_is_405_with_allow(
    ready_client: TestClient, method: str
) -> None:
    response = ready_client.request(method, "/health")

    assert response.status_code == 405
    assert response.json() == {"error": "method not allowed", "code": "method_not_allowed"}
    assert {m.strip() for m in response.headers["allow"].split(",")} == {"GET", "HEAD"}


def test_bc_18_options_with_cors_off_is_405_with_allow(ready_client: TestClient) -> None:
    """BC-18: C++ answered OPTIONS without CORS with 404; with CORS off it is a wrong method."""
    response = ready_client.options("/health", headers={"Origin": "http://example.test"})

    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/audio/convert"),
        ("GET", "/"),
        ("GET", "/app.js"),
        ("GET", "/style.css"),
        # FastAPI's generated docs are not part of the contract either.
        ("GET", "/docs"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
    ],
)
def test_fr_001_excluded_routes_are_404_envelope(
    ready_client: TestClient, method: str, path: str
) -> None:
    response = ready_client.request(method, path)

    assert response.status_code == 404
    assert response.json() == {"error": "not found", "code": "not_found"}


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/health"), ("HEAD", "/health"), ("GET", "/nope"), ("POST", "/health")],
)
def test_fr_037a_version_header_on_every_response(
    ready_client: TestClient, method: str, path: str
) -> None:
    response = ready_client.request(method, path)

    assert response.headers["x-breeze-version"] == __version__


def test_fr_037a_version_header_while_loading(client: TestClient) -> None:
    assert client.get("/health").headers["x-breeze-version"] == __version__


def test_require_ready_gives_other_routes_the_loading_503_then_the_runtime(
    readiness: Readiness,
) -> None:
    """Every non-health route depends on `require_ready` (FR-036)."""
    app = FastAPI()
    install_health(app, readiness, lambda: 0)

    @app.get("/probe")
    async def probe(
        runtime: Annotated[Any, Depends(readiness.require_ready)],
    ) -> dict[str, int]:
        return {"sample_rate": runtime.sample_rate}

    client = TestClient(app)
    loading = client.get("/probe")
    readiness.mark_ready(FakeRuntime())
    ready = client.get("/probe")

    assert loading.status_code == 503
    assert loading.json() == LOADING
    assert ready.json() == {"sample_rate": 24000}


GPU_UNAVAILABLE = {
    "status": "error",
    "error": "gpu is not responding",
    "code": "gpu_unavailable",
}


def test_health_is_503_gpu_unavailable_once_the_gpu_stops_responding(
    readiness: Readiness, ready_client: TestClient
) -> None:
    readiness.mark_unhealthy()
    response = ready_client.get("/health")

    assert response.status_code == 503
    assert response.json() == GPU_UNAVAILABLE


def test_unhealthy_wins_over_a_later_ready_and_over_loading(
    readiness: Readiness, client: TestClient
) -> None:
    readiness.mark_unhealthy()
    assert client.get("/health").json() == GPU_UNAVAILABLE
    readiness.mark_ready(FakeRuntime())
    assert client.get("/health").json() == GPU_UNAVAILABLE
    assert readiness.runtime is None
