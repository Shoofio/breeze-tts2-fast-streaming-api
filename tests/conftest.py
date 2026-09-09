from __future__ import annotations

import os
from pathlib import Path

import pytest

GPU_ENV = "BREEZE_MODEL"


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get(GPU_ENV):
        return
    skip = pytest.mark.skip(reason=f"set {GPU_ENV}=/path/to/checkpoint to run gpu tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def breeze_model() -> Path:
    value = os.environ.get(GPU_ENV)
    if not value:
        pytest.skip(f"{GPU_ENV} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{GPU_ENV}={value} is not a directory")
    return path
