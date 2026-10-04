from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mlx_runtime(mlx_model: Path):
    """Load the MLX runtime once per session.

    The loader arrives with the runtime module, so until then tests that need
    the runtime skip instead of failing.
    """
    pytest.skip("T014")
