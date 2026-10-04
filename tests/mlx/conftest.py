from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mlx_runtime(mlx_model: Path):
    """Load the MLX runtime once per session: one model per process, since a 16 GB Mac has
    no room for two."""
    from models.mlx_streaming import load_mlx_runtime

    return load_mlx_runtime(mlx_model)
