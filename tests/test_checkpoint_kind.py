"""`checkpoint_kind`: what a checkpoint's config.json says about its format and precision."""

from __future__ import annotations

from typing import Any

import pytest

from breeze_infer.settings import CheckpointKind, checkpoint_kind

MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "breeze"}, CheckpointKind("pytorch", "bf16")),
        ({"model_type": "breeze_tts"}, CheckpointKind("mlx", "bf16")),
        ({"model_type": "breeze_tts", "quantization": MXFP8}, CheckpointKind("mlx", "8bit")),
    ],
)
def test_supported_checkpoints(config: dict[str, Any], expected: CheckpointKind) -> None:
    assert checkpoint_kind(config) == expected


def test_unsupported_quantization_is_refused() -> None:
    config = {"model_type": "breeze_tts", "quantization": MXFP4}
    with pytest.raises(ValueError) as refused:
        checkpoint_kind(config, "snap")
    assert str(refused.value) == (
        "snap is 4-bit mxfp4; the MLX backend supports bf16 and 8-bit (mxfp8)"
    )


@pytest.mark.parametrize("config", [{"model_type": "llama"}, {}])
def test_unknown_model_type_is_refused(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        checkpoint_kind(config)
