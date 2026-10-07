"""`MlxSpeedOptions` (the MLX runtime's opt-in speed settings) and `quantize_parts`. Model-free;
the quantization test needs mlx (Apple Silicon) and is skipped elsewhere."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.mlx_streaming import MlxSpeedOptions, quantize_parts


def test_defaults_change_nothing():
    speed = MlxSpeedOptions.from_env({})
    assert speed == MlxSpeedOptions()
    assert speed.quantize_parts() == []
    assert speed.report() == {"quantize": None, "cache_limit_gb": None, "compile_frame": False, "fast_first_frames": 0}


def test_from_env_reads_every_setting():
    speed = MlxSpeedOptions.from_env(
        {"BREEZE_MLX_QUANT": " depth:8, backbone:4 ", "BREEZE_MLX_GROUP": "32", "BREEZE_MLX_CACHE_GB": "1.5",
         "BREEZE_MLX_COMPILE": "1", "BREEZE_MLX_FAST_FIRST": "6"}
    )
    assert speed.quantize_parts() == [("depth", 8), ("backbone", 4)]
    assert speed.group_size == 32
    assert speed.cache_limit_gb == 1.5
    assert speed.compile_frame
    assert speed.fast_first_frames == 6


@pytest.mark.parametrize("spec", ["depth", "depth:eight", "text_encoder:8", "depth:7", "depth:16"])
def test_bad_quantize_specs_are_refused(spec):
    with pytest.raises(ValueError, match="BREEZE_MLX_QUANT"):
        MlxSpeedOptions(quantize=spec)


def test_quantize_parts_touches_only_the_named_part():
    pytest.importorskip("mlx.core")
    from mlx import nn

    model = SimpleNamespace(
        backbone_model=nn.Sequential(nn.Linear(64, 64)),
        depth_decoder=nn.Sequential(nn.Linear(64, 64), nn.Embedding(32, 64)),
    )
    quantize_parts(model, MlxSpeedOptions(quantize="depth:8"))
    assert isinstance(model.depth_decoder.layers[0], nn.QuantizedLinear)
    assert model.depth_decoder.layers[0].bits == 8
    assert type(model.depth_decoder.layers[1]) is nn.Embedding  # lookups stay exact
    assert type(model.backbone_model.layers[0]) is nn.Linear
