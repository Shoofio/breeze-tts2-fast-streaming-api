from __future__ import annotations

from pathlib import Path

import pytest

from breeze_infer import model_loading
from breeze_infer.compile_cache import CACHE_DIR_ENV
from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from breeze_infer.settings import settings_from_args
from models.fast_streaming import FastBreezeStreamingRuntime

MODEL_DIR = str(Path(__file__).parent)  # any existing directory; nothing loads it


def test_runtime_fast_properties_return_values_not_methods() -> None:
    runtime = FastBreezeStreamingRuntime.__new__(FastBreezeStreamingRuntime)
    runtime._fast_text_encoder = False
    runtime._fast_backbone_prefill = False
    runtime._fast_backbone_decode = False
    runtime._fast_depth_decoder = False
    runtime._fast_codec = False
    runtime._codec_chunk_frames = 2

    assert runtime.fast_enabled is False
    assert runtime.codec_chunk_frames == 2

    runtime._fast_codec = True
    runtime._codec_chunk_frames = 1

    assert runtime.fast_enabled is True
    assert runtime.codec_chunk_frames == 1


def test_configure_compile_cache_exports_chosen_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_loading, "pin_torch_key", lambda cache_dir: "skipped")
    environ: dict[str, str] = {}

    cache_dir, torch_key = model_loading.configure_compile_cache(tmp_path / "cc", environ)

    assert cache_dir == (tmp_path / "cc").resolve()
    assert environ[CACHE_DIR_ENV] == str(cache_dir)
    assert torch_key == "skipped"


def test_streaming_config_maps_the_fast_flags_from_settings() -> None:
    settings = settings_from_args(
        [MODEL_DIR, "--fast-backbone-decode", "--fast-codec", "--no-fast-text-encoder"]
    )

    config = model_loading.streaming_config(settings)

    assert config.fast_all is None
    assert config.stage_fast("backbone_decode") is True
    assert config.stage_fast("codec") is True
    assert config.stage_fast("text_encoder") is False
    assert config.stage_fast("backbone_prefill") is False
    assert config.max_new_tokens == MAX_NEW_TOKENS_CEILING
    assert config.max_seq_len == model_loading.MAX_SEQ_LEN == 2048


def test_streaming_config_fast_all_overrides_each_stage() -> None:
    settings = settings_from_args([MODEL_DIR, "--fast-all"])

    config = model_loading.streaming_config(settings)

    assert all(
        config.stage_fast(stage)
        for stage in ("text_encoder", "backbone_prefill", "backbone_decode", "depth_decoder", "codec")
    )


def test_load_model_passes_settings_and_device_to_load_runtime(
    tmp_path, monkeypatch
) -> None:
    class _Sentinel(Exception):
        pass

    recorded: dict[str, object] = {}

    def fake_load_runtime(model, *, device, attn_implementation):
        recorded.update(model=model, device=device, attn_implementation=attn_implementation)
        raise _Sentinel

    monkeypatch.setattr(model_loading, "pin_torch_key", lambda cache_dir: "skipped")
    monkeypatch.setattr(model_loading, "load_runtime", fake_load_runtime)
    settings = settings_from_args(
        [
            MODEL_DIR,
            "--attn-implementation",
            "sdpa",
            "--compile-cache-dir",
            str(tmp_path / "cc"),
        ]
    )

    with pytest.raises(_Sentinel):
        model_loading.load_model(settings, "cuda:1", {})

    assert recorded == {
        "model": Path(MODEL_DIR),
        "device": "cuda:1",
        "attn_implementation": "sdpa",
    }


def test_warmup_report_has_timing_and_cache_counters() -> None:
    manifest = {
        "total_elapsed_ms": 1234.567,
        "compile_cache": {
            "counters": {"inductor.fxgraph_cache_hit": 7, "inductor.fxgraph_cache_miss": 2}
        },
    }

    assert model_loading.warmup_report(manifest) == {
        "warmup_ms": 1234.57,
        "fx_graph_cache_hits": 7,
        "fx_graph_cache_misses": 2,
    }
