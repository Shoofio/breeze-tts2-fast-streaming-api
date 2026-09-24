from __future__ import annotations

from models.fast_streaming import FastBreezeStreamingRuntime


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
    import os

    from breeze_infer.api import ApiSettings, configure_compile_cache
    from breeze_infer.compile_cache import CACHE_DIR_ENV

    settings = ApiSettings(
        model=tmp_path / "model",
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        compile_cache_dir=tmp_path / "cc",
    )

    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(
        "breeze_infer.api.pin_torch_key", lambda cache_dir: "skipped"
    )
    configured = configure_compile_cache(settings)
    assert configured.compile_cache_dir == (tmp_path / "cc").resolve()
    assert os.environ[CACHE_DIR_ENV] == str(configured.compile_cache_dir)


def test_load_app_passes_attn_implementation_to_load_runtime(
    tmp_path, monkeypatch
) -> None:
    import pytest
    from fastapi import FastAPI

    from breeze_infer import api

    class _Sentinel(Exception):
        pass

    recorded: dict[str, object] = {}

    def fake_load_runtime(model, *, device, attn_implementation):
        recorded["attn_implementation"] = attn_implementation
        raise _Sentinel

    monkeypatch.setattr(api, "load_runtime", fake_load_runtime)

    settings = api.ApiSettings(
        model=tmp_path / "model",
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        attn_implementation="sdpa",
    )
    with pytest.raises(_Sentinel):
        api._load_app(FastAPI(), settings)

    assert recorded["attn_implementation"] == "sdpa"


def test_api_settings_default_to_eager_attention(tmp_path) -> None:
    from breeze_infer.api import ApiSettings

    settings = ApiSettings(
        model=tmp_path / "model",
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
    )
    assert settings.attn_implementation == "eager"
