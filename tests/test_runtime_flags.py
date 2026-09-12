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


def test_configure_compile_cache_exports_chosen_dir(tmp_path) -> None:
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

    saved = os.environ.get(CACHE_DIR_ENV)
    try:
        chosen = configure_compile_cache(settings)
        assert chosen == (tmp_path / "cc").resolve()
        assert os.environ[CACHE_DIR_ENV] == str(chosen)
    finally:
        if saved is None:
            os.environ.pop(CACHE_DIR_ENV, None)
        else:
            os.environ[CACHE_DIR_ENV] = saved
