"""The MLX speed settings (`MlxSpeedOptions`) on the real checkpoint. Needs BREEZE_MLX_MODEL on Apple Silicon."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.synthesis import NoRef, prepare_piece
from models.fast_streaming import select_fast_cfg
from models.mlx_streaming import (
    _CONDITIONAL,
    _NEGATIVE,
    _frame_function,
    _Generation,
    _to_mlx_prompt,
)

pytestmark = pytest.mark.mlx

SENTENCE = "Hello there, this is a short test of the streaming voice."
FRAMES = 12


def _inputs(runtime, cfg_scale: float):
    instruction = "Speak warmly and gently." if cfg_scale != 1.0 else DEFAULT_INSTRUCTION
    return prepare_piece(runtime.tokenizer, runtime.model, NoRef(), SENTENCE, instruction, cfg_scale)


def _generation(runtime, inputs, *, compile_frame: bool = False) -> _Generation:
    cfg = select_fast_cfg(inputs)
    rows = [_CONDITIONAL, _NEGATIVE] if cfg.mode == "single_cfg" else [_CONDITIONAL]
    model = runtime._mlx_model
    return _Generation(
        model,
        [_to_mlx_prompt(inputs, keys, model.config) for keys in rows],
        frames=FRAMES + 1,
        guidance=cfg.guidance_scale,
        backbone_sampling=runtime._backbone_sampling,
        depth_sampling=runtime._depth_sampling,
        repetition_penalty=runtime.config.repetition_penalty,
        seed=42,
        compile_frame=compile_frame,
    )


@pytest.mark.parametrize("cfg_scale", [1.0, 2.0], ids=["plain", "cfg"])
def test_compiled_frame_matches_the_uncompiled_graph(mlx_runtime, cfg_scale):
    """The same forced codes through the compiled and the uncompiled frame graph. Under CFG the depth logits
    are identical; on a plain line the fused graph rounds differently in bf16 (measured: KL ~6e-5, top-1 99.4 %
    on the bf16 checkpoint), far below what the int8 depth decoder changes (KL ~2e-3)."""
    import mlx.core as mx

    generation = _generation(mlx_runtime, _inputs(mlx_runtime, cfg_scale))
    settings = (mlx_runtime._backbone_sampling, mlx_runtime._depth_sampling, mlx_runtime.config.repetition_penalty)
    model = mlx_runtime._mlx_model
    draw = _frame_function(model, *settings, compiled=False)
    plain = _frame_function(model, *settings, forced=True, compiled=False)
    compiled = _frame_function(model, *settings, forced=True)
    kls, agree = [], []
    for _ in range(FRAMES):
        key = generation._next_key()
        guidance = mx.array(generation._guidance, dtype=mx.float32)
        codes = draw(generation._hidden, generation._seen, guidance, key)
        _, a = plain(generation._hidden, generation._seen, guidance, key, codes)
        _, b = compiled(generation._hidden, generation._seen, guidance, key, codes)
        if cfg_scale != 1.0:
            np.testing.assert_array_equal(np.array(a), np.array(b))
        la = a - mx.logsumexp(a, axis=-1, keepdims=True)
        lb = b - mx.logsumexp(b, axis=-1, keepdims=True)
        finite = mx.isfinite(la)
        kls += np.array(mx.sum(mx.where(finite, mx.exp(la) * (la - mx.where(finite, lb, 0)), 0), axis=-1)).tolist()
        agree += np.array(mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)).tolist()
        generation.advance(codes[:-1])
    assert max(kls) < 5e-3
    assert np.mean(agree) >= 0.98


def test_compiled_frames_stream_audio(mlx_runtime):
    speed = dataclasses.replace(mlx_runtime.speed, compile_frame=True)
    previous, mlx_runtime.speed = mlx_runtime.speed, speed
    try:
        chunks = list(mlx_runtime.iter_audio_chunks(_inputs(mlx_runtime, 2.0), seed=42, max_new_tokens=FRAMES))
    finally:
        mlx_runtime.speed = previous
    audio = np.concatenate([c.audio for c in chunks])
    assert audio.size > 0 and np.isfinite(audio).all()
    assert chunks[-1].is_final
