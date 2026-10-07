"""`patch_codec_overlap_bias`: mlx-audio's streaming codec decode matches a one-shot decode at any
chunk size. Model-free; needs mlx and mlx-audio (Apple Silicon), skipped elsewhere."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
speech_tokenizer = pytest.importorskip("mlx_audio.tts.models.qwen3_tts.speech_tokenizer")

from models.mlx_streaming import patch_codec_overlap_bias

CHUNKINGS = ([1] * 12, [2] * 6, [3, 1, 4, 4], [12])


def _block(upsample_rate: int):
    mx.random.seed(0)
    block = speech_tokenizer.DecoderBlockUpsample(in_dim=6, out_dim=5, upsample_rate=upsample_rate)
    # A non-zero bias: a zero bias hides a doubled one.
    block.conv.bias = mx.random.normal(block.conv.bias.shape)
    return block


def _streamed(block, x, chunk_sizes):
    block.reset_state()
    out, start = [], 0
    for size in chunk_sizes:
        out.append(block.step(x[:, start : start + size, :]))
        start += size
    return mx.concatenate(out, axis=1)


@pytest.mark.parametrize("upsample_rate", [2, 4, 8])
@pytest.mark.parametrize("chunk_sizes", CHUNKINGS, ids=lambda c: "-".join(map(str, c)))
def test_streamed_decode_matches_one_call(upsample_rate, chunk_sizes):
    patch_codec_overlap_bias()
    block = _block(upsample_rate)
    x = mx.random.normal((1, 12, 6))
    np.testing.assert_allclose(np.array(_streamed(block, x, chunk_sizes)), np.array(block(x)), atol=1e-5)


def test_patch_is_idempotent():
    patch_codec_overlap_bias()
    step = speech_tokenizer.DecoderBlockUpsample.step
    patch_codec_overlap_bias()
    assert speech_tokenizer.DecoderBlockUpsample.step is step
