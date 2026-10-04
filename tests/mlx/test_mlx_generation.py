"""The MLX runtime's `iter_audio_chunks` on the real checkpoint (contracts/runtime-seam.md).
Needs BREEZE_MLX_MODEL on Apple Silicon."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.synthesis import NoRef, prepare_piece
from models.cudagraph.sampling import NonFiniteLogitsError

pytestmark = pytest.mark.mlx

SENTENCE = "Hello there, this is a short test of the streaming voice."
THREE_SENTENCES = (
    "The lighthouse keeper climbed the stairs at dusk. "
    "He trimmed the wick and polished the lens. "
    "Ships passed far out on the dark water."
)
# Short runs keep the suite fast; 24 frames is about two seconds of audio.
SHORT = 24


def inputs_for(runtime, text: str, *, instruction: str = DEFAULT_INSTRUCTION, cfg_scale: float = 1.0):
    """The inputs the server builds for a piece with no reference."""
    return prepare_piece(runtime.tokenizer, runtime.model, NoRef(), text, instruction, cfg_scale)


def audio_of(chunks) -> np.ndarray:
    return np.concatenate([chunk.audio for chunk in chunks])


def test_audio_streams_in_several_chunks(mlx_runtime) -> None:
    chunks = list(mlx_runtime.iter_audio_chunks(inputs_for(mlx_runtime, THREE_SENTENCES), seed=1))
    assert len(chunks) > 1
    assert not chunks[0].is_final
    assert chunks[-1].is_final
    for chunk in chunks:
        assert chunk.audio.dtype == np.float32 and chunk.audio.ndim == 1
        assert chunk.sample_rate == 24000
        assert chunk.audio.shape[0] == chunk.codec_frames * 1920


def test_same_seed_gives_the_same_audio(mlx_runtime) -> None:
    inputs = inputs_for(mlx_runtime, SENTENCE)

    def run(seed: int) -> np.ndarray:
        return audio_of(mlx_runtime.iter_audio_chunks(inputs, seed=seed, max_new_tokens=SHORT))

    first = run(7)
    assert np.array_equal(first, run(7))
    other = run(8)
    assert first.shape != other.shape or not np.array_equal(first, other)


def test_close_aborts_quickly_and_the_next_request_works(mlx_runtime) -> None:
    chunks = mlx_runtime.iter_audio_chunks(inputs_for(mlx_runtime, THREE_SENTENCES), seed=3)
    for _ in range(3):
        next(chunks)
    started = time.perf_counter()
    chunks.close()
    elapsed = time.perf_counter() - started
    print(f"close() took {elapsed * 1000:.1f} ms")
    assert elapsed < 1.0
    after = list(mlx_runtime.iter_audio_chunks(inputs_for(mlx_runtime, SENTENCE), seed=3, max_new_tokens=SHORT))
    assert after and audio_of(after).size > 0


def test_max_new_tokens_caps_the_frames(mlx_runtime) -> None:
    observed: list[torch.Tensor] = []
    chunks = list(
        mlx_runtime.iter_audio_chunks(
            inputs_for(mlx_runtime, THREE_SENTENCES), seed=5, max_new_tokens=SHORT, token_observer=observed.append
        )
    )
    assert len(observed) <= SHORT
    assert sum(chunk.codec_frames for chunk in chunks) <= SHORT


def test_token_observer_sees_every_frame(mlx_runtime) -> None:
    observed: list[torch.Tensor] = []
    chunks = list(
        mlx_runtime.iter_audio_chunks(
            inputs_for(mlx_runtime, SENTENCE), seed=11, max_new_tokens=SHORT, token_observer=observed.append
        )
    )
    codebooks = mlx_runtime.model.config.num_codebooks
    # The backbone's code plus the depth decoder's, as CUDA's `torch.cat([token, depth_tokens])`.
    assert all(frame.dtype == torch.long and tuple(frame.shape) == (codebooks,) for frame in observed)
    # The backbone never samples the pad id, so every observed frame is decoded.
    assert len(observed) == sum(chunk.codec_frames for chunk in chunks)
    codes = torch.stack(observed)
    assert 0 <= int(codes.min()) and int(codes.max()) < mlx_runtime.model.config.codec_config.codebook_size


def test_cfg_with_an_instruction_produces_audio(mlx_runtime) -> None:
    inputs = inputs_for(mlx_runtime, SENTENCE, instruction="A calm, warm voice.", cfg_scale=4.0)
    assert inputs["cfg_scale"] == 4.0 and "cfg_negative_prompt_ids" in inputs
    chunks = list(mlx_runtime.iter_audio_chunks(inputs, seed=2, max_new_tokens=SHORT))
    assert audio_of(chunks).size > 0


def test_cfg_scale_one_runs_without_a_negative_prompt(mlx_runtime) -> None:
    inputs = inputs_for(mlx_runtime, SENTENCE, cfg_scale=1.0)
    assert not [key for key in inputs if key.startswith("cfg_negative_")]
    chunks = list(mlx_runtime.iter_audio_chunks(inputs, seed=2, max_new_tokens=SHORT))
    assert audio_of(chunks).size > 0


def test_cfg_scale_zero_runs_the_negative_prompt_alone(mlx_runtime) -> None:
    inputs = inputs_for(mlx_runtime, SENTENCE, cfg_scale=0.0)
    chunks = list(mlx_runtime.iter_audio_chunks(inputs, seed=2, max_new_tokens=SHORT))
    assert audio_of(chunks).size > 0


def test_dual_cfg_is_rejected_on_the_first_next(mlx_runtime) -> None:
    inputs = {**inputs_for(mlx_runtime, SENTENCE), "cfg_scale_ref": 2.0}
    chunks = mlx_runtime.iter_audio_chunks(inputs)
    with pytest.raises(ValueError, match="dual CFG"):
        next(chunks)


@pytest.mark.parametrize(
    ("part", "message"),
    [
        ("backbone", "backbone logits contain NaN or +inf, or no finite value; cannot sample"),
        ("depth", "depth decoder logits contain NaN or +inf, or no finite value; cannot sample"),
    ],
)
def test_nonfinite_logits_fail_before_the_frame_is_observed(mlx_runtime, monkeypatch, part, message) -> None:
    """A NaN norm weight (mlx-audio's module, restored after the test) makes that part's logits
    NaN; the request fails with CUDA's error before anything sees the frame."""
    import mlx.core as mx

    model = mlx_runtime._mlx_model
    norm = model.backbone_model.norm if part == "backbone" else model.depth_decoder.model.norm
    monkeypatch.setitem(norm, "weight", mx.full(norm.weight.shape, mx.nan, norm.weight.dtype))
    observed: list[torch.Tensor] = []
    chunks = mlx_runtime.iter_audio_chunks(
        inputs_for(mlx_runtime, SENTENCE), seed=4, max_new_tokens=SHORT, token_observer=observed.append
    )
    with pytest.raises(NonFiniteLogitsError) as raised:
        next(chunks)
    assert str(raised.value) == message
    assert observed == []


def test_first_chunk_is_quick_after_warmup(mlx_runtime) -> None:
    warmup_ms = mlx_runtime.warmup()
    started = time.perf_counter()
    chunks = mlx_runtime.iter_audio_chunks(inputs_for(mlx_runtime, SENTENCE), seed=6)
    next(chunks)
    first_chunk = time.perf_counter() - started
    chunks.close()
    print(f"warmup {warmup_ms:.0f} ms, time to first chunk {first_chunk * 1000:.0f} ms")
    assert warmup_ms > 0
    assert first_chunk < 2.0
