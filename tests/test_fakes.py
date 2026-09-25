"""`tests/fakes.py`'s own behavior: the fakes are test infrastructure (Complexity Tracking,
Principle V), so a bug in them would silently corrupt every test that depends on them.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from models.fast_streaming import FastStreamingChunk
from tests.fakes import CODEC_CODEBOOK_SIZE, FakeCodec, FakeRuntime, codec_frame_count

# Long enough never to fire on a healthy run; only there so a bug hangs a test, not the suite.
TIMEOUT = 5.0


def _drain(runtime: FakeRuntime, **kwargs) -> list[FastStreamingChunk]:
    return list(runtime.iter_audio_chunks({"input_ids": np.zeros((1, 3))}, **kwargs))


def test_reports_sample_rate_24000() -> None:
    assert FakeRuntime().sample_rate == 24000


def test_records_seed_overrides_reference_and_prefix_per_piece() -> None:
    runtime = FakeRuntime(chunks=1)
    _drain(
        runtime,
        request_id="req-0",
        seed=7,
        reference="voice-codes",
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=300,
        prefix="cached-kv",
    )
    _drain(runtime, request_id="req-0", seed=8, reference="voice-codes")

    assert len(runtime.calls) == 2
    first, second = runtime.calls
    assert first["seed"] == 7
    assert first["reference"] == "voice-codes"
    assert first["temperature"] == 0.8
    assert first["top_k"] == 50
    assert first["top_p"] == 0.9
    assert first["repetition_penalty"] == 1.2
    assert first["max_new_tokens"] == 300
    assert first["prefix"] == "cached-kv"
    # Piece 1 reuses the same (identical) reference object rather than re-encoding it,
    # which is exactly what a test for T038's "inline reference encoded once" needs to see.
    assert second["seed"] == 8
    assert second["reference"] is first["reference"]


def test_chunks_are_shaped_like_the_real_streaming_chunk() -> None:
    runtime = FakeRuntime(chunks=3, samples=480)
    chunks = _drain(runtime)

    assert len(chunks) == 3
    assert all(isinstance(chunk, FastStreamingChunk) for chunk in chunks)
    assert all(chunk.audio.dtype == np.float32 for chunk in chunks)
    assert all(chunk.audio.shape == (480,) for chunk in chunks)
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    # Different calls (pieces) are told apart by their constant sample value.
    other_piece = _drain(runtime)
    assert chunks[0].audio[0] == 0.0
    assert other_piece[0].audio[0] == pytest.approx(0.01)


def test_gate_holds_generation_mid_piece_until_released() -> None:
    gate = threading.Event()
    gate_reached = threading.Event()
    runtime = FakeRuntime(chunks=3, gate=gate, gate_reached=gate_reached)
    seen: list[int] = []

    def consume() -> None:
        for index, _chunk in enumerate(runtime.iter_audio_chunks({})):
            seen.append(index)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert gate_reached.wait(TIMEOUT)
        # Chunk 0 (before the gate, at index gate_at=1) is out; chunk 1 is held back.
        assert seen == [0]
        assert thread.is_alive()
    finally:
        gate.set()
        thread.join(TIMEOUT)
    assert not thread.is_alive()
    assert seen == [0, 1, 2]


def test_fail_after_n_raises_after_n_chunks_and_still_records_close() -> None:
    runtime = FakeRuntime(chunks=5, fail_after=2)
    gen = runtime.iter_audio_chunks({})

    collected = [next(gen), next(gen)]
    assert len(collected) == 2
    with pytest.raises(RuntimeError, match="CUDA error"):
        next(gen)
    assert runtime.closed == 1


def test_generator_close_is_recorded() -> None:
    runtime = FakeRuntime(chunks=5)
    gen = runtime.iter_audio_chunks({})
    next(gen)
    assert runtime.closed == 0
    gen.close()
    assert runtime.closed == 1


def test_codec_frame_count_matches_the_resample_then_frame_formula() -> None:
    # No resampling: a plain ceiling division by the 1920-sample frame.
    assert codec_frame_count(1920, 24000) == 1
    assert codec_frame_count(2400, 24000) == 2  # 2400 / 1920 = 1.25 -> 2
    # Resampling first (librosa's own ceil(len * target_sr / orig_sr)), then framing:
    # 3840 samples @ 48 kHz -> ceil(3840 * 24000 / 48000) = 1920 -> 1 frame exactly.
    assert codec_frame_count(3840, 48000) == 1
    # One sample over that boundary pushes the resampled count past 1920, so it costs a
    # whole extra frame: ceil(3841 * 24000 / 48000) = 1921 -> ceil(1921 / 1920) = 2.
    assert codec_frame_count(3841, 48000) == 2
    assert codec_frame_count(0, 24000) == 0


def test_codec_encode_is_deterministic_shaped_and_in_range() -> None:
    codec = FakeCodec()
    wav = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)

    codes_a = codec.encode(wav, 24000)
    codes_b = codec.encode(wav, 24000)

    assert codes_a.dtype == np.int16
    assert codes_a.shape == (codec_frame_count(4800, 24000), FakeCodec.CODEBOOKS)
    assert np.array_equal(codes_a, codes_b)
    assert codes_a.min() >= 0
    assert codes_a.max() < CODEC_CODEBOOK_SIZE
    assert codec.encode_calls == 2
    assert codec.last_sr == 24000

    different_wav = wav + 1.0
    codes_c = codec.encode(different_wav, 24000)
    assert not np.array_equal(codes_a, codes_c)
