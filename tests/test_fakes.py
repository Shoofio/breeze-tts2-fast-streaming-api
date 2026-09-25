"""`tests/fakes.py`'s own behavior: the fakes are test infrastructure (Complexity Tracking,
Principle V), so a bug in them would silently corrupt every test that depends on them.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from models.fast_streaming import FastStreamingChunk
from tests.fakes import (
    CODEC_CODEBOOK_SIZE,
    CODEC_SAMPLES_PER_FRAME,
    FakeCodec,
    FakeRuntime,
    codec_frame_count,
)

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


def test_chunk_samples_are_derived_from_frames_per_chunk() -> None:
    runtime = FakeRuntime(chunks=3, frames_per_chunk=2)
    chunks = _drain(runtime)

    assert len(chunks) == 3
    assert all(isinstance(chunk, FastStreamingChunk) for chunk in chunks)
    assert all(chunk.audio.dtype == np.float32 for chunk in chunks)
    # 2 codec frames/chunk * 1920 samples/frame, not a hard-coded sample count.
    assert all(chunk.audio.shape == (2 * CODEC_SAMPLES_PER_FRAME,) for chunk in chunks)
    assert all(chunk.codec_frames == 2 for chunk in chunks)
    # Different calls (pieces) are told apart by their constant sample value.
    other_piece = _drain(runtime)
    assert chunks[0].audio[0] == 0.0
    assert other_piece[0].audio[0] == pytest.approx(0.01)


def test_no_chunk_is_final_by_default() -> None:
    # The fast-codec path's common case: generation ends via EOS with an empty buffer, so
    # the real generator never yields an ``is_final=True`` chunk — it just stops.
    chunks = _drain(FakeRuntime(chunks=3))
    assert [chunk.is_final for chunk in chunks] == [False, False, False]
    assert [chunk.timing["is_final"] for chunk in chunks] == [False, False, False]


def test_is_final_on_last_can_simulate_reaching_the_token_limit() -> None:
    chunks = _drain(FakeRuntime(chunks=3, is_final_on_last=True))
    assert [chunk.is_final for chunk in chunks] == [False, False, True]


def test_timing_keys_match_the_real_per_chunk_dict() -> None:
    chunks = _drain(FakeRuntime(chunks=2))
    always = {
        "chunk_index",
        "codec_frames",
        "decode_launch_ms",
        "total_frames",
        "is_final",
        "codec_launch_ms",
        "audio_d2h_ms",
    }
    assert always <= chunks[0].timing.keys()
    assert always <= chunks[1].timing.keys()
    # Only chunk 0 carries the first-time-to-audio fields.
    assert "ttfa_internal_ms" in chunks[0].timing
    assert "prefill_path" in chunks[0].timing
    assert "ttfa_internal_ms" not in chunks[1].timing
    assert "prefill_path" not in chunks[1].timing
    # prefill_gpu_ms only appears with collect_timing=True.
    assert "prefill_gpu_ms" not in chunks[0].timing
    with_timing = _drain(FakeRuntime(chunks=1, collect_timing=True))
    assert "prefill_gpu_ms" in with_timing[0].timing
    assert with_timing[0].timing["chunk_index"] == 0
    assert with_timing[0].timing["total_frames"] == with_timing[0].timing["codec_frames"]


def test_token_observer_is_called_once_per_frame_interleaved_with_yields() -> None:
    frames = [torch.tensor([i]) for i in range(4)]
    runtime = FakeRuntime(chunks=2, frames_per_chunk=2, frames=frames)
    observed: list[torch.Tensor] = []
    gen = runtime.iter_audio_chunks({}, token_observer=observed.append)

    # Nothing is observed before the caller pulls the first chunk...
    assert observed == []
    next(gen)
    # ...but by the time the first chunk is out, exactly its 2 frames have been observed,
    # not all 4 (i.e. observation is interleaved with yields, not all done up front).
    assert [t.item() for t in observed] == [0, 1]
    next(gen)
    assert [t.item() for t in observed] == [0, 1, 2, 3]


def test_token_observer_sees_leftover_frames_even_with_zero_chunks() -> None:
    # An all-pad piece: every frame is observed (so the anchor logic can see they're all
    # pad), but nothing is decoded into an audio chunk.
    frames = [torch.tensor([i]) for i in range(3)]
    runtime = FakeRuntime(chunks=0, frames=frames)
    observed: list[torch.Tensor] = []

    chunks = list(runtime.iter_audio_chunks({}, token_observer=observed.append))

    assert chunks == []
    assert [t.item() for t in observed] == [0, 1, 2]


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
    assert codec_frame_count(0, 24000) == 0
    # librosa's ratio-first rounding (``ratio = float(target)/orig`` computed once, then
    # ``n * ratio``) gives a different sample count than a naive ``n * target / orig``
    # computed the other order, at some rates. 44.1 kHz x 30.0 s is exactly the case the
    # review flagged: 376 frames, one more than the naive computation's 375.
    assert codec_frame_count(44100 * 30, 44100) == 376
    naive_375 = -(-int(44100 * 30 * 24000 / 44100) // 1920)
    assert naive_375 == 375
    # The same rounding difference shows up at every other rate the review named.
    for sr, expected in ((22050, 376), (11025, 376), (88200, 376), (176400, 376)):
        assert codec_frame_count(sr * 30, sr) == expected


def test_codec_frame_count_skips_resampling_exactly_at_24khz() -> None:
    # At 24 kHz the real tokenizer never calls librosa.resample at all (``int(sr) !=
    # target_sr`` is false), so this must be an exact ceiling division with no float
    # rounding from a resample step.
    assert codec_frame_count(24000 * 30, 24000) == 375  # 720000 / 1920 exactly


def test_codec_encode_is_deterministic_shaped_and_in_range() -> None:
    codec = FakeCodec()
    wav = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)

    codes_a = codec.encode(wav, 24000)["audio_codes"][0]
    codes_b = codec.encode(wav, 24000)["audio_codes"][0]

    assert isinstance(codes_a, torch.Tensor)
    assert codes_a.dtype == torch.int64
    assert tuple(codes_a.shape) == (codec_frame_count(4800, 24000), FakeCodec.CODEBOOKS)
    assert torch.equal(codes_a, codes_b)
    assert int(codes_a.min()) >= 0
    assert int(codes_a.max()) < CODEC_CODEBOOK_SIZE
    assert codec.encode_calls == 2
    assert codec.last_sr == 24000

    different_wav = wav + 1.0
    codes_c = codec.encode(different_wav, 24000)["audio_codes"][0]
    assert not torch.equal(codes_a, codes_c)


def test_codec_encode_does_not_overflow_over_many_random_wavs() -> None:
    # Regression for the uint64-hash-into-int64-array overflow: a wide sweep of lengths
    # and sample rates (including ones that force resampling) must never raise.
    rng = np.random.default_rng(12345)
    codec = FakeCodec()
    sample_rates = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
    for _ in range(200):
        sr = int(rng.choice(sample_rates))
        num_samples = int(rng.integers(1, 200_000))
        wav = rng.uniform(-1.0, 1.0, size=num_samples).astype(np.float32)

        codes = codec.encode(wav, sr)["audio_codes"][0]

        assert codes.dtype == torch.int64
        assert tuple(codes.shape) == (codec_frame_count(num_samples, sr), FakeCodec.CODEBOOKS)
        assert int(codes.min()) >= 0
        assert int(codes.max()) < CODEC_CODEBOOK_SIZE
