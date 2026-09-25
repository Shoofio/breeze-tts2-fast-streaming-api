"""`tests/fakes.py`'s own behavior: the fakes are test infrastructure (Complexity Tracking,
Principle V), so a bug in them would silently corrupt every test that depends on them.
"""

from __future__ import annotations

import threading

import librosa
import numpy as np
import pytest
import torch

from models.fast_streaming import FastStreamingChunk
from tests.fakes import (
    CODEC_CODEBOOK_SIZE,
    CODEC_SAMPLE_RATE,
    CODEC_SAMPLES_PER_FRAME,
    FakeCodec,
    FakeRuntime,
    FakeStreamingConfig,
    codec_frame_count,
)

# Long enough never to fire on a healthy run; only there so a bug hangs a test, not the suite.
TIMEOUT = 5.0


_DEFAULT_INPUTS = {"input_ids": np.zeros((1, 3))}


def _drain(runtime: FakeRuntime, *, inputs=None, **kwargs) -> list[FastStreamingChunk]:
    return list(
        runtime.iter_audio_chunks(
            _DEFAULT_INPUTS if inputs is None else inputs, **kwargs
        )
    )


def test_reports_sample_rate_24000() -> None:
    assert FakeRuntime().sample_rate == 24000


def test_records_seed_overrides_inputs_and_prefix_per_piece() -> None:
    # No ``reference=`` here (or on the real runtime, models/fast_streaming.py): a
    # reference reaches iter_audio_chunks either baked into ``inputs`` (ref_audio_codes,
    # the "codes" Reference variant) or as ``prefix`` (a cached-KV "prefix" variant).
    # synthesis.py's resolve_reference/prepare_piece build one of those two shapes; a
    # test for "the inline reference is encoded once and reused for every piece" asserts
    # that by passing the *same* ``inputs``/``prefix`` object to every call, as below.
    runtime = FakeRuntime(chunks=1)
    shared_inputs = {"input_ids": np.zeros((1, 3)), "attention_mask": np.ones((1, 3))}
    _drain(
        runtime,
        inputs=shared_inputs,
        request_id="req-0",
        seed=7,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=300,
        prefix="cached-kv",
    )
    _drain(runtime, inputs=shared_inputs, request_id="req-0", seed=8, prefix="cached-kv")

    assert len(runtime.calls) == 2
    first, second = runtime.calls
    assert first["seed"] == 7
    assert first["inputs"] is shared_inputs
    assert first["temperature"] == 0.8
    assert first["top_k"] == 50
    assert first["top_p"] == 0.9
    assert first["repetition_penalty"] == 1.2
    assert first["max_new_tokens"] == 300
    assert first["prefix"] == "cached-kv"
    # Piece 1 reuses the same (identical) inputs/prefix rather than re-encoding a
    # reference, which is exactly what T038's "inline reference encoded once" needs.
    assert second["seed"] == 8
    assert second["inputs"] is first["inputs"]
    assert second["prefix"] == first["prefix"]


_INVALID_OVERRIDES = [
    {"temperature": 0.0},
    {"top_k": 0},
    {"top_p": -0.1},
    {"repetition_penalty": 0.0},
    {"max_new_tokens": 0},
    {"max_new_tokens": -3},
    {"temperature": float("nan")},
    {"temperature": float("inf")},
    {"top_k": float("nan")},
    # Counts must be integers; the real runtime does not truncate them.
    {"top_k": 2.5},
    {"max_new_tokens": 2.5},
    # bool is an int subclass in Python, but the real runtime rejects it explicitly
    # (tests/test_fast_streaming.py's own _INVALID_OVERRIDES doesn't cover this case).
    {"top_k": True},
    {"max_new_tokens": False},
]


@pytest.mark.parametrize("override", _INVALID_OVERRIDES)
def test_iter_audio_chunks_rejects_invalid_overrides_like_the_real_runtime(
    override,
) -> None:
    runtime = FakeRuntime(chunks=1)
    gen = runtime.iter_audio_chunks(_DEFAULT_INPUTS, **override)

    # A generator function's body -- real and fake alike -- doesn't run until the first
    # next(), so validation (and the ValueError) only happens there.
    with pytest.raises(ValueError, match=next(iter(override))):
        next(gen)


def test_iter_audio_chunks_accepts_none_overrides() -> None:
    # None means "use the default" and must never raise.
    chunks = _drain(FakeRuntime(chunks=1))
    assert len(chunks) == 1


def _room_inputs(prompt_len: int) -> dict[str, np.ndarray]:
    return {"attention_mask": np.ones((1, prompt_len))}


def test_max_new_tokens_room_rejects_invalid_requested() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        FakeRuntime().max_new_tokens_room(0, _room_inputs(10))
    with pytest.raises(ValueError, match="max_new_tokens"):
        FakeRuntime().max_new_tokens_room(2.5, _room_inputs(10))


def test_max_new_tokens_room_none_uses_default_then_config_ceiling() -> None:
    # No default_max_new_tokens given: falls back to config.max_new_tokens (750).
    assert FakeRuntime().max_new_tokens_room(None, _room_inputs(10)) == 750
    # An explicit model default (mirrors model.generation_config.max_new_tokens) is used
    # instead, same as the real runtime's _frame_cap.
    runtime = FakeRuntime(default_max_new_tokens=5)
    assert runtime.max_new_tokens_room(None, _room_inputs(10)) == 5


def test_max_new_tokens_room_clamped_by_config_ceiling() -> None:
    runtime = FakeRuntime(config=FakeStreamingConfig(max_new_tokens=4, max_seq_len=2048))
    assert runtime.max_new_tokens_room(50, _room_inputs(10)) == 4


def test_max_new_tokens_room_shrinks_with_prompt_and_prefix_length() -> None:
    runtime = FakeRuntime(config=FakeStreamingConfig(max_new_tokens=1500, max_seq_len=2048))
    room_short = runtime.max_new_tokens_room(1500, _room_inputs(10))
    room_long = runtime.max_new_tokens_room(1500, _room_inputs(1000))
    room_with_prefix = runtime.max_new_tokens_room(1500, _room_inputs(10), prefix_len=1000)
    assert room_short == 1500  # plenty of context room
    assert room_long < room_short
    assert room_with_prefix < room_short


def test_flush_frames_must_be_smaller_than_a_full_chunk() -> None:
    # 0 < flush_frames < frames_per_chunk (module docstring: a full chunk would have
    # flushed inside the main loop already, and 0 leaves nothing to flush).
    with pytest.raises(ValueError, match="flush_frames"):
        FakeRuntime(frames_per_chunk=2, flush_frames=2)
    with pytest.raises(ValueError, match="flush_frames"):
        FakeRuntime(frames_per_chunk=2, flush_frames=3)
    with pytest.raises(ValueError, match="flush_frames"):
        FakeRuntime(frames_per_chunk=2, flush_frames=0)
    # A valid flush_frames (< frames_per_chunk) is fine.
    FakeRuntime(frames_per_chunk=2, flush_frames=1)


def test_is_final_on_last_and_flush_frames_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="is_final_on_last"):
        FakeRuntime(frames_per_chunk=2, flush_frames=1, is_final_on_last=True)


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


def test_flush_frames_models_the_post_loop_leftover_buffer_chunk() -> None:
    # The non-fast-codec case: 2 regular (frames_per_chunk=2) chunks, then a piece ends
    # with exactly 1 frame still buffered -- the real post-loop ``if chunk_buffer:``
    # flush, not another full frames_per_chunk-sized chunk.
    chunks = _drain(FakeRuntime(chunks=2, frames_per_chunk=2, flush_frames=1))

    assert len(chunks) == 3
    flush = chunks[-1]
    assert flush.codec_frames == 1
    assert flush.audio.shape == (1 * CODEC_SAMPLES_PER_FRAME,)
    assert flush.is_final is True
    assert flush.timing["chunk_index"] == 2
    assert flush.timing["codec_frames"] == 1
    assert flush.timing["total_frames"] == 2 * 2 + 1
    # Never the chunk-0 enrichment, even though this is the only (and so index-0) chunk
    # when there are no regular chunks at all.
    assert "ttfa_internal_ms" not in flush.timing
    assert "prefill_path" not in flush.timing

    only_flush = _drain(FakeRuntime(chunks=0, frames_per_chunk=2, flush_frames=1))
    assert len(only_flush) == 1
    assert only_flush[0].timing["chunk_index"] == 0
    assert "ttfa_internal_ms" not in only_flush[0].timing
    assert "prefill_path" not in only_flush[0].timing


def test_default_frames_let_token_observer_run_with_no_arguments() -> None:
    # A caller that passes token_observer but no explicit frames= must still get one
    # observer call per frame the call is going to produce, not silent no-ops.
    observed: list[torch.Tensor] = []
    chunks = _drain(
        FakeRuntime(chunks=2, frames_per_chunk=2, flush_frames=1),
        token_observer=observed.append,
    )
    assert len(chunks) == 3
    assert len(observed) == 2 * 2 + 1


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
    # 48 kHz boundary: exactly 1920 resampled samples is 1 frame; one more costs a whole
    # extra frame (ceil(3841 * 24000/48000) = 1921 -> ceil(1921/1920) = 2).
    assert codec_frame_count(3840, 48000) == 1
    assert codec_frame_count(3841, 48000) == 2
    # librosa's ratio-first rounding (``ratio = float(target)/orig`` computed once, then
    # ``n * ratio``) gives a different sample count than a naive ``n * target / orig``
    # computed the other order, at some rates. 44.1 kHz x 30.0 s is exactly the case the
    # review flagged: 376 frames, one more than a naive
    # ``-(-int(44100 * 30 * 24000 / 44100) // 1920)`` (== 375) would give.
    assert codec_frame_count(44100 * 30, 44100) == 376
    # The same rounding difference shows up at every other rate the review named.
    for sr, expected in ((22050, 376), (11025, 376), (88200, 376), (176400, 376)):
        assert codec_frame_count(sr * 30, sr) == expected


def test_codec_frame_count_skips_resampling_exactly_at_24khz() -> None:
    # At 24 kHz the real tokenizer never calls librosa.resample at all (``int(sr) !=
    # target_sr`` is false), so this must be an exact ceiling division with no float
    # rounding from a resample step.
    assert codec_frame_count(24000 * 30, 24000) == 375  # 720000 / 1920 exactly


def test_codec_frame_count_matches_actual_librosa_resample() -> None:
    """Root-cause test: compares against a real call to ``librosa.resample``, not just
    another implementation of the same formula (two independent implementations of a
    wrong formula would still agree with each other). Mirrors qwen_tts's exact call
    (``librosa.resample(y=a, orig_sr=int(sr), target_sr=target_sr)``, no explicit
    ``res_type``) -- librosa's ``fix=True`` default forces the returned length to exactly
    ``ceil(len(y) * target_sr / orig_sr)`` no matter which resampler backend runs, so this
    is checking the framing step and the real resample step together."""
    sample_rates = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400)
    lengths = (1, 100, 1920, 1921, 3840, 3841, 44100 * 30)
    for sr in sample_rates:
        for n in lengths:
            wav = np.zeros(n, dtype=np.float32)
            if sr == CODEC_SAMPLE_RATE:
                resampled_len = n  # qwen_tts skips the call entirely at this rate too.
            else:
                resampled_len = len(librosa.resample(y=wav, orig_sr=sr, target_sr=CODEC_SAMPLE_RATE))
            expected = -(-resampled_len // CODEC_SAMPLES_PER_FRAME)
            assert codec_frame_count(n, sr) == expected, (n, sr)


def test_codec_encode_is_deterministic_shaped_and_in_range() -> None:
    codec = FakeCodec()
    wav = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)

    result_a = codec.encode(wav, 24000)
    codes_a = result_a["audio_codes"][0]
    codes_b = codec.encode(wav, 24000)["audio_codes"][0]

    assert isinstance(codes_a, torch.Tensor)
    assert codes_a.dtype == torch.int64
    assert tuple(codes_a.shape) == (codec_frame_count(4800, 24000), FakeCodec.CODEBOOKS)
    # Non-contiguous, like the real encode()'s ``code[..., :T].transpose(0, 1)``.
    assert codes_a.shape[0] > 1  # otherwise a size-1 dim trivially reads as "contiguous"
    assert not codes_a.is_contiguous()
    assert torch.equal(codes_a, codes_b)
    assert int(codes_a.min()) >= 0
    assert int(codes_a.max()) < CODEC_CODEBOOK_SIZE
    assert codec.encode_calls == 2
    assert codec.last_sr == 24000

    different_wav = wav + 1.0
    codes_c = codec.encode(different_wav, 24000)["audio_codes"][0]
    assert not torch.equal(codes_a, codes_c)


def test_codec_encode_result_supports_attribute_and_key_access() -> None:
    # The real Qwen3TTSTokenizerV2EncoderOutput is a transformers ModelOutput, so both
    # ``result.audio_codes`` and ``result["audio_codes"]`` work; this fake must too.
    result = FakeCodec().encode(np.zeros(4800, dtype=np.float32), 24000)
    assert result.audio_codes is result["audio_codes"]


def test_codec_encode_result_supports_integer_indexing_like_model_output() -> None:
    # transformers' ModelOutput indexes an int key into to_tuple() (its non-None values
    # in field order); with exactly one field, index 0 unwraps straight to audio_codes,
    # not a 1-tuple containing it.
    result = FakeCodec().encode(np.zeros(4800, dtype=np.float32), 24000)
    assert result[0] is result.audio_codes


def test_codec_encode_return_dict_false_gives_the_plain_tuple() -> None:
    # The real encode() takes return_dict (as transformers model forwards do); False
    # returns to_tuple() directly instead of the attribute/dict-style wrapper.
    wav = np.zeros(4800, dtype=np.float32)
    codec = FakeCodec()

    as_dict = codec.encode(wav, 24000, return_dict=True)
    as_tuple = codec.encode(wav, 24000, return_dict=False)

    assert isinstance(as_tuple, tuple)
    assert len(as_tuple) == 1
    # audio_codes is itself a 1-element list (the real return type: "audio_codes:
    # List[torch.LongTensor]"), so unwrap one more level to reach the tensor.
    assert torch.equal(as_tuple[0][0], as_dict.audio_codes[0])
    # return_dict=True is still the default.
    assert isinstance(codec.encode(wav, 24000), type(as_dict))


def test_codec_encode_raises_on_empty_wav() -> None:
    with pytest.raises(RuntimeError):
        FakeCodec().encode(np.zeros(0, dtype=np.float32), 24000)


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
