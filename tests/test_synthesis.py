"""Tests for `breeze_infer/synthesis.py` (specs/003-cpp-compatible-api/tasks.md T038, the
single-reference part).

Uses `tests/fakes.py`'s GPU-free stand-ins (`FakeRuntime`, `FakeCodec`, `FakeTokenizer`,
`fake_model`) -- the same Principle V deviation the rest of the no-GPU suite relies on
(see that module's docstring).
"""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from breeze_infer.http_fields import InlineRef, NoReference, VoiceRef
from breeze_infer.reference_audio import DecodedAudio
from breeze_infer.synthesis import (
    CodesRef,
    NoRef,
    generate_piece,
    piece_seed,
    prepare_piece,
    ramp_pcm,
    resolve_reference,
)
from tests.fakes import (
    CODEC_SAMPLES_PER_FRAME,
    FakeCodec,
    FakeRuntime,
    FakeTokenizer,
    fake_model,
)


def _model_with_codec_facts():
    """``fake_model()`` plus the codec facts ``templates._codec_facts`` requires
    (``codec_config.codebook_size``, cross-checked against ``codebook_pad_token_id``).
    ``tests/fakes.py`` deliberately leaves ``fake_model()`` without ``codebook_size``
    (another agent owns that file's ``fake_model()``; `tests/test_templates.py`'s own
    ``_model_with_codec_facts`` tests that omission on purpose), so this augments the
    ``SimpleNamespace`` locally, in this test module only -- same pattern, same values
    (2048/2050) as `tests/test_templates.py`'s helper.
    """
    model = fake_model()
    model.config.codec_config.codebook_size = 2048
    return model


class _SyncGpu:
    """Duck-typed stand-in for `breeze_infer.gpu.GpuThread`'s ``run``.

    `resolve_reference` only needs ``await gpu.run(fn, *args)`` to execute ``fn(*args)``
    and return its result -- not the real thread/executor machinery -- so this runs the
    call inline rather than pulling in `breeze_infer.gpu` (a module other agents are
    editing concurrently; nothing here needs its actual behavior, only its `run` shape).
    """

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, fn, *args):
        self.calls += 1
        return fn(*args)


def _decoded_audio(seconds: float = 1.0, sample_rate: int = 24000) -> DecodedAudio:
    samples = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    return DecodedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=seconds,
        predicted_frames=8,
    )


# --- piece_seed ------------------------------------------------------------------------


def test_piece_seed_adds_the_index_to_the_request_seed() -> None:
    assert piece_seed(42, 0) == 42
    assert piece_seed(42, 3) == 45


def test_piece_seed_wraps_to_uint32() -> None:
    assert piece_seed(0xFFFFFFFF, 1) == 0
    assert piece_seed(0xFFFFFFFE, 5) == 3


# --- resolve_reference -------------------------------------------------------------------


def test_resolve_reference_no_reference_is_noref_and_never_touches_the_codec() -> None:
    codec = FakeCodec()
    gpu = _SyncGpu()

    reference = asyncio.run(
        resolve_reference(NoReference(), decoded_audio=None, audio_tokenizer=codec, gpu=gpu)
    )

    assert reference == NoRef()
    assert codec.encode_calls == 0
    assert gpu.calls == 0


def test_resolve_reference_inline_ref_encodes_once_on_the_gpu() -> None:
    codec = FakeCodec()
    gpu = _SyncGpu()
    spec = InlineRef(audio_bytes=b"already-decoded-elsewhere", ref_text="hello there")
    decoded = _decoded_audio()

    reference = asyncio.run(
        resolve_reference(spec, decoded_audio=decoded, audio_tokenizer=codec, gpu=gpu)
    )

    assert isinstance(reference, CodesRef)
    assert reference.ref_text == "hello there"
    # The encode itself ran through gpu.run (i.e. on "the GPU thread"), exactly once.
    assert gpu.calls == 1
    assert codec.encode_calls == 1
    assert codec.last_sr == 24000


def test_resolve_reference_voice_ref_is_a_documented_stub_until_t066() -> None:
    with pytest.raises(NotImplementedError, match="T066"):
        asyncio.run(
            resolve_reference(
                VoiceRef(voice_id="v1", ref_text_override=None),
                decoded_audio=None,
                audio_tokenizer=FakeCodec(),
                gpu=_SyncGpu(),
            )
        )


def test_resolve_reference_rejects_an_unknown_spec() -> None:
    with pytest.raises(TypeError, match="unknown reference spec"):
        asyncio.run(
            resolve_reference(object(), decoded_audio=None, audio_tokenizer=FakeCodec(), gpu=_SyncGpu())
        )


# --- prepare_piece -----------------------------------------------------------------------


def test_prepare_piece_no_reference_uses_the_instruction_template() -> None:
    inputs = prepare_piece(
        FakeTokenizer(), _model_with_codec_facts(), NoRef(), "hello", "calm voice", 1.0
    )

    assert "input_ids" in inputs
    assert inputs.get("input_values") is None  # no reference audio baked in
    assert "cfg_negative_prompt_ids" not in inputs  # cfg_scale == 1.0: no negative branch


def test_prepare_piece_codes_reference_bakes_in_ref_audio_codes() -> None:
    codec = FakeCodec()
    wav = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)
    codes = codec.encode(wav, 24000)["audio_codes"][0]
    reference = CodesRef(codes=codes, ref_text="the reference transcript")

    inputs = prepare_piece(
        FakeTokenizer(), _model_with_codec_facts(), reference, "piece text", "ins", 1.0
    )

    assert inputs["input_values"] is not None


def test_prepare_piece_rejects_an_unknown_reference() -> None:
    with pytest.raises(TypeError, match="unknown reference"):
        prepare_piece(FakeTokenizer(), fake_model(), object(), "hi", "ins", 1.0)


def test_inline_reference_is_encoded_once_and_reused_across_pieces_and_cfg_rows() -> None:
    """T038's own requirement: an inline reference is encoded once on the GpuThread and
    reused for every piece, and (inside a single-CFG piece) both the guided and
    unguided rows. Verified the way `tests/fakes.py`'s `FakeRuntime` docstring says a
    test for this should: a call count on `FakeCodec.encode`, not by inspecting some
    ``reference=`` field (there is none on the real runtime)."""
    codec = FakeCodec()
    gpu = _SyncGpu()
    spec = InlineRef(audio_bytes=b"raw", ref_text="ref text")
    decoded = _decoded_audio()

    reference = asyncio.run(
        resolve_reference(spec, decoded_audio=decoded, audio_tokenizer=codec, gpu=gpu)
    )
    assert codec.encode_calls == 1

    tokenizer, model = FakeTokenizer(), _model_with_codec_facts()
    # Three pieces; the second and third use cfg_scale != 1.0, so each of those also
    # builds the unguided (negative) row -- both rows share the one reference.
    for text, cfg_scale in [("first piece", 1.0), ("second piece", 2.5), ("third piece", 2.5)]:
        inputs = prepare_piece(tokenizer, model, reference, text, "voice design", cfg_scale)
        assert inputs["input_values"] is not None
        if cfg_scale != 1.0:
            assert inputs["cfg_negative_input_values"] is not None

    # Still exactly one encode: the reference was never touched again after resolve.
    assert codec.encode_calls == 1


# --- ramp_pcm --------------------------------------------------------------------------


def _streaming_chunk(value: float, codec_frames: int) -> SimpleNamespace:
    """A `models.fast_streaming.FastStreamingChunk`-shaped object with the two fields
    ``ramp_pcm`` reads (``.audio``, ``.codec_frames``) -- a plain `SimpleNamespace`
    rather than the real class, so these tests never pay to import
    ``models.fast_streaming`` (finding #8; the real class is exercised for real by the
    `generate_piece` tests below, which go through `FakeRuntime`)."""
    return SimpleNamespace(
        audio=np.full(codec_frames * CODEC_SAMPLES_PER_FRAME, value, dtype=np.float32),
        codec_frames=codec_frames,
    )


def _frame_count(flush: bytes) -> int:
    assert len(flush) % 2 == 0
    samples = len(flush) // 2
    assert samples % CODEC_SAMPLES_PER_FRAME == 0
    return samples // CODEC_SAMPLES_PER_FRAME


def _assert_ramp_growth(sizes: list[int], chunk_max: int) -> None:
    """Every flush is <= ``chunk_max``, and non-decreasing -- except possibly the very
    last one, which can be a shorter leftover if the piece ends mid-ramp (the "flushes a
    short leftover" test below covers that case on its own)."""
    assert all(size <= chunk_max for size in sizes)
    for previous, current in itertools.pairwise(sizes[:-1]):
        assert current >= previous


def test_ramp_pcm_grows_from_chunk_first_to_chunk_max_preserving_the_total() -> None:
    total_input_frames = 80
    chunks = [_streaming_chunk(i / 100.0, codec_frames=1) for i in range(total_input_frames)]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25))
    sizes = [_frame_count(flush) for flush in flushes]

    assert sizes[0] == 1  # never held back past chunk_first frames
    _assert_ramp_growth(sizes, chunk_max=25)
    assert sum(sizes) == total_input_frames  # every frame is accounted for


def test_ramp_pcm_flushes_a_short_leftover_when_the_piece_ends() -> None:
    # 3 frames with chunk_max=25: the first (and only) flush is whatever the ramp has
    # buffered when the input runs out, not padded up to chunk_max.
    chunks = [_streaming_chunk(0.1, codec_frames=1) for _ in range(3)]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25))

    assert sum(_frame_count(flush) for flush in flushes) == 3


def test_ramp_pcm_skips_empty_chunks() -> None:
    chunks = [
        SimpleNamespace(audio=np.zeros(0, dtype=np.float32), codec_frames=0),
        _streaming_chunk(0.5, codec_frames=1),
    ]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=5))

    assert sum(_frame_count(flush) for flush in flushes) == 1


def test_ramp_pcm_on_no_chunks_yields_nothing() -> None:
    assert list(ramp_pcm([], chunk_first=1, chunk_max=25)) == []


def test_ramp_pcm_clamps_chunk_first_above_chunk_max() -> None:
    chunks = [_streaming_chunk(0.2, codec_frames=1) for _ in range(10)]

    flushes = list(ramp_pcm(chunks, chunk_first=100, chunk_max=4))

    assert _frame_count(flushes[0]) == 4  # chunk_first clamped down to chunk_max
    assert sum(_frame_count(flush) for flush in flushes) == 10


def test_ramp_pcm_handles_multi_frame_chunks() -> None:
    # frames_per_chunk=2 (the non-fast codec path): each streamed chunk already carries
    # 2 codec frames, so samples_per_frame must be derived per chunk, not assumed to be
    # the whole chunk's sample count.
    chunks = [_streaming_chunk(0.3, codec_frames=2) for _ in range(6)]  # 12 frames total

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25))

    assert sum(_frame_count(flush) for flush in flushes) == 12


# --- generate_piece --------------------------------------------------------------------


def test_generate_piece_passes_the_seed_and_overrides_to_the_runtime() -> None:
    runtime = FakeRuntime(chunks=4, frames_per_chunk=1)
    inputs = {"input_ids": np.zeros((1, 3))}

    pcm = b"".join(
        generate_piece(
            runtime,
            inputs,
            request_id="req-1",
            seed=piece_seed(42, 2),
            chunk_first=1,
            chunk_max=25,
            temperature=0.5,
            top_k=10,
            top_p=0.8,
            repetition_penalty=1.3,
            max_new_tokens=200,
        )
    )

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["inputs"] is inputs
    assert call["request_id"] == "req-1"
    assert call["seed"] == piece_seed(42, 2) == 44
    assert call["temperature"] == 0.5
    assert call["top_k"] == 10
    assert call["top_p"] == 0.8
    assert call["repetition_penalty"] == 1.3
    assert call["max_new_tokens"] == 200
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0  # s16le: every sample is 2 bytes
    # The piece ran to exhaustion, so the runtime's own generator was closed too (its
    # ``finally`` runs on normal completion, not only on an early ``close()``).
    assert runtime.closed == 1


def test_generate_piece_overrides_default_to_none() -> None:
    runtime = FakeRuntime(chunks=1)
    list(generate_piece(runtime, {}, request_id="r", seed=1, chunk_first=1, chunk_max=25))

    call = runtime.calls[0]
    for name in ("temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"):
        assert call[name] is None


def test_generate_piece_ramps_growing_pcm_chunks() -> None:
    runtime = FakeRuntime(chunks=20, frames_per_chunk=1)
    gen = generate_piece(runtime, {}, request_id="r", seed=1, chunk_first=1, chunk_max=5)

    flushes = list(gen)
    sizes = [_frame_count(flush) for flush in flushes]

    assert sizes[0] == 1
    _assert_ramp_growth(sizes, chunk_max=5)
    assert sum(sizes) == 20


def test_generate_piece_close_propagates_to_the_runtime_generator() -> None:
    # A client disconnect closes the piece's GpuSession, which closes this generator --
    # that must reach the runtime's own generator so it releases whatever it holds.
    runtime = FakeRuntime(chunks=10, frames_per_chunk=1)
    gen = generate_piece(runtime, {}, request_id="r", seed=1, chunk_first=1, chunk_max=25)

    next(gen)  # pull the first PCM flush; the runtime generator is now mid-piece
    assert runtime.closed == 0

    gen.close()

    assert runtime.closed == 1


def test_generate_piece_carries_the_prefix_through() -> None:
    runtime = FakeRuntime(chunks=1)
    list(
        generate_piece(
            runtime, {}, request_id="r", seed=1, chunk_first=1, chunk_max=25, prefix="cached-kv"
        )
    )

    assert runtime.calls[0]["prefix"] == "cached-kv"
