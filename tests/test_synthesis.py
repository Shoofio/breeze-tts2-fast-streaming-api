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
from typing import Any

import numpy as np
import pytest
import torch

from breeze_infer.http_fields import InlineRef, NoReference, VoiceRef
from breeze_infer.reference_audio import DecodedAudio
from breeze_infer.synthesis import (
    AnchorSizing,
    CodesRef,
    NoRef,
    PieceRoom,
    anchor_sizing,
    codec_samples_per_frame,
    generate_piece,
    piece_frame_limit,
    piece_room,
    piece_seed,
    predicted_room,
    prepare_piece,
    ramp_pcm,
    resolve_reference,
    stand_in_reference,
)
from models.fast_streaming import NoRoomError, PromptLength, prompt_length
from tests.fakes import (
    CODEC_SAMPLES_PER_FRAME,
    FakeCodec,
    FakeRuntime,
    FakeTokenizer,
    RecordingEvents,
    fake_model,
    model_with_codec_facts,
)

# `model_with_codec_facts` (tests/fakes.py): `fake_model()` plus the codec facts
# `templates._codec_facts` requires (`codec_config.codebook_size`), used everywhere below
# that needs a model `prepare_piece` can actually run -- review-agent pass 1, finding 9,
# which moved this out of a local duplicate here (and in tests/test_routes_speech.py) into
# the shared fake, since `tests/test_templates.py`'s own `_model_with_codec_facts` still
# tests the *omission* on purpose and stays local to that module.


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
        FakeTokenizer(), model_with_codec_facts(), NoRef(), "hello", "calm voice", 1.0
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
        FakeTokenizer(), model_with_codec_facts(), reference, "piece text", "ins", 1.0
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
    ``reference=`` field (there is none on the real runtime) -- and (review finding #10)
    by checking that the *same* encoded codes, value-equal, actually reach every piece's
    inputs through both `prepare_piece` and `generate_piece`, not just that some
    non-``None`` value is present."""
    codec = FakeCodec()
    gpu = _SyncGpu()
    spec = InlineRef(audio_bytes=b"raw", ref_text="ref text")
    decoded = _decoded_audio()

    reference = asyncio.run(
        resolve_reference(spec, decoded_audio=decoded, audio_tokenizer=codec, gpu=gpu)
    )
    assert codec.encode_calls == 1

    tokenizer, model = FakeTokenizer(), model_with_codec_facts()
    runtime = FakeRuntime(chunks=1)
    # Three pieces; the second and third use cfg_scale != 1.0, so each of those also
    # builds the unguided (negative) row -- both rows, and every piece, must carry the
    # exact same encoded codes.
    for index, (text, cfg_scale) in enumerate(
        [("first piece", 1.0), ("second piece", 2.5), ("third piece", 2.5)]
    ):
        inputs = prepare_piece(tokenizer, model, reference, text, "voice design", cfg_scale)
        # A fresh tensor every call (`_resolve_segment_audio_codes`/`_collate_inputs`
        # copy it via .to()/.contiguous()/torch.cat), so this is value equality, not
        # identity -- the same content the codec produced exactly once above.
        assert torch.equal(inputs["input_values"][0], reference.codes)
        if cfg_scale != 1.0:
            # Both CFG rows (the guided and unguided branches of one piece) embed the
            # same reference, not two different encodes of it.
            assert torch.equal(inputs["cfg_negative_input_values"][0], reference.codes)

        # And downstream, generate_piece must hand the runtime these exact inputs
        # unchanged -- the encoded reference travels through to the runtime call too.
        list(
            generate_piece(
                runtime,
                inputs,
                request_id="r",
                seed=piece_seed(0, index),
                chunk_first=1,
                chunk_max=25,
                samples_per_frame=CODEC_SAMPLES_PER_FRAME,
            )
        )
        assert runtime.calls[index]["inputs"] is inputs
        assert torch.equal(runtime.calls[index]["inputs"]["input_values"][0], reference.codes)

    # Still exactly one encode: the reference was never touched again after resolve.
    assert codec.encode_calls == 1


# --- ramp_pcm --------------------------------------------------------------------------


def _streaming_chunk(value: float, codec_frames: int) -> SimpleNamespace:
    """A `models.fast_streaming.FastStreamingChunk`-shaped object with the one field
    ``ramp_pcm`` reads (``.audio``; ``samples_per_frame`` is now a caller-supplied
    parameter, review finding #8, so ``.codec_frames`` is no longer read by ``ramp_pcm``
    itself -- it only sizes ``.audio`` here) -- a plain `SimpleNamespace` rather than the
    real class, so these tests never pay to import ``models.fast_streaming`` (finding #8;
    the real class is exercised for real by the `generate_piece` tests below, which go
    through `FakeRuntime`)."""
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
    # review finding #6: the exact sequence _ramp_pcm's growth rule (chunk // 3 + 1,
    # capped at chunk_max) produces for chunk_first=1, chunk_max=25, computed by hand
    # from the ported rule itself (A:api.py's _ramp_pcm), not just checked for "grows":
    # 1 -> 2 -> 3 -> 5 -> 7 -> 10 -> 14 -> 19 -> capped at 25, but only 19 frames of
    # input are left by then (80 - 61 = 19 < 25), so the last flush is that 19-frame
    # leftover, not a full 25.
    total_input_frames = 80
    chunks = [_streaming_chunk(i / 100.0, codec_frames=1) for i in range(total_input_frames)]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25, samples_per_frame=CODEC_SAMPLES_PER_FRAME))
    sizes = [_frame_count(flush) for flush in flushes]

    assert sizes == [1, 2, 3, 5, 7, 10, 14, 19, 19]
    assert sum(sizes) == total_input_frames  # every frame is accounted for


def test_ramp_pcm_flushes_a_short_leftover_when_the_piece_ends() -> None:
    # 3 frames with chunk_max=25: the first (and only) flush is whatever the ramp has
    # buffered when the input runs out, not padded up to chunk_max.
    chunks = [_streaming_chunk(0.1, codec_frames=1) for _ in range(3)]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25, samples_per_frame=CODEC_SAMPLES_PER_FRAME))

    assert sum(_frame_count(flush) for flush in flushes) == 3


def test_ramp_pcm_skips_empty_chunks() -> None:
    chunks = [
        SimpleNamespace(audio=np.zeros(0, dtype=np.float32), codec_frames=0),
        _streaming_chunk(0.5, codec_frames=1),
    ]

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=5, samples_per_frame=CODEC_SAMPLES_PER_FRAME))

    assert sum(_frame_count(flush) for flush in flushes) == 1


def test_ramp_pcm_on_no_chunks_yields_nothing() -> None:
    assert list(ramp_pcm([], chunk_first=1, chunk_max=25, samples_per_frame=CODEC_SAMPLES_PER_FRAME)) == []


def test_ramp_pcm_clamps_chunk_first_above_chunk_max() -> None:
    chunks = [_streaming_chunk(0.2, codec_frames=1) for _ in range(10)]

    flushes = list(ramp_pcm(chunks, chunk_first=100, chunk_max=4, samples_per_frame=CODEC_SAMPLES_PER_FRAME))

    assert _frame_count(flushes[0]) == 4  # chunk_first clamped down to chunk_max
    assert sum(_frame_count(flush) for flush in flushes) == 10


def test_ramp_pcm_handles_multi_frame_chunks() -> None:
    # frames_per_chunk=2 (the non-fast codec path): each streamed chunk already carries
    # 2 codec frames worth of samples; ramp_pcm must sum a piece's frames correctly
    # regardless of how many frames arrive per streamed chunk.
    chunks = [_streaming_chunk(0.3, codec_frames=2) for _ in range(6)]  # 12 frames total

    flushes = list(ramp_pcm(chunks, chunk_first=1, chunk_max=25, samples_per_frame=CODEC_SAMPLES_PER_FRAME))

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
            samples_per_frame=CODEC_SAMPLES_PER_FRAME,
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
    list(
        generate_piece(
            runtime,
            {},
            request_id="r",
            seed=1,
            chunk_first=1,
            chunk_max=25,
            samples_per_frame=CODEC_SAMPLES_PER_FRAME,
        )
    )

    call = runtime.calls[0]
    for name in ("temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"):
        assert call[name] is None


def test_generate_piece_ramps_growing_pcm_chunks() -> None:
    runtime = FakeRuntime(chunks=20, frames_per_chunk=1)
    gen = generate_piece(
        runtime,
        {},
        request_id="r",
        seed=1,
        chunk_first=1,
        chunk_max=5,
        samples_per_frame=CODEC_SAMPLES_PER_FRAME,
    )

    flushes = list(gen)
    sizes = [_frame_count(flush) for flush in flushes]

    assert sizes[0] == 1
    _assert_ramp_growth(sizes, chunk_max=5)
    assert sum(sizes) == 20


def test_generate_piece_close_propagates_to_the_runtime_generator() -> None:
    # A client disconnect closes the piece's GpuSession, which closes this generator --
    # that must reach the runtime's own generator so it releases whatever it holds.
    runtime = FakeRuntime(chunks=10, frames_per_chunk=1)
    gen = generate_piece(
        runtime,
        {},
        request_id="r",
        seed=1,
        chunk_first=1,
        chunk_max=25,
        samples_per_frame=CODEC_SAMPLES_PER_FRAME,
    )

    next(gen)  # pull the first PCM flush; the runtime generator is now mid-piece
    assert runtime.closed == 0

    gen.close()

    assert runtime.closed == 1


def test_generate_piece_carries_the_prefix_through() -> None:
    runtime = FakeRuntime(chunks=1)
    # A real-shaped stand-in, not a bare string: FakeRuntime reads prefix.prefix_len
    # strictly, like the real runtime's own ReferencePrefix (review finding #3).
    prefix_stub = SimpleNamespace(prefix_len=64)
    list(
        generate_piece(
            runtime,
            {},
            request_id="r",
            seed=1,
            chunk_first=1,
            chunk_max=25,
            samples_per_frame=CODEC_SAMPLES_PER_FRAME,
            prefix=prefix_stub,
        )
    )

    assert runtime.calls[0]["prefix"] is prefix_stub


def test_generate_piece_requires_samples_per_frame() -> None:
    # No default and no fallback (review finding #10): omitting it is a TypeError at the
    # call site, not a silently-wrong ramp.
    with pytest.raises(TypeError):
        generate_piece(FakeRuntime(chunks=1), {}, request_id="r", seed=1, chunk_first=1, chunk_max=25)


def test_codec_samples_per_frame_uses_the_audio_tokenizers_accessor() -> None:
    runtime = SimpleNamespace(audio_tokenizer=FakeCodec())

    assert codec_samples_per_frame(runtime) == CODEC_SAMPLES_PER_FRAME


def test_codec_samples_per_frame_without_the_accessor_raises() -> None:
    # No fallback: the real wrapper's `config` is None unless built by from_pretrained, and
    # a guessed frame size would mis-size every PCM flush.
    runtime = SimpleNamespace(audio_tokenizer=SimpleNamespace(config=None))

    with pytest.raises(TypeError, match="get_decode_upsample_rate"):
        codec_samples_per_frame(runtime)


def test_piece_room_takes_the_cap_from_the_runtimes_public_frame_cap() -> None:
    runtime = SimpleNamespace(
        frame_cap=lambda requested: 40,
        max_new_tokens_room=lambda requested, inputs: 25,
    )

    assert piece_room(runtime, {}, None) == PieceRoom(cap=40, room=25)


# --- predicted_room: piece 0's room before the gate (review 26b #7) ----------------------------


@pytest.mark.parametrize("cfg_scale", [1.0, 2.5, 0.0])
def test_predicted_room_matches_the_room_of_the_real_inputs(cfg_scale: float) -> None:
    """`predicted_room` builds its inputs with a model view that has only `config` and
    `device`. This fails (AttributeError) if `templates.prepare_inputs` ever reads more of the
    model, on any branch: no CFG, CFG with a negative prompt, and a reference."""
    runtime = FakeRuntime()
    runtime.model = model_with_codec_facts()
    codes = torch.zeros((7, 16), dtype=torch.int16)
    for reference in (NoRef(), CodesRef(codes=codes, ref_text="a reference transcript")):
        real = prepare_piece(
            FakeTokenizer(), runtime.model, reference, "Hello there.", "Speak.", cfg_scale
        )

        predicted = predicted_room(
            runtime, FakeTokenizer(), reference, "Hello there.", "Speak.", cfg_scale, None
        )

        assert predicted == piece_room(runtime, real, None)


# --- anchor sizing: later pieces' lengths, measured before the gate ---------------------------


ANCHOR_TEXT = "The opening piece, which becomes the anchor."
LATER_TEXTS = ["A second piece.", "A third, somewhat longer piece of the text.", "Fourth."]


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0, 0.0])
def test_anchor_sizing_predicts_every_anchored_prompt_length(cfg_scale: float) -> None:
    """The anchor check after piece 0 is arithmetic on lengths measured before the gate; it
    must give exactly the length the real anchored prompt has, for any frame count, on every
    CFG branch shape."""
    runtime = FakeRuntime()
    runtime.model = model_with_codec_facts()

    sizing = anchor_sizing(
        runtime, FakeTokenizer(), ANCHOR_TEXT, LATER_TEXTS, "Speak.", cfg_scale
    )

    def real_length(reference: Any, text: str) -> PromptLength:
        return prompt_length(
            prepare_piece(FakeTokenizer(), runtime.model, reference, text, "Speak.", cfg_scale)
        )

    assert sizing.later_lengths == tuple(real_length(NoRef(), text) for text in LATER_TEXTS)
    for frames in (1, 2, 7, 40):
        anchor = CodesRef(
            codes=torch.zeros((frames, 16), dtype=torch.int16), ref_text=ANCHOR_TEXT
        )
        for text, length in zip(LATER_TEXTS, sizing.later_lengths, strict=True):
            assert sizing.anchored(length, frames) == real_length(anchor, text)


def test_anchor_sizing_needs_no_gpu_model() -> None:
    """It runs on the CPU executor before the gate: a model view with only `config` works."""
    runtime = SimpleNamespace(model=SimpleNamespace(config=model_with_codec_facts().config))

    sizing = anchor_sizing(runtime, FakeTokenizer(), ANCHOR_TEXT, LATER_TEXTS, "Speak.", 1.0)

    assert isinstance(sizing, AnchorSizing)
    assert len(sizing.later_lengths) == len(LATER_TEXTS)


@pytest.mark.xfail(
    reason="T066: a voice_id reference has no stand-in until the voice registry is wired; "
    "a voice_id request must not 500 before the gate (review 26b #6)",
    raises=TypeError,
    strict=True,
)
def test_a_voice_reference_has_a_stand_in_for_the_pre_gate_room_check() -> None:
    stand_in_reference(VoiceRef(voice_id="alice", ref_text_override=None), None, 16)


# --- piece_frame_limit: clamping (FR-036a, review 26b #9) ---------------------------------------


def test_a_clamped_piece_reports_the_clients_request_the_cap_and_the_room() -> None:
    events = RecordingEvents()

    limit = piece_frame_limit(
        PieceRoom(cap=750, room=30), events, request_id="r", piece_index=2, requested=None
    )

    assert limit == 30
    assert events.calls == [
        (
            "speech.piece_clamped",
            {
                "level": "warning",
                "request_id": "r",
                "piece_index": 2,
                "requested": None,
                "cap": 750,
                "room": 30,
            },
        )
    ]


def test_a_piece_within_its_cap_is_not_clamped_and_no_room_raises() -> None:
    events = RecordingEvents()

    assert piece_frame_limit(
        PieceRoom(cap=50, room=50), events, request_id="r", piece_index=0, requested=50
    ) == 50
    with pytest.raises(NoRoomError):
        piece_frame_limit(
            PieceRoom(cap=50, room=0), events, request_id="r", piece_index=1, requested=50
        )
    assert events.calls == []
