"""Per-piece synthesis: reference resolution, prompt preparation and PCM streaming.

specs/003-cpp-compatible-api/tasks.md T038 and T052 (data-model.md "Reference", "Piece").
Only ``NoReference`` and ``InlineRef`` (`breeze_infer/http_fields.py` `ReferenceSpec`) are
resolved here; ``VoiceRef`` is a documented stub until T066 wires the voice registry and its
cached-KV prefix path. For long text, `anchor_codes` turns a no-reference request's piece 0
into the reference for every later piece, and `piece_room`/`piece_frame_limit` size each
piece against the room `FastBreezeStreamingRuntime.max_new_tokens_room` reports (clamping or
refusing it). The piece loop itself is `routes_speech._iter_pieces`.

Nothing here reads ``app.state``: every dependency (the tokenizer, the model, the audio
tokenizer, the GPU thread) is passed in by the caller (`breeze_infer/routes_speech.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from breeze_infer.audio import encode_prompt_waveform, pcm16
from breeze_infer.http_fields import InlineRef, NoReference, ReferenceSpec, VoiceRef
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import NoRoomError, PromptLength, prompt_length

if TYPE_CHECKING:
    from breeze_infer.reference_audio import DecodedAudio
    from models.fast_streaming import FastStreamingChunk

# --- Reference (synthesis-internal; data-model.md "Reference") ---------------------


@dataclass(frozen=True)
class NoRef:
    """No reference: voice design from ``instruction`` alone."""


@dataclass(frozen=True)
class CodesRef:
    """A reference by codes: an inline upload (encoded once, here) or, from T066 on, a
    saved voice with an overridden transcript."""

    codes: Any  # int tensor [frames, codebooks]
    ref_text: str


# ``PrefixRef`` (cached backbone KV for a saved or unnamed voice with no override) is
# deliberately not built yet: nothing before T066 resolves a ``VoiceRef`` at all, so a
# reference variant only the voice path would produce has no caller in this phase.
Reference = NoRef | CodesRef


async def resolve_reference(
    spec: ReferenceSpec,
    *,
    decoded_audio: DecodedAudio | None,
    audio_tokenizer: Any,
    gpu: Any,
) -> Reference:
    """Resolve one request's ``ReferenceSpec`` into a synthesis ``Reference``, once.

    The caller resolves a request's reference exactly once and reuses the result for
    every piece (and, inside a single-CFG piece, both the guided and unguided rows --
    they share one ``ref_audio_codes`` entry in the segments `templates.py` builds): an
    ``InlineRef``'s bytes are decoded off the GPU thread (`reference_audio.decode`,
    already done by the time this runs -- `decoded_audio` is the result) and encoded
    into codec tokens here, the one step that needs the codec model, so it needs to run
    on the GPU thread. ``gpu`` is duck-typed to `breeze_infer.gpu.GpuThread`'s
    ``async def run(fn, *args)`` (run ``fn(*args)`` on the GPU thread, return its
    result) rather than importing that type, since nothing else here needs it.
    """
    if isinstance(spec, NoReference):
        return NoRef()
    if isinstance(spec, InlineRef):
        if decoded_audio is None:
            # A caller bug, not a request error: routes_speech.py always decodes an
            # InlineRef's bytes (reference_audio.decode) before resolving it, so a
            # missing DecodedAudio here means that step was skipped, not that the
            # request itself was invalid -- an assert would vanish under `-O` and let
            # the encode below crash on `None.samples` instead.
            raise ValueError("InlineRef requires a decoded clip (decoded_audio is None)")
        codes = await gpu.run(
            encode_prompt_waveform,
            audio_tokenizer,
            decoded_audio.samples,
            decoded_audio.sample_rate,
        )
        return CodesRef(codes=codes, ref_text=spec.ref_text)
    if isinstance(spec, VoiceRef):
        # T066 connects this to the voice registry: no override uses the cached-KV
        # prefix path, an override re-encodes with the given text on the codes path.
        raise NotImplementedError(
            "voice_id references are not wired until T066 (voice registry lookup)"
        )
    raise TypeError(f"unknown reference spec: {spec!r}")


def anchor_codes(frames: list[Any], pad_id: int) -> Any | None:
    """The codes piece 0 leaves behind as every later piece's reference, or ``None``.

    ``frames`` are the frames `iter_audio_chunks` handed its ``token_observer``, one 1-D
    tensor of every codebook's code per generated frame, pad frames included. The runtime
    decodes only frames that are not all-pad (``_frame_flags``: ``(frame ==
    codebook_pad_token_id).all()``), and C++ ``generate_chunk`` keeps the same ones, so the
    anchor is exactly the audio piece 0 produced. A frame with only some codebooks at the pad
    id is kept, as both of those keep it. An EOS ends the loop before it becomes a frame, so
    there is none to drop. Ported from ``A:api.py`` ``_anchor_codes`` (~1002-1010).

    ``None`` when nothing is left (piece 0 produced zero non-pad frames): there is no audio
    to anchor on, so the later pieces stay voice design (data-model.md "Reference").
    """
    if not frames:
        return None
    stacked = torch.stack(frames)
    kept = stacked[~(stacked == pad_id).all(dim=1)]
    if kept.shape[0] == 0:
        return None
    return kept.to(device="cpu", dtype=torch.int16)


def stand_in_reference(spec: ReferenceSpec, predicted_frames: int | None, codebooks: int) -> Reference:
    """A reference with the shape ``resolve_reference`` will give, before the codec has run.

    Used only to size piece 0's prompt for its room check before the GPU gate is taken
    (FR-007: a ``400`` comes before ``409 busy``). An ``InlineRef`` becomes zero codes of
    its *predicted* frame count (`reference_audio.predicted_frames`): a prompt's length
    depends on how many frames the reference has, never on their values. ``codebooks`` is
    the model's ``num_codebooks``, the width `templates.py` checks reference codes against.
    """
    if isinstance(spec, NoReference):
        return NoRef()
    if isinstance(spec, InlineRef):
        if predicted_frames is None:
            raise ValueError("InlineRef requires its predicted frame count")
        codes = torch.zeros((predicted_frames, codebooks), dtype=torch.int16)
        return CodesRef(codes=codes, ref_text=spec.ref_text)
    raise TypeError(f"no stand-in for reference spec: {spec!r}")


def piece_seed(seed: int, index: int) -> int:
    """data-model.md "Piece": piece ``index``'s seed, wrapped to the runtime's uint32."""
    return (seed + index) & 0xFFFFFFFF


def prepare_piece(
    tokenizer: Any,
    model: Any,
    reference: Reference,
    text: str,
    instruction: str,
    cfg_scale: float,
) -> dict[str, Any]:
    """Build one piece's model inputs (sync; runs on the GPU thread).

    ``tokenizer``/``model`` are the plain model parts `templates.prepare_inputs` already
    takes, not the streaming runtime -- the runtime doesn't itself need to be on the GPU
    thread for this step, only its tokenizer and model do, and keeping them separate
    matches `tests/gpu/conftest.py`'s own ``synthesize`` helper and lets tests pass
    ``tests/fakes.py``'s ``FakeTokenizer``/``fake_model()`` without a `FakeRuntime` at all.

    No CFG negative/dual-branch inputs are built here (``guidance_scale_ref``/``_ins``
    stay ``None``): T038 only exercises single-reference, single-CFG pieces, same as
    `iter_audio_chunks`'s ``no_cfg``/``single_cfg`` paths -- ``cfg_scale`` alone selects
    between them inside `prepare_inputs`.
    """
    request: dict[str, Any] = {"text": text, "instruction": instruction}
    if isinstance(reference, CodesRef):
        request["ref_text"] = reference.ref_text
        request["ref_audio_codes"] = reference.codes
        template = get_template("ref_edit_tata")
    elif isinstance(reference, NoRef):
        template = get_template("tts_instruction")
    else:
        raise TypeError(f"unknown reference: {reference!r}")

    return prepare_inputs(
        tokenizer,
        model,
        [request],
        template,
        guidance_scale=cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )


# --- Room (data-model.md "Piece"; FR-036a, BC-47) -----------------------------------------


@dataclass(frozen=True)
class PieceRoom:
    """How many frames one piece may generate.

    ``cap`` is what the request allows: its ``max_new_tokens``, or the model default,
    clamped to the server ceiling. ``room`` is what the piece will really get:
    ``min(cap, frames the context leaves after its prompt)``; ``<= 0`` means none.
    """

    cap: int
    room: int


def piece_room(runtime: Any, inputs: dict[str, Any], requested: int | None) -> PieceRoom:
    """``inputs``' room, from the runtime's own estimate (`max_new_tokens_room`, which shares
    ``_prefill_plan`` with the decode loop, so it stops exactly where the loop would).

    The cap comes from the runtime's ``frame_cap``, the one place that resolves ``None`` to
    the model default and applies the ceiling; re-deriving it here would be a second copy of
    that rule to keep in step.
    """
    return PieceRoom(
        cap=runtime.frame_cap(requested),
        room=runtime.max_new_tokens_room(requested, inputs),
    )


def predicted_room(
    runtime: Any,
    tokenizer: Any,
    reference: Reference,
    text: str,
    instruction: str,
    cfg_scale: float,
    requested: int | None,
) -> PieceRoom:
    """A piece's room, computed from inputs built on the CPU and then dropped.

    The speech route uses it for piece 0 before the GPU gate is taken (`routes_speech`).

    Builds the piece's inputs exactly as `prepare_piece` does, but on the CPU (`_cpu_model`),
    so this needs neither the GPU thread nor the gate, and leaves no tensor on the device.
    That view is safe exactly as long as templates read nothing else from the model: any
    other attribute is an AttributeError here, and
    `tests/test_synthesis.py::test_predicted_room_matches_the_room_of_the_real_inputs` runs
    every template branch through it to catch that.
    For piece 0, ``reference`` is `stand_in_reference`'s result, so the prompt has the length
    the real one will have, and the route still checks the real inputs on the GPU thread
    afterwards, in case the codec's frame count differs from the prediction.

    ``tokenizer`` must be one no other thread is using at the same time: the route's own copy
    (`routes_speech.CpuTokenizer`), never ``runtime.tokenizer``, which the GPU thread uses.
    """
    inputs = prepare_piece(
        tokenizer, _cpu_model(runtime), reference, text, instruction, cfg_scale
    )
    return piece_room(runtime, inputs, requested)


def _cpu_model(runtime: Any) -> Any:
    """The model view `predicted_room` and `anchor_sizing` build prompts with: its ``config``
    and ``device="cpu"``, the two attributes `templates.prepare_inputs` reads."""
    return SimpleNamespace(config=runtime.model.config, device="cpu")


@dataclass(frozen=True)
class AnchorSizing:
    """What deciding on piece 0's anchor needs, measured before piece 0 exists
    (`anchor_sizing`), so the decision itself is arithmetic.

    ``later_lengths`` are the later pieces' prompt lengths without an anchor, in order. An
    anchor of ``frames`` frames adds ``anchor_tokens + frames * tokens_per_frame`` tokens to
    every row of each of them (`anchored`).
    """

    later_lengths: tuple[PromptLength, ...]
    anchor_tokens: int
    tokens_per_frame: int

    def anchored(self, length: PromptLength, frames: int) -> PromptLength:
        """``length`` with an anchor of ``frames`` frames in front of every row."""
        added = self.anchor_tokens + frames * self.tokens_per_frame
        return length._replace(seq_len=length.seq_len + added)


def anchor_sizing(
    runtime: Any,
    tokenizer: Any,
    anchor_text: str,
    later_texts: list[str],
    instruction: str,
    cfg_scale: float,
) -> AnchorSizing:
    """Measure what `AnchorSizing` holds, from prompts built on the CPU and then dropped.

    ``anchor_text`` is piece 0's text, the anchor's transcript; its frames are not known yet.
    What the anchor adds is measured, not re-derived from `templates.py`'s rules: piece 1 is
    built with stand-in anchors of one and of two frames, and the differences give the
    per-frame and the fixed part. That the addition is the same for every later piece, every
    CFG row and any frame count holds because the reference segments come first on every row,
    ahead of the same text segment, and the tokenizer splits the prompt at the audio markers
    between them; `tests/test_synthesis.py` checks it with the fake tokenizer and
    `tests/gpu/test_speech_long_text.py` with the real one.

    ``tokenizer`` follows `predicted_room`'s rule: one no other thread is using.
    """
    cpu_model = _cpu_model(runtime)
    codebooks = int(runtime.model.config.num_codebooks)

    def length(reference: Reference, text: str) -> PromptLength:
        return prompt_length(
            prepare_piece(tokenizer, cpu_model, reference, text, instruction, cfg_scale)
        )

    def stand_in(frames: int) -> CodesRef:
        codes = torch.zeros((frames, codebooks), dtype=torch.int16)
        return CodesRef(codes=codes, ref_text=anchor_text)

    later_lengths = tuple(length(NoRef(), text) for text in later_texts)
    one_frame = length(stand_in(1), later_texts[0]).seq_len
    two_frames = length(stand_in(2), later_texts[0]).seq_len
    tokens_per_frame = two_frames - one_frame
    return AnchorSizing(
        later_lengths=later_lengths,
        anchor_tokens=one_frame - tokens_per_frame - later_lengths[0].seq_len,
        tokens_per_frame=tokens_per_frame,
    )


def piece_frame_limit(
    room: PieceRoom,
    events: Any,
    *,
    request_id: str,
    piece_index: int,
    requested: int | None,
) -> int:
    """The ``max_new_tokens`` to generate a piece with, or ``NoRoomError`` if it has no room.

    A room below the cap clamps the piece: it is generated up to the room and ends
    normally, as reaching ``max_new_tokens`` does, and ``speech.piece_clamped`` records it
    (FR-036a) with the client's ``requested`` value (``None``: the model default), the
    server's ``cap`` for it and the ``room``. Passing the room as ``max_new_tokens`` (rather than letting the context stop
    the loop) makes the runtime mark the last chunk final, as at any other token limit.

    No room at all raises. For piece 0 the route has already turned that into ``400
    text_too_long`` before streaming; for a later piece the ``200`` is already out, so the
    exception aborts the stream (FR-013, BC-47).
    """
    if room.room <= 0:
        raise NoRoomError(f"piece {piece_index} leaves no room to generate")
    if room.room < room.cap:
        events.emit(
            "speech.piece_clamped",
            level="warning",
            request_id=request_id,
            piece_index=piece_index,
            requested=requested,
            cap=room.cap,
            room=room.room,
        )
    return room.room


# --- PCM ramp (contracts/http-api.md "Splitting": "chunks grow from --chunk-first to
# --chunk-max codec frames") ----------------------------------------------------------


def codec_samples_per_frame(runtime: Any) -> int:
    """The codec's frame size in samples (qwen-tts's ``decode_upsample_rate``): 1,920 on
    the bundled checkpoint, for a 12.5 fps frame rate at the 24 kHz output.

    Read through the loaded audio tokenizer's own accessor,
    ``runtime.audio_tokenizer.get_decode_upsample_rate()`` (``qwen_tts``'s
    ``Qwen3TTSTokenizer`` wrapper, which asks its codec model;
    ``FastBreezeStreamingRuntime`` keeps the wrapper at ``self.audio_tokenizer``). Not
    ``audio_tokenizer.config``: the wrapper sets that only in ``from_pretrained`` and leaves
    it ``None`` otherwise. Not ``config.json`` on disk either: the caller has the runtime,
    not a checkpoint path. There is no fallback value: a tokenizer without the accessor
    raises, since a guessed frame size would silently mis-size every PCM flush.

    The route calls this once per request and passes the result as ``generate_piece``'s
    ``samples_per_frame`` (review finding #10) -- it is not called inside
    ``generate_piece`` itself, so a piece's hot loop never re-reads it.
    """
    accessor = getattr(runtime.audio_tokenizer, "get_decode_upsample_rate", None)
    if accessor is None:
        raise TypeError(
            "the audio tokenizer has no get_decode_upsample_rate(); cannot tell the "
            f"codec's samples per frame ({type(runtime.audio_tokenizer).__name__})"
        )
    return int(accessor())


def ramp_pcm(
    chunks: Iterable[FastStreamingChunk], chunk_first: int, chunk_max: int, samples_per_frame: int
) -> Iterator[bytes]:
    """Regroup one piece's streamed chunks into growing PCM byte flushes.

    Ported from ``A:api.py``'s ``_ramp_pcm`` (`git show api-alignment:breeze_infer/api.py`
    ~199-238, itself porting the C++ server's ``generate_chunk`` flush from
    ``generation.cpp``), fused with that module's ``_chunk_audio`` (pulling ``.audio`` off
    each `FastStreamingChunk`) and ``_pcm16`` (this repo's `breeze_infer.audio.pcm16`) so
    this yields wire-ready bytes directly instead of intermediate float32 arrays.

    The first flush is ``chunk_first`` codec frames; each flush after it grows by
    ``chunk // 3 + 1`` frames, capped at ``chunk_max``. A flush goes out as soon as enough
    audio is buffered (so the first one is never held back past ``chunk_first`` frames),
    and whatever is left when the piece ends goes out as one final, possibly short,
    flush. The ramp restarts for every piece (a fresh ``ramp_pcm`` call per piece), since
    C++ starts it afresh in each ``generate_chunk`` call too.

    ``samples_per_frame`` is the caller's to supply (review finding #8), not inferred from
    the first chunk: inferring it silently accepts a piece with zero non-empty chunks (and
    so never actually measures anything) as if that were fine, and ties this function's
    correctness to chunk arrival order rather than a fact the caller already has.
    """
    chunk_max = max(1, chunk_max)
    chunk = min(max(1, chunk_first), chunk_max)
    pending: list[Any] = []
    pending_samples = 0
    for streaming_chunk in chunks:
        audio = streaming_chunk.audio
        if audio.size == 0:
            continue
        pending.append(audio)
        pending_samples += audio.size
        if pending_samples < chunk * samples_per_frame:
            continue
        buffered = pending[0] if len(pending) == 1 else np.concatenate(pending)
        start = 0
        while buffered.size - start >= chunk * samples_per_frame:
            end = start + chunk * samples_per_frame
            yield pcm16(buffered[start:end])
            start = end
            chunk = min(chunk + chunk // 3 + 1, chunk_max)
        rest = buffered[start:]
        pending = [rest] if rest.size else []
        pending_samples = rest.size
    if pending:
        yield pcm16(pending[0] if len(pending) == 1 else np.concatenate(pending))


def generate_piece(
    runtime: Any,
    inputs: dict[str, Any],
    *,
    request_id: str,
    seed: int,
    chunk_first: int,
    chunk_max: int,
    samples_per_frame: int,
    prefix: Any | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_new_tokens: int | None = None,
    token_observer: Callable[[Any], None] | None = None,
) -> Iterator[bytes]:
    """One piece's PCM bytes, a sync generator the route steps on the GPU thread.

    Each ``next()`` -- one `breeze_infer.gpu.GpuSession.step()` call -- runs the runtime
    forward (via ``ramp_pcm``'s wrapped ``chunks``) until one growing-ramp PCM flush is
    ready, so a single step never does more GPU work than one flush needs. The sampling
    overrides are the `SpeechRequest`'s own (``None`` means the runtime's default); the
    caller passes ``piece_seed(request.seed, index)`` for ``seed``.

    ``samples_per_frame`` is required, not defaulted or inferred (review finding #10):
    the caller passes ``codec_samples_per_frame(runtime)`` (above), computed once per
    request rather than re-read on every piece.

    ``token_observer`` is passed straight to the runtime, which calls it with every
    generated frame (pad frames included); the route uses it to collect piece 0's frames
    for `anchor_codes`.

    ``chunks.close()`` always runs, even if this generator itself is closed early (a
    client disconnect closes the piece's `breeze_infer.gpu.GpuSession`, which closes this
    generator, which then closes ``chunks`` here) -- the runtime's own generator must be
    closed to release whatever it holds, exactly as `models/fast_streaming.py`'s own
    callers already do.
    """
    chunks = runtime.iter_audio_chunks(
        inputs,
        request_id=request_id,
        seed=seed,
        token_observer=token_observer,
        prefix=prefix,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )
    try:
        yield from ramp_pcm(chunks, chunk_first, chunk_max, samples_per_frame)
    finally:
        chunks.close()
