"""Per-piece synthesis: reference resolution, prompt preparation and PCM streaming.

specs/003-cpp-compatible-api/tasks.md T038, the single-reference part (data-model.md
"Reference", "Piece"). Only ``NoReference`` and ``InlineRef`` (`breeze_infer/http_fields.py`
`ReferenceSpec`) are resolved here; ``VoiceRef`` is a documented stub until T066 wires the
voice registry and its cached-KV prefix path. Multi-piece concerns -- anchoring a
no-reference request's first piece onto its own audio, and clamping a piece to the room
`FastBreezeStreamingRuntime.max_new_tokens_room` reports -- are T052's job, not this
module's yet.

Nothing here reads ``app.state``: every dependency (the tokenizer, the model, the audio
tokenizer, the GPU thread) is passed in by the caller (`breeze_infer/routes_speech.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from breeze_infer.audio import encode_prompt_waveform, pcm16
from breeze_infer.http_fields import InlineRef, NoReference, ReferenceSpec, VoiceRef
from breeze_infer.templates import get_template, prepare_inputs

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


# --- PCM ramp (contracts/http-api.md "Splitting": "chunks grow from --chunk-first to
# --chunk-max codec frames") ----------------------------------------------------------


def codec_samples_per_frame(runtime: Any) -> int:
    """The codec's frame size in samples (qwen-tts's ``decode_upsample_rate``): 1,920 on
    the bundled checkpoint, for a 12.5 fps frame rate at the 24 kHz output.

    Read off the already-loaded audio tokenizer's own config
    (``runtime.audio_tokenizer.config.decode_upsample_rate``, a live
    ``Qwen3TTSTokenizerV2Config`` attribute -- ``FastBreezeStreamingRuntime`` keeps the
    loaded tokenizer at ``self.audio_tokenizer``), not from ``config.json`` on disk: that
    is where ``breeze_infer.audio``'s codec-identity fingerprint reads the same field
    from (``_TOP_LEVEL_IDENTITY_FIELDS``), but a caller preparing to stream a piece
    already has the runtime, not a checkpoint path. The route calls this once per request
    (or once at startup) and passes the result as ``generate_piece``'s
    ``samples_per_frame`` (review finding #10) -- it is not called inside
    ``generate_piece`` itself, so a piece's hot loop never re-reads it.
    """
    return int(runtime.audio_tokenizer.config.decode_upsample_rate)


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
