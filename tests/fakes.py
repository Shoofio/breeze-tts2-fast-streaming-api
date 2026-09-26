"""GPU-free stand-ins used at the model edge by unit tests.

These replace only the pieces that need a CUDA device and a checkpoint (the audio
tokenizer/codec, the text tokenizer, and the streaming runtime). Everything the tests
exercise above that edge (templates, store, routes, the session machine, the segmenter)
is the real code.

**Principle V deviation** (plan.md "Complexity Tracking", re-recorded from 001's deviation
at `api-alignment:specs/001-saved-voice-references/plan.md:167-176`): API, streaming and
WebSocket tests use `FakeRuntime` in place of the real model because the checkpoint needs
a CUDA GPU and about 8 GiB. The fake stands in at the GPU edge only. The real path is
covered by the `gpu` test suite and the live SillyTavern gates, which must pass before
each phase closes.

**Expiry**: delete this module and switch the no-GPU suite to the real runtime once a
CPU-loadable miniature Breeze checkpoint fixture exists under `tests/fixtures/`.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from breeze_infer.routes_voices import VoiceServices
from breeze_infer.voice_prefix import VoicePrefixCache
from breeze_infer.voice_registry import VoiceRegistry
from breeze_infer.voice_store import VoiceStore

# ``models.fast_streaming`` pulls in the cudagraph submodules and costs real wall-clock
# time to import (finding #8, T021 review 2) even though most of this module's own
# consumers (FakeCodec, codec_frame_count, RecordingEvents, FakeTokenizer) need none of
# it. So the real import is deferred to inside ``FakeRuntime.iter_audio_chunks`` and
# ``FakeRuntime.max_new_tokens_room`` below; this one is TYPE_CHECKING-only (never runs)
# purely so the return-type annotation resolves for a type checker/linter without paying
# for it at runtime.
if TYPE_CHECKING:
    from models.fast_streaming import FastStreamingChunk


class RecordingEvents:
    """A small fake for the `events` argument `install_error_handlers`/middleware take: records
    every emitted event instead of writing anywhere. Shared by test_version_header,
    test_api_errors and test_body_limit so there's one definition of "a fake event sink" (V6).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def emit(self, name: str, **fields: object) -> None:
        self.calls.append((name, fields))


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, list[int] | torch.Tensor]:
        del add_special_tokens
        # Position-independent ids so that tokenizing two halves separately
        # and concatenating equals tokenizing the whole string.
        ids = [ord(char) + 2 for char in text]
        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, input_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "x" * len(input_ids)


def fake_model():
    return SimpleNamespace(
        config=SimpleNamespace(
            num_codebooks=16,
            codebook_pad_token_id=2050,
            # frame_size == samples per codec frame (Mimi's ``MimiConfig``
            # property); 1920 matches 24000 Hz / 12.5 fps, same as the real
            # checkpoint, so voice "seconds" math is exercisable in tests.
            codec_config=SimpleNamespace(frame_size=1920),
        ),
        device="cpu",
    )


def model_with_codec_facts():
    """``fake_model()`` plus the codec facts ``templates._codec_facts`` requires
    (``codec_config.codebook_size``, cross-checked against ``codebook_pad_token_id``).
    ``fake_model()`` deliberately leaves this out (``tests/test_templates.py``'s own
    ``_model_with_codec_facts`` tests that omission on purpose), so this augments the
    ``SimpleNamespace`` here instead -- shared by every other caller that needs a model
    ``prepare_piece``/``prepare_inputs`` can actually use (review-agent pass 1, finding 9:
    previously duplicated in ``tests/test_synthesis.py`` and ``tests/test_routes_speech.py``).
    """
    model = fake_model()
    model.config.codec_config.codebook_size = 2048
    return model


# --- FakeCodec: the 12 Hz audio tokenizer at the reference-encode edge -------------

# The real tokenizer is qwen-tts's ``tokenizer_12hz`` (its class name is literally the
# "12 Hz" codec the spec means, though its true frame rate is 12.5 fps):
# ``input_sample_rate = output_sample_rate = 24_000`` and
# ``encode_downsample_rate = 1920`` samples/frame (both from
# ``qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2``'s
# ``Qwen3TTSTokenizerV2Config``, read from the installed package since no checkpoint is
# downloaded in this repo — see the module docstring). Its ``encoder_config`` is a
# ``transformers`` ``MimiConfig`` with its own ``num_quantizers`` (32, the codec's full
# quantizer stack) -- but ``Qwen3TTSTokenizerV2Config.encoder_valid_num_quantizers`` (16)
# is what actually caps it: ``encode()`` slices
# ``audio_codes[:, :self.encoder_valid_num_quantizers]``
# (``qwen_tts...modeling_qwen3_tts_tokenizer_v2.py``), and that field, not the encoder's
# own ``num_quantizers``, is the source of ``CODEC_CODEBOOKS`` (16) below. This is
# unrelated to the *backbone*'s own ``num_codebooks`` config default of 32
# (``models/breeze_base_config.py``), a different field on a different model that just
# happens to share the number.
CODEC_SAMPLE_RATE = 24000
CODEC_SAMPLES_PER_FRAME = 1920
CODEC_CODEBOOKS = 16
# If a real checkpoint is ever added under tests/fixtures/, verify this against
# ``<checkpoint>/audio_tokenizer/config.json``'s ``codebook_size`` — the same directory
# ``breeze_infer/runtime.py:load_runtime`` loads the bundled audio tokenizer from. 2048 is
# not verified against such a file (none exists in this repo); it matches the Mimi codec's
# default (``transformers.models.mimi.configuration_mimi.MimiConfig.codebook_size``),
# which qwen-tts's 12 Hz tokenizer config also carries.
CODEC_CODEBOOK_SIZE = 2048


def codec_frame_count(num_samples: int, sr: int) -> int:
    """How many codec frames a wav of ``num_samples`` samples at ``sr`` Hz becomes.

    Two steps, both copied exactly from the real tokenizer so a test can assert an exact
    frame count (T036's ``predicted_frames`` is expected to use this same formula, and
    T049 asserts it against the real encode):

    1. **Resample to 24 kHz**, unless ``sr`` is already 24 kHz (``Qwen3TTSTokenizer``'s
       ``_normalize_audio_inputs``/``load_audio`` only call ``librosa.resample`` when
       ``int(sr) != target_sr``, so exactly-24 kHz audio skips this step entirely).
       ``librosa.resample`` computes the resampled length as
       ``int(np.ceil(y.shape[axis] * ratio))`` with ``ratio = float(target_sr) / orig_sr``
       computed *first*, as one float division, and *then* multiplied by the sample count
       (``librosa/core/audio.py:resample``). That is not the same value as
       ``ceil(num_samples * target_sr / orig_sr)`` computed the other order — float
       rounding differs at some rates (44.1/22.05/11.025/88.2/176.4 kHz), e.g. 30 seconds
       at 44.1 kHz resamples to one sample more than the "obvious" computation, which
       costs a whole extra frame at the boundary. Do not simplify this back to a single
       division.
    2. **Frame it**: the tokenizer's own integer ceiling division,
       ``-(-mask.sum() // encode_downsample_rate)`` -- not ``int(np.ceil(a / b))``, which
       is exact for these sizes but is float division first; the integer form is both
       what the real code does and cannot round differently no matter how large the
       sample count gets.
    """
    if num_samples <= 0:
        return 0
    sr = int(sr)
    if sr != CODEC_SAMPLE_RATE:
        ratio = float(CODEC_SAMPLE_RATE) / sr
        num_samples = int(np.ceil(num_samples * ratio))
    return -(-num_samples // CODEC_SAMPLES_PER_FRAME)


class _EncoderOutput:
    """Minimal stand-in for transformers' ``ModelOutput``, giving attribute, string-key
    and integer-index access to one field.

    The real return type, ``Qwen3TTSTokenizerV2EncoderOutput``
    (``qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2``), literally *is* a
    ``transformers.utils.ModelOutput`` subclass with exactly this one field,
    ``audio_codes``. This fake does not subclass or import the real ``ModelOutput``:
    importing ``transformers`` cold took several minutes of wall-clock time in this
    sandbox (mostly filesystem stats across its many submodules on a slow mount, not CPU
    -- the same class of cost finding #8 raised about ``models.fast_streaming``), which
    isn't worth paying for one field's dict/attribute duality.

    ``result[0]`` and ``result["audio_codes"]`` both give ``audio_codes`` itself (not a
    1-tuple containing it), matching ``ModelOutput.__getitem__``: an integer key indexes
    ``to_tuple()`` (the object's non-``None`` values, in field order), and with exactly
    one field that tuple is ``(audio_codes,)``, so index ``0`` unwraps straight back to
    ``audio_codes``.
    """

    def __init__(self, audio_codes: list[torch.Tensor]) -> None:
        self.audio_codes = audio_codes

    def to_tuple(self) -> tuple[list[torch.Tensor]]:
        return (self.audio_codes,)

    def __getitem__(self, key: str | int) -> list[torch.Tensor]:
        if isinstance(key, str):
            return getattr(self, key)
        return self.to_tuple()[key]


class FakeCodec:
    """Stands in for the bundled `qwen_tts` audio tokenizer at the point it turns a
    waveform into codes (`breeze_infer/audio.py:encode_prompt_waveform` indexes
    ``encode(wav, sr)["audio_codes"][0]``, exactly as the real ``Qwen3TTSTokenizer``'s
    ``ModelOutput`` return is indexed -- ``encode(...).audio_codes`` also works, on both
    the fake and the real thing).

    ``audio_codes`` is a one-element list holding a single ``int64`` ``LongTensor`` shaped
    ``[frames, codebooks]`` — the real return type
    (`qwen_tts/inference/qwen3_tts_tokenizer.py`'s `encode()` docstring: "12Hz: ...
    audio_codes: List[torch.LongTensor]"), not a numpy array. ``frames`` follows
    ``codec_frame_count`` and ``codebooks == CODEC_CODEBOOKS`` (16). Each tensor is a
    ``.transpose(0, 1)`` view (built codebooks-first, then transposed), not a fresh
    contiguous array — the real `encode()` does the same
    (``code[..., :T].transpose(0, 1)``), and code downstream of it must not assume
    contiguity.

    Codes are deterministic — a function of ``wav`` and ``sr`` only (a content hash, not
    randomness) — so the same reference always encodes to the same codes and re-encoding
    is detectable by a test. The hash is reduced modulo ``CODEBOOK_SIZE`` immediately after
    extraction, before any arithmetic with the ``int64`` codes array: a raw 8-byte hash is
    an unreduced value up to ``2**64 - 1``, which overflows ``int64`` (max ``2**63 - 1``)
    the moment it is added to one.

    A 0-sample wav raises ``RuntimeError``: the real Mimi encoder's convolution stack
    similarly rejects input shorter than its kernel (its exact exception type is not
    verified here -- no checkpoint is available in this repo to exercise it against).
    """

    SAMPLE_RATE = CODEC_SAMPLE_RATE
    SAMPLES_PER_FRAME = CODEC_SAMPLES_PER_FRAME
    CODEBOOKS = CODEC_CODEBOOKS
    CODEBOOK_SIZE = CODEC_CODEBOOK_SIZE
    def __init__(self) -> None:
        self.encode_calls = 0
        self.last_wav: np.ndarray | None = None
        self.last_sr: int | None = None

    def get_decode_upsample_rate(self) -> int:
        """The real wrapper's accessor (``Qwen3TTSTokenizer.get_decode_upsample_rate``),
        which `breeze_infer.synthesis.codec_samples_per_frame` reads: waveform samples per
        codec frame, the same value as SAMPLES_PER_FRAME."""
        return self.SAMPLES_PER_FRAME

    def encode(
        self, wav: np.ndarray, sr: int, return_dict: bool = True
    ) -> _EncoderOutput | tuple[list[torch.Tensor]]:
        if len(wav) == 0:
            raise RuntimeError(
                "FakeCodec.encode: a 0-sample wav can't be encoded (mirrors the real "
                "Mimi encoder's Conv1d stack rejecting input shorter than its kernel; "
                "see the class docstring)"
            )
        self.encode_calls += 1
        self.last_wav = wav
        self.last_sr = sr
        frames = codec_frame_count(len(wav), sr)
        wav_bytes = np.ascontiguousarray(wav, dtype=np.float32).tobytes()
        digest = hashlib.sha256(wav_bytes + int(sr).to_bytes(4, "little")).digest()
        seed = int.from_bytes(digest[:8], "little") % self.CODEBOOK_SIZE
        # Built (codebooks, frames) -- i.e. codebook-major -- then transposed, so the
        # returned tensor is a non-contiguous view, like the real ``encode()``'s
        # ``code[..., :T].transpose(0, 1)``, not a fresh (frames, codebooks) array.
        codebook_idx = np.arange(self.CODEBOOKS, dtype=np.int64).reshape(-1, 1)
        frame_idx = np.arange(frames, dtype=np.int64).reshape(1, -1)
        codes_codebook_major = (
            seed + frame_idx * self.CODEBOOKS + codebook_idx
        ) % self.CODEBOOK_SIZE
        codes = torch.as_tensor(codes_codebook_major, dtype=torch.int64).transpose(0, 1)
        output = _EncoderOutput([codes])
        # Like the real ``ModelOutput``-returning ``encode()``: ``return_dict=False``
        # gives the plain tuple (``to_tuple()``) instead of the attribute/dict-style
        # wrapper, matching how e.g. ``transformers`` model forwards accept the same
        # flag (review finding: encode()'s real signature has this parameter and the
        # fake didn't).
        return output if return_dict else output.to_tuple()


# --- FakeRuntime: the streaming runtime at the generation edge ---------------------


@dataclass(frozen=True)
class FakeStreamingConfig:
    """The subset of ``models.fast_streaming.FastStreamingConfig`` that
    ``FakeRuntime.max_new_tokens_room``/``frame_cap``/``_context_room`` read
    (``max_new_tokens``, the per-request ceiling; ``max_seq_len``, the context length;
    ``fast_backbone_prefill``, whether the prefill bucket-pads). A plain local
    dataclass, not the real one: constructing a `FakeRuntime` must never import
    ``models.fast_streaming`` just to build its default ``config`` (the module-import
    cost this file's docstring and finding #8 already avoid). Field defaults match
    ``FastStreamingConfig``'s own defaults.
    """

    max_new_tokens: int = 1500
    max_seq_len: int = 1024
    # Real default is False (``_prefill_plan``): the prefill runs at the exact prompt
    # length until the fast backbone-prefill path is turned on, which is what pads it
    # up to the nearest ``_PREFILL_TOKEN_GRANULARITY`` bucket. See ``_context_room``.
    fast_backbone_prefill: bool = False


class FakeRuntime:
    """Yields constant-valued chunks shaped like ``FastStreamingChunk``; records every
    call so tests can inspect it.

    Chunk shape and the loop's own structure are copied from
    ``models/fast_streaming.py:FastBreezeStreamingRuntime.iter_audio_chunks`` (roughly
    lines 1062-1178):
    - **Samples per chunk** = ``frames_per_chunk * CODEC_SAMPLES_PER_FRAME`` (derived, not
      hard-coded): the real runtime decodes ``codec_frames`` codec frames per chunk, and
      each frame is ``CODEC_SAMPLES_PER_FRAME`` (1920) samples. ``frames_per_chunk``
      mirrors the real ``_codec_chunk_frames`` (1 with the fast codec path, 2 without),
      and is configurable for the same reason.
    - **No chunk is final by default** (``is_final_on_last=False``, ``flush_frames=None``):
      with the fast codec (``_codec_chunk_frames = 1``), every generated frame is
      immediately flushed as its own chunk, so by the time generation ends via EOS (the
      common case, not the ``max_new_tokens`` limit), the buffer that would have carried a
      final, ``is_final=True`` chunk is already empty — the real generator simply stops
      without ever yielding one. **Consumers must treat generator exhaustion itself as the
      end of a piece, not the presence of an ``is_final`` chunk.** Pass
      ``is_final_on_last=True`` to instead simulate reaching the ``max_new_tokens`` limit,
      where the real code does mark the last chunk final; pass ``flush_frames`` to
      simulate the other case that produces a final chunk, below.
    - **``flush_frames`` models the real post-loop flush** (the ``if chunk_buffer:`` block
      right after the ``for step_idx in range(...)`` loop): with the non-fast codec
      (``frames_per_chunk=2``), a piece can end (EOS, or the sequence-length guard) with
      exactly one decoded frame still buffered -- one short of a full chunk, so it was
      never flushed inside the loop. The real code yields that leftover as one more
      chunk, marked ``is_final=True``, carrying only its own (smaller) frame/sample count
      -- *not* another ``frames_per_chunk``-sized chunk. Default ``None`` means no such
      chunk (the common fast-codec case above). This is on top of, not instead of, the
      ``chunks``/``frames_per_chunk`` chunks from the main loop.
    - **Timing keys** match the real per-chunk dict exactly: every chunk from the main
      loop has ``chunk_index``, ``codec_frames``, ``decode_launch_ms``, ``total_frames``,
      ``is_final``, ``codec_launch_ms`` and ``audio_d2h_ms``; chunk 0 of that loop
      additionally has ``ttfa_internal_ms`` and ``prefill_path``, plus ``prefill_gpu_ms``
      only when ``collect_timing=True`` (the real code only records the CUDA prefill
      events, and so only has a value to report, when timing collection is on). The
      ``flush_frames`` chunk, if any, never gets that chunk-0 enrichment even when it is
      the only (and so index-0) chunk yielded overall (``chunks=0``): the real post-loop
      flush is a separate branch of the generator from the mid-loop one that does the
      enrichment, and it does not repeat that step. The fake values themselves (0.0,
      "graph") are placeholders — only the keys and their gating are real.
    - **``token_observer`` is called once per generated frame, interleaved with the
      yields** (not all up front): the real loop calls it for every backbone step as soon
      as that frame is sampled, before deciding whether a codec chunk is ready to flush.
      Here, the ``frames_per_chunk`` (or ``flush_frames``) frames due before a chunk are
      observed immediately before that chunk is yielded (and, since the fake's blocking
      ``gate`` stands in for "no further generation has happened yet", is waited on
      *before* those frames are observed). Any frames left over after the last chunk (e.g.
      an all-pad piece with ``chunks=0`` and no ``flush_frames``, or trailing pad frames
      after the last decoded chunk) are still observed once the loop ends, since the real
      loop calls ``token_observer`` on pad frames too — only frames ``_frame_flags`` marks
      as not a terminal pad feed the audio buffer. ``frames`` defaults to one dummy
      tensor per frame this call is going to observe
      (``chunks * frames_per_chunk + (flush_frames or 0)``), so
      ``token_observer`` is exercised with no arguments beyond it, rather than silently
      observing nothing because the caller forgot to size a ``frames=`` list to match.

    Alignment with the real runtime (review finding: this fake used to accept calls the
    real runtime rejects, so a test could pass here and still raise in production):
    - ``iter_audio_chunks`` takes exactly the real method's keyword set
      (``models/fast_streaming.py:FastBreezeStreamingRuntime.iter_audio_chunks``):
      ``request_id``, ``seed``, ``token_observer``, ``prefix``, and the five per-request
      sampling overrides (``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
      ``max_new_tokens``). There is no ``reference=`` keyword on the real method and none
      here either -- a reference reaches the runtime two ways, both already covered by
      existing parameters: baked into ``inputs`` as ``ref_audio_codes`` (by
      `breeze_infer.templates.prepare_inputs`, for the "codes" `Reference` variant), or
      as a cached-KV ``prefix`` (the "prefix" variant, `build_reference_prefix`'s
      result, or a stand-in with the same ``prefix_len`` attribute -- see the next
      bullet). `synthesis.py`'s `resolve_reference`/`prepare_piece` build one of those
      two shapes; a test asserting "the inline reference is encoded once and reused for
      every piece and both CFG rows" (T038) does so by asserting `FakeCodec.encode_calls`
      stays 1 while the same encoded codes turn up, value-equal, in every recorded call's
      ``inputs["input_values"]`` (a fresh tensor each time -- `templates.py`'s
      ``_resolve_segment_audio_codes``/``_collate_inputs`` copy it via ``.to()``/
      ``.contiguous()`` on every `prepare_piece` call, so it is never literally the same
      object -- see `tests/test_synthesis.py`), not by inspecting a ``reference`` field
      on these calls -- there is none.
    - ``prefix`` is read strictly, like the real runtime's own ``prefix.prefix_len``: a
      caller passing a reference prefix must give a real-shaped object (the real
      ``ReferencePrefix``, or any stand-in with a ``prefix_len`` attribute, e.g.
      ``SimpleNamespace(prefix_len=64)``) -- not an arbitrary value that happens not to
      crash a lenient lookup.
    - The five overrides are validated exactly as the real ``_require_valid_overrides``
      does, by calling that same function (imported lazily, like ``FastStreamingChunk``
      below, so constructing or draining a `FakeRuntime` that never overrides anything
      still never pays to import ``models.fast_streaming``): ``None`` is the default,
      anything else must satisfy the real per-override rules (``_OVERRIDE_RULES``: a
      range, and an integer for ``top_k``/``max_new_tokens``, never a bool). Since ``iter_audio_chunks`` is a generator function, validation
      -- here and on the real runtime -- runs on the first ``next()``, not at call time.
    - ``max_new_tokens_room`` approximates the real method's room estimate (frame cap via
      ``config``/``default_max_new_tokens`` -- ``default_max_new_tokens`` defaults to
      750, the deployed model's own ``generation_config.max_new_tokens``, and
      ``FakeStreamingConfig.max_new_tokens`` defaults to 1500, the deployed server's
      ceiling -- then the context room from the prefill length) by calling the real
      ``_require_valid_overrides`` and ``prompt_length`` helpers and porting
      ``frame_cap``'s two-line rule directly. ``_context_room`` uses the **exact**
      prompt length by default and bucket-pads to the nearest
      ``_PREFILL_TOKEN_GRANULARITY`` (32) only when ``self.config.fast_backbone_prefill``
      is set -- mirroring ``_prefill_plan``'s own default (``FastStreamingConfig
      .fast_backbone_prefill`` is ``False`` unless a profile turns the fast backbone-
      prefill path on).
      **Approximation gap**: a frozen warmup cache can still refuse an unwarmed bucket
      and fall back to an exact-length eager prefill even with the flag on
      (`models/fast_streaming.py:955-998`); this fake has no captured prefill graphs or
      warmup state to consult, so with the flag on it always bucket-pads as if every
      bucket were warmed. A test asserting an *exact* room number against a frozen/eager
      fallback boundary needs the real runtime (`tests/gpu/`), not this.
    - ``iter_audio_chunks`` raises the real runtime's ``ValueError`` when
      ``max_new_tokens_room`` is ``<= 0``, before recording the call -- inputs with no
      ``attention_mask`` (the bare ``{}`` a few call sites still pass, where there is no
      prompt to measure) skip this and every room/cap check below entirely, staying
      unconstrained. Otherwise, when that room allows fewer frames than ``chunks``
      (``* frames_per_chunk``, plus any ``flush_frames``) would naturally produce, the
      call is cut short at that many frames instead, with its last chunk marked
      ``is_final=True`` -- the real loop's ``reached_limit``, whether the cap came from
      ``max_new_tokens`` or the context room (both fold into one ``limit`` here, as they
      do into the real loop's own two stop conditions). A natural (uncapped) ending keeps
      using ``is_final_on_last``/``flush_frames`` exactly as documented above.
    - ``build_reference_prefix`` (the cached-KV "prefix" `Reference` variant) is
      deliberately not faked here: none of the tasks that use `FakeRuntime` today
      exercise a saved voice with no override, only the "codes" variant. Add it when a
      task needs it.

    ``fail_after`` raises ``RuntimeError`` once that many chunks of a call have been
    yielded, standing in for a mid-stream CUDA error (checked, like the gate, before that
    chunk's frames are observed or it is yielded). ``gate`` (a ``threading.Event``) is
    waited on just before chunk index ``gate_at`` (default 1, i.e. mid-piece for the
    default ``chunks=2``), so a test can hold generation there from another thread;
    ``gate_reached``, if given, is set right before that wait so the test can synchronize
    on "generation is now blocked" instead of racing a timeout. Chunk ``i`` of call ``n``
    has sample value ``n / 100`` so tests can tell pieces apart in the PCM body.
    """

    sample_rate = 24000

    def __init__(
        self,
        chunks: int = 2,
        *,
        frames_per_chunk: int = 1,
        flush_frames: int | None = None,
        frames: list[torch.Tensor] | None = None,
        fail_after: int | None = None,
        gate: threading.Event | None = None,
        gate_at: int = 1,
        gate_reached: threading.Event | None = None,
        is_final_on_last: bool = False,
        collect_timing: bool = False,
        prefill_path: str = "graph",
        config: FakeStreamingConfig | None = None,
        default_max_new_tokens: int | None = 750,
    ) -> None:
        # The real post-loop flush (module docstring, ``flush_frames`` bullet) only
        # exists because the main loop's ``chunk_ready`` check
        # (``len(chunk_buffer) >= self._codec_chunk_frames``) can leave 1..frames_per_chunk-1
        # frames buffered when generation ends -- never a full chunk's worth (the loop
        # would have flushed that already) and never zero (there would be nothing left to
        # flush). And a piece ends exactly one way: either it runs out of frames inside
        # the loop (a possible ``is_final_on_last`` chunk) or it has leftovers after
        # (``flush_frames``), never both, since the loop's own ``reached_limit`` branch
        # already flushes and breaks before any post-loop code can run.
        if flush_frames is not None and not (0 < flush_frames < frames_per_chunk):
            raise ValueError(
                "flush_frames must be > 0 and < frames_per_chunk (a full chunk would "
                f"have flushed already), got flush_frames={flush_frames!r} "
                f"frames_per_chunk={frames_per_chunk!r}"
            )
        if is_final_on_last and flush_frames:
            raise ValueError(
                "is_final_on_last and flush_frames both simulate how a piece ends and "
                "are mutually exclusive: the real loop's reached_limit branch already "
                "flushes and breaks, so there is never a post-loop flush after it"
            )
        self.chunks = chunks
        self.frames_per_chunk = frames_per_chunk
        self.flush_frames = flush_frames
        self.config = config or FakeStreamingConfig()
        self.default_max_new_tokens = default_max_new_tokens
        if frames is None:
            # One dummy tensor per frame this call will observe, so token_observer is
            # exercised even when the caller passes no ``frames=`` (finding #1).
            total_frames = chunks * frames_per_chunk + (flush_frames or 0)
            frames = [torch.tensor([i]) for i in range(total_frames)]
        self.frames = frames
        self.fail_after = fail_after
        self.gate = gate
        self.gate_at = gate_at
        self.gate_reached = gate_reached
        self.is_final_on_last = is_final_on_last
        self.collect_timing = collect_timing
        self.prefill_path = prefill_path
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def frame_cap(self, requested: int | None) -> int:
        """Ports ``FastBreezeStreamingRuntime.frame_cap`` exactly (it's two lines and
        pure Python -- no need to import the real one just for this)."""
        if requested is None:
            requested = self.default_max_new_tokens or self.config.max_new_tokens
        return min(int(requested), self.config.max_new_tokens)

    def max_new_tokens_room(
        self, requested: int | None, inputs: dict[str, Any], *, prefix_len: int = 0
    ) -> int:
        """Approximates ``FastBreezeStreamingRuntime.max_new_tokens_room`` -- see the
        class docstring's "Approximation gap" paragraph for what this can't reproduce
        without real captured prefill graphs."""
        # Lazy for the same reason as in iter_audio_chunks below (finding #8).
        from models.fast_streaming import prompt_length

        return self.room_for_length(requested, prompt_length(inputs), prefix_len=prefix_len)

    def room_for_length(self, requested: int | None, length: Any, *, prefix_len: int = 0) -> int:
        """``FastBreezeStreamingRuntime.room_for_length``: the room above, for a prompt known
        only by its ``PromptLength``."""
        from models.fast_streaming import _require_valid_overrides

        _require_valid_overrides(max_new_tokens=requested)
        return min(self.frame_cap(requested), self._context_room(length.seq_len, prefix_len))

    def _context_room(self, seq_len: int, prefix_len: int) -> int:
        """The real ``_context_room``/``_prefill_plan`` rule: exact prompt length by
        default, bucketed only when ``self.config.fast_backbone_prefill`` is set (review
        finding #2 -- the real ``_prefill_plan`` only pads when the fast backbone-prefill
        path is on, ``FastStreamingConfig.fast_backbone_prefill`` default ``False``; this
        used to bucket unconditionally, which is backwards from the real default).

        With the flag on, the bucketed length still falls back to the exact length
        whenever bucketing would leave less than ``MIN_SUFFIX_FRAMES`` (12) of room --
        not only when it would overflow ``max_seq_len`` outright (review finding #4,
        f021d7b's real ``_prefill_plan``): padding a short suffix up to its bucket could
        otherwise eat most of a registered prefix's promised room, so the graph path is
        abandoned a bit before the hard overflow point, and every longer prompt after
        that point stays eager too (keeps room monotonic in prompt length -- see the
        real docstring).
        """
        from models.fast_streaming import _PREFILL_TOKEN_GRANULARITY, MIN_SUFFIX_FRAMES

        exact_len = prefix_len + seq_len
        if not self.config.fast_backbone_prefill:
            return self.config.max_seq_len - exact_len - 1
        bucketed_len = (
            prefix_len
            + -(-seq_len // _PREFILL_TOKEN_GRANULARITY) * _PREFILL_TOKEN_GRANULARITY
        )
        bucketed_room = self.config.max_seq_len - bucketed_len - 1
        if bucketed_room < MIN_SUFFIX_FRAMES:
            return self.config.max_seq_len - exact_len - 1
        return bucketed_room

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
        seed: int | None = None,
        token_observer: Callable[[torch.Tensor], None] | None = None,
        prefix: Any | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[FastStreamingChunk]:
        # Lazy: models.fast_streaming (cudagraph submodules) is slow to import, and most
        # of tests/fakes.py's own consumers never call this method (finding #8).
        from models.fast_streaming import (
            FastStreamingChunk,
            NoRoomError,
            _require_valid_overrides,
        )

        # Validated -- and, like the real generator, only once the caller starts
        # iterating, not at call time -- with the exact same rules the real runtime
        # uses, by calling that same function rather than re-implementing its rules
        # (review finding: the fake used to accept calls the real runtime rejects).
        _require_valid_overrides(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        # Strict, like the real runtime's own ``prefix.prefix_len`` (no fallback): a
        # ``prefix`` must be a real-shaped object (``ReferencePrefix``, or a stand-in
        # with the same attribute), not any value that happens not to crash a lenient
        # ``getattr`` (review finding #3).
        prefix_len = 0 if prefix is None else int(prefix.prefix_len)

        # How many frames this call may produce at most, mirroring the real loop's own
        # combination of the ``max_new_tokens`` cap and the context-room boundary
        # (``max_new_tokens_room``, and the per-step ``prefill_len + step_idx >=
        # max_seq_len - 1`` break) -- review finding #1. Skipped for ``inputs`` with no
        # ``attention_mask`` (the bare ``{}`` some tests still pass): there is no prompt
        # to measure, so those calls stay unconstrained, as before this fake modeled a
        # cap at all.
        limit: int | None = None
        if "attention_mask" in inputs:
            limit = self.max_new_tokens_room(max_new_tokens, inputs, prefix_len=prefix_len)
            if limit <= 0:
                # The real runtime's own exception type (not a bare ValueError), so a
                # route that maps NoRoomError specifically to 400 text_too_long (and
                # leaves every other ValueError as an unhandled 500 -- T046 review,
                # finding 1) gets the same 400 here that the real runtime would give.
                # The message text still matches the real one, so tests matching on it
                # keep working unchanged.
                raise NoRoomError(
                    "prompt leaves no room to generate in the "
                    f"{self.config.max_seq_len}-token context"
                )

        call_index = len(self.calls)
        self.calls.append(
            {
                "inputs": inputs,
                "request_id": request_id,
                "seed": seed,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "prefix": prefix,
                "observed": token_observer is not None,
            }
        )

        # ``chunks``/``frames_per_chunk``/``flush_frames`` describe how a call would end
        # on its own (EOS with an empty buffer, a token-limit simulation, or a post-loop
        # leftover flush -- the docstring bullets above); ``limit``, when it constrains
        # fewer frames than that, cuts the call short instead, exactly as the real
        # runtime's own cap/room would. A call that produces fewer frames than
        # requested, cut short by ``limit``, gets its last chunk marked final -- the real
        # loop's ``reached_limit``.
        natural_frames = self.chunks * self.frames_per_chunk + (self.flush_frames or 0)
        capped = limit is not None and limit < natural_frames
        if capped:
            regular_chunks = min(self.chunks, limit // self.frames_per_chunk)
            cap_leftover = limit - regular_chunks * self.frames_per_chunk
            if cap_leftover <= 0:
                cap_leftover = None
        else:
            regular_chunks = self.chunks
            cap_leftover = None

        frame_iter = iter(self.frames)
        total_frames = 0
        try:
            for index in range(regular_chunks):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("CUDA error: an illegal memory access (fake)")
                if self.gate is not None and index == self.gate_at:
                    if self.gate_reached is not None:
                        self.gate_reached.set()
                    self.gate.wait()
                if token_observer is not None:
                    for _ in range(self.frames_per_chunk):
                        frame = next(frame_iter, None)
                        if frame is not None:
                            token_observer(frame)
                total_frames += self.frames_per_chunk
                if capped:
                    is_final = cap_leftover is None and index == regular_chunks - 1
                else:
                    is_final = self.is_final_on_last and index == self.chunks - 1
                timing: dict[str, float | int | bool | str] = {
                    "chunk_index": index,
                    "codec_frames": self.frames_per_chunk,
                    "decode_launch_ms": 0.0,
                    "total_frames": total_frames,
                    "is_final": is_final,
                    "codec_launch_ms": 0.0,
                    "audio_d2h_ms": 0.0,
                }
                if index == 0:
                    timing["ttfa_internal_ms"] = 0.0
                    timing["prefill_path"] = self.prefill_path
                    if self.collect_timing:
                        timing["prefill_gpu_ms"] = 0.0
                yield FastStreamingChunk(
                    audio=np.full(
                        self.frames_per_chunk * CODEC_SAMPLES_PER_FRAME,
                        call_index / 100,
                        dtype=np.float32,
                    ),
                    sample_rate=self.sample_rate,
                    codec_frames=self.frames_per_chunk,
                    is_final=is_final,
                    timing=timing,
                )
            if capped:
                if cap_leftover is not None:
                    if token_observer is not None:
                        for _ in range(cap_leftover):
                            frame = next(frame_iter, None)
                            if frame is not None:
                                token_observer(frame)
                    total_frames += cap_leftover
                    yield FastStreamingChunk(
                        audio=np.full(
                            cap_leftover * CODEC_SAMPLES_PER_FRAME,
                            call_index / 100,
                            dtype=np.float32,
                        ),
                        sample_rate=self.sample_rate,
                        codec_frames=cap_leftover,
                        is_final=True,
                        timing={
                            "chunk_index": regular_chunks,
                            "codec_frames": cap_leftover,
                            "decode_launch_ms": 0.0,
                            "total_frames": total_frames,
                            "is_final": True,
                            "codec_launch_ms": 0.0,
                            "audio_d2h_ms": 0.0,
                        },
                    )
            elif self.flush_frames is not None:
                if token_observer is not None:
                    for _ in range(self.flush_frames):
                        frame = next(frame_iter, None)
                        if frame is not None:
                            token_observer(frame)
                total_frames += self.flush_frames
                # No chunk-0 enrichment here even when self.chunks == 0 (see the
                # flush_frames docstring bullet): the real post-loop flush branch never
                # goes through the mid-loop's ``if chunk_index == 0`` step.
                yield FastStreamingChunk(
                    audio=np.full(
                        self.flush_frames * CODEC_SAMPLES_PER_FRAME,
                        call_index / 100,
                        dtype=np.float32,
                    ),
                    sample_rate=self.sample_rate,
                    codec_frames=self.flush_frames,
                    is_final=True,
                    timing={
                        "chunk_index": self.chunks,
                        "codec_frames": self.flush_frames,
                        "decode_launch_ms": 0.0,
                        "total_frames": total_frames,
                        "is_final": True,
                        "codec_launch_ms": 0.0,
                        "audio_d2h_ms": 0.0,
                    },
                )
            # Not when capped: the real loop samples nothing more once the cap/room
            # stops it, so there are no further "trailing pad frames" to observe --
            # unlike the natural-EOS case below, where frames already sampled after the
            # last decoded chunk are still real observations to replay.
            if not capped and token_observer is not None:
                for frame in frame_iter:
                    token_observer(frame)
        finally:
            self.closed += 1


def open_no_voices(_runtime: Any) -> VoiceServices:
    """A stand-in for `api.Components.open_voices` in tests that don't use the voice routes:
    empty voice services that never touch the disk (nothing scans `voices_dir`, and nothing
    in those tests writes a voice)."""
    return VoiceServices(
        store=VoiceStore(
            Path("unused-test-voices"),
            codebooks=CODEC_CODEBOOKS,
            codebook_size=CODEC_CODEBOOK_SIZE,
            codec_fingerprint="0" * 64,
            events=RecordingEvents(),
            clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        registry=VoiceRegistry(clock=lambda: 0.0),
        prefix_cache=VoicePrefixCache(bytes_per_token=1, on_event=lambda *_a, **_k: None),
    )
