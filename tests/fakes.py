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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

# ``models.fast_streaming`` pulls in the cudagraph submodules and costs real wall-clock
# time to import (finding #8, T021 review 2) even though most of this module's own
# consumers (FakeCodec, codec_frame_count, RecordingEvents, FakeTokenizer) need none of
# it. So the real import is deferred to inside ``FakeRuntime.iter_audio_chunks`` below;
# this one is TYPE_CHECKING-only (never runs) purely so the return-type annotation
# resolves for a type checker/linter without paying for it at runtime.
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
    """Minimal stand-in for transformers' ``ModelOutput``, giving both attribute and
    string-key access to one field.

    The real return type, ``Qwen3TTSTokenizerV2EncoderOutput``
    (``qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2``), literally *is* a
    ``transformers.utils.ModelOutput`` subclass with exactly this one field,
    ``audio_codes``. This fake does not subclass or import the real ``ModelOutput``:
    importing ``transformers`` cold took several minutes of wall-clock time in this
    sandbox (mostly filesystem stats across its many submodules on a slow mount, not CPU
    -- the same class of cost finding #8 raised about ``models.fast_streaming``), which
    isn't worth paying for one field's dict/attribute duality.
    """

    def __init__(self, audio_codes: list[torch.Tensor]) -> None:
        self.audio_codes = audio_codes

    def __getitem__(self, key: str) -> list[torch.Tensor]:
        return getattr(self, key)


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

    def encode(self, wav: np.ndarray, sr: int) -> _EncoderOutput:
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
        return _EncoderOutput([codes])


# --- FakeRuntime: the streaming runtime at the generation edge ---------------------


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
      loop calls ``token_observer`` on pad frames too — only ``should_decode_codec_frame``
      frames feed the audio buffer. ``frames`` defaults to one dummy tensor per frame this
      call is going to observe (``chunks * frames_per_chunk + (flush_frames or 0)``), so
      ``token_observer`` is exercised with no arguments beyond it, rather than silently
      observing nothing because the caller forgot to size a ``frames=`` list to match.

    Assumptions, to reconcile once `synthesis.py` (T038) lands and the real runtime
    changes (R12) are made:
    - ``iter_audio_chunks`` on the real runtime (``models/fast_streaming.py`` today) takes
      only ``inputs`` plus ``request_id``, ``seed``, ``token_observer`` and ``prefix``.
      This fake additionally accepts ``reference`` and the five per-request sampling
      overrides (``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
      ``max_new_tokens``) that R12 decision 1 plans to add to that same method. They are
      recorded, not applied to the fake chunks.
    - ``reference`` is whatever `synthesis.py`'s ``resolve_reference``/``prepare_piece``
      pass through (a `Reference` variant, or ``None``); recording it lets a test assert
      an inline reference is encoded once and the same object is reused for every piece
      and both CFG rows, instead of being re-encoded per piece (T038).
    - ``max_new_tokens_room`` mirrors the *planned* runtime addition of the same name
      (R12 decision 3, `research.md` "R12"; not implemented in `models/fast_streaming.py`
      yet) rather than a method that exists today, since T050's room-clamp tests need it.
    - ``build_reference_prefix`` (the cached-KV "prefix" `Reference` variant) is
      deliberately not faked here: none of T036/T038/T040/T050 exercise a saved voice
      with no override, only the "codes" variant. Add it when a task needs it.

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
    ) -> None:
        self.chunks = chunks
        self.frames_per_chunk = frames_per_chunk
        self.flush_frames = flush_frames
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

    def max_new_tokens_room(
        self, requested: int | None, inputs: dict[str, Any], *, prefix_len: int = 0
    ) -> int:
        """The planned rule (R12 #3) minus prefill bucket padding (no graphs here)."""
        cap = requested if requested is not None and 0 < requested < float("inf") else 750
        prompt_len = int(inputs["input_ids"].shape[1])
        negative = inputs.get("cfg_negative_prompt_ids")
        if negative is not None:
            prompt_len = max(prompt_len, int(negative.shape[1]))
        return min(int(cap), 1500, 2048 - (prompt_len + prefix_len) - 1)

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
        seed: int | None = None,
        reference: Any = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        token_observer: Callable[[torch.Tensor], None] | None = None,
        prefix: Any | None = None,
    ) -> Iterator[FastStreamingChunk]:
        # Lazy: models.fast_streaming (cudagraph submodules) is slow to import, and most
        # of tests/fakes.py's own consumers never call this method (finding #8).
        from models.fast_streaming import FastStreamingChunk

        call_index = len(self.calls)
        self.calls.append(
            {
                "inputs": inputs,
                "request_id": request_id,
                "seed": seed,
                "reference": reference,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "prefix": prefix,
                "observed": token_observer is not None,
            }
        )
        frame_iter = iter(self.frames)
        total_frames = 0
        try:
            for index in range(self.chunks):
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
            if self.flush_frames is not None:
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
            if token_observer is not None:
                for frame in frame_iter:
                    token_observer(frame)
        finally:
            self.closed += 1
