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
import math
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from models.fast_streaming import FastStreamingChunk


class FakeAudioTokenizer:
    """Fakes the bundled `qwen_tts` audio tokenizer's `.encode()` used by
    `breeze_infer/audio.py:encode_prompt_audio` (dict-of-list-of-array return shape)."""

    def __init__(self, frames: int = 4) -> None:
        self.frames = frames
        self.last_wav: np.ndarray | None = None
        self.last_sr: int | None = None
        self.encode_calls = 0

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, list[np.ndarray]]:
        self.encode_calls += 1
        self.last_wav = wav
        self.last_sr = sr
        return {"audio_codes": [np.zeros((self.frames, 16), dtype=np.int16)]}


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


# --- FakeCodec: the 12 Hz audio codec at the reference-encode edge -----------------

# The real tokenizer is qwen-tts's ``tokenizer_12hz`` (its class name is literally the
# "12 Hz" codec the spec means, though its true frame rate is 12.5 fps):
# ``input_sample_rate = output_sample_rate = 24_000`` and
# ``encode_downsample_rate = 1920`` samples/frame (both from
# ``qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2``, read from the
# installed package since no checkpoint config.json is available in this repo — see the
# module docstring). A wav at another sample rate is resampled to 24 kHz first; the
# tokenizer uses librosa, whose ``resample`` picks the output length as
# ``ceil(len(wav) * target_sr / orig_sr)`` (``librosa/core/audio.py``). Framing is then a
# ceiling division by the frame size (the tokenizer's own
# ``-(-mask.sum() // encode_downsample_rate)``). ``reference_audio.predicted_frames``
# (T036) is expected to use this same two-step formula.
CODEC_SAMPLE_RATE = 24000
CODEC_SAMPLES_PER_FRAME = 1920
CODEC_CODEBOOKS = 16
# Not discoverable from a real checkpoint in this repo (none is downloaded; that GPU/
# checkpoint requirement is exactly the Principle V deviation this module documents).
# 2048 matches the Mimi codec's default (``transformers.models.mimi.configuration_mimi
# .MimiConfig.codebook_size``), which qwen-tts's 12 Hz tokenizer config also carries.
CODEC_CODEBOOK_SIZE = 2048


def codec_frame_count(num_samples: int, sr: int) -> int:
    """How many codec frames a wav of ``num_samples`` samples at ``sr`` Hz becomes.

    See the comment above ``CODEC_SAMPLE_RATE`` for the formula's origin.
    """
    if num_samples <= 0:
        return 0
    if sr != CODEC_SAMPLE_RATE:
        num_samples = math.ceil(num_samples * CODEC_SAMPLE_RATE / sr)
    return math.ceil(num_samples / CODEC_SAMPLES_PER_FRAME)


class FakeCodec:
    """Stands in for the audio tokenizer at the point it turns a waveform into codes.

    ``encode(wav, sr)`` returns deterministic ``int16`` codes shaped
    ``[frames, codebooks]`` (data-model Voice's `codes` field), with ``frames`` following
    ``codec_frame_count`` and ``codebooks == CODEC_CODEBOOKS`` (16, the real tokenizer's
    codebook count — distinct from the backbone's own 32-codebook config). "Deterministic"
    means a function of ``wav`` and ``sr`` only (a content hash, not randomness), so the
    same reference always encodes to the same codes and re-encoding is detectable.
    """

    SAMPLE_RATE = CODEC_SAMPLE_RATE
    SAMPLES_PER_FRAME = CODEC_SAMPLES_PER_FRAME
    CODEBOOKS = CODEC_CODEBOOKS
    CODEBOOK_SIZE = CODEC_CODEBOOK_SIZE

    def __init__(self) -> None:
        self.encode_calls = 0
        self.last_wav: np.ndarray | None = None
        self.last_sr: int | None = None

    def encode(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self.encode_calls += 1
        self.last_wav = wav
        self.last_sr = sr
        frames = codec_frame_count(len(wav), sr)
        wav_bytes = np.ascontiguousarray(wav, dtype=np.float32).tobytes()
        digest = hashlib.sha256(wav_bytes + sr.to_bytes(4, "little", signed=False)).digest()
        seed = int.from_bytes(digest[:8], "little")
        rows = np.arange(frames, dtype=np.int64).reshape(-1, 1)
        cols = np.arange(self.CODEBOOKS, dtype=np.int64).reshape(1, -1)
        codes = (seed + rows * self.CODEBOOKS + cols) % self.CODEBOOK_SIZE
        return codes.astype(np.int16)


# --- FakeRuntime: the streaming runtime at the generation edge ---------------------


class FakeRuntime:
    """Yields constant-valued chunks shaped like ``FastStreamingChunk``; records every
    call so tests can inspect it.

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
    - A chunk is ``FastStreamingChunk`` (imported from `models.fast_streaming`, not a
      duplicate type): ``audio`` is float32 PCM, plus ``sample_rate``, ``codec_frames``
      and ``is_final`` as the real runtime reports them.

    ``fail_after`` raises ``RuntimeError`` once that many chunks of a call have been
    yielded, standing in for a mid-stream CUDA error. ``gate`` (a ``threading.Event``) is
    waited on just before yielding chunk index ``gate_at`` (default 1, i.e. mid-piece for
    the default ``chunks=2``), so a test can hold generation there from another thread;
    ``gate_reached``, if given, is set right before that wait so the test can synchronize
    on "generation is now blocked" instead of racing a timeout. Chunk ``i`` of call ``n``
    has sample value ``n / 100`` so tests can tell pieces apart in the PCM body.
    """

    sample_rate = 24000

    def __init__(
        self,
        chunks: int = 2,
        *,
        samples: int = 480,
        frames: list[torch.Tensor] | None = None,
        fail_after: int | None = None,
        gate: threading.Event | None = None,
        gate_at: int = 1,
        gate_reached: threading.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.samples = samples
        self.frames = frames or []
        self.fail_after = fail_after
        self.gate = gate
        self.gate_at = gate_at
        self.gate_reached = gate_reached
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
        try:
            if token_observer is not None:
                for frame in self.frames:
                    token_observer(frame)
            for index in range(self.chunks):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("CUDA error: an illegal memory access (fake)")
                if self.gate is not None and index == self.gate_at:
                    if self.gate_reached is not None:
                        self.gate_reached.set()
                    self.gate.wait()
                yield FastStreamingChunk(
                    audio=np.full(self.samples, call_index / 100, dtype=np.float32),
                    sample_rate=self.sample_rate,
                    codec_frames=1,
                    is_final=index == self.chunks - 1,
                    timing={"prefill_gpu_ms": 1.5, "prefill_path": "graph"},
                )
        finally:
            self.closed += 1
