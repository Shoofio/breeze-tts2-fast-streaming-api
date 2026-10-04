"""The MLX streaming runtime for Apple Silicon Macs (specs/005-mlx-mac-inference).

`MlxBreezeStreamingRuntime` implements the server's runtime seam
(contracts/runtime-seam.md) on top of mlx-audio's Breeze port (research R1), so the routes,
templates, synthesis and voice code run unchanged on a Mac.

mlx exists only on macOS arm64, and this module must still import everywhere (the room tests
run on Linux CI), so every `mlx` and `mlx_audio` import is inside the function that needs it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from breeze_infer.reference_audio import predicted_frames

from .cudagraph.sampling import require_number
from .fast_streaming import (
    _POSITIVE_INTEGER_RULE,
    FastStreamingConfig,
    PromptLength,
    _require_valid_overrides,
    prompt_length,
)

# The backbone's context length. The CUDA runtime is launched with the same value, and request
# validation must match it (contracts/runtime-seam.md, `config.max_seq_len`).
MAX_SEQ_LEN = 2048


@dataclass(frozen=True)
class MlxCodecConfig:
    """The codec fields the server reads from `model.config.codec_config`."""

    codebook_size: int
    sampling_rate: int


@dataclass(frozen=True)
class MlxModelConfig:
    """A read-only view of the MLX checkpoint's `config.json`, with the attribute names the
    server reads from the CUDA model's config (contracts/runtime-seam.md, `model.config`).

    The values come from the same top-level JSON keys the official `BreezeConfig` takes its
    attributes from. The MLX conversion keeps them unchanged.
    """

    num_codebooks: int
    codebook_pad_token_id: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    codec_config: MlxCodecConfig

    @classmethod
    def from_config_json(cls, config: Mapping[str, Any]) -> MlxModelConfig:
        codec = config["codec_config"]
        return cls(
            num_codebooks=int(config["num_codebooks"]),
            codebook_pad_token_id=int(config["codebook_pad_token_id"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            hidden_size=int(config["hidden_size"]),
            head_dim=int(config["head_dim"]),
            codec_config=MlxCodecConfig(
                codebook_size=int(codec["codebook_size"]),
                sampling_rate=int(codec["sampling_rate"]),
            ),
        )


class MlxAudioTokenizer:
    """The server's `audio_tokenizer` (contracts/runtime-seam.md) over mlx-audio's Qwen3-TTS
    codec: reference audio in, codec codes out, as the official tokenizer returns them."""

    def __init__(self, codec: Any) -> None:
        self._codec = codec

    def get_decode_upsample_rate(self) -> int:
        return int(self._codec.decode_upsample_rate)

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, list[torch.Tensor]]:
        """Encode one waveform into `{"audio_codes": [LongTensor[frames, codebooks]]}`.

        The steps are qwen_tts's (`Qwen3TTSTokenizer._normalize_audio_inputs`, then the 12 Hz
        model's `encode`): downmix, `librosa.resample` with its defaults unless `sr` is already
        the codec's rate, encode, then keep the frames that cover the audio. That frame count
        is `reference_audio.predicted_frames`, which the server already uses to bound a
        reference before encoding it.
        """
        # librosa takes seconds to import (numba), and only reference encoding needs it.
        import librosa
        import mlx.core as mx

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        frames = predicted_frames(wav.shape[0], int(sr))
        target_sr = int(self._codec.input_sample_rate)
        if int(sr) != target_sr:
            wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=target_sr)
        # The codec takes [batch, channels, samples] and returns [batch, codebooks, frames].
        codes = self._codec.encode(mx.array(wav.astype(np.float32))[None, None, :])
        codes = np.array(codes[0, :, :frames]).T.astype(np.int64)
        return {"audio_codes": [torch.from_numpy(codes)]}


class MlxBreezeStreamingRuntime:
    """The runtime seam (contracts/runtime-seam.md) on MLX.

    `load_mlx_runtime` builds it with the loaded mlx-audio model, its codec and the tokenizer.
    The room rules read only `config` and `generation_config`'s `max_new_tokens`, so the
    model-free tests build it with `None` for the loaded parts.
    """

    # The KV cache is bf16 at both precisions (mxfp8 quantizes weights, not activations), so
    # the voice prefix cache's byte estimate is the same as on CUDA.
    dtype = torch.bfloat16
    # MLX warms up through its own branch in model loading, not the CUDA graph warmup.
    fast_enabled = False
    # Frames per yielded chunk, as on the CUDA path without --fast-codec (research R6).
    codec_chunk_frames = 2

    def __init__(
        self,
        *,
        mlx_model: Any,
        codec: Any,
        tokenizer: Any,
        model_config: MlxModelConfig | None,
        generation_config: Mapping[str, Any],
    ) -> None:
        self._mlx_model = mlx_model
        self._codec = codec
        self.audio_tokenizer = MlxAudioTokenizer(codec)
        self.tokenizer = tokenizer
        # The server only reads `.config` and uses `.device` as the `.to()` target for template
        # tensors, which are converted to MLX per request (research R4).
        self.model = SimpleNamespace(config=model_config, device="cpu")
        self.config = FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS_CEILING, max_seq_len=MAX_SEQ_LEN
        )
        default = generation_config.get("max_new_tokens")
        # Validated at load, as the CUDA runtime does: it sizes every request that sets none.
        if default is not None:
            require_number("generation_config.max_new_tokens", default, _POSITIVE_INTEGER_RULE)
        self._default_max_new_tokens = default

    @property
    def sample_rate(self) -> int:
        return self.model.config.codec_config.sampling_rate

    def frame_cap(self, requested: int | None) -> int:
        """Frames a request may generate, before any context limit: `requested`, or the
        checkpoint's default when it is `None`, clamped to the ceiling. The same rule as
        `FastBreezeStreamingRuntime.frame_cap`."""
        if requested is None:
            requested = self._default_max_new_tokens or self.config.max_new_tokens
        return min(int(requested), self.config.max_new_tokens)

    def max_new_tokens_room(
        self, requested: int | None, inputs: dict[str, Any], *, prefix_len: int = 0
    ) -> int:
        """Frames one request for `inputs` will produce at most; `<= 0` means none."""
        return self.room_for_length(requested, prompt_length(inputs), prefix_len=prefix_len)

    def room_for_length(
        self, requested: int | None, length: PromptLength, *, prefix_len: int = 0
    ) -> int:
        """`max_new_tokens_room` for a prompt known only by its `PromptLength`.

        MLX has no prefill buckets, so the prefill is always the exact length: CUDA's rule
        without --fast-backbone-prefill (research R4). The decode loop stops once
        `prefill_len + step >= max_seq_len - 1`.
        """
        _require_valid_overrides(max_new_tokens=requested)
        prefill_len = prefix_len + length.seq_len
        return min(self.frame_cap(requested), self.config.max_seq_len - prefill_len - 1)


def load_mlx_runtime(path: Path) -> MlxBreezeStreamingRuntime:
    """Load an MLX Breeze checkpoint (bf16 or mxfp8) into a runtime."""
    from mlx_audio.tts.utils import load
    from transformers import AutoTokenizer

    # mlx-audio's Breeze loader validates the weights and, in its post-load hook, loads the
    # codec from `path / "audio_tokenizer"` with its strict codec-weights check.
    mlx_model = load(path)
    # AutoTokenizer loads the MLX checkpoint directly with transformers 4.57.3, which this
    # backend pins (research R2): `tokenizer_config.json` names `GemmaTokenizerFast`, so the
    # unknown `breeze_tts` model type never comes into it. The files are the official ones
    # (research R3), loaded the way `breeze_infer.runtime` loads them on CUDA. mlx-audio's own
    # `mlx_model.tokenizer` is not used.
    tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
    return MlxBreezeStreamingRuntime(
        mlx_model=mlx_model,
        codec=mlx_model.audio_tokenizer,
        tokenizer=tokenizer,
        model_config=MlxModelConfig.from_config_json(config),
        generation_config=generation_config,
    )
