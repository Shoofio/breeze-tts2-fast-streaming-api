"""Reference-audio encoding, PCM packing and codec identity for Breeze inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def encode_prompt_waveform(
    audio_tokenizer: Any, wav: np.ndarray, sample_rate: int
) -> torch.Tensor:
    """Encode a mono float32 waveform into codec tokens ``int16[frames, codebooks]``."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    encoded = audio_tokenizer.encode(wav, sr=int(sample_rate))
    codes = torch.as_tensor(encoded["audio_codes"][0], dtype=torch.int16)
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D audio codes, got shape {tuple(codes.shape)}")
    return codes.cpu().contiguous()


def pcm16(audio: np.ndarray) -> bytes:
    """Pack float32 samples as little-endian int16 PCM bytes.

    Order matters (review #3): NaN maps to 0 first (``np.nan_to_num``, which also
    folds +-inf to a large finite value), *then* the result is clipped to [-1, 1],
    scaled by 32767 and rounded to the nearest integer (``np.rint``, not truncation)
    before the final cast. Doing it in this order means no NaN/inf ever reaches the
    multiply, so it raises no ``RuntimeWarning``.

    The scale is the *symmetric* int16 range: both +1.0 and -1.0 map to +-32767, one
    short of the asymmetric int16 minimum (-32768), so the positive and negative
    ends round-trip by the same factor instead of one direction clipping a fraction
    harder than the other.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return np.rint(audio * 32767.0).astype("<i2", copy=False).tobytes()


# The codec config fields that determine what a stored code means (review #4): change
# any of these and the same integer in ``codes`` decodes to different audio, so the
# fingerprint must cover exactly this set -- no more (irrelevant fields like training
# hyperparameters would churn the fingerprint for no reason) and no less.
#
# Read from ``<audio_tokenizer_dir>/config.json``, the ``Qwen3TTSTokenizerV2Config``
# the bundled qwen-tts tokenizer loads (``breeze_infer/runtime.py``):
#   - top level: ``input_sample_rate``, ``output_sample_rate`` (24000/24000 for the
#     bundled tokenizer), ``encoder_valid_num_quantizers`` (how many of the encoder's
#     quantizers are actually emitted -- 16 of 32), ``encode_downsample_rate`` (the
#     encode-side frame rate: samples per codec frame, 1920);
#   - ``encoder_config`` (a ``transformers.MimiConfig``): ``codebook_size`` (2048),
#     ``codebook_dim``, ``num_quantizers`` (the encoder's full quantizer stack, 32,
#     before ``encoder_valid_num_quantizers`` slices it down to 16);
#   - ``decoder_config``: ``upsample_rates`` (how the decoder turns codes back into a
#     waveform -- relevant because the same codec instance also decodes *generated*
#     speech, not only reference audio).
_CODEC_IDENTITY_FIELDS = (
    "input_sample_rate",
    "output_sample_rate",
    "encoder_valid_num_quantizers",
    "encode_downsample_rate",
)
_CODEC_ENCODER_IDENTITY_FIELDS = ("codebook_size", "codebook_dim", "num_quantizers")
_CODEC_DECODER_IDENTITY_FIELDS = ("upsample_rates",)


def _codec_identity_fields(config: dict[str, Any]) -> dict[str, Any]:
    encoder_config = config.get("encoder_config") or {}
    decoder_config = config.get("decoder_config") or {}
    identity = {field: config.get(field) for field in _CODEC_IDENTITY_FIELDS}
    identity.update({field: encoder_config.get(field) for field in _CODEC_ENCODER_IDENTITY_FIELDS})
    identity.update({field: decoder_config.get(field) for field in _CODEC_DECODER_IDENTITY_FIELDS})
    return identity


def _safetensors_header_bytes(path: Path) -> bytes:
    """The JSON header of a safetensors file, without reading the tensor data.

    A safetensors file starts with an 8-byte little-endian ``uint64`` giving the
    header's byte length, followed by that many bytes of JSON listing every tensor's
    name, shape and dtype (https://github.com/huggingface/safetensors -- "Format").
    That header is exactly the "are these the same weights" fact; the multi-GB tensor
    payload after it is not read.
    """
    with path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        return f.read(header_len)


def codec_fingerprint(audio_tokenizer_dir: str | Path) -> str:
    """Identify the codec that produced (and will decode) a set of codec tokens.

    Saved voices store this next to their codes (data-model.md "Voice file v1").
    There is no re-encode path for a saved voice -- the original recording isn't
    kept, only its codes -- so a fingerprint mismatch at startup means that voice is
    *skipped*, not rebuilt.

    The fingerprint covers two things, hashed together in order:
    (a) the canonical JSON (``sort_keys=True``, compact separators, so re-serializing
        the same file with different key order or whitespace doesn't change the
        fingerprint) of the codec config fields that determine code meaning -- see
        ``_codec_identity_fields`` and the module-level field list above;
    (b) every ``*.safetensors`` file's header under ``audio_tokenizer_dir`` (sorted by
        filename, for the sharded-weights case), which names every tensor's shape and
        dtype without reading the tensor data itself.

    Directory-only, not path plus caller-supplied facts (review #4's "read them
    itself, or take them from the caller" choice): the caller already has this same
    directory (it's the one ``breeze_infer/runtime.py`` loads the audio tokenizer
    from), so a directory is no more information to thread through than the codec
    facts were, and it keeps the ``qwen_tts`` config schema (the nested
    ``encoder_config``/``decoder_config`` indirection above) known in exactly one
    place instead of duplicated at every call site.

    Never hashes ``audio_tokenizer_dir`` itself (R13): moving the checkpoint to a new
    location on disk must not invalidate every saved voice.
    """
    directory = Path(audio_tokenizer_dir)
    config = json.loads((directory / "config.json").read_text())
    identity = _codec_identity_fields(config)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors weights found under {directory}")
    for weight_path in weight_files:
        digest.update(_safetensors_header_bytes(weight_path))

    return digest.hexdigest()
