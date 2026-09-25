"""Reference-audio encoding, PCM packing and codec identity for Breeze inference."""

from __future__ import annotations

import hashlib
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
    """Clip float32 samples to [-1, 1] and pack as little-endian int16 PCM bytes."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


def codec_fingerprint(
    audio_tokenizer_dir: str | Path,
    *,
    codebooks: int,
    codebook_size: int,
    sample_rate: int,
) -> str:
    """Identify the codec config that produced a set of codec tokens.

    Saved voices store this next to their codes (data-model.md "Voice file v1"); a
    fingerprint mismatch at startup means the codes must be rebuilt from the
    original recording rather than trusted as-is.

    The hash covers ``<audio_tokenizer_dir>/config.json``'s raw bytes plus
    ``codebooks``, ``codebook_size`` and ``sample_rate`` -- never the checkpoint's
    absolute path (R13), so moving the checkpoint to a new location on disk doesn't
    invalidate every saved voice.

    ``codebooks`` and ``codebook_size`` are the caller's job to supply: the caller
    reads them off the already-loaded codec config (16 valid quantizers and a
    ``codebook_size`` of 2048 for the bundled tokenizer's own
    ``<model>/audio_tokenizer/config.json``) rather than this function re-parsing
    config.json's schema itself, since which JSON fields mean "valid codebook
    count" is the loader's knowledge, not this module's.
    """
    config_path = Path(audio_tokenizer_dir) / "config.json"
    digest = hashlib.sha256(config_path.read_bytes())
    digest.update(int(codebooks).to_bytes(4, "little"))
    digest.update(int(codebook_size).to_bytes(4, "little"))
    digest.update(int(sample_rate).to_bytes(4, "little"))
    return digest.hexdigest()
