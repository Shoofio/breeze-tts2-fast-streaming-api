from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from breeze_infer.audio import codec_fingerprint, encode_prompt_waveform, pcm16
from tests.fakes import FakeCodec, codec_frame_count


def test_encode_prompt_waveform_downmixes_stereo() -> None:
    tokenizer = FakeCodec()
    wav = np.stack(
        [
            np.linspace(-0.5, 0.5, 8, dtype=np.float32),
            np.linspace(0.5, -0.5, 8, dtype=np.float32),
        ],
        axis=1,
    )

    codes = encode_prompt_waveform(tokenizer, wav, 24000)

    assert isinstance(codes, torch.Tensor)
    assert codes.dtype == torch.int16
    assert tuple(codes.shape) == (codec_frame_count(8, 24000), 16)
    assert tokenizer.last_sr == 24000
    assert tokenizer.last_wav is not None
    assert tokenizer.last_wav.shape == (8,)
    np.testing.assert_allclose(tokenizer.last_wav, np.mean(wav, axis=1), atol=1e-4)


def test_encode_prompt_waveform_leaves_mono_input_unchanged() -> None:
    tokenizer = FakeCodec()
    wav = np.linspace(-1.0, 1.0, 1920, dtype=np.float32)

    codes = encode_prompt_waveform(tokenizer, wav, 24000)

    assert tuple(codes.shape) == (codec_frame_count(1920, 24000), 16)
    np.testing.assert_allclose(tokenizer.last_wav, wav)


def test_encode_prompt_waveform_rejects_non_2d_codes() -> None:
    class _FlatCodec:
        def encode(self, wav, sr):
            del wav, sr
            return {"audio_codes": [torch.zeros(16, dtype=torch.int64)]}

    with pytest.raises(ValueError, match="2D"):
        encode_prompt_waveform(_FlatCodec(), np.zeros(1920, dtype=np.float32), 24000)


def test_pcm16_clips_out_of_range_samples() -> None:
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    np.testing.assert_array_equal(samples, [-32767, -32767, 0, 32767, 32767])


def test_pcm16_packs_in_range_samples_as_little_endian_int16() -> None:
    audio = np.array([-0.5, 0.25], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    np.testing.assert_array_equal(samples, (audio * 32767.0).astype("<i2"))


def _write_codec_config(directory, **extra_fields) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"codebook_size": 2048, "sample_rate": 24000, **extra_fields}
    (directory / "config.json").write_text(json.dumps(payload))


def test_codec_fingerprint_is_unchanged_when_the_checkpoint_moves(tmp_path) -> None:
    """R13: the fingerprint hashes config.json's bytes plus the codebook/sample-rate
    facts, never the checkpoint's own absolute path -- so re-pointing --model-path at
    a copy of the same checkpoint doesn't strand every previously saved voice.
    """
    first = tmp_path / "checkpoint_a" / "audio_tokenizer"
    second = tmp_path / "somewhere" / "else" / "checkpoint_b" / "audio_tokenizer"
    _write_codec_config(first)
    _write_codec_config(second)
    assert first.resolve() != second.resolve()

    fingerprint_a = codec_fingerprint(
        first, codebooks=16, codebook_size=2048, sample_rate=24000
    )
    fingerprint_b = codec_fingerprint(
        second, codebooks=16, codebook_size=2048, sample_rate=24000
    )

    assert fingerprint_a == fingerprint_b


def test_codec_fingerprint_changes_when_config_json_bytes_differ(tmp_path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _write_codec_config(first)
    _write_codec_config(second, extra="a different config")

    fingerprint_a = codec_fingerprint(
        first, codebooks=16, codebook_size=2048, sample_rate=24000
    )
    fingerprint_b = codec_fingerprint(
        second, codebooks=16, codebook_size=2048, sample_rate=24000
    )

    assert fingerprint_a != fingerprint_b


def test_codec_fingerprint_changes_when_declared_codebook_facts_differ(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_config(directory)

    baseline = codec_fingerprint(
        directory, codebooks=16, codebook_size=2048, sample_rate=24000
    )

    assert baseline != codec_fingerprint(
        directory, codebooks=32, codebook_size=2048, sample_rate=24000
    )
    assert baseline != codec_fingerprint(
        directory, codebooks=16, codebook_size=1024, sample_rate=24000
    )
    assert baseline != codec_fingerprint(
        directory, codebooks=16, codebook_size=2048, sample_rate=16000
    )
