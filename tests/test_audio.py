from __future__ import annotations

import json
import warnings

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


# --- pcm16 --------------------------------------------------------------------------


def test_pcm16_clips_out_of_range_samples() -> None:
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    np.testing.assert_array_equal(samples, [-32767, -32767, 0, 32767, 32767])


def test_pcm16_maps_specific_values_to_the_documented_int16_scale() -> None:
    """The scale is symmetric (documented on ``pcm16``): both +-1.0 map to +-32767,
    not the asymmetric int16 range [-32768, 32767].
    """
    audio = np.array([0.99999, -1.0, 1.0, 0.0], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    np.testing.assert_array_equal(samples, [32767, -32767, 32767, 0])


def test_pcm16_rounds_to_nearest_rather_than_truncating() -> None:
    # 0.6 / 32767 * 32767 == 0.6, which np.rint rounds up to 1; plain truncation
    # (int(0.6)) would give 0. (0.5 is avoided deliberately: np.rint's round-half-to-
    # even would round that particular tie down to 0, which would prove nothing.)
    audio = np.array([0.6 / 32767.0], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    assert samples[0] == 1


def test_pcm16_maps_nan_to_zero_without_a_runtimewarning() -> None:
    audio = np.array([float("nan"), 0.5, float("inf"), float("-inf")], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        samples = np.frombuffer(pcm16(audio), dtype="<i2")

    assert samples[0] == 0
    assert samples[2] == 32767
    assert samples[3] == -32767


# --- codec_fingerprint ----------------------------------------------------------


def _write_fake_safetensors(path, header: dict) -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes)


_DEFAULT_TENSOR_HEADER = {
    "decoder.weight": {"dtype": "F32", "shape": [1536, 1024], "data_offsets": [0, 6291456]}
}


def _write_codec_dir(directory, *, config: dict, tensor_header: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("config.json").write_text(json.dumps(config))
    _write_fake_safetensors(
        directory / "model.safetensors", tensor_header or _DEFAULT_TENSOR_HEADER
    )


def _codec_config(**overrides) -> dict:
    """A config.json shaped like the real ``Qwen3TTSTokenizerV2Config`` (see
    ``qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2`` and
    ``transformers.MimiConfig``): top-level sample rates/quantizer-count/frame rate,
    plus nested ``encoder_config``/``decoder_config`` blocks.
    """
    config = {
        "input_sample_rate": 24000,
        "output_sample_rate": 24000,
        "encoder_valid_num_quantizers": 16,
        "encode_downsample_rate": 1920,
        "decode_upsample_rate": 1920,
        "model_type": "qwen3_tts_tokenizer_12hz",
        "encoder_config": {
            "codebook_size": 2048,
            "codebook_dim": 256,
            "num_quantizers": 32,
            "frame_rate": 12.5,
            "hidden_size": 512,
        },
        "decoder_config": {
            "codebook_size": 2048,
            "num_quantizers": 16,
            "upsample_rates": [8, 5, 4, 3],
            "hidden_size": 1024,
        },
    }
    config.update(overrides)
    return config


def test_codec_fingerprint_is_unchanged_when_the_checkpoint_moves(tmp_path) -> None:
    """R13: never hash the checkpoint's own path, so re-pointing --model-path at a
    copy of the same checkpoint doesn't strand every previously saved voice.
    """
    first = tmp_path / "checkpoint_a" / "audio_tokenizer"
    second = tmp_path / "somewhere" / "else" / "checkpoint_b" / "audio_tokenizer"
    _write_codec_dir(first, config=_codec_config())
    _write_codec_dir(second, config=_codec_config())
    assert first.resolve() != second.resolve()

    assert codec_fingerprint(first) == codec_fingerprint(second)


def test_codec_fingerprint_ignores_key_order_and_whitespace(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    config = _codec_config()
    _write_fake_safetensors(directory / "model.safetensors", _DEFAULT_TENSOR_HEADER)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    baseline = codec_fingerprint(directory)

    # Re-serialize with reversed top-level key order and no whitespace at all.
    reordered = dict(reversed(list(config.items())))
    (directory / "config.json").write_text(
        json.dumps(reordered, separators=(",", ":"))
    )

    assert codec_fingerprint(directory) == baseline


def test_codec_fingerprint_ignores_fields_outside_the_identity_set(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(
        directory,
        config=_codec_config(model_type="something_else", unrelated_hyperparameter=1),
    )

    assert codec_fingerprint(directory) == baseline


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_sample_rate": 16000},
        {"encoder_valid_num_quantizers": 8},
        {"encode_downsample_rate": 960},
    ],
)
def test_codec_fingerprint_changes_with_top_level_identity_fields(tmp_path, overrides) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(directory, config=_codec_config(**overrides))

    assert codec_fingerprint(directory) != baseline


@pytest.mark.parametrize(
    "encoder_overrides",
    [{"codebook_size": 1024}, {"codebook_dim": 128}, {"num_quantizers": 16}],
)
def test_codec_fingerprint_changes_with_encoder_identity_fields(
    tmp_path, encoder_overrides
) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    config = _codec_config()
    config["encoder_config"].update(encoder_overrides)
    _write_codec_dir(directory, config=config)

    assert codec_fingerprint(directory) != baseline


def test_codec_fingerprint_changes_with_decoder_upsample_rates(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    config = _codec_config()
    config["decoder_config"]["upsample_rates"] = [4, 4, 4, 4]
    _write_codec_dir(directory, config=config)

    assert codec_fingerprint(directory) != baseline


def test_codec_fingerprint_changes_when_a_weight_shape_differs(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    different_shape = {
        "decoder.weight": {"dtype": "F32", "shape": [1536, 2048], "data_offsets": [0, 12582912]}
    }
    _write_codec_dir(directory, config=_codec_config(), tensor_header=different_shape)

    assert codec_fingerprint(directory) != baseline


def test_codec_fingerprint_covers_every_safetensors_shard(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    single_shard = codec_fingerprint(directory)

    _write_fake_safetensors(
        directory / "model-00002-of-00002.safetensors",
        {"encoder.weight": {"dtype": "F32", "shape": [256, 256], "data_offsets": [0, 262144]}},
    )

    assert codec_fingerprint(directory) != single_shard


def test_codec_fingerprint_requires_safetensors_weights(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))

    with pytest.raises(FileNotFoundError):
        codec_fingerprint(directory)
