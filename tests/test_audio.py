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


def test_encode_prompt_waveform_scopes_cudnn_to_deterministic_no_benchmark() -> None:
    """`--fast-all` leaves `torch.backends.cudnn.benchmark = True` set for the whole
    process (models/stream_runtime/stream/runtime.py), so cuDNN autotunes a conv
    algorithm on first use and reuses it -- different processes can pick different
    algorithms, which round differently and give ~1% different fine-codebook codes for
    the identical reference wav (2026-09-25 finding). `encode_prompt_waveform` must
    force `benchmark=False, deterministic=True` for just the codec's `encode` call so
    the same wav always encodes to the same codes, regardless of what the rest of the
    process has autotuned.
    """
    seen_flags: dict[str, bool] = {}

    class _FlagRecordingCodec(FakeCodec):
        def encode(self, wav, sr, return_dict=True):
            seen_flags["benchmark"] = torch.backends.cudnn.benchmark
            seen_flags["deterministic"] = torch.backends.cudnn.deterministic
            return super().encode(wav, sr, return_dict=return_dict)

    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False):
        encode_prompt_waveform(
            _FlagRecordingCodec(), np.zeros(1920, dtype=np.float32), 24000
        )

        assert seen_flags == {"benchmark": False, "deterministic": True}
        # Restored to the caller's own flags once the encode call returns, not left
        # at the scoped values.
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False


def test_encode_prompt_waveform_restores_cudnn_flags_when_encode_raises() -> None:
    class _RaisingCodec(FakeCodec):
        def encode(self, wav, sr, return_dict=True):
            raise RuntimeError("boom")

    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False):
        with pytest.raises(RuntimeError, match="boom"):
            encode_prompt_waveform(
                _RaisingCodec(), np.zeros(1920, dtype=np.float32), 24000
            )

        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False


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


def test_pcm16_rounds_half_to_even() -> None:
    # 0.5 * 32767 == 16383.5, exactly halfway between 16383 (odd) and 16384 (even);
    # np.rint's round-half-to-even rounds that tie up to 16384.
    audio = np.array([0.5], dtype=np.float32)

    samples = np.frombuffer(pcm16(audio), dtype="<i2")

    assert samples[0] == 16384


def test_pcm16_rounds_to_nearest_rather_than_truncating() -> None:
    # 0.6 / 32767 * 32767 == 0.6, which np.rint rounds up to 1; plain truncation
    # (int(0.6)) would give 0. (0.5 is covered separately above, since that tie
    # rounds a specific, non-obvious way.)
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
    "decoder.weight": {"dtype": "F32", "shape": [1536, 1024], "data_offsets": [0, 6291456]},
    "__metadata__": {"format": "pt"},
}


def _write_codec_dir(directory, *, config: dict, tensor_header: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("config.json").write_text(json.dumps(config))
    _write_fake_safetensors(
        directory / "model.safetensors", tensor_header or _DEFAULT_TENSOR_HEADER
    )


def _codec_config(*, encoder_overrides=None, decoder_overrides=None, **top_overrides) -> dict:
    """A config.json shaped like the real bundled checkpoint's
    ``audio_tokenizer/config.json`` (``Qwen3TTSTokenizerV2Config``): top-level sample
    rates/quantizer-count/frame rates, plus nested ``encoder_config``/
    ``decoder_config`` blocks whose ``codebook_size``/``codebook_dim``/
    ``num_quantizers`` deliberately differ from each other, exactly as the real file's
    do (encoder codebook_dim 256 vs. decoder 512; encoder num_quantizers 32 vs.
    decoder 16).
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
            "_frame_rate": 12.5,
            "hidden_size": 512,
            **(encoder_overrides or {}),
        },
        "decoder_config": {
            "codebook_size": 2048,
            "codebook_dim": 512,
            "num_quantizers": 16,
            "semantic_codebook_size": 4096,
            "num_semantic_quantizers": 1,
            # Two distinct fields, both required (review #4): the real config has
            # both, at this same path, and the decoder's total upsampling factor is
            # prod(upsample_rates + upsampling_ratios).
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
            "hidden_size": 512,
            **(decoder_overrides or {}),
        },
    }
    config.update(top_overrides)
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

    reordered = dict(reversed(list(config.items())))
    (directory / "config.json").write_text(json.dumps(reordered, separators=(",", ":")))

    assert codec_fingerprint(directory) == baseline


def test_codec_fingerprint_ignores_fields_outside_the_identity_set(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(
        directory, config=_codec_config(model_type="something_else", architectures=["X"])
    )

    assert codec_fingerprint(directory) == baseline


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_sample_rate": 16000},
        {"encoder_valid_num_quantizers": 8},
        {"encode_downsample_rate": 960},
        {"decode_upsample_rate": 960},
    ],
)
def test_codec_fingerprint_changes_with_top_level_identity_fields(tmp_path, overrides) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(directory, config=_codec_config(**overrides))

    assert codec_fingerprint(directory) != baseline


@pytest.mark.parametrize(
    "overrides", [{"codebook_size": 1024}, {"codebook_dim": 128}, {"num_quantizers": 16}]
)
def test_codec_fingerprint_changes_with_encoder_identity_fields(tmp_path, overrides) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(directory, config=_codec_config(encoder_overrides=overrides))

    assert codec_fingerprint(directory) != baseline


@pytest.mark.parametrize(
    "overrides",
    [
        {"codebook_size": 1024},
        {"codebook_dim": 256},
        {"num_quantizers": 32},
        {"semantic_codebook_size": 2048},
        {"num_semantic_quantizers": 2},
        {"upsample_rates": [4, 4, 4, 4]},
        {"upsampling_ratios": [4, 1]},
    ],
    ids=[
        "codebook_size",
        "codebook_dim",
        "num_quantizers",
        "semantic_codebook_size",
        "num_semantic_quantizers",
        "upsample_rates",
        "upsampling_ratios",
    ],
)
def test_codec_fingerprint_changes_with_decoder_identity_fields(tmp_path, overrides) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_codec_dir(directory, config=_codec_config(decoder_overrides=overrides))

    assert codec_fingerprint(directory) != baseline


@pytest.mark.parametrize(
    "drop_path",
    [
        ("input_sample_rate",),
        ("encoder_config", "codebook_size"),
        ("decoder_config", "num_semantic_quantizers"),
        ("decoder_config", "upsample_rates"),
        ("decoder_config", "upsampling_ratios"),
    ],
    ids=["top-level", "encoder", "decoder", "decoder-upsample_rates", "decoder-upsampling_ratios"],
)
def test_codec_fingerprint_requires_every_identity_field(tmp_path, drop_path) -> None:
    """``upsample_rates`` and ``upsampling_ratios`` are both required independently
    (review #4): dropping either one alone (not just both together) must raise.
    """
    directory = tmp_path / "codec"
    config = _codec_config()
    target = config
    for key in drop_path[:-1]:
        target = target[key]
    del target[drop_path[-1]]
    _write_codec_dir(directory, config=config)

    with pytest.raises(ValueError, match="required"):
        codec_fingerprint(directory)


def test_codec_fingerprint_changes_when_a_weight_shape_differs(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    different_shape = {
        "decoder.weight": {"dtype": "F32", "shape": [1536, 2048], "data_offsets": [0, 12582912]}
    }
    _write_codec_dir(directory, config=_codec_config(), tensor_header=different_shape)

    assert codec_fingerprint(directory) != baseline


def test_codec_fingerprint_cannot_detect_a_retrain_of_the_same_shapes(tmp_path) -> None:
    """Pins the documented limitation: the fingerprint is built only from tensor
    names/dtypes/shapes (never the trained values, which aren't read), so retraining
    a checkpoint without changing its architecture is invisible to it.
    """
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    original = codec_fingerprint(directory)

    # Same tensor names/dtypes/shapes as _DEFAULT_TENSOR_HEADER, but this stands in
    # for "retrained weights": codec_fingerprint has no way to see that the actual
    # float values on disk would differ, since it never reads past the header.
    _write_codec_dir(directory, config=_codec_config(), tensor_header=_DEFAULT_TENSOR_HEADER)

    assert codec_fingerprint(directory) == original


def test_codec_fingerprint_ignores_a_stray_safetensors_file(tmp_path) -> None:
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    baseline = codec_fingerprint(directory)

    _write_fake_safetensors(
        directory / "backup-unused.safetensors",
        {"unrelated": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
    )

    assert codec_fingerprint(directory) == baseline


def test_codec_fingerprint_uses_every_shard_listed_by_the_index(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    _write_fake_safetensors(
        directory / "model-00001-of-00002.safetensors",
        {"encoder.weight": {"dtype": "F32", "shape": [256, 256], "data_offsets": [0, 262144]}},
    )
    _write_fake_safetensors(
        directory / "model-00002-of-00002.safetensors",
        {"decoder.weight": {"dtype": "F32", "shape": [512, 512], "data_offsets": [0, 1048576]}},
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.weight": "model-00001-of-00002.safetensors",
                    "decoder.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    sharded = codec_fingerprint(directory)

    single = tmp_path / "single"
    _write_codec_dir(
        single,
        config=_codec_config(),
        tensor_header={
            "encoder.weight": {"dtype": "F32", "shape": [256, 256], "data_offsets": [0, 262144]},
            "decoder.weight": {"dtype": "F32", "shape": [512, 512], "data_offsets": [0, 1048576]},
        },
    )

    assert sharded == codec_fingerprint(single)


def test_codec_fingerprint_requires_safetensors_weights(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))

    with pytest.raises(FileNotFoundError):
        codec_fingerprint(directory)


def test_codec_fingerprint_prefers_the_single_file_over_a_stale_index(tmp_path) -> None:
    """review #3: transformers 4.57's own resolution order
    (``_get_resolved_checkpoint_files``, ``modeling_utils.py`` ~936-947) tries
    ``model.safetensors`` *before* ``model.safetensors.index.json`` -- so a directory
    that happens to have both (e.g. a stale index left over from a resharding) must
    be fingerprinted from the single file, the one the real loader would actually use.
    """
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config())
    single_file_fingerprint = codec_fingerprint(directory)

    # A stale index pointing at a shard file with a *different* tensor shape --
    # if _weight_files preferred the index, this would change the fingerprint.
    _write_fake_safetensors(
        directory / "stale-00001-of-00001.safetensors",
        {"decoder.weight": {"dtype": "F32", "shape": [9999, 9999], "data_offsets": [0, 4]}},
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"decoder.weight": "stale-00001-of-00001.safetensors"}}
        )
    )

    assert codec_fingerprint(directory) == single_file_fingerprint


def test_codec_fingerprint_rejects_a_pytorch_only_checkpoint(tmp_path) -> None:
    """review #3: a codec shipped only as pytorch_model.bin is real and loadable by
    transformers, but this fingerprint only parses the safetensors header format, so
    it must fail clearly rather than report "no weights at all".
    """
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    (directory / "pytorch_model.bin").write_bytes(b"not really a torch checkpoint")

    with pytest.raises(ValueError, match="safetensors"):
        codec_fingerprint(directory)


@pytest.mark.parametrize(
    "bad_header",
    [{"decoder.weight": 5}, {"decoder.weight": {}}, {"decoder.weight": [1, 2, 3]}],
    ids=["scalar", "empty_object", "list"],
)
def test_codec_fingerprint_rejects_a_malformed_tensor_entry(tmp_path, bad_header) -> None:
    """review #9: a header that parses as JSON but has the wrong shape for a
    safetensors header (not {name: {dtype, shape, ...}}) must raise ValueError, not
    TypeError (indexing an int/list) or KeyError (a dict missing 'dtype'/'shape').
    """
    directory = tmp_path / "codec"
    _write_codec_dir(directory, config=_codec_config(), tensor_header=bad_header)

    with pytest.raises(ValueError, match="not a safetensors file"):
        codec_fingerprint(directory)


@pytest.mark.parametrize(
    "bad_index",
    [
        {"weight_map": 5},
        {"weight_map": ["model.safetensors"]},
        {"weight_map": {"decoder.weight": 5}},
        {"weight_map": {"decoder.weight": ["model.safetensors"]}},
        {"weight_map": {"decoder.weight": {"nested": "model.safetensors"}}},
        ["model.safetensors"],
    ],
    ids=[
        "scalar_map",
        "list_map",
        "int_filename",
        "list_filename",
        "dict_filename",
        "list_index",
    ],
)
def test_codec_fingerprint_rejects_a_malformed_index(tmp_path, bad_index) -> None:
    """review #9: same guarantee as the header case, for model.safetensors.index.json."""
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    (directory / "model.safetensors.index.json").write_text(json.dumps(bad_index))

    with pytest.raises(ValueError):
        codec_fingerprint(directory)


@pytest.mark.parametrize(
    "bad_filename",
    [
        "../other.safetensors",
        "/etc/other.safetensors",
        "subdir/model.safetensors",
        "subdir\\model.safetensors",
        ".",
        "..",
        "model.bin",
        "",
    ],
    ids=[
        "parent_traversal",
        "absolute_path",
        "forward_slash",
        "backslash",
        "dot",
        "dotdot",
        "wrong_extension",
        "empty",
    ],
)
def test_codec_fingerprint_rejects_a_weight_map_filename_that_is_not_plain(
    tmp_path, bad_filename
) -> None:
    """review #5: a weight_map value must be a plain filename inside the checkpoint
    directory -- no path separator (which could reach outside the directory via
    '../' or an absolute path), not '.'/'..', and it must end in '.safetensors'.
    Each value is validated before the (deduplicating) set of filenames is built, so
    a single bad entry can't be lost to deduplication ahead of being checked.
    """
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"decoder.weight": bad_filename}})
    )

    with pytest.raises(ValueError):
        codec_fingerprint(directory)


def test_codec_fingerprint_rejects_a_git_lfs_pointer_without_a_memoryerror(tmp_path) -> None:
    """review #3: a git-lfs pointer file is a few hundred bytes of plain text, not a
    real safetensors file -- its first 8 bytes decode to an arbitrary uint64, which
    must be rejected by a size/plausibility check rather than trigger a multi-GB read.
    """
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    (directory / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000\n"
        "size 682293092\n"
    )

    with pytest.raises(ValueError, match="not a safetensors file"):
        codec_fingerprint(directory)


def test_codec_fingerprint_rejects_a_file_too_small_to_have_a_header(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    (directory / "model.safetensors").write_bytes(b"short")

    with pytest.raises(ValueError, match="not a safetensors file"):
        codec_fingerprint(directory)


def test_codec_fingerprint_rejects_a_header_longer_than_the_file(tmp_path) -> None:
    directory = tmp_path / "codec"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_codec_config()))
    # Declares a header far larger than the (short) file actually is.
    (directory / "model.safetensors").write_bytes((10_000).to_bytes(8, "little") + b"{}")

    with pytest.raises(ValueError, match="not a safetensors file"):
        codec_fingerprint(directory)
