"""Reference-audio encoding, PCM packing and codec identity for Breeze inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# transformers.utils.SAFE_WEIGHTS_NAME / SAFE_WEIGHTS_INDEX_NAME / WEIGHTS_NAME /
# WEIGHTS_INDEX_NAME, and the exact resolution order `PreTrainedModel.from_pretrained`
# checks them in (review #3, this round -- verified against transformers 4.57's
# `modeling_utils.py`, `_get_resolved_checkpoint_files`, ~lines 936-958): the
# single-file safetensors name first, then the sharded safetensors index, and only
# then (never for this codebase, since nothing here passes `use_safetensors=False`)
# a bare or sharded PyTorch `.bin`. `Qwen3TTSTokenizerV2Model.from_pretrained` (what
# `breeze_infer/runtime.py` calls to load the bundled audio tokenizer) uses this same
# resolution, so this is the one/few file(s) that actually determine what the loaded
# codec's weights are -- not every `*.safetensors` file that happens to sit in the
# directory (review #1/#5, prior round).
_SAFETENSORS_SINGLE_FILE = "model.safetensors"
_SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
_PYTORCH_SINGLE_FILE = "pytorch_model.bin"
_PYTORCH_INDEX_FILE = "pytorch_model.bin.index.json"

_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB; a real header is a few KB.


def encode_prompt_waveform(
    audio_tokenizer: Any, wav: np.ndarray, sample_rate: int
) -> torch.Tensor:
    """Encode a mono float32 waveform into codec tokens ``int16[frames, codebooks]``."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    # `--fast-all` turns cudnn.benchmark on process-wide, so each server process can autotune
    # a different conv algorithm; they round differently and flip about 1% of the fine codes,
    # changing the audio for the same reference after a restart (research.md R18). Only the
    # encode is pinned; decode keeps its tuned algorithms.
    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True
    ):
        encoded = audio_tokenizer.encode(wav, sr=int(sample_rate))
    codes = torch.as_tensor(encoded["audio_codes"][0], dtype=torch.int16)
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D audio codes, got shape {tuple(codes.shape)}")
    return codes.cpu().contiguous()


def pcm16(audio: np.ndarray) -> bytes:
    """Pack float32 samples as little-endian int16 PCM bytes.

    Order matters (review #3, prior round): NaN maps to 0 first (``np.nan_to_num``,
    which also folds +-inf to a large finite value), *then* the result is clipped to
    [-1, 1], scaled by 32767 and rounded to the nearest integer (``np.rint``, ties to
    even -- so ``pcm16([0.5 / 32767])`` gives 0, not 1) before the final cast. Doing
    it in this order means no NaN/inf ever reaches the multiply, so it raises no
    ``RuntimeWarning``.

    The scale is the *symmetric* int16 range: both +1.0 and -1.0 map to +-32767, one
    short of the asymmetric int16 minimum (-32768), so the positive and negative
    ends round-trip by the same factor instead of one direction clipping a fraction
    harder than the other.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return np.rint(audio * 32767.0).astype("<i2", copy=False).tobytes()


# The codec config fields that determine what a stored code means (review #4/#1,
# this round): change any of these and the same integer in `codes` decodes to
# different audio, so the fingerprint must cover exactly this set -- no more
# (irrelevant fields like training hyperparameters would churn the fingerprint for
# no reason) and no less. Every field is REQUIRED: a missing one means the config
# can't say what its own codes mean, which is worth a clear error, not a silent gap
# in the fingerprint.
#
# Read from ``<audio_tokenizer_dir>/config.json``, the ``Qwen3TTSTokenizerV2Config``
# the bundled qwen-tts tokenizer loads (``breeze_infer/runtime.py``), verified against
# the real bundled checkpoint's own file
# (``<model>/audio_tokenizer/config.json``):
#   - top level: ``input_sample_rate``, ``output_sample_rate`` (24000/24000),
#     ``encoder_valid_num_quantizers`` (how many of the encoder's quantizers are
#     actually emitted -- 16 of 32), ``encode_downsample_rate`` (the encode-side
#     frame rate: samples per codec frame, 1920), ``decode_upsample_rate`` (the
#     decode-side counterpart, 1920);
#   - ``encoder_config``: ``codebook_size`` (2048), ``codebook_dim`` (256),
#     ``num_quantizers`` (the encoder's full quantizer stack, 32, before
#     ``encoder_valid_num_quantizers`` slices it down to 16);
#   - ``decoder_config``: ``codebook_size`` (2048), ``codebook_dim`` (512 -- not the
#     same value as the encoder's), ``num_quantizers`` (16), ``semantic_codebook_size``
#     (4096), ``num_semantic_quantizers`` (1), and **both** ``upsample_rates``
#     ([8, 5, 4, 3]) and ``upsampling_ratios`` ([2, 2]) -- these are two distinct
#     fields, not aliases of each other: the decoder's total upsampling factor is
#     ``prod(upsample_rates + upsampling_ratios)``, so a change to either one changes
#     what a code decodes to. Both are present, at this path, in the real bundled
#     checkpoint's ``decoder_config`` and both are required.
_TOP_LEVEL_IDENTITY_FIELDS = (
    "input_sample_rate",
    "output_sample_rate",
    "encoder_valid_num_quantizers",
    "encode_downsample_rate",
    "decode_upsample_rate",
)
_ENCODER_IDENTITY_FIELDS = ("codebook_size", "codebook_dim", "num_quantizers")
_DECODER_IDENTITY_FIELDS = (
    "codebook_size",
    "codebook_dim",
    "num_quantizers",
    "semantic_codebook_size",
    "num_semantic_quantizers",
    "upsample_rates",
    "upsampling_ratios",
)


def _require_field(config: dict[str, Any], key: str, *, where: str) -> Any:
    value = config.get(key)
    if value is None:
        raise ValueError(f"codec config missing required {where} field '{key}'")
    return value


def _require_object_field(config: dict[str, Any], key: str, *, where: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        # ValueError, not TypeError (noqa: TRY004): every other identity-field
        # failure in this module is a ValueError, and callers (a future voices.py
        # loading a saved voice) need one exception type to catch for "bad config",
        # whether the problem is a missing field or the wrong shape of config.
        raise ValueError(f"{where} missing required '{key}' object")  # noqa: TRY004
    return value


def _codec_identity_fields(config: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``config.json`` that determines code meaning, nested by which
    sub-model it describes (encoder vs. decoder) so that a same-named field on each
    side -- ``codebook_size``, ``codebook_dim`` and ``num_quantizers`` all differ
    between the two in the real checkpoint -- can't collide and silently overwrite
    one with the other in a flattened dict.
    """
    if not isinstance(config, dict):
        raise ValueError("codec config.json must be a JSON object")  # noqa: TRY004
    encoder_config = _require_object_field(config, "encoder_config", where="codec config")
    decoder_config = _require_object_field(config, "decoder_config", where="codec config")

    return {
        "top": {
            field: _require_field(config, field, where="top-level")
            for field in _TOP_LEVEL_IDENTITY_FIELDS
        },
        "encoder": {
            field: _require_field(encoder_config, field, where="encoder_config")
            for field in _ENCODER_IDENTITY_FIELDS
        },
        "decoder": {
            field: _require_field(decoder_config, field, where="decoder_config")
            for field in _DECODER_IDENTITY_FIELDS
        },
    }


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors file's JSON header without reading the tensor payload.

    The format: an 8-byte little-endian uint64 header length, then that many bytes of
    JSON (https://github.com/huggingface/safetensors -- "Format"). The length prefix
    is untrusted data until checked against the file's own size and a sane upper
    bound (review #3): a git-lfs pointer file (a few hundred bytes of plain text, not
    real weights) has *some* 8 bytes at its start that decode to an arbitrary
    ``uint64``, and reading that many bytes without a cap could demand gigabytes for
    a file that is not a safetensors file at all.
    """
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"not a safetensors file: {path} is only {size} bytes")
    with path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if header_len > _MAX_SAFETENSORS_HEADER_BYTES or header_len > size - 8:
            raise ValueError(
                f"not a safetensors file: {path} declares a {header_len}-byte "
                f"header, which is not plausible for a {size}-byte file"
            )
        header_bytes = f.read(header_len)
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"not a safetensors file: {path} has an invalid JSON header"
        ) from exc
    if not isinstance(header, dict):
        raise ValueError(f"not a safetensors file: {path} has a non-object header")  # noqa: TRY004
    return header


def _tensor_identity(header: dict[str, Any]) -> dict[str, list[Any]]:
    """``{tensor_name: [dtype, shape]}``, excluding ``data_offsets``, any padding
    key and ``__metadata__`` (review #1/#5): those describe *where* a tensor's bytes
    sit in the file, not what the tensor logically is, and would make the fingerprint
    depend on a resharding or a writer's padding choices rather than the weights
    themselves. This is also, by construction, the limit of what this fingerprint can
    see: two checkpoints with identical tensor names/dtypes/shapes but different
    trained values hash the same. It detects an architecture, shape or dtype change,
    not a retrain of the same-shaped weights.

    Each entry is validated (review #9): a header that parses as JSON but has the
    wrong shape for a safetensors header (a tensor entry that isn't an object, or is
    an object missing ``dtype``/``shape``) raises ``ValueError`` here rather than
    crashing this function with a ``TypeError``/``KeyError`` when it's indexed.
    """
    identity: dict[str, list[Any]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(  # noqa: TRY004 -- ValueError for consistency, see module note above
                f"not a safetensors file: tensor entry '{name}' is not an object"
            )
        if "dtype" not in entry or "shape" not in entry:
            raise ValueError(
                f"not a safetensors file: tensor entry '{name}' is missing "
                "'dtype' or 'shape'"
            )
        identity[name] = [entry["dtype"], entry["shape"]]
    return identity


def _validate_weight_filename(name: Any) -> str:
    """A ``weight_map`` value must be a plain filename inside the checkpoint
    directory, not a path (review #5): no ``/`` or ``\\`` (which could reach outside
    the directory, e.g. an absolute path or ``../other.safetensors``), not the bare
    special names ``.``/``..``, and it must end in ``.safetensors`` -- the only
    format ``_safetensors_header`` knows how to parse.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"weight_map value must be a filename string, got {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError(f"weight_map value must not contain a path separator: {name!r}")
    if name in (".", ".."):
        raise ValueError(f"weight_map value must not be '.' or '..': {name!r}")
    if not name.endswith(".safetensors"):
        raise ValueError(f"weight_map value must end in .safetensors: {name!r}")
    return name


def _weight_map_files(index_path: Path) -> list[Path]:
    """The shard filenames a ``*.safetensors.index.json`` lists, validated (review
    #9, prior round; #5, this round): a same-shaped-but-wrong index (``weight_map``
    not an object, or a value that isn't a plain ``*.safetensors`` filename) raises
    ``ValueError`` here instead of surfacing as a ``TypeError`` from
    ``Path.__truediv__`` or an ``AttributeError`` from ``dict.get`` on something
    that isn't a dict -- or, worse, silently resolving a path outside the checkpoint
    directory. Every value is validated *before* ``filenames`` (the deduplicated
    set) is built, so a single bad entry is never lost to set deduplication ahead of
    being checked.
    """
    index = json.loads(index_path.read_text())
    if not isinstance(index, dict):
        raise ValueError(f"{index_path} must be a JSON object")  # noqa: TRY004
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} has no usable 'weight_map'")
    for name in weight_map.values():
        _validate_weight_filename(name)
    filenames = set(weight_map.values())
    return [index_path.parent / name for name in sorted(filenames)]


def _weight_files(directory: Path) -> list[Path]:
    """The weight file(s) ``Qwen3TTSTokenizerV2Model.from_pretrained`` actually loads
    from ``audio_tokenizer/`` -- HuggingFace's own resolution order (review #3, this
    round; see the module-level comment by ``_SAFETENSORS_SINGLE_FILE``): the
    single-file safetensors name first, *then* the sharded index, never the other
    way around (a directory could in principle carry a stale index next to a real
    single file, and the loader would still prefer the single file). A stray extra
    ``*.safetensors`` file the loader would never touch (an old backup, an unrelated
    shard) is ignored rather than folded into the fingerprint (review #1/#5, prior
    round).

    A directory with only a PyTorch ``.bin`` checkpoint (no safetensors at all) is a
    real, loadable codec -- just not one this fingerprint can identify by tensor
    header without adding a second, heavier parser -- so it's a clear, named
    ``ValueError`` rather than the generic ``FileNotFoundError`` for "no weights at
    all".
    """
    single_path = directory / _SAFETENSORS_SINGLE_FILE
    if single_path.is_file():
        return [single_path]

    index_path = directory / _SAFETENSORS_INDEX_FILE
    if index_path.is_file():
        return _weight_map_files(index_path)

    if (directory / _PYTORCH_SINGLE_FILE).is_file() or (
        directory / _PYTORCH_INDEX_FILE
    ).is_file():
        raise ValueError("the codec weights must be safetensors to fingerprint")

    raise FileNotFoundError(
        f"Neither {_SAFETENSORS_SINGLE_FILE} nor {_SAFETENSORS_INDEX_FILE} "
        f"found under {directory}"
    )


def codec_fingerprint(audio_tokenizer_dir: str | Path) -> str:
    """Identify the codec that produced (and will decode) a set of codec tokens.

    Saved voices store this next to their codes (data-model.md "Voice file v1").
    There is no re-encode path for a saved voice -- the original recording isn't
    kept, only its codes -- so a fingerprint mismatch at startup means that voice is
    *skipped*, not rebuilt.

    The fingerprint covers two things, hashed together in order:
    (a) the canonical JSON (``sort_keys=True``, compact separators, so re-serializing
        the same file with different key order or whitespace doesn't change the
        fingerprint) of the codec config's identity fields -- see
        ``_codec_identity_fields`` and the module-level field list above. Every field
        is required; a config missing one raises rather than silently leaving it out.
    (b) the tensor identity (``{name: [dtype, shape]}``, not the trained values --
        see ``_tensor_identity``) of the ONE weight file (or, for a sharded
        checkpoint, every shard the loader's own index lists) that
        ``Qwen3TTSTokenizerV2Model.from_pretrained`` actually loads.

    Consequently, this fingerprint detects an **architecture, shape or dtype**
    change -- a different codec entirely, a retrained model with a different
    quantizer count, etc. -- not a **retrain of weights with identical shapes**: two
    checkpoints that differ only in trained values hash the same. That's an accepted
    limitation of a fingerprint cheap enough to compute at every startup without
    reading multi-GB tensor payloads, not an oversight (see
    ``test_codec_fingerprint_cannot_detect_a_retrain_of_the_same_shapes`` in
    ``tests/test_audio.py``).

    Directory-only, not path plus caller-supplied facts: the caller already has this
    same directory (it's the one ``breeze_infer/runtime.py`` loads the audio
    tokenizer from), so a directory is no more information to thread through than
    the codec facts were, and it keeps the ``qwen_tts`` config/weights-loading
    conventions (the nested config blocks, the single-file-vs-sharded resolution)
    known in exactly one place instead of duplicated at every call site.

    Never hashes ``audio_tokenizer_dir`` itself (R13): moving the checkpoint to a new
    location on disk must not invalidate every saved voice.
    """
    directory = Path(audio_tokenizer_dir)
    config = json.loads((directory / "config.json").read_text())
    identity = _codec_identity_fields(config)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    tensor_identity: dict[str, list[Any]] = {}
    for weight_path in _weight_files(directory):
        tensor_identity.update(_tensor_identity(_safetensors_header(weight_path)))
    digest.update(
        json.dumps(tensor_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )

    return digest.hexdigest()
