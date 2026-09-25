from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

AUDIO_TAG = "<|AUDIO|>"
AUDIO_EOS = "<|audio_eos|>"
INSTRUCTION_BOS = "<ins_bos>"
INSTRUCTION_EOS = "<ins_eos>"


Segment = dict[str, Any]
Request = dict[str, Any]


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    required_fields: tuple[str, ...]
    build_segments: Callable[[Request], list[Segment]]
    build_negative_segments: Callable[[Request], list[Segment]] | None = None
    build_dual_branches: Callable[[Request], dict[str, list[Segment]]] | None = None
    reference_audio: bool = False


def _speaker_prefix(request: Request) -> str:
    speaker = request.get("speaker", "S0")
    if speaker in (None, ""):
        return ""
    if isinstance(speaker, str) and speaker.startswith("[") and speaker.endswith("]"):
        return speaker
    return f"[{speaker}]"


def _tts_plain_segments(request: Request) -> list[Segment]:
    return [{"type": "text", "text": f"{_speaker_prefix(request)}{request['text']}"}]


def _tts_instruction_segments(request: Request) -> list[Segment]:
    prefix = _speaker_prefix(request)
    return [
        {
            "type": "text",
            "text": f"{prefix}{INSTRUCTION_BOS}{request['instruction']}{INSTRUCTION_EOS}{request['text']}",
        }
    ]


def _tts_instruction_negative_segments(request: Request) -> list[Segment]:
    return _tts_plain_segments(request)


def _ref_audio_segment(
    request: Request,
    *,
    append_eos: bool = True,
    drop_last_frame: bool = False,
) -> Segment:
    """Reference audio always arrives pre-encoded as ``ref_audio_codes`` (an inline
    upload, a saved voice's stored codes, or the CLI's own ``--ref-audio`` encode) --
    there is no path-based variant; see ``breeze_infer.audio.encode_prompt_waveform``.
    """
    segment: Segment = {
        "type": "audio",
        "append_eos": append_eos,
        "drop_last_frame": drop_last_frame,
    }
    if request.get("ref_audio_codes") is not None:
        segment["audio_codes"] = request["ref_audio_codes"]
    return segment


def _ref_prefix_segments(request: Request) -> list[Segment]:
    """The reference-only part of ``ref_edit_tata`` (``ref_text`` + the reference
    audio segment): the common prefix that ``_ref_edit_tata_segments``,
    ``_ref_clone_tata_segments`` and ``split_reference_prefix`` all build on (review
    #6 -- there is exactly one place this pair of segments is assembled).
    """
    prefix = _speaker_prefix(request)
    return [
        {"type": "text", "text": f"{prefix}{request['ref_text']}"},
        _ref_audio_segment(request),
    ]


def _ref_clone_tata_segments(request: Request) -> list[Segment]:
    """``ref_edit_tata`` with no instruction: the reference prefix followed by the
    plain ``[speaker]text`` segment -- the same trailing segment ``tts_instruction``'s
    negative branch (``_tts_instruction_negative_segments``) uses.
    """
    return _ref_prefix_segments(request) + _tts_plain_segments(request)


def _ref_edit_tata_segments(request: Request) -> list[Segment]:
    """The reference prefix followed by the instruction-wrapped text segment -- the
    same trailing segment the plain ``tts_instruction`` template uses.
    """
    return _ref_prefix_segments(request) + _tts_instruction_segments(request)


def _ref_edit_tata_negative_segments(request: Request) -> list[Segment]:
    return _ref_clone_tata_segments(request)


def _ref_edit_tata_dual_branches(request: Request) -> dict[str, list[Segment]]:
    return {
        "uncond": _tts_plain_segments(request),
        "ref": _ref_clone_tata_segments(request),
        "ins": _tts_instruction_segments(request),
    }


TEMPLATES: dict[str, TemplateSpec] = {
    "tts_instruction": TemplateSpec(
        name="tts_instruction",
        required_fields=("text", "instruction"),
        build_segments=_tts_instruction_segments,
        build_negative_segments=_tts_instruction_negative_segments,
    ),
    "ref_edit_tata": TemplateSpec(
        name="ref_edit_tata",
        required_fields=("text", "instruction", "ref_text"),
        build_segments=_ref_edit_tata_segments,
        build_negative_segments=_ref_edit_tata_negative_segments,
        build_dual_branches=_ref_edit_tata_dual_branches,
        reference_audio=True,
    ),
}


def split_reference_prefix(
    request: Request,
) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """Split ``ref_edit_tata`` into (prefix, guided suffix, unguided suffix).

    ``prefix + guided`` renders exactly what ``build_segments`` renders, and
    ``prefix + unguided`` exactly what ``build_negative_segments`` renders, so a
    prefix processed once can be continued by either suffix -- true by construction,
    since ``_ref_edit_tata_segments``/``_ref_clone_tata_segments`` are themselves
    ``_ref_prefix_segments(request) + <this same guided/unguided piece>`` (review #6).
    """
    return (
        _ref_prefix_segments(request),
        _tts_instruction_segments(request),
        _tts_plain_segments(request),
    )


def get_template(name: str) -> TemplateSpec:
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown template '{name}'. Available: {sorted(TEMPLATES)}"
        ) from exc


def _normalize_codes_array(codes: np.ndarray) -> torch.Tensor:
    """Validate and normalize a numpy ``audio_codes`` array into a plain, native-
    byte-order, contiguous ``int64`` tensor (review #2).

    Checked on the *original* array, before any cast:
    - an object-dtype array (e.g. a ragged/mixed-type array) is rejected with a
      specific message, since ``np.issubdtype(object, np.integer)`` is silently
      ``False`` and would otherwise fall into the generic "not an integer dtype"
      message;
    - any other non-integer dtype (float, bool, complex) is rejected the same way
      as a torch tensor's dtype is (``_resolve_segment_audio_codes``).

    Only once the dtype class is known-good does ``np.ascontiguousarray(codes,
    dtype=np.int64)`` run, in one call: it makes a negative-stride view (e.g.
    ``codes[::-1]``) into a real contiguous copy (``torch.from_numpy`` outright
    refuses negative strides), it byte-swaps a non-native-endian array (e.g.
    ``dtype('>i2')``) into native order the same way ``.astype`` does (``torch``
    tensors have no non-native-byte-order concept at all), and it upcasts a numpy
    integer width/signedness that ``torch`` doesn't support well (``uint16``,
    ``uint32``, ``uint64``) into a signed width every torch build does. Values are
    still checked against ``codebook_size`` by the caller once this returns, so the
    generic upcast is safe: a real code is always tiny compared to ``int64``.
    """
    if codes.dtype == np.dtype("O"):
        raise ValueError("audio_codes must not be an object array")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"audio_codes must have an integer dtype, got {codes.dtype}")
    return torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64))


def _require_integer_tensor_dtype(codes: torch.Tensor) -> None:
    """The one shared dtype check (review #10) for every already-a-tensor path:
    a torch tensor has no object/byte-order/stride quirks to normalize, only a
    dtype that might be float, complex or bool instead of integer.
    """
    if codes.dtype is torch.bool or torch.is_floating_point(codes) or torch.is_complex(codes):
        raise ValueError(f"audio_codes must have an integer dtype, got {codes.dtype}")


def _resolve_segment_audio_codes(
    segment: Segment, *, codebooks: int, codebook_size: int
) -> torch.Tensor:
    """Reference audio is always pre-encoded before it reaches a template (see
    ``_ref_audio_segment``), so this only normalizes and validates an already-encoded
    codes array/tensor -- nothing here re-encodes audio, so it no longer takes an
    ``audio_tokenizer`` (review #6, prior round).

    Validation runs on the caller's own dtype and values, *before* the int16 cast: a
    float/bool/object array, a byte-order or stride quirk, or an out-of-range code
    that slipped through would otherwise be silently coerced or crash the conversion,
    and would only surface later as a CUDA device assert deep in the backbone, which
    poisons the whole process (review #1/#2).
    """
    codes = segment.get("audio_codes")
    if codes is None:
        raise ValueError("Audio segment must include audio_codes")

    if isinstance(codes, np.ndarray):
        codes = _normalize_codes_array(codes)
    elif isinstance(codes, torch.Tensor):
        _require_integer_tensor_dtype(codes)
    else:
        # A plain list/tuple (or anything else torch will try to interpret): torch's
        # own exception for e.g. a value too big for its inferred dtype (2**70) or a
        # string element is not always a ValueError (review #10), so it's wrapped
        # into one with the offending input still named in the message.
        try:
            codes = torch.as_tensor(codes)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise ValueError(f"audio_codes could not be read as a tensor: {exc}") from exc
        _require_integer_tensor_dtype(codes)

    if codes.ndim != 2:
        raise ValueError(
            f"audio_codes must be 2D [frames, codebooks], got {tuple(codes.shape)}"
        )
    if codes.shape[0] == 0:
        raise ValueError("audio_codes must have at least 1 frame")
    if codes.shape[1] != codebooks:
        raise ValueError(
            f"audio_codes must have {codebooks} codebooks, got {codes.shape[1]}"
        )
    low, high = int(codes.min()), int(codes.max())
    if low < 0 or high >= codebook_size:
        raise ValueError(
            f"audio_codes values must be in [0, {codebook_size}), "
            f"got range [{low}, {high}]"
        )

    return codes.to(torch.int16).cpu().contiguous()


def _missing_fields(request: Request, fields: tuple[str, ...]) -> list[str]:
    """Required fields that are absent, ``None``, or blank/whitespace-only.

    A string value is stripped before the truthy check, so ``" "`` counts as missing
    the same as ``""`` (review #10, prior round).

    ``ref_audio_codes`` never belongs in ``fields``: it's array-valued, and a bare
    ``bool()`` on a multi-element array raises, so it's checked separately by
    ``_check_reference_source`` instead of this truthy check.
    """
    missing = []
    for field in fields:
        value = request.get(field)
        if isinstance(value, str):
            value = value.strip()
        if not value:
            missing.append(field)
    return missing


def _check_reference_source(template: TemplateSpec, request: Request) -> None:
    if not template.reference_audio:
        return
    if request.get("ref_audio_codes") is None:
        raise ValueError(
            f"Request {request.get('id')} must provide ref_audio_codes"
        )


def _validate_fields(request: Request, fields: tuple[str, ...]) -> None:
    """The field-presence half of validation, usable on its own (review #4/#9, this
    round): ``prepare_suffix_inputs`` needs exactly this and nothing more -- no
    ``ref_audio_codes`` check, since the suffix carries no audio at all.
    """
    missing = _missing_fields(request, fields)
    if missing:
        raise ValueError(
            f"Request {request.get('id')} missing template fields: {missing}"
        )


def _validate_reference_request(
    template: TemplateSpec, request: Request, fields: tuple[str, ...]
) -> None:
    """Shared by ``prepare_inputs`` and ``prepare_prefix_inputs`` (review #8, prior
    round): both need the same two checks -- the given ``fields`` are present, and
    (for a reference-audio template) ``ref_audio_codes`` is too -- just against a
    different field tuple (the full template vs. the reference-prefix-only fields).
    Split into ``_validate_fields`` (the field-presence half) plus
    ``_check_reference_source`` (review #4/#9, this round) so a caller that only
    needs the first half -- ``prepare_suffix_inputs`` -- can use just that.
    """
    _validate_fields(request, fields)
    _check_reference_source(template, request)


def _codec_facts(model_config: Any) -> tuple[int, int]:
    """``(codebooks, codebook_size)``.

    ``codebooks`` keeps the existing fallback convention (``getattr`` with the real
    checkpoint's own value, 16, as the default) since it only affects the shape of an
    empty placeholder tensor when a request has no audio at all.

    ``codebook_size`` has no such fallback (review #8): it reaches templates as a
    **model attribute**, ``model_config.codec_config.codebook_size`` -- the backbone
    config's own record of the codec it was loaded with, the same attribute
    ``models/fast_streaming.py`` already reads as ``self._codec_codebook_size``. A
    missing value means the loaded model can't say what its own codec's valid code
    range is, so this raises rather than quietly assuming 2048.

    It is then cross-checked against the backbone embedding's own layout via
    ``codebook_pad_token_id``: ``models/fast_streaming.py``'s
    ``range(self._codec_codebook_size, int(self.model.config.vocab_size))`` treats
    every vocab id from ``codebook_size`` up to (not including) ``vocab_size`` as a
    reserved special id, and ``codebook_pad_token_id`` is one of those ids (2050 for
    the real checkpoint, with ``codebook_size`` 2048 and ``vocab_size`` 2051 -- 2
    reserved slots below the pad id, then the backbone's own EOS at ``vocab_size``).
    So ``codebook_pad_token_id`` must be at or above ``codebook_size``; if it's
    below, the codec's own valid codes would collide with the pad token, which means
    the codec and backbone configs were not loaded as a matching pair.
    """
    codebooks = getattr(model_config, "num_codebooks", 16)

    codec_config = getattr(model_config, "codec_config", None)
    codebook_size = getattr(codec_config, "codebook_size", None)
    if codebook_size is None:
        raise ValueError(
            "model_config.codec_config.codebook_size is required to validate "
            "reference audio codes, but the loaded model doesn't have it set"
        )

    pad_token_id = getattr(model_config, "codebook_pad_token_id", None)
    if pad_token_id is None:
        raise ValueError(
            "model_config.codebook_pad_token_id is required to cross-check "
            "codebook_size against the backbone embedding, but the loaded model "
            "doesn't have it set"
        )
    if pad_token_id < codebook_size:
        raise ValueError(
            f"codebook_pad_token_id ({pad_token_id}) must be >= codebook_size "
            f"({codebook_size}) -- the codec and backbone configs are inconsistent"
        )

    return codebooks, codebook_size


def _prepare_one(
    tokenizer: Any,
    model_config: Any,
    segments: list[Segment],
) -> dict[str, torch.Tensor]:
    rendered_segments: list[dict[str, str]] = []
    audio_tokens_list: list[torch.Tensor] = []
    codebooks, codebook_size = _codec_facts(model_config)

    for segment in segments:
        segment_type = segment["type"]
        if segment_type == "text":
            encoded = tokenizer(segment["text"], add_special_tokens=True)
            rendered = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
            rendered_segments.append({"type": "text", "value": rendered})
            continue

        if segment_type != "audio":
            raise ValueError(f"Unknown segment type: {segment_type}")

        codes = _resolve_segment_audio_codes(
            segment, codebooks=codebooks, codebook_size=codebook_size
        )
        if segment.get("drop_last_frame", False):
            if codes.shape[0] <= 1:
                raise ValueError(
                    "Cannot drop the last frame from an audio segment with <= 1 frame"
                )
            codes = codes[:-1].contiguous()
        audio_tokens_list.append(codes)

        placeholders = AUDIO_TAG * codes.shape[0]
        if segment.get("append_eos", False):
            placeholders += AUDIO_EOS
        rendered_segments.append({"type": "audio", "value": placeholders})

    final_text = "".join(segment["value"] for segment in rendered_segments)
    encoded = tokenizer(final_text, add_special_tokens=False, return_tensors="pt")

    text_ids_mask: list[bool] = []
    text_ids_len: list[int] = []
    for segment in rendered_segments:
        segment_len = len(
            tokenizer(segment["value"], add_special_tokens=False)["input_ids"]
        )
        if segment["type"] == "text":
            text_ids_mask.extend([True] * segment_len)
            text_ids_len.append(segment_len)
        else:
            text_ids_mask.extend([False] * segment_len)

    if audio_tokens_list:
        audio_tokens = torch.cat(audio_tokens_list, dim=0).unsqueeze(0)
    else:
        audio_tokens = torch.zeros((1, 0, codebooks), dtype=torch.int16)

    encoded["audio_tokens"] = audio_tokens
    encoded["text_ids_mask"] = torch.tensor([text_ids_mask], dtype=torch.bool)
    encoded["text_ids_len"] = torch.tensor(text_ids_len, dtype=torch.long)
    return encoded


def _collate_inputs(
    tokenizer: Any, inputs_list: list[dict[str, torch.Tensor]], device: str
) -> dict[str, torch.Tensor | None]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id")

    audio_tokens = torch.cat([item["audio_tokens"] for item in inputs_list], dim=1)
    max_len = max(item["input_ids"].shape[1] for item in inputs_list)

    input_ids_list = []
    attention_mask_list = []
    text_ids_mask_list = []
    for item in inputs_list:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        text_ids_mask = item["text_ids_mask"]
        pad_len = max_len - input_ids.shape[1]
        if pad_len > 0:
            input_ids = F.pad(input_ids, (pad_len, 0), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)
            text_ids_mask = F.pad(text_ids_mask, (pad_len, 0), value=False)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        text_ids_mask_list.append(text_ids_mask)

    input_values = audio_tokens.to(device) if audio_tokens.shape[1] > 0 else None
    return {
        "input_ids": torch.cat(input_ids_list, dim=0).to(device),
        "attention_mask": torch.cat(attention_mask_list, dim=0).to(device),
        "text_ids_mask": torch.cat(text_ids_mask_list, dim=0).to(device),
        "text_ids_len": torch.cat(
            [item["text_ids_len"] for item in inputs_list], dim=0
        ).to(device),
        "input_values": input_values,
    }


def _prepare_segment_batches(
    tokenizer: Any,
    model_config: Any,
    device: str,
    segment_batches: list[list[Segment]],
) -> dict[str, torch.Tensor | None]:
    inputs_list = [
        _prepare_one(tokenizer, model_config, segments) for segments in segment_batches
    ]
    return _collate_inputs(tokenizer, inputs_list, device)


def prepare_inputs(
    tokenizer: Any,
    model: Any,
    requests: list[Request],
    template: TemplateSpec,
    *,
    guidance_scale: float,
    guidance_scale_ref: float | None,
    guidance_scale_ins: float | None,
) -> dict[str, torch.Tensor | None | float]:
    for request in requests:
        _validate_reference_request(template, request, template.required_fields)

    positive_segments = [template.build_segments(request) for request in requests]
    inputs = _prepare_segment_batches(
        tokenizer,
        model.config,
        model.device,
        positive_segments,
    )

    use_dual_cfg = (
        guidance_scale_ref is not None
        and guidance_scale_ins is not None
        and template.build_dual_branches is not None
    )
    if use_dual_cfg:
        branch_batches = [template.build_dual_branches(request) for request in requests]
        for branch_name, prefix in [
            ("uncond", "cfg_uncond"),
            ("ref", "cfg_ref"),
            ("ins", "cfg_ins"),
        ]:
            branch_inputs = _prepare_segment_batches(
                tokenizer,
                model.config,
                model.device,
                [branches[branch_name] for branches in branch_batches],
            )
            inputs[f"{prefix}_prompt_ids"] = branch_inputs["input_ids"]
            inputs[f"{prefix}_prompt_attention_mask"] = branch_inputs["attention_mask"]
            inputs[f"{prefix}_text_ids_mask"] = branch_inputs["text_ids_mask"]
            inputs[f"{prefix}_text_ids_len"] = branch_inputs["text_ids_len"]
        inputs["cfg_scale_ref"] = guidance_scale_ref
        inputs["cfg_scale_ins"] = guidance_scale_ins
        return inputs

    if guidance_scale != 1.0:
        if template.build_negative_segments is None:
            raise ValueError(
                f"Template '{template.name}' does not define a negative prompt but cfg_scale={guidance_scale}"
            )
        negative_inputs = _prepare_segment_batches(
            tokenizer,
            model.config,
            model.device,
            [template.build_negative_segments(request) for request in requests],
        )
        inputs["cfg_negative_prompt_ids"] = negative_inputs["input_ids"]
        inputs["cfg_negative_prompt_attention_mask"] = negative_inputs["attention_mask"]
        inputs["cfg_negative_text_ids_mask"] = negative_inputs["text_ids_mask"]
        inputs["cfg_negative_text_ids_len"] = negative_inputs["text_ids_len"]
        if negative_inputs["input_values"] is not None:
            inputs["cfg_negative_input_values"] = negative_inputs["input_values"]
        inputs["cfg_scale"] = guidance_scale

    return inputs


def prepare_prefix_inputs(
    tokenizer: Any,
    model: Any,
    request: Request,
) -> dict[str, torch.Tensor | None]:
    """Batch-1 inputs for the reference prefix of ``ref_edit_tata`` alone."""
    template = get_template("ref_edit_tata")
    _validate_reference_request(template, request, ("ref_text",))
    return _prepare_segment_batches(
        tokenizer,
        model.config,
        model.device,
        [_ref_prefix_segments(request)],
    )


def prepare_suffix_inputs(
    tokenizer: Any,
    model: Any,
    request: Request,
    *,
    guidance_scale: float,
) -> dict[str, torch.Tensor | None | float]:
    """Inputs for the part of ``ref_edit_tata`` that follows a cached prefix.

    The result has the same keys as ``prepare_inputs`` for a single-CFG request
    (``input_ids`` plus ``cfg_negative_*`` when ``guidance_scale != 1``) but
    carries no audio, so the runtime's branch builder can consume it unchanged.

    Unlike ``prepare_inputs``/``prepare_prefix_inputs``, this needs neither
    ``ref_audio_codes`` nor ``ref_text`` (review #4/#9, this round): the suffix is
    pure text, continuing a prefix that ``prepare_prefix_inputs`` already prepared
    separately for the same request. So it builds ``guided``/``unguided`` directly
    from ``_tts_instruction_segments``/``_tts_plain_segments`` rather than calling
    ``split_reference_prefix``, which also computes the reference prefix (and so
    reads ``request['ref_text']``, even though that piece is then discarded) --
    a request with only ``id``/``text``/``instruction`` must work here.
    """
    _validate_fields(request, ("text", "instruction"))
    guided = _tts_instruction_segments(request)
    inputs = _prepare_segment_batches(tokenizer, model.config, model.device, [guided])
    if guidance_scale != 1.0:
        unguided = _tts_plain_segments(request)
        negative = _prepare_segment_batches(
            tokenizer, model.config, model.device, [unguided]
        )
        inputs["cfg_negative_prompt_ids"] = negative["input_ids"]
        inputs["cfg_negative_prompt_attention_mask"] = negative["attention_mask"]
        inputs["cfg_negative_text_ids_mask"] = negative["text_ids_mask"]
        inputs["cfg_negative_text_ids_len"] = negative["text_ids_len"]
        inputs["cfg_scale"] = guidance_scale
    return inputs
