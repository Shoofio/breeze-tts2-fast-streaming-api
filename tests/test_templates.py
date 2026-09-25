from __future__ import annotations

import numpy as np
import pytest
import torch

from breeze_infer.templates import (
    get_template,
    prepare_inputs,
    prepare_prefix_inputs,
    prepare_suffix_inputs,
    split_reference_prefix,
)
from tests.fakes import FakeTokenizer, fake_model


def _model_with_codec_facts(*, codebook_size=2048, codebook_pad_token_id=2050):
    """``fake_model()`` plus the codec facts ``_codec_facts`` now requires (review
    #8): ``tests/fakes.py`` doesn't set ``model.config.codec_config.codebook_size``
    (another agent owns that file, so it isn't touched here) -- this augments the
    ``SimpleNamespace`` ``fake_model()`` returns locally, in this test module only.
    ``codebook_pad_token_id`` is already 2050 on ``fake_model()``; it's set again
    here just so a test can override it independently of ``codebook_size``.
    """
    model = fake_model()
    model.config.codec_config.codebook_size = codebook_size
    model.config.codebook_pad_token_id = codebook_pad_token_id
    return model


def _reference_request(**overrides):
    request = {
        "id": "r1",
        "text": "hello world",
        "instruction": "calm",
        "speaker": "S0",
        "ref_text": "the transcript",
    }
    request.update(overrides)
    return request


def _prepare(codes, *, model=None, guidance_scale=1.0):
    return prepare_inputs(
        FakeTokenizer(),
        model or _model_with_codec_facts(),
        [_reference_request(ref_audio_codes=codes)],
        get_template("ref_edit_tata"),
        guidance_scale=guidance_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )


def test_ref_edit_tata_uses_pre_encoded_reference_codes() -> None:
    inputs = _prepare(np.zeros((7, 16), dtype=np.int16))

    assert tuple(inputs["input_values"].shape) == (1, 7, 16)
    placeholder_chars = len("<|AUDIO|>") * 7 + len("<|audio_eos|>")
    assert int((~inputs["text_ids_mask"]).sum()) == placeholder_chars


def test_ref_edit_tata_requires_reference_codes() -> None:
    with pytest.raises(ValueError, match="ref_audio_codes"):
        prepare_inputs(
            FakeTokenizer(),
            _model_with_codec_facts(),
            [_reference_request()],
            get_template("ref_edit_tata"),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )


# --- codes validation (review #2/#1) ------------------------------------------


def test_ref_edit_tata_accepts_a_torch_tensor_directly() -> None:
    codes = torch.zeros((5, 16), dtype=torch.int64)

    inputs = _prepare(codes)

    assert tuple(inputs["input_values"].shape) == (1, 5, 16)


def test_ref_edit_tata_accepts_negative_strided_codes() -> None:
    # arr[::-1] is a valid numpy view with a negative stride on axis 0; torch.from_numpy
    # refuses that directly, so this exercises the np.ascontiguousarray normalization.
    base = np.arange(3 * 16, dtype=np.int16).reshape(3, 16) % 2048
    strided = base[::-1]
    assert strided.strides[0] < 0

    inputs = _prepare(strided)

    expected = torch.from_numpy(base[::-1].copy().astype(np.int16))
    assert torch.equal(inputs["input_values"][0], expected)


def test_ref_edit_tata_accepts_big_endian_codes() -> None:
    native = (np.arange(4 * 16, dtype=np.int64).reshape(4, 16) % 2048).astype(np.int16)
    big_endian = native.astype(">i2")
    assert big_endian.dtype.byteorder == ">"

    inputs = _prepare(big_endian)

    assert torch.equal(inputs["input_values"][0], torch.from_numpy(native))


def test_ref_edit_tata_accepts_uint16_codes() -> None:
    codes = np.zeros((3, 16), dtype=np.uint16)
    codes[0, 0] = 2047

    inputs = _prepare(codes)

    assert inputs["input_values"][0, 0, 0].item() == 2047


def test_ref_edit_tata_rejects_object_arrays() -> None:
    codes = np.empty((3, 16), dtype=object)
    codes.fill(0)

    with pytest.raises(ValueError, match="object array"):
        _prepare(codes)


@pytest.mark.parametrize(
    "codes",
    [
        np.zeros((5, 16), dtype=np.float32),
        np.zeros((5, 16), dtype=bool),
    ],
    ids=["float", "bool"],
)
def test_ref_edit_tata_rejects_non_integer_codes(codes) -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        _prepare(codes)


def test_ref_edit_tata_rejects_the_wrong_codebook_count() -> None:
    with pytest.raises(ValueError, match="16 codebooks"):
        _prepare(np.zeros((5, 8), dtype=np.int16))


def test_ref_edit_tata_rejects_zero_frames() -> None:
    with pytest.raises(ValueError, match="at least 1 frame"):
        _prepare(np.zeros((0, 16), dtype=np.int16))


@pytest.mark.parametrize("bad_value", [-1, 2048], ids=["negative", "at_codebook_size"])
def test_ref_edit_tata_rejects_out_of_range_codes(bad_value) -> None:
    codes = np.zeros((3, 16), dtype=np.int64)
    codes[1, 4] = bad_value

    with pytest.raises(ValueError, match=r"\[0, 2048\)"):
        _prepare(codes)


# --- _codec_facts (review #8) ---------------------------------------------------


def test_codec_facts_raises_when_codebook_size_is_not_set_on_the_model() -> None:
    # Plain fake_model(): its config.codec_config has no codebook_size (see the
    # module docstring above) -- proving there is no silent 2048 fallback.
    with pytest.raises(ValueError, match="codec_config.codebook_size"):
        _prepare(np.zeros((3, 16), dtype=np.int16), model=fake_model())


def test_codec_facts_raises_when_pad_token_id_is_below_codebook_size() -> None:
    model = _model_with_codec_facts(codebook_size=2048, codebook_pad_token_id=100)

    with pytest.raises(ValueError, match="codebook_pad_token_id"):
        _prepare(np.zeros((3, 16), dtype=np.int16), model=model)


def test_missing_fields_treats_whitespace_only_as_missing() -> None:
    request = _reference_request(
        instruction="   ", ref_audio_codes=np.zeros((2, 16), dtype=np.int16)
    )

    with pytest.raises(ValueError, match=r"\['instruction'\]"):
        prepare_inputs(
            FakeTokenizer(),
            _model_with_codec_facts(),
            [request],
            get_template("ref_edit_tata"),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )


# --- segment de-duplication (review #6) -----------------------------------------


def test_split_reference_prefix_reproduces_both_branches() -> None:
    template = get_template("ref_edit_tata")
    request = _reference_request(ref_audio_codes=np.zeros((3, 16), dtype=np.int16))

    prefix, guided, unguided = split_reference_prefix(request)

    assert prefix + guided == template.build_segments(request)
    assert prefix + unguided == template.build_negative_segments(request)
    assert prefix[1]["type"] == "audio"


def test_dual_branches_uncond_matches_the_plain_negative_branch() -> None:
    """review #6: the dual-CFG 'uncond' branch and ref_edit_tata's own negative
    branch's trailing segment must be the literal same text -- both are
    ``_tts_plain_segments`` now, not two independent copies of the f-string.
    """
    template = get_template("ref_edit_tata")
    request = _reference_request(ref_audio_codes=np.zeros((3, 16), dtype=np.int16))

    branches = template.build_dual_branches(request)

    assert branches["uncond"] == template.build_negative_segments(request)[-1:]


# --- prefix/suffix ----------------------------------------------------------------


def test_prefix_and_suffix_inputs_concatenate_to_the_full_prompt() -> None:
    tokenizer = FakeTokenizer()
    model = _model_with_codec_facts()
    request = _reference_request(ref_audio_codes=np.ones((5, 16), dtype=np.int16))

    full = prepare_inputs(
        tokenizer,
        model,
        [request],
        get_template("ref_edit_tata"),
        guidance_scale=4.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    prefix = prepare_prefix_inputs(tokenizer, model, request)
    suffix = prepare_suffix_inputs(tokenizer, model, request, guidance_scale=4.0)

    joined = torch.cat([prefix["input_ids"], suffix["input_ids"]], dim=1)
    assert torch.equal(joined, full["input_ids"])
    joined_neg = torch.cat(
        [prefix["input_ids"], suffix["cfg_negative_prompt_ids"]], dim=1
    )
    assert torch.equal(joined_neg, full["cfg_negative_prompt_ids"])
    assert suffix["input_values"] is None
    assert torch.equal(prefix["input_values"], full["input_values"])
    assert suffix["cfg_scale"] == 4.0
    assert "cfg_negative_prompt_ids" not in prepare_suffix_inputs(
        tokenizer, model, request, guidance_scale=1.0
    )


def test_prefix_inputs_need_only_the_reference_fields() -> None:
    request = {
        "id": "p",
        "speaker": "S0",
        "ref_text": "the transcript",
        "ref_audio_codes": np.zeros((4, 16), dtype=np.int16),
    }

    prefix = prepare_prefix_inputs(
        FakeTokenizer(), _model_with_codec_facts(), request
    )

    assert tuple(prefix["input_values"].shape) == (1, 4, 16)
    assert int(prefix["attention_mask"].sum()) == prefix["input_ids"].shape[1]


def test_prefix_inputs_also_require_reference_codes() -> None:
    request = {"id": "p", "speaker": "S0", "ref_text": "the transcript"}

    with pytest.raises(ValueError, match="ref_audio_codes"):
        prepare_prefix_inputs(FakeTokenizer(), _model_with_codec_facts(), request)


def test_suffix_inputs_use_the_shared_validator() -> None:
    """review #7: prepare_suffix_inputs now runs the same reference-source check as
    prepare_inputs/prepare_prefix_inputs, so a request missing ref_audio_codes is
    rejected before any text is prepared, with the same message shape.
    """
    request = {"id": "s", "speaker": "S0", "text": "hi", "instruction": "calm"}

    with pytest.raises(ValueError, match="ref_audio_codes"):
        prepare_suffix_inputs(
            FakeTokenizer(), _model_with_codec_facts(), request, guidance_scale=1.0
        )
