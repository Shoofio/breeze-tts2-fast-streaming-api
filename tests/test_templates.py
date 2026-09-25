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
from tests.fakes import FakeCodec, FakeTokenizer, fake_model


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


def test_ref_edit_tata_uses_pre_encoded_reference_codes() -> None:
    audio_tokenizer = FakeCodec()
    codes = np.zeros((7, 16), dtype=np.int16)

    inputs = prepare_inputs(
        FakeTokenizer(),
        audio_tokenizer,
        fake_model(),
        [_reference_request(ref_audio_codes=codes)],
        get_template("ref_edit_tata"),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    # No call ever reaches the codec: codes handed in are used as-is, never re-encoded.
    assert audio_tokenizer.encode_calls == 0
    assert tuple(inputs["input_values"].shape) == (1, 7, 16)
    placeholder_chars = len("<|AUDIO|>") * 7 + len("<|audio_eos|>")
    assert int((~inputs["text_ids_mask"]).sum()) == placeholder_chars


def test_ref_edit_tata_requires_reference_codes() -> None:
    with pytest.raises(ValueError, match="ref_audio_codes"):
        prepare_inputs(
            FakeTokenizer(),
            FakeCodec(),
            fake_model(),
            [_reference_request()],
            get_template("ref_edit_tata"),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )


def test_split_reference_prefix_reproduces_both_branches() -> None:
    template = get_template("ref_edit_tata")
    request = _reference_request(ref_audio_codes=np.zeros((3, 16), dtype=np.int16))

    prefix, guided, unguided = split_reference_prefix(request)

    assert prefix + guided == template.build_segments(request)
    assert prefix + unguided == template.build_negative_segments(request)
    assert prefix[1]["type"] == "audio"


def test_prefix_and_suffix_inputs_concatenate_to_the_full_prompt() -> None:
    tokenizer = FakeTokenizer()
    audio_tokenizer = FakeCodec()
    model = fake_model()
    request = _reference_request(ref_audio_codes=np.ones((5, 16), dtype=np.int16))

    full = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template("ref_edit_tata"),
        guidance_scale=4.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    prefix = prepare_prefix_inputs(tokenizer, audio_tokenizer, model, request)
    suffix = prepare_suffix_inputs(
        tokenizer, audio_tokenizer, model, request, guidance_scale=4.0
    )

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
        tokenizer, audio_tokenizer, model, request, guidance_scale=1.0
    )


def test_prefix_inputs_need_only_the_reference_fields() -> None:
    request = {
        "id": "p",
        "speaker": "S0",
        "ref_text": "the transcript",
        "ref_audio_codes": np.zeros((4, 16), dtype=np.int16),
    }

    prefix = prepare_prefix_inputs(FakeTokenizer(), FakeCodec(), fake_model(), request)

    assert tuple(prefix["input_values"].shape) == (1, 4, 16)
    assert int(prefix["attention_mask"].sum()) == prefix["input_ids"].shape[1]
