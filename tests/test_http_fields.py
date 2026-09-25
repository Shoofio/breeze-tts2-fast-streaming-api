"""`http_fields` tests (tasks.md T037): the happy path only.

Table-driven tests for the malformed corpus (bad numbers, ranges, duplicates, control
characters) arrive with T044/T048; this file only checks that valid input parses to the
defaults and values the contract promises, and that the handful of rules T037 already
owns (BC-02 empty-means-absent, BC-09 blank instruction, BC-10 required text, FR-006's
`0` -> `None`) hold.

A tiny FastAPI app drives `read_fields` through a real `Request`, exercising both
`multipart/form-data` (with a real file part for `ref_audio`) and
`application/x-www-form-urlencoded` bodies, plus the query string -- `TestClient` builds
the real ASGI request `read_fields` has to parse, rather than constructing a `Fields` by
hand and skipping that parsing entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from breeze_infer.errors import install_error_handlers
from breeze_infer.http_fields import (
    DEFAULT_CFG_SCALE,
    DEFAULT_INSTRUCTION,
    DEFAULT_SEED,
    InlineRef,
    NoReference,
    ReferenceSpec,
    SpeechRequest,
    VoiceRef,
    parse_speech,
    read_fields,
)
from breeze_infer.settings import Settings, settings_from_args
from tests.fakes import RecordingEvents


def _settings(tmp_path: Path) -> Settings:
    return settings_from_args([str(tmp_path)])


def _reference_payload(reference: ReferenceSpec) -> dict[str, object]:
    """A JSON-safe summary of a `ReferenceSpec`, tagged by variant, for the test route's
    response -- the dataclasses themselves (and `InlineRef`'s raw bytes) aren't JSON."""
    if isinstance(reference, NoReference):
        return {"kind": "none"}
    if isinstance(reference, VoiceRef):
        return {
            "kind": "voice",
            "voice_id": reference.voice_id,
            "ref_text_override": reference.ref_text_override,
        }
    if isinstance(reference, InlineRef):
        return {
            "kind": "inline",
            "audio_bytes": reference.audio_bytes.decode("latin-1"),
            "ref_text": reference.ref_text,
        }
    raise AssertionError(f"unhandled ReferenceSpec variant: {reference!r}")  # pragma: no cover


def _speech_payload(parsed: SpeechRequest) -> dict[str, object]:
    return {
        "text": parsed.text,
        "instruction": parsed.instruction,
        "reference": _reference_payload(parsed.reference),
        "cfg_scale": parsed.cfg_scale,
        "seed": parsed.seed,
        "temperature": parsed.temperature,
        "top_k": parsed.top_k,
        "top_p": parsed.top_p,
        "repetition_penalty": parsed.repetition_penalty,
        "max_new_tokens": parsed.max_new_tokens,
        "split_chars": parsed.split_chars,
    }


def _client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    install_error_handlers(app, RecordingEvents())
    settings = _settings(tmp_path)

    @app.post("/speech")
    async def speech(request: Request) -> JSONResponse:
        fields = await read_fields(request)
        parsed = parse_speech(fields, settings)
        return JSONResponse(_speech_payload(parsed))

    return TestClient(app, raise_server_exceptions=False)


# --- defaults ---------------------------------------------------------------------


def test_defaults_with_only_text_given(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hello there"})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "text": "hello there",
        "instruction": DEFAULT_INSTRUCTION,
        "reference": {"kind": "none"},
        "cfg_scale": DEFAULT_CFG_SCALE,
        "seed": DEFAULT_SEED,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
        "max_new_tokens": None,
        "split_chars": 600,  # Settings.split_chars default (settings.py DEFAULT_SPLIT_CHARS)
    }


# --- parsing valid values, from the body and from the query -----------------------


_VALID_FORM = {
    "text": "hello there",
    "instruction": "Sound cheerful.",
    "cfg_scale": "2.5",
    "seed": "123",
    "temperature": "0.8",
    "top_k": "40",
    "top_p": "0.9",
    "repetition_penalty": "1.1",
    "max_new_tokens": "500",
    "split_chars": "300",
}

_EXPECTED_FOR_VALID_FORM = {
    "text": "hello there",
    "instruction": "Sound cheerful.",
    "reference": {"kind": "none"},
    "cfg_scale": 2.5,
    "seed": 123,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_new_tokens": 500,
    "split_chars": 300,
}


def test_parses_valid_values_from_the_form_body(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data=_VALID_FORM)

    assert response.status_code == 200
    assert response.json() == _EXPECTED_FOR_VALID_FORM


def test_parses_valid_values_from_the_query_string(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", params=_VALID_FORM)

    assert response.status_code == 200
    assert response.json() == _EXPECTED_FOR_VALID_FORM


def test_fields_may_be_split_across_body_and_query(tmp_path: Path) -> None:
    """Nothing in T037 forbids mixing sources for *different* fields (T048 adds the
    duplicate-field check for the *same* field appearing in both)."""
    body = {"text": "hello there", "seed": "7"}
    query = {"cfg_scale": "3.0"}

    response = _client(tmp_path).post("/speech", data=body, params=query)

    assert response.status_code == 200
    body_json = response.json()
    assert body_json["seed"] == 7
    assert body_json["cfg_scale"] == 3.0


# --- FR-006: 0 means the model default (None) --------------------------------------


@pytest.mark.parametrize(
    "field", ["temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]
)
def test_zero_means_default_for_sampling_fields(tmp_path: Path, field: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: "0"})

    assert response.status_code == 200
    assert response.json()[field] is None


# --- BC-09: a blank instruction uses the default -----------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_instruction_uses_the_default(tmp_path: Path, blank: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "instruction": blank})

    assert response.status_code == 200
    assert response.json()["instruction"] == DEFAULT_INSTRUCTION


def test_non_blank_instruction_is_kept_exactly(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "instruction": "  Speak slowly.  "}
    )

    assert response.status_code == 200
    assert response.json()["instruction"] == "  Speak slowly.  "


# --- BC-10: missing or blank text is 400 text_required -----------------------------


def test_missing_text_gets_400_text_required(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_text_gets_400_text_required(tmp_path: Path, blank: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": blank})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


# --- BC-02: an empty value means absent ---------------------------------------------


def test_empty_seed_falls_back_to_the_default(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "seed": ""})

    assert response.status_code == 200
    assert response.json()["seed"] == DEFAULT_SEED


def test_empty_voice_id_means_no_reference(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "voice_id": ""})

    assert response.status_code == 200
    assert response.json()["reference"] == {"kind": "none"}


# --- ReferenceSpec variants ----------------------------------------------------------


def test_voice_id_alone_gives_voice_ref_with_no_override(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "alice"}
    )

    assert response.status_code == 200
    assert response.json()["reference"] == {
        "kind": "voice",
        "voice_id": "alice",
        "ref_text_override": None,
    }


def test_voice_id_with_ref_text_overrides_the_stored_transcript(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "alice", "ref_text": "hello world"}
    )

    assert response.status_code == 200
    assert response.json()["reference"] == {
        "kind": "voice",
        "voice_id": "alice",
        "ref_text_override": "hello world",
    }


def test_ref_audio_and_ref_text_give_inline_ref(tmp_path: Path) -> None:
    audio_bytes = b"not-real-audio-bytes"
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "ref_text": "a voice saying hello"},
        files={"ref_audio": ("ref.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["reference"] == {
        "kind": "inline",
        "audio_bytes": audio_bytes.decode("latin-1"),
        "ref_text": "a voice saying hello",
    }


def test_voice_id_with_ref_audio_gets_reference_conflict(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "voice_id": "alice", "ref_text": "hello"},
        files={"ref_audio": ("ref.wav", b"anything", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "voice_id and ref_audio cannot be used together",
        "code": "reference_conflict",
    }


def test_ref_audio_without_ref_text_gets_ref_text_required(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi"},
        files={"ref_audio": ("ref.wav", b"anything", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text is required with ref_audio",
        "code": "ref_text_required",
    }


def test_ref_text_alone_gets_reference_required(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "ref_text": "hello world"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text needs ref_audio or voice_id",
        "code": "reference_required",
    }
