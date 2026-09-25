"""`http_fields` tests (tasks.md T037): the happy path only.

Table-driven tests for the malformed corpus (bad numbers, ranges, duplicates, control
characters) arrive with T044/T048; this file only checks that valid input parses to the
defaults and values the contract promises, and that the handful of rules T037 already
owns (BC-02 empty-means-absent, BC-09 blank instruction, BC-10 required text, FR-006's
`0` -> `None`) hold, plus the T037 review 1 findings listed below `# --- review 1 findings`.

A tiny FastAPI app drives `read_fields` through a real `Request`, exercising both
`multipart/form-data` (with a real file part for `ref_audio`) and
`application/x-www-form-urlencoded` bodies, plus the query string -- `TestClient` builds
the real ASGI request `read_fields` has to parse, rather than constructing a `Fields` by
hand and skipping that parsing entirely.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from breeze_infer import http_fields
from breeze_infer.errors import install_error_handlers
from breeze_infer.http_fields import (
    DEFAULT_CFG_SCALE,
    DEFAULT_INSTRUCTION,
    DEFAULT_SEED,
    FORM_MAX_PART_SIZE,
    InlineRef,
    NoReference,
    ReferenceSpec,
    SpeechRequest,
    VoiceRef,
    parse_speech,
    read_fields,
)
from breeze_infer.limits import MAX_TEXT_CHARS
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


# --- review 1 findings ---------------------------------------------------------------


def _multipart_body(fields: dict[str, bytes]) -> tuple[bytes, str]:
    """A hand-built `multipart/form-data` body with raw byte field values -- so a test can
    send bytes httpx's own multipart encoder would never produce (invalid UTF-8), the same
    way `tests/test_body_limit.py`'s `_oversize_multipart_body` does."""
    boundary = "xxxxBOUNDARYxxxx"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii")
        + value
        + b"\r\n"
        for name, value in fields.items()
    ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


# finding #1 (HIGH): a file part under any field name other than ref_audio is rejected.


@pytest.mark.parametrize(
    "field", ["text", "instruction", "voice_id", "ref_text", "seed"]
)
def test_file_part_under_a_text_field_name_gets_400(tmp_path: Path, field: str) -> None:
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi"} if field != "text" else {},
        files={field: ("f.txt", b"not text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": f"{field} must be a text field",
        "code": "invalid_field",
    }


# finding #2: a string ref_audio (form field or query string) is rejected.


def test_ref_audio_as_a_multipart_text_field_gets_400(tmp_path: Path) -> None:
    body, content_type = _multipart_body({"text": b"hi", "ref_audio": b"not a file"})

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": content_type}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


def test_ref_audio_as_a_urlencoded_field_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "ref_audio": "not a file"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


def test_ref_audio_in_the_query_string_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi"}, params={"ref_audio": "not a file"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


# finding #3: FORM_MAX_PART_SIZE is derived so a full-length CJK/emoji text field's
# percent-encoded urlencoded form fits.


def test_form_max_part_size_is_derived_from_max_text_chars() -> None:
    assert FORM_MAX_PART_SIZE == MAX_TEXT_CHARS * 12


def test_a_full_length_four_byte_char_text_field_fits_urlencoded(tmp_path: Path) -> None:
    # "🎉" is a 4-byte UTF-8 code point; percent-encoded it's exactly 12 ASCII bytes
    # ("%XX" x 4), so MAX_TEXT_CHARS of them is exactly at FORM_MAX_PART_SIZE.
    text = "\U0001f389" * MAX_TEXT_CHARS
    encoded_value = quote(text, safe="")
    assert len(encoded_value) == FORM_MAX_PART_SIZE
    body = f"text={encoded_value}".encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == text


def test_an_oversize_urlencoded_value_gets_400(tmp_path: Path) -> None:
    encoded_value = quote("a" * (FORM_MAX_PART_SIZE + 1), safe="")
    body = f"text={encoded_value}".encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# finding #4: strict UTF-8, for both content types, plus a declared non-UTF-8 multipart
# charset.


def test_valid_cjk_text_round_trips_urlencoded(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "你好"})

    assert response.status_code == 200
    assert response.json()["text"] == "你好"


def test_valid_cjk_text_round_trips_multipart(tmp_path: Path) -> None:
    body, content_type = _multipart_body({"text": "你好".encode()})

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": content_type}
    )

    assert response.status_code == 200
    assert response.json()["text"] == "你好"


def test_invalid_utf8_urlencoded_value_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"text=%FF%FE",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "text must be UTF-8 text", "code": "invalid_field"}


def test_invalid_utf8_multipart_value_gets_400(tmp_path: Path) -> None:
    body, content_type = _multipart_body({"text": b"\xff\xfe"})

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": content_type}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "text must be UTF-8 text", "code": "invalid_field"}


def test_multipart_declaring_a_non_utf8_charset_gets_400(tmp_path: Path) -> None:
    boundary = "xxxxBOUNDARYxxxx"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"\r\n\r\n'
        "hi"
        f"\r\n--{boundary}--\r\n"
    ).encode("ascii")
    content_type = f"multipart/form-data; boundary={boundary}; charset=iso-8859-1"

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": content_type}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "text must be UTF-8 text", "code": "invalid_field"}


# finding #5/#6: the form and its UploadFile are closed after reading, and ref_audio is
# read bounded to MAX_AUDIO_BYTES + 1.


def test_ref_audio_read_is_bounded_to_max_audio_bytes_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_fields, "MAX_AUDIO_BYTES", 10)

    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "ref_text": "hello"},
        files={"ref_audio": ("ref.wav", b"x" * 100, "audio/wav")},
    )

    assert response.status_code == 200
    audio_bytes = response.json()["reference"]["audio_bytes"]
    assert len(audio_bytes) == 11  # MAX_AUDIO_BYTES (patched to 10) + 1


# finding #7: an unsupported content type with a body is rejected; query-only still works.


def test_unsupported_content_type_with_a_body_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", content=b'{"text": "hi"}', headers={"content-type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "unsupported content type",
        "code": "invalid_field",
    }


def test_query_only_request_with_no_body_still_works(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", params={"text": "hi"})

    assert response.status_code == 200
    assert response.json()["text"] == "hi"


# finding #9: _optional_number's TypeVar keeps int/float sampling fields distinct -- a
# type-checker concern, verified here only by the existing zero-means-default tests still
# passing for both an int field (top_k) and a float field (temperature).


# finding #10: form values take precedence over query values (no duplicate check yet).


def test_form_value_takes_precedence_over_query_value(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "seed": "111"}, params={"seed": "222"}
    )

    assert response.status_code == 200
    assert response.json()["seed"] == 111
