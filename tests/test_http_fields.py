"""`http_fields` tests (tasks.md T037): the happy path only.

Table-driven tests for the malformed corpus (bad numbers, ranges, duplicates, control
characters) arrive with T044/T048; this file only checks that valid input parses to the
defaults and values the contract promises, and that the handful of rules T037 already
owns (BC-02 empty-means-absent, BC-09 blank instruction, BC-10 required text, FR-006's
`0` -> `None`) hold, plus the review 1 (`# --- review 1 findings`) and review 2
(`# --- review 2 findings`) findings below.

A tiny FastAPI app drives `read_fields` through a real `Request`, exercising both
`multipart/form-data` (with a real file part for `ref_audio`) and
`application/x-www-form-urlencoded` bodies, plus the query string -- `TestClient` builds
the real ASGI request `read_fields` has to parse, rather than constructing a `Fields` by
hand and skipping that parsing entirely.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.datastructures import FormData

from breeze_infer import http_fields
from breeze_infer.errors import install_error_handlers
from breeze_infer.http_fields import (
    DEFAULT_CFG_SCALE,
    DEFAULT_INSTRUCTION,
    DEFAULT_SEED,
    FORM_MAX_FIELDS,
    FORM_MAX_PART_SIZE,
    InlineRef,
    NoReference,
    ReferenceSpec,
    SpeechRequest,
    VoiceRef,
    parse_speech,
    read_fields,
)
from breeze_infer.limits import (
    MAX_INSTRUCTION_CHARS,
    MAX_NEW_TOKENS_CEILING,
    MAX_REF_TEXT_CHARS,
    MAX_TEXT_CHARS,
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
    # review 2 finding #4: doubled from the exact 12-bytes-per-4-byte-codepoint figure, so
    # the field-level length check (T048's text_too_long) is always what actually rejects
    # an over-length `text`, not this module's own generic parser-error message landing on
    # the boundary first.
    assert FORM_MAX_PART_SIZE == 2 * MAX_TEXT_CHARS * 12


def test_a_full_length_four_byte_char_text_field_fits_urlencoded(tmp_path: Path) -> None:
    # "🎉" is a 4-byte UTF-8 code point; percent-encoded it's exactly 12 ASCII bytes
    # ("%XX" x 4), so MAX_TEXT_CHARS of them is exactly at half of FORM_MAX_PART_SIZE --
    # comfortably under it, with headroom to spare (review 2 finding #4).
    text = "\U0001f389" * MAX_TEXT_CHARS
    encoded_value = quote(text, safe="")
    assert len(encoded_value) < FORM_MAX_PART_SIZE
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
        "error": (
            "content type must be multipart/form-data or application/x-www-form-urlencoded"
        ),
        "code": "invalid_field",
    }


def test_query_only_request_with_no_body_still_works(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", params={"text": "hi"})

    assert response.status_code == 200
    assert response.json()["text"] == "hi"


# review 1 finding #9: _optional_number's TypeVar keeps int/float sampling fields distinct
# -- a type-checker concern, verified here only by the existing zero-means-default tests
# still passing for both an int field (top_k) and a float field (temperature).


# review 1 finding #10 (form precedence over query, with no duplicate check) is superseded
# by T048/BC-08: the same field in both the form and the query is now `400 duplicate_field`
# -- see test_bc_08_duplicate_field_in_body_query_or_both_gets_400 below.


# --- review 2 findings ----------------------------------------------------------------


def _multipart_field_part(name: str, value: str) -> bytes:
    return (
        f'--xxxxBOUNDARYxxxx\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("ascii")


def _multipart_close() -> bytes:
    return b"--xxxxBOUNDARYxxxx--\r\n"


_MULTIPART_CONTENT_TYPE = "multipart/form-data; boundary=xxxxBOUNDARYxxxx"


# findings #1+2+9: the urlencoded parser is rebuilt on parse_qsl over raw bytes, decoding
# strictly by hand (parse_qsl's own encoding/errors parameters are silently ignored for a
# bytes input), and capping the field count before doing any per-field work.


def test_raw_unescaped_utf8_urlencoded_value_is_decoded(tmp_path: Path) -> None:
    """"你好" sent as raw UTF-8 bytes, not percent-escaped."""
    response = _client(tmp_path).post(
        "/speech",
        content="text=你好".encode(),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "你好"


def test_raw_unescaped_invalid_utf8_urlencoded_value_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"text=\xff\xfe",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "text must be UTF-8 text", "code": "invalid_field"}


def test_urlencoded_field_count_over_the_cap_gets_400(tmp_path: Path) -> None:
    extra = "&".join(f"f{i}=v" for i in range(FORM_MAX_FIELDS + 1))
    body = f"text=hi&{extra}".encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


def test_a_huge_body_of_many_pairs_fails_fast_on_the_field_cap(tmp_path: Path) -> None:
    """A ~26 MiB urlencoded body of millions of tiny fields must be rejected by parse_qsl's
    max_num_fields guard -- one pass over the bytes -- not by actually splitting and
    decoding each field (which would take far longer for this many)."""
    pair = b"a=1&"
    body = pair * (27_000_000 // len(pair))  # ~26 MiB, millions of pairs

    start = time.monotonic()
    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    elapsed = time.monotonic() - start

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"
    assert elapsed < 5.0, f"took {elapsed:.2f}s -- did not fail fast on the field cap"


# finding #3: a text-then-file ref_audio is rejected regardless of order.


def test_ref_audio_text_then_file_gets_400(tmp_path: Path) -> None:
    """`form.get("ref_audio")` alone only sees the *last* same-named entry -- a valid file,
    here -- which would silently hide an earlier, invalid text `ref_audio` sent first."""
    body = (
        _multipart_field_part("ref_audio", "not a file")
        + (
            "--xxxxBOUNDARYxxxx\r\n"
            'Content-Disposition: form-data; name="ref_audio"; filename="ref.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            "later, valid-looking bytes\r\n"
        ).encode("ascii")
        + _multipart_close()
    )

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": _MULTIPART_CONTENT_TYPE}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


# finding #5: the query string is parsed by the same strict routine as the body.


def test_invalid_utf8_query_string_value_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech?text=%FF%FE")

    assert response.status_code == 400
    assert response.json() == {"error": "text must be UTF-8 text", "code": "invalid_field"}


# finding #6: accept utf-8/utf8 case-insensitively; reject any other declared charset.


@pytest.mark.parametrize("charset", ["utf-8", "UTF-8", "utf8", "UTF8"])
def test_accepted_charset_spellings_for_urlencoded(tmp_path: Path, charset: str) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"text=hi",
        headers={"content-type": f"application/x-www-form-urlencoded; charset={charset}"},
    )

    assert response.status_code == 200


def test_urlencoded_declaring_a_non_utf8_charset_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"text=hi",
        headers={"content-type": "application/x-www-form-urlencoded; charset=iso-8859-1"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "request body must be UTF-8 text",
        "code": "invalid_field",
    }


@pytest.mark.parametrize("charset", ["utf-8", "UTF8"])
def test_accepted_charset_spellings_for_multipart(tmp_path: Path, charset: str) -> None:
    body = _multipart_field_part("text", "hi") + _multipart_close()
    content_type = f"multipart/form-data; boundary=xxxxBOUNDARYxxxx; charset={charset}"

    response = _client(tmp_path).post(
        "/speech", content=body, headers={"content-type": content_type}
    )

    assert response.status_code == 200


# finding #7: a client-controlled field name is truncated before it's echoed into an error.


def test_long_field_name_is_truncated_in_the_error_message(tmp_path: Path) -> None:
    long_name = "x" * 100
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi"},
        files={long_name: ("f.txt", b"not text", "text/plain")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_field"
    assert body["error"] == "x" * 64 + "… must be a text field"


# finding #8: an unsupported content type only matters when the body is actually non-empty.


def test_unsupported_content_type_with_empty_body_still_works(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech?text=hi", headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    assert response.json()["text"] == "hi"


# finding #10: the multipart form (and its UploadFile) is actually closed.


def test_multipart_form_is_closed_after_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed_calls: list[FormData] = []
    original_close = FormData.close

    async def spy_close(self: FormData) -> None:
        closed_calls.append(self)
        await original_close(self)

    monkeypatch.setattr(FormData, "close", spy_close)

    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "ref_text": "hello"},
        files={"ref_audio": ("ref.wav", b"audio bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert len(closed_calls) == 1


# --- T044: the malformed corpus (tasks.md Phase 5, US2) --------------------------------
#
# T037's tests above only exercise the happy path; these are table-driven against the C++
# server behavior each one replaces (contracts/http-api.md's field table, number grammar,
# reference rules and error table; data-model.md's SpeechRequest ranges; spec.md's
# FR-004-FR-011 and BC-01 through BC-46). Written before T048's implementation, so they are
# expected to fail (mostly by getting 200 or the wrong error) until T048 lands.


# BC-01: an unparseable number gets 400 invalid_field naming the field, never silently 0.
# The grammar is ASCII-only -- Python's bare `\d` also matches non-ASCII decimal digits
# (e.g. U+0661 ARABIC-INDIC DIGIT ONE), which the contract's grammar must reject the same
# as any other non-numeral text.


@pytest.mark.parametrize(
    ("field", "raw", "expected_message"),
    [
        ("cfg_scale", "banana", "cfg_scale must be a number"),
        ("cfg_scale", "1e", "cfg_scale must be a number"),
        ("cfg_scale", "0x10", "cfg_scale must be a number"),
        ("cfg_scale", "inf", "cfg_scale must be a number"),
        ("cfg_scale", "nan", "cfg_scale must be a number"),
        ("seed", "1.5", "seed must be an integer"),  # a decimal for an integer field
        ("seed", "١٢", "seed must be an integer"),  # Arabic-Indic digits, not ASCII
        ("cfg_scale", " 12", "cfg_scale must be a number"),  # leading whitespace
        ("cfg_scale", "12 ", "cfg_scale must be a number"),  # trailing whitespace
        ("seed", " 12", "seed must be an integer"),
        ("seed", "12 ", "seed must be an integer"),
    ],
)
def test_bc_01_unparseable_numbers_get_400(
    tmp_path: Path, field: str, raw: str, expected_message: str
) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: raw})

    assert response.status_code == 400
    assert response.json() == {"error": expected_message, "code": "invalid_field"}


# BC-02: an empty value means the field is absent, for every field, not just text/instruction.


def test_bc_02_empty_value_is_absent(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        data={
            "text": "hi",
            "instruction": "",
            "cfg_scale": "",
            "seed": "",
            "temperature": "",
            "top_k": "",
            "top_p": "",
            "repetition_penalty": "",
            "max_new_tokens": "",
            "split_chars": "",
            "voice_id": "",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["instruction"] == DEFAULT_INSTRUCTION
    assert body["cfg_scale"] == DEFAULT_CFG_SCALE
    assert body["seed"] == DEFAULT_SEED
    assert body["temperature"] is None
    assert body["top_k"] is None
    assert body["top_p"] is None
    assert body["repetition_penalty"] is None
    assert body["max_new_tokens"] is None
    assert body["split_chars"] == 600
    assert body["reference"] == {"kind": "none"}


# BC-03: negative, NaN or out-of-range sampling values get 400, never silently clamped,
# defaulted or passed through. `0` is the one value that still means "use the model
# default" (test_zero_means_default_for_sampling_fields above).


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("temperature", "-1"),
        ("temperature", "nan"),
        ("top_p", "1.5"),
        ("top_p", "-0.5"),
        ("cfg_scale", "101"),
        ("cfg_scale", "-1"),
        ("cfg_scale", "nan"),
        ("repetition_penalty", "0.00005"),  # below 1e-4
        ("repetition_penalty", "11"),
        ("top_k", "-5"),
        ("seed", "-1"),
    ],
)
def test_bc_03_out_of_range_sampling_values_get_400(
    tmp_path: Path, field: str, raw: str
) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: raw})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# BC-04: max_new_tokens has a server ceiling (MAX_NEW_TOKENS_CEILING, 1,500), unlike the
# C++ server which let it grow unbounded.


@pytest.mark.parametrize("raw", [str(MAX_NEW_TOKENS_CEILING + 1), "999999"])
def test_bc_04_max_new_tokens_over_ceiling_gets_400(tmp_path: Path, raw: str) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "max_new_tokens": raw}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# BC-05: text and instruction are bounded (the C++ server let them grow unbounded); ref_text
# shares instruction's limit and its own invalid_field wording, not text_too_long.


def test_bc_05_text_or_instruction_too_long_gets_400(tmp_path: Path) -> None:
    text_response = _client(tmp_path).post(
        "/speech", data={"text": "a" * (MAX_TEXT_CHARS + 1)}
    )
    assert text_response.status_code == 400
    assert text_response.json() == {"error": "text is too long", "code": "text_too_long"}

    instruction_response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "instruction": "a" * (MAX_INSTRUCTION_CHARS + 1)},
    )
    assert instruction_response.status_code == 400
    assert instruction_response.json() == {
        "error": "instruction must be at most 2000 characters",
        "code": "invalid_field",
    }

    ref_text_response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "ref_text": "a" * (MAX_REF_TEXT_CHARS + 1)}
    )
    assert ref_text_response.status_code == 400
    assert ref_text_response.json() == {
        "error": "ref_text must be at most 2000 characters",
        "code": "invalid_field",
    }


# BC-08: a field repeated within the form, within the query, or split across both, is
# always 400 duplicate_field -- the C++ server resolved this silently (and this module's
# own T037 precedence, review 1 finding #10, resolved it silently too).


def test_bc_08_duplicate_field_in_body_query_or_both_gets_400(tmp_path: Path) -> None:
    # httpx's `data=` only form-urlencodes a Mapping -- a list of tuples (which would
    # otherwise be the natural way to send the same key twice) is instead treated as raw
    # streamed `content`, so the duplicate-within-the-form body is built by hand here, the
    # same way the rest of this file sends bodies httpx's own encoder can't produce.
    within_form = _client(tmp_path).post(
        "/speech",
        content=b"text=hi&seed=1&seed=2",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert within_form.status_code == 400
    assert within_form.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }

    within_query = _client(tmp_path).post(
        "/speech", data={"text": "hi"}, params=[("seed", "1"), ("seed", "2")]
    )
    assert within_query.status_code == 400
    assert within_query.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }

    across_both = _client(tmp_path).post(
        "/speech", data={"text": "hi", "seed": "1"}, params={"seed": "2"}
    )
    assert across_both.status_code == 400
    assert across_both.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }


# BC-09: a blank (or absent) instruction uses the default -- the C++ server used the
# literal empty string.


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_bc_09_blank_instruction_uses_default(tmp_path: Path, blank: str) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "instruction": blank}
    )

    assert response.status_code == 200
    assert response.json()["instruction"] == DEFAULT_INSTRUCTION


# BC-10: whitespace-only text is rejected -- the C++ server accepted it.


@pytest.mark.parametrize("blank", ["   ", "\t\t", "\n\n", " \t\n "])
def test_bc_10_whitespace_text_gets_400(tmp_path: Path, blank: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": blank})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


# BC-46: control characters other than TAB, CR and LF are rejected in text, instruction
# and ref_text -- the C++ server accepted them (and treated NUL as sentence-closing
# punctuation).

_DISALLOWED_CONTROL_CHARS = ["\x00", "\x07", "\x1b", "\x7f", "\x85"]
_ALLOWED_WHITESPACE_CONTROL_CHARS = ["\t", "\r", "\n"]


@pytest.mark.parametrize("bad", _DISALLOWED_CONTROL_CHARS)
def test_bc_46_control_characters_get_400(tmp_path: Path, bad: str) -> None:
    text_response = _client(tmp_path).post(
        "/speech", data={"text": f"hello{bad}there"}
    )
    assert text_response.status_code == 400
    assert text_response.json() == {
        "error": "text must be free of control characters",
        "code": "invalid_field",
    }

    instruction_response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "instruction": f"speak{bad}slowly"}
    )
    assert instruction_response.status_code == 400
    assert instruction_response.json() == {
        "error": "instruction must be free of control characters",
        "code": "invalid_field",
    }

    ref_text_response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "voice_id": "alice", "ref_text": f"hello{bad}world"},
    )
    assert ref_text_response.status_code == 400
    assert ref_text_response.json() == {
        "error": "ref_text must be free of control characters",
        "code": "invalid_field",
    }


@pytest.mark.parametrize("good", _ALLOWED_WHITESPACE_CONTROL_CHARS)
def test_bc_46_tab_cr_lf_are_allowed_in_text(tmp_path: Path, good: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": f"hello{good}there"})

    assert response.status_code == 200
    assert response.json()["text"] == f"hello{good}there"


# seed range: 0-4294967295 (data-model.md SpeechRequest).


@pytest.mark.parametrize("raw", ["-1", "4294967296", "99999999999"])
def test_seed_out_of_range_gets_400(tmp_path: Path, raw: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "seed": raw})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


@pytest.mark.parametrize("raw", ["0", "4294967295"])
def test_seed_boundary_values_are_accepted(tmp_path: Path, raw: str) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "seed": raw})

    assert response.status_code == 200
    assert response.json()["seed"] == int(raw)


# split_chars range: 0-10,000 (data-model.md SpeechRequest).


@pytest.mark.parametrize("raw", ["-1", "10001"])
def test_split_chars_out_of_range_gets_400(tmp_path: Path, raw: str) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "split_chars": raw}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


@pytest.mark.parametrize("raw", ["0", "10000"])
def test_split_chars_boundary_values_are_accepted(tmp_path: Path, raw: str) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "split_chars": raw}
    )

    assert response.status_code == 200
    assert response.json()["split_chars"] == int(raw)


# Review finding: a nonzero literal that underflows to 0.0 in float parsing (e.g.
# `1e-400`) must still be `400 invalid_field` -- "0 means default" is decided from the
# literal text (every digit is `0`), not from the parsed float, since Python's `float()`
# silently underflows a tiny-enough nonzero literal to exactly `0.0`.


@pytest.mark.parametrize("field", ["repetition_penalty", "temperature", "top_p"])
def test_underflowing_nonzero_literal_is_400_not_default(
    tmp_path: Path, field: str
) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: "1e-400"})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# --- more malformed-corpus coverage (tasks.md T044/T048, Phase 5 checkpoint follow-ups) --


def test_bc_01_integer_longer_than_20_digits_gets_400(tmp_path: Path) -> None:
    """BC-01: an integer literal too long to ever be in range (over 20 digits) is rejected
    with `400` naming the field's own range, not left to reach `int()` -- which, past
    Python's 4,300-digit string-conversion ceiling, raises unhandled and would otherwise
    surface as a `500`. The C++ server parsed a number with `atoi`/`strtod`-family
    functions, which don't raise for an over-long digit string at all (they just saturate
    or give undefined results); this server must never crash on one instead.
    """
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "seed": "9" * 4400}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "seed must be an integer between 0 and 4294967295",
        "code": "invalid_field",
    }


@pytest.mark.parametrize("bad", ["\x1c", "\x1d", "\x1e", "\x1f", "\x85"])
def test_bc_46_control_character_that_python_calls_whitespace_is_rejected(
    tmp_path: Path, bad: str
) -> None:
    """BC-46: `\\x1c`-`\\x1f` and `\\x85` (NEL) are control characters this contract
    rejects, even though Python's own `str.strip()`/`str.isspace()` treat them as
    whitespace -- a `text` or `instruction` consisting of *only* one of these must not be
    mistaken for a blank field (`text_required`, or `instruction`'s default) instead of the
    control-character violation it actually is. The C++ server accepted every control
    character in these fields (and treated NUL as sentence-closing punctuation).
    """
    text_response = _client(tmp_path).post("/speech", data={"text": bad})
    assert text_response.status_code == 400
    assert text_response.json() == {
        "error": "text must be free of control characters",
        "code": "invalid_field",
    }

    instruction_response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "instruction": bad}
    )
    assert instruction_response.status_code == 400
    assert instruction_response.json() == {
        "error": "instruction must be free of control characters",
        "code": "invalid_field",
    }


def test_bc_08_duplicate_ref_audio_file_parts_get_400(tmp_path: Path) -> None:
    """BC-08: two `ref_audio` file parts under the same name -- both otherwise valid -- is
    `400 duplicate_field`, not the last one silently winning. The C++ server (and a naive
    `form.get`) would just use whichever file part it read last.
    """
    boundary = "xxxxBOUNDARYxxxx"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"\r\n\r\n'
        "hi\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="ref_audio"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "first file bytes\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="ref_audio"; filename="b.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "second file bytes\r\n"
        f"--{boundary}--\r\n"
    ).encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio was given more than once",
        "code": "duplicate_field",
    }


def test_bc_08_duplicate_detection_runs_before_other_field_errors(
    tmp_path: Path,
) -> None:
    """BC-08: duplicate-field detection is one upfront pass over every key present, run
    before any single field's own syntax is checked -- a too-long `text` (which would
    otherwise be `400 text_too_long`) must not shadow an unrelated duplicated `seed`. The
    C++ server resolved a duplicate field silently, using whichever value it parsed last,
    and never checked one field's validity before another's.
    """
    body = ("text=" + "a" * (MAX_TEXT_CHARS + 1) + "&seed=1&seed=2").encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }


def test_bc_08_duplicate_unknown_field_gets_400(tmp_path: Path) -> None:
    """BC-08: "a field present more than once, anywhere, gets 400" applies to every key on
    the wire, not just the ones `parse_speech` reads -- `foo` isn't a field this contract
    defines at all, but repeating it is still rejected. The C++ server ignored unknown
    fields entirely, duplicated or not.
    """
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi"}, params=[("foo", "1"), ("foo", "2")]
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "foo was given more than once",
        "code": "duplicate_field",
    }


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("temperature", "10.0000000000000001"),
        ("top_p", "1.00000000000000001"),
        ("repetition_penalty", "0.00009999999999999999999"),
    ],
)
def test_bc_03_decimal_range_bound_is_exact(
    tmp_path: Path, field: str, raw: str
) -> None:
    """BC-03: a decimal literal just past a range boundary is rejected even when `float`'s
    limited precision would otherwise round it into range (`float("10.0000000000000001")`
    is exactly `10.0`) -- the bound comparison is exact, against the literal digits, not the
    parsed float. The C++ server didn't range-check these fields at all.
    """
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: raw})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# voice_id is validated here as a lookup key -- either a saved voice's name
# ([A-Za-z0-9_-]{1,64}, contracts/http-api.md POST /v1/voices, which can never itself start
# with v_ per BC-26) or an unnamed voice's v_ + 16 lowercase hex id. Whether the id actually
# names a registered voice is Phase 7's concern (the stub 404 lookup); these only check its
# shape, which the C++ server never validated at all (any string reached its lookup as-is).


def test_voice_id_with_invalid_characters_gets_400(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "not a valid name!"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "voice_id must be a voice name or v_ id",
        "code": "invalid_field",
    }


def test_voice_id_saved_name_is_accepted(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "alice-1_2"}
    )

    assert response.status_code == 200
    assert response.json()["reference"]["voice_id"] == "alice-1_2"


def test_voice_id_v_id_is_accepted(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "v_" + "a1b2c3d4e5f60789"}
    )

    assert response.status_code == 200
    assert response.json()["reference"]["voice_id"] == "v_a1b2c3d4e5f60789"


def test_voice_id_v_prefixed_but_not_valid_hex_gets_400(tmp_path: Path) -> None:
    """A `v_`-prefixed string that isn't 16 lowercase hex characters can't be a real
    unnamed-voice id, and structurally can't be a saved name either (BC-26: a saved name
    can never start with `v_`) -- so it's rejected, even though its characters alone would
    otherwise fit the general name pattern."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "v_not-a-real-id"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "voice_id must be a voice name or v_ id",
        "code": "invalid_field",
    }


def test_blank_ref_text_counts_as_absent_with_voice_id(tmp_path: Path) -> None:
    """A whitespace-only `ref_text` counts as absent (BC-02's general rule, extended to
    "blank" the same way `instruction`'s own default check works), decided only after the
    control-character check -- so a `ref_text` that's genuinely a control character is
    still rejected, never silently treated as blank. Paired with `voice_id`, a blank
    `ref_text` means no override of the voice's stored transcript.
    """
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "alice", "ref_text": "   "}
    )

    assert response.status_code == 200
    assert response.json()["reference"] == {
        "kind": "voice",
        "voice_id": "alice",
        "ref_text_override": None,
    }


def test_blank_ref_text_counts_as_absent_needing_ref_text_required_with_ref_audio(
    tmp_path: Path,
) -> None:
    """A whitespace-only `ref_text` counts as absent -- paired with `ref_audio`, that means
    `ref_text` is missing, so this is `400 ref_text_required`, the same as omitting it
    outright."""
    response = _client(tmp_path).post(
        "/speech",
        data={"text": "hi", "ref_text": "   "},
        files={"ref_audio": ("ref.wav", b"anything", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_text is required with ref_audio",
        "code": "ref_text_required",
    }
