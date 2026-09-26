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

import itertools
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


# T048 post-final review findings #4/#7: `ref_audio` in the query string is knowable
# without ever touching the body, so it's rejected before the body's content type is even
# sniffed -- not shadowed by a body-shaped problem (a bad content type, a rejected charset)
# that was only ever going to be a distraction from the query's own, already-certain error.


def test_ref_audio_in_query_wins_over_an_unsupported_content_type(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"irrelevant body",
        headers={"content-type": "text/plain"},
        params={"ref_audio": "not a file"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


def test_ref_audio_in_query_wins_over_a_rejected_body_charset(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech",
        content=b"text=hi",
        headers={"content-type": "application/x-www-form-urlencoded; charset=iso-8859-1"},
        params={"ref_audio": "not a file"},
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
    # U+20000, a CJK Extension B ideograph, is a 4-byte UTF-8 code point; percent-encoded
    # it's exactly 12 ASCII bytes ("%XX" x 4), so MAX_TEXT_CHARS of them is exactly at half
    # of FORM_MAX_PART_SIZE -- comfortably under it, with headroom to spare (review 2
    # finding #4). A CJK ideograph, not an emoji, specifically because it's a letter
    # (Unicode category Lo): text_split.py's speakable rule (mirrored in this module) drops
    # text with no letter or digit at all, which an emoji-only string would trip on.
    text = "\U00020000" * MAX_TEXT_CHARS
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
    max_num_fields guard -- one pass over the bytes -- not by a real per-field parse (which
    would take far longer for this many). T048 post-final review finding #3:
    `_urlencoded_pairs_sync`'s fallback for that guard now also checks for a duplicate name,
    but only via a single cheap split (`_names_from_raw_urlencoded`), no percent-decoding or
    per-field UTF-8 validation -- so this still has to fail fast, and every one of these
    millions of pairs is named `a`, a genuine duplicate, so the answer is now the more useful
    `duplicate_field`, not the generic error."""
    pair = b"a=1&"
    body = pair * (27_000_000 // len(pair))  # ~26 MiB, millions of pairs

    start = time.monotonic()
    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"took {elapsed:.2f}s -- did not fail fast on the field cap"
    assert response.status_code == 400
    assert response.json() == {
        "error": "a was given more than once",
        "code": "duplicate_field",
    }


def test_a_huge_body_of_distinct_names_fails_fast_on_the_field_cap(tmp_path: Path) -> None:
    """T048 post-final review finding #2 (HIGH, DoS): an earlier `_names_from_raw_urlencoded`
    split the *whole* body on `&` and ran `Counter` over every piece -- ~1.25s and ~1.1GB RSS
    for a 26 MiB body of two-byte names, with the GIL held throughout. The test above
    (`test_a_huge_body_of_many_pairs_fails_fast_on_the_field_cap`) only passed because its
    single-letter names are small-string singletons CPython already caches, hiding that cost
    -- this one uses distinct multi-byte names, which aren't. The fallback now bounds its own
    split to `FORM_MAX_FIELDS + 1` pieces -- exactly enough to see whatever caused
    `parse_qsl`'s own guard to trip, and no more -- so this must still fail fast regardless of
    how large or distinct the rest of the body is."""
    distinct_names = [
        "".join(letters)
        for letters in itertools.islice(
            itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2), 40
        )
    ]
    head = "&".join(f"{name}=1" for name in distinct_names).encode("ascii")
    filler = b"zz9=1&"  # three bytes, distinct from every two-letter name above
    body = head + b"&" + filler * (27_000_000 // len(filler))  # ~26 MiB total

    start = time.monotonic()
    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"took {elapsed:.2f}s -- did not fail fast on the field cap"
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


# finding #3: a text-then-file ref_audio is rejected regardless of order.


def test_ref_audio_text_then_file_gets_400(tmp_path: Path) -> None:
    """`form.get("ref_audio")` alone only sees the *last* same-named entry -- a valid file,
    here -- which would silently hide an earlier, invalid text `ref_audio` sent first.

    Two parts named `ref_audio`, regardless of their types, is also two occurrences of the
    same key -- BC-08's duplicate-field pass runs before any per-field check, including the
    file-vs-text one that would otherwise explain *which* `ref_audio` was invalid, so this
    is `400 duplicate_field`, not `invalid_field`."""
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
        "error": "ref_audio was given more than once",
        "code": "duplicate_field",
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
        "error": "instruction must be at most 2,000 characters",
        "code": "invalid_field",
    }

    ref_text_response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "ref_text": "a" * (MAX_REF_TEXT_CHARS + 1)}
    )
    assert ref_text_response.status_code == 400
    assert ref_text_response.json() == {
        "error": "ref_text must be at most 2,000 characters",
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


# --- final-review follow-ups (Phase 5 checkpoint) ---------------------------------------


def test_bc_01_decimal_with_huge_exponent_gets_400(tmp_path: Path) -> None:
    """BC-01: a decimal literal with an enormous exponent is rejected with 400, not left to
    crash. DECISION (final review): the grammar doesn't cap the exponent's digit count --
    the contract's own grammar allows any exponent, and this shouldn't be stricter than the
    contract -- so a literal like this reaches `_check_decimal_range`, whose own
    `decimal.Decimal(literal)` call is guarded by `try`/`except InvalidOperation`
    (`Decimal`'s default context bounds an exponent to roughly +/-999999 and would otherwise
    raise, unhandled, for one outside that). The C++ server's strtod-family parsing has no
    such limit and would just saturate to infinity instead of raising.
    """
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "cfg_scale": "1e1000000000000000000"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


def test_decimal_zero_with_huge_exponent_still_means_the_default(tmp_path: Path) -> None:
    """`_is_zero_literal` decides the "0 means default" sentinel from the literal's digits
    before `_check_decimal_range` -- and therefore `decimal.Decimal` -- ever sees it: a zero
    mantissa with an exponent too large for `Decimal`'s own context (e.g. `0e99999`, though
    this one alone fits) never has to construct a `Decimal` at all to know it's zero."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "temperature": "0e99999"}
    )

    assert response.status_code == 200
    assert response.json()["temperature"] is None


def test_decimal_boundary_value_with_a_padded_exponent_is_accepted(tmp_path: Path) -> None:
    """`temperature=1e00001` is exactly `10.0` (temperature's inclusive upper bound) --
    written with a padded, multi-digit exponent, to prove the grammar really does accept any
    exponent width now, not just up to some digit count."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "temperature": "1e00001"}
    )

    assert response.status_code == 200
    assert response.json()["temperature"] == 10.0


# T048 post-final review finding #1: cfg_scale has no `_is_zero_literal` pre-check the way
# the optional sampling fields do (0 is an ordinary in-range value for it, not a "use the
# default" sentinel) -- so, unlike `test_decimal_zero_with_huge_exponent_still_means_the_
# default` above (which never reaches `Decimal` at all), these literals actually reach
# `_check_decimal_range`'s `except InvalidOperation` branch and must still resolve to the
# value C++'s strtod would give.


@pytest.mark.parametrize(
    "literal", ["0e1000000000000000000", "0e-999999999999999999999"],
    ids=["huge-positive-exponent", "huge-negative-exponent"],
)
def test_cfg_scale_unrepresentable_zero_is_accepted(tmp_path: Path, literal: str) -> None:
    """A zero mantissa with an exponent `Decimal` can't represent is still exactly zero --
    in range for `cfg_scale` (`[0, 100]`) -- regardless of which direction the unrepresentable
    exponent points."""
    response = _client(tmp_path).post("/speech", data={"text": "hi", "cfg_scale": literal})

    assert response.status_code == 200
    assert response.json()["cfg_scale"] == 0.0


def test_cfg_scale_tiny_value_with_unrepresentable_exponent_is_accepted(
    tmp_path: Path,
) -> None:
    """A *nonzero* mantissa with a huge negative exponent -- `Decimal` can't represent the
    literal, but `float(literal)` still underflows to exactly `0.0`, same as C++'s strtod --
    is in range for `cfg_scale` (0 inclusive), unlike the representable-underflow case
    (`test_underflowing_nonzero_literal_is_400_not_default`), which is a different field's
    range rule (temperature's `(0, 10]` excludes 0), not a different code path."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "cfg_scale": "1e-999999999999999999999"}
    )

    assert response.status_code == 200
    assert response.json()["cfg_scale"] == 0.0


def test_cfg_scale_negative_tiny_value_with_unrepresentable_exponent_gets_400(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #1: the fallback used to lose the literal's sign --
    `float("-1e-999999999999999999999")` underflows to exactly `-0.0`, and
    `Decimal(-0.0) >= Decimal("0")` is true (decimal treats `-0` and `0` as equal for
    comparison), so a genuinely negative `cfg_scale` this tiny was wrongly accepted as
    in-range, with the stored value `-0.0` -- while the representable `cfg_scale=-1e-400`
    correctly got 400. The fallback now rebuilds a `Decimal` straight from the literal's own
    sign and digits (only the exponent is re-anchored to something `Decimal` accepts), so
    this is out of range the same way `-1e-400` already is."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "cfg_scale": "-1e-999999999999999999999"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


def test_cfg_scale_huge_positive_exponent_with_nonzero_mantissa_gets_400(
    tmp_path: Path,
) -> None:
    """A nonzero mantissa with a huge *positive* exponent overflows towards infinity, never
    in range for any of this contract's (finite) upper bounds -- the same literal
    `test_bc_01_decimal_with_huge_exponent_gets_400` already covers, named here to sit next
    to the rest of this InvalidOperation-reaching group."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "cfg_scale": "1e1000000000000000000"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


@pytest.mark.parametrize(
    "literal", ["1e-999999999999999999999", "1e1000000000000000000"],
    ids=["huge-negative-exponent", "huge-positive-exponent"],
)
def test_temperature_unrepresentable_exponent_gets_400(tmp_path: Path, literal: str) -> None:
    """The same `except InvalidOperation` fallback, exercised for an *optional* field:
    unlike `test_decimal_zero_with_huge_exponent_still_means_the_default` (a zero mantissa,
    caught by `_is_zero_literal` before `Decimal` is ever involved), a nonzero mantissa still
    reaches `_check_decimal_range` and actually raises `InvalidOperation` -- and, for
    temperature (`(0, 10]`, 0 excluded), both directions land outside the range: the tiny
    literal underflows to `0.0`, which fails the low-exclusive bound the same way
    `1e-400` already does; the huge literal overflows to infinity, which fails the high
    bound."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "temperature": literal}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


@pytest.mark.parametrize(
    ("field", "literal"),
    [
        ("cfg_scale", "1e" + "9" * 5000),
        ("temperature", "1e-" + "9" * 5000),
    ],
    ids=["cfg_scale-huge-positive", "temperature-huge-negative"],
)
def test_decimal_exponent_over_4300_digits_doesnt_crash(
    tmp_path: Path, field: str, literal: str
) -> None:
    """T048 post-final review finding #1 (HIGH, a 500 regression): a 5,000-digit exponent is
    well past `sys.get_int_max_str_digits()` (4,300 by default) -- `Decimal(literal)` raises
    `InvalidOperation` for it same as any other unrepresentable exponent, but the fallback
    used to call bare `int(exponent_digits)` on the *full* exponent string to re-anchor it,
    which itself raises `ValueError`, unhandled, a `500`. This must still be the ordinary 400
    an out-of-range value gets."""
    response = _client(tmp_path).post("/speech", data={"text": "hi", field: literal})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


def test_decimal_exponent_of_5000_leading_zero_digits_then_one_is_accepted(
    tmp_path: Path,
) -> None:
    """The digit-count cap above must count *significant* exponent digits, not raw ones on
    the wire -- an exponent padded with thousands of leading zeros in front of a single `1`
    is exactly `1e1`, not an unrepresentable magnitude, and must resolve to that value, not
    be clamped as if it were huge."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "cfg_scale": "1e" + "0" * 4998 + "1"}
    )

    assert response.status_code == 200
    assert response.json()["cfg_scale"] == 10.0


def test_bc_01_leading_zeros_dont_count_against_the_digit_cap(tmp_path: Path) -> None:
    """The integer literal digit cap (test_bc_01_integer_longer_than_20_digits_gets_400)
    counts significant digits, not raw digits on the wire -- a value padded with leading
    zeros must be judged by its actual magnitude, not its length."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "seed": "0" * 25 + "1"}
    )

    assert response.status_code == 200
    assert response.json()["seed"] == 1


def test_all_zero_integer_literal_still_means_the_default_however_padded(
    tmp_path: Path,
) -> None:
    """An all-zero integer literal is exactly 0 -- FR-006's "0 means the model default" --
    no matter how many leading zeros pad it, not rejected as too long."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "top_k": "0" * 25}
    )

    assert response.status_code == 200
    assert response.json()["top_k"] is None


def test_top_k_range_message_matches_the_contract_exactly(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "top_k": "-1"})

    assert response.status_code == 400
    assert response.json() == {
        "error": "top_k must be 0, or an integer between 1 and 10,000",
        "code": "invalid_field",
    }


def test_max_new_tokens_range_message_matches_the_contract_exactly(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "max_new_tokens": str(MAX_NEW_TOKENS_CEILING + 1)}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": f"max_new_tokens must be 0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING:,}",
        "code": "invalid_field",
    }


def test_split_chars_range_message_matches_the_contract_exactly(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "hi", "split_chars": "-1"})

    assert response.status_code == 400
    assert response.json() == {
        "error": "split_chars must be an integer between 0 and 10,000",
        "code": "invalid_field",
    }


def test_bc_08_three_ref_audio_file_parts_get_400(tmp_path: Path) -> None:
    """BC-08: three ref_audio file parts (not just two) still resolve to duplicate_field --
    the multipart parser is given enough headroom to actually see every one of them itself,
    rather than tripping its own generic file-count limit on the third one first."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(n: int) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="ref_audio"; filename="{n}.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            f"file bytes {n}\r\n"
        )

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"\r\n\r\n'
        "hi\r\n" + part(1) + part(2) + part(3) + f"--{boundary}--\r\n"
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


def test_bc_08_five_ref_audio_file_parts_still_get_duplicate_field(tmp_path: Path) -> None:
    """T048 post-final review finding #3: five `ref_audio` file parts is one past
    `FORM_MAX_FILES` (4) -- Starlette's own parser refuses to hand back that many file parts
    at all, raising `MultiPartException` before BC-08's own duplicate pass ever gets raw
    items to count. `_parse_multipart_form` falls back to the same `_check_no_duplicate_names`
    BC-08 already uses elsewhere, run over every name the parser had already produced
    (`parser.items`) plus the one that tripped the limit (`parser._current_part.field_name`)
    -- finding all five named `ref_audio` -- rather than leaving the client with the generic
    parser error, which wouldn't name the field at all."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(n: int) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="ref_audio"; filename="{n}.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            f"file bytes {n}\r\n"
        )

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"\r\n\r\n'
        "hi\r\n" + part(1) + part(2) + part(3) + part(4) + part(5) + f"--{boundary}--\r\n"
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


def test_bc_08_ref_audio_before_and_after_three_foo_files_still_gets_duplicate_field(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #3: the general rule doesn't care *where* in the run
    of file parts a repeated name falls -- `ref_audio`, three unrelated `foo` files, then a
    second `ref_audio` still trips `FORM_MAX_FILES` on the fifth part, and the first
    `ref_audio` is still sitting in `parser.items` by the time the duplicate check runs."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(name: str, n: int) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{n}.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            f"file bytes {n}\r\n"
        )

    body = (
        part("ref_audio", 1)
        + part("foo", 2)
        + part("foo", 3)
        + part("foo", 4)
        + part("ref_audio", 5)
        + f"--{boundary}--\r\n"
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


def test_bc_08_four_ref_audio_files_then_one_foo_still_gets_duplicate_field(
    tmp_path: Path,
) -> None:
    """The name that trips the limit doesn't have to be the repeated one itself -- four
    `ref_audio` files fill every slot `FORM_MAX_FILES` allows, and an unrelated fifth
    (`foo`) is what actually raises, but `ref_audio` is still the name repeated four times
    over in `parser.items`."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(name: str, n: int) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{n}.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            f"file bytes {n}\r\n"
        )

    body = (
        part("ref_audio", 1)
        + part("ref_audio", 2)
        + part("ref_audio", 3)
        + part("ref_audio", 4)
        + part("foo", 5)
        + f"--{boundary}--\r\n"
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


def test_bc_08_five_foo_file_parts_get_duplicate_field_not_just_ref_audio(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #3: the old mechanism only ever recognised `ref_audio`
    specifically (it matched Starlette's "Too many files" message text, then checked the
    field name against that one literal string); the general rule catches any repeated
    name, `ref_audio` or not."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(n: int) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="foo"; filename="{n}.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            f"file bytes {n}\r\n"
        )

    body = (part(1) + part(2) + part(3) + part(4) + part(5) + f"--{boundary}--\r\n").encode(
        "ascii"
    )

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "foo was given more than once",
        "code": "duplicate_field",
    }


def test_bc_08_five_distinctly_named_file_parts_keep_the_generic_error(
    tmp_path: Path,
) -> None:
    """A file-count limit hit by five *distinct* names has no duplicate to report -- the
    generic parser error is still what the client gets, same as before this rule existed."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(name: str) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{name}.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            f"file bytes {name}\r\n"
        )

    body = (
        part("a") + part("b") + part("c") + part("d") + part("e") + f"--{boundary}--\r\n"
    ).encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_multipart_missing_boundary_with_an_empty_query_name_keeps_the_generic_error(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #3 (review 31b): `MultipartPart.field_name` defaults
    to `""`, not `None` -- a request whose multipart parsing fails before any part ever gets
    a real name (here, a missing boundary raises immediately) still has a `_current_part`
    with that default, empty name. `?=1` puts one query field with an empty name alongside
    it, and `""` (not `None`) used to get appended to the names list regardless, so the two
    empty names looked like the same field given twice -- `duplicate_field` naming no field
    at all, instead of the generic parser error this malformed request should get."""
    response = _client(tmp_path).post(
        "/speech?=1", content=b"", headers={"content-type": "multipart/form-data"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_bc_08_multipart_limit_hit_with_parser_missing_private_attribute_stays_400_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T048 post-final review finding #3: `parser.items` and `parser._current_part` are
    read through `getattr` guards specifically because neither is a stable, documented
    Starlette attribute. Simulated here by deleting `_current_part` at the exact moment a
    `MultiPartException` is about to propagate (as if a future Starlette release renamed or
    dropped it) -- the five distinct file names below give no duplicate to find from
    `parser.items` alone even with the attribute intact, so this must still fall back to the
    generic error, not crash with a 500."""

    class _ParserMissingCurrentPart(http_fields._StrictMultiPartParser):
        def on_headers_finished(self) -> None:
            try:
                super().on_headers_finished()
            except http_fields.MultiPartException:
                del self._current_part
                raise

    monkeypatch.setattr(http_fields, "_StrictMultiPartParser", _ParserMissingCurrentPart)

    boundary = "xxxxBOUNDARYxxxx"

    def part(name: str) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{name}.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            f"file bytes {name}\r\n"
        )

    body = (
        part("a") + part("b") + part("c") + part("d") + part("e") + f"--{boundary}--\r\n"
    ).encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "could not parse the request body",
        "code": "invalid_field",
    }


def test_bc_08_33_same_named_multipart_text_fields_get_duplicate_field(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #3: 33 `seed` text fields is one past
    `FORM_MAX_FIELDS` (32) -- the *field* limit, not the file limit, but
    `on_headers_finished` sets `_current_part.field_name` before checking either one, so the
    same general rule applies regardless of which limit actually trips."""
    boundary = "xxxxBOUNDARYxxxx"

    def part(n: int) -> str:
        return (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="seed"\r\n\r\n'
            f"{n}\r\n"
        )

    body = ("".join(part(n) for n in range(33)) + f"--{boundary}--\r\n").encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }


def test_bc_08_33_same_named_urlencoded_fields_get_duplicate_field(tmp_path: Path) -> None:
    """T048 post-final review finding #3: the urlencoded equivalent -- `parse_qsl`'s own
    `max_num_fields` guard (`test_urlencoded_field_count_over_the_cap_gets_400`) fires before
    it has split or decoded a single field, so `_urlencoded_pairs_sync` re-derives just the
    first `FORM_MAX_FIELDS + 1` field names (findings #2/#4/#5: bounded, and decoded the same
    way the real parse would) to run the same duplicate check against."""
    body = "&".join(f"seed={n}" for n in range(FORM_MAX_FIELDS + 1)).encode("ascii")

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


def test_urlencoded_fallback_decodes_body_names_before_comparing_avoids_false_duplicate(
    tmp_path: Path,
) -> None:
    """T048 post-final review findings #4/#5: comparing an undecoded body name against the
    query's own already-decoded names can report a false duplicate. `?a%2Bb=1` decodes to
    the name `a+b`; a body field literally spelled `a+b` decodes to `a b` (a real `+` in a
    urlencoded body always means space) -- two different names that only looked equal when
    the body side wasn't decoded the same way. 32 unrelated filler fields plus this one push
    the body to 33 fields, one past `FORM_MAX_FIELDS`, so the fallback runs."""
    body = ("&".join(f"f{i}=1" for i in range(FORM_MAX_FIELDS)) + "&a+b=x").encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
        params={"a+b": "1"},  # httpx percent-encodes this to ?a%2Bb=1 on the wire
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_field"


def test_urlencoded_fallback_decodes_body_names_before_comparing_still_finds_real_duplicate(
    tmp_path: Path,
) -> None:
    """The flip side of the test above: `seed` in the query next to `se%65d` in the body (33
    fields deep, past `FORM_MAX_FIELDS`) are the same name once both sides are decoded the
    same way (`%65` is `e`) -- decoding the body side properly must still catch a real
    duplicate, not just avoid a false one."""
    body = ("&".join(f"f{i}=1" for i in range(FORM_MAX_FIELDS)) + "&se%65d=2").encode("ascii")

    response = _client(tmp_path).post(
        "/speech",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
        params={"seed": "1"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }


def test_text_length_is_checked_before_control_characters(tmp_path: Path) -> None:
    """When text is both over-length and made entirely of control characters, the (cheap,
    O(1)) length check runs first and wins -- BC-05's text_too_long, not BC-46's control-
    character message -- so an over-length value is never scanned for control characters at
    all."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "\x07" * (MAX_TEXT_CHARS + 1)}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "text is too long", "code": "text_too_long"}


def test_instruction_blank_check_runs_before_the_length_check(tmp_path: Path) -> None:
    """DECISION (final review): length-before-blank is specific to `text`; `instruction`
    keeps blank-before-length -- a whitespace-only instruction still means the default no
    matter how long it is, not `invalid_field`."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "instruction": " " * (MAX_INSTRUCTION_CHARS + 1)}
    )

    assert response.status_code == 200
    assert response.json()["instruction"] == DEFAULT_INSTRUCTION


def test_ref_text_blank_check_runs_before_the_length_check(tmp_path: Path) -> None:
    """Same restored ordering as instruction: a whitespace-only ref_text still counts as
    absent no matter how long it is, not `invalid_field`."""
    response = _client(tmp_path).post(
        "/speech",
        data={
            "text": "hi",
            "voice_id": "alice",
            "ref_text": " " * (MAX_REF_TEXT_CHARS + 1),
        },
    )

    assert response.status_code == 200
    assert response.json()["reference"]["ref_text_override"] is None


def test_voice_id_uppercase_v_prefix_is_rejected(tmp_path: Path) -> None:
    """BC-26: names are unique ignoring case, so the v_ prefix is reserved case-
    insensitively too -- V_ + 16 hex characters can't be treated as an ordinary
    saved-name-shaped string just because its case doesn't literally match "v_"; a real
    unnamed-voice id must still be lowercase, exactly as before."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "voice_id": "V_" + "a1b2c3d4e5f60789"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "voice_id must be a voice name or v_ id",
        "code": "invalid_field",
    }


def test_punctuation_only_text_gets_text_required(tmp_path: Path) -> None:
    """text_split.py's own rule drops a piece with no letter or digit (punctuation-only
    text can't be spoken); caught here too, so it's 400 text_required at the field stage
    rather than something split_text silently discards downstream."""
    response = _client(tmp_path).post("/speech", data={"text": "..."})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


def test_emoji_only_text_gets_text_required(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/speech", data={"text": "\U0001f389\U0001f389"})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


def test_unspeakable_text_gets_text_required_before_reference_checks(tmp_path: Path) -> None:
    """Stage 2 (field syntax) runs before stage 3 (reference consistency), per FR-007:
    text="..." (unspeakable) together with a bare ref_text (which, on its own, would
    otherwise be 400 reference_required) must still be 400 text_required -- the field-stage
    problem with `text` is caught before `_build_reference` ever looks at `ref_text`."""
    response = _client(tmp_path).post("/speech", data={"text": "...", "ref_text": "t"})

    assert response.status_code == 400
    assert response.json() == {"error": "text is required", "code": "text_required"}


def test_bc_01_integer_literal_over_4300_raw_digits_doesnt_crash(tmp_path: Path) -> None:
    """Regression: the digit cap (test_bc_01_integer_longer_than_20_digits_gets_400) counts
    significant digits, but a fix that only fixed the counting -- and still called int() on
    the original, unstripped literal -- would still crash for a literal with more raw digits
    than sys.get_int_max_str_digits() (4,300 by default), even though its actual value is
    tiny. int() must only ever be called on the stripped digits."""
    response = _client(tmp_path).post(
        "/speech", data={"text": "hi", "seed": "0" * 5000 + "1"}
    )

    assert response.status_code == 200
    assert response.json()["seed"] == 1


def test_ref_audio_in_query_wins_over_an_unrelated_body_only_duplicate(
    tmp_path: Path,
) -> None:
    """T048 post-final review findings #4/#7: this used to be `duplicate_field` (BC-08's
    combined query+body pass ran before `ref_audio`-as-text, in every branch) -- but
    reaching that combined pass at all means the body has already been read and parsed,
    which is exactly what `ref_audio` in the query must be rejected *before*, per
    findings #4/#7 above. Query-only checks run first now, so a `ref_audio` already known to
    be wrong from the query string alone is reported even when the body, once parsed, turns
    out to have its own, unrelated problem."""
    response = _client(tmp_path).post(
        "/speech",
        content=b"seed=1&seed=2",
        headers={"content-type": "application/x-www-form-urlencoded"},
        params={"ref_audio": "x"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


def test_ref_audio_in_query_wins_over_a_cross_source_seed_duplicate(
    tmp_path: Path,
) -> None:
    """T048 post-final review finding #8: the query holds both `ref_audio` and one `seed`;
    the body holds a second `seed` -- a genuine duplicate only once the two are combined,
    which is exactly what the second, per-branch `_check_no_duplicate_names` pass (query
    names plus the body's) exists to catch. But `ref_audio` in the query is rejected by the
    query-only pass *before* that second pass -- or the body itself -- is ever reached, so
    `ref_audio must be a file part` wins over `duplicate_field` for `seed`, exactly as the
    module docstring's query-first ordering intends."""
    response = _client(tmp_path).post(
        "/speech",
        content=b"seed=2",
        headers={"content-type": "application/x-www-form-urlencoded"},
        params={"ref_audio": "x", "seed": "1"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "ref_audio must be a file part",
        "code": "invalid_field",
    }


def test_bc_08_duplicate_pass_still_runs_before_ref_audio_as_text_check_within_the_body(
    tmp_path: Path,
) -> None:
    """BC-08's duplicate pass still runs before the ref_audio-as-text check for whatever a
    single body branch produces -- both `ref_audio` (as text) and the duplicated `seed` are
    in the body here, not split across the query and the body, so there's no query-only
    check to short-circuit first; the combined duplicate check (run once per branch) still
    wins over that same branch's own `_reject_ref_audio_text` call."""
    response = _client(tmp_path).post(
        "/speech",
        content=b"ref_audio=x&seed=1&seed=2",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "seed was given more than once",
        "code": "duplicate_field",
    }


def test_voice_id_names_are_checked_by_voice_file_is_valid_name(monkeypatch) -> None:
    """Review finding #10: one saved-name rule, `voice_file.is_valid_name`, shared with the
    voice store -- not a second copy of the pattern here. Swapping the shared helper for a
    stand-in changes what `voice_id` accepts; the unnamed `v_` branch is unaffected."""
    from breeze_infer import voice_file

    monkeypatch.setattr(voice_file, "is_valid_name", lambda name: name == "only-this")
    assert http_fields.is_valid_voice_id("only-this") is True
    assert http_fields.is_valid_voice_id("alice") is False
    assert http_fields.is_valid_voice_id("v_0123456789abcdef") is True
