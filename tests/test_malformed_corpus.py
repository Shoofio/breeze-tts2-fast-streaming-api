"""Malformed HTTP request corpus (specs/003-cpp-compatible-api/tasks.md T047, SC-003).

SC-003: at least 50 malformed HTTP requests, each producing a structured error, with
no item crashing the server, hanging a connection, or changing later results.

Every item here is an `_Item`: a name, the `TestClient.post(SPEECH_PATH, **kwargs)`
keyword arguments, and the exact `(status, code, message)` the request must produce --
read straight off `breeze_infer/http_fields.py`'s own rule strings and
`contracts/http-api.md`'s error table (T045's `test_routes_reference.py` covers the
audio-decode messages), not just a generic "some 4xx" check. The corpus covers, per the
task: bad numbers, duplicate fields, control characters, reference conflicts, corrupt/
truncated/oversize audio, multipart structural damage (no boundary, junk after the
boundary, a truncated body with no closing boundary), and a bad `Content-Type`, plus a
bad charset and an oversized multipart/urlencoded part.

T048 (http_fields.py's strict grammar, duplicate detection and control-character rule)
has landed, so every item below is asserted for real -- nothing here is `xfail`.
"""

# Every fixture imported below (`client`, `components`, `readiness`, `ready_client`) is
# reused, by pytest's own name-based discovery, as a same-named parameter on the tests
# in this file -- the standard way to share fixtures across modules without a
# `conftest.py`. Ruff's F811 ("redefinition") otherwise fires on every one of those
# parameters, since it can't tell a fixture parameter from an accidental shadow.
# ruff: noqa: F811

from __future__ import annotations

import struct
from typing import NamedTuple

import pytest
from fastapi.testclient import TestClient

from breeze_infer.limits import (
    MAX_AUDIO_BYTES,
    MAX_INSTRUCTION_CHARS,
    MAX_NEW_TOKENS_CEILING,
    MAX_REF_TEXT_CHARS,
    MAX_TEXT_CHARS,
)
from tests.test_routes_speech import (  # noqa: F401 -- fixtures reused by name
    SPEECH_PATH,
    _wav_bytes,
    client,
    components,
    readiness,
    ready_client,
)


class _Item(NamedTuple):
    """One corpus entry: the request to send, and the exact envelope it must produce.

    `status`/`code`/`message` are asserted exactly (review finding 2), not just "some
    4xx" -- each was read off `http_fields.py`'s own rule strings (or, for the audio
    items, `reference_audio.py`'s fixed messages) rather than guessed, so a wording
    change anywhere in that module is a real, visible corpus failure, not something
    this file silently tolerates.
    """

    name: str
    kwargs: dict[str, object]
    status: int
    code: str
    message: str


# --- helpers to build the corpus ----------------------------------------------------


def _numeric(field: str, value: str, status: int, code: str, message: str) -> _Item:
    return _Item(f"bad_number:{field}={value!r}", {"data": {"text": "hello", field: value}}, status, code, message)


def _urlencoded(name: str, body: bytes, status: int, code: str, message: str, *, content_type: str = "application/x-www-form-urlencoded") -> _Item:
    """review finding 1: httpx 0.28's `TestClient.post(data=[(k, v), ...])` can't build
    a request from a list of tuples at all (`TypeError: sequence item 0: expected a
    bytes-like object, tuple found`, raised deep in httpx's own content encoding) --
    which the app then sees as a client that hung up mid-request, answered `500`. A
    duplicate key needs raw, hand-built wire bytes instead; this is exactly what
    `application/x-www-form-urlencoded` (and multipart) already are on the wire, so a
    literal `b"text=one&text=two"` is both correct and simple.
    """
    return _Item(name, {"content": body, "headers": {"content-type": content_type}}, status, code, message)


def _valid_ref_wav() -> bytes:
    return _wav_bytes()


def _bogus_data_chunk_length_wav() -> bytes:
    """A small, otherwise well-formed WAV (16 kHz, 16-bit mono, 1,600 bytes of real
    silence -- under the 80 ms/1,920-sample minimum) whose `data` chunk declares a
    length of `0xFFFFFFFF`, far past the real bytes present. libsndfile reads only the
    real bytes present (BC-15's original defect was over-reading past them), which are
    themselves too short -- `audio_too_short`, a safe, structured `400` (mirrors
    `tests/test_routes_reference.py`'s `_with_bogus_data_length`)."""
    channels, sample_rate, bits_per_sample = 1, 16000, 16
    data = bytes(1600)
    block_align = channels * bits_per_sample // 8
    fmt_chunk = struct.pack(
        "<HHIIHH", 1, channels, sample_rate, sample_rate * block_align, block_align, bits_per_sample
    )
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
        + b"data" + struct.pack("<I", 0xFFFFFFFF) + data
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


_CONTENT_TYPE_ERROR = (
    400,
    "invalid_field",
    "content type must be multipart/form-data or application/x-www-form-urlencoded",
)


def _corpus() -> list[_Item]:
    items: list[_Item] = []

    # --- bad numbers (BC-01/BC-03/BC-04): every message below is `http_fields.py`'s own
    # rule string, read from a real run against the live module (this file's own docstring
    # explains why: the coordinator's wording pass can still move these under us). ---
    bad_number_cases: list[tuple[str, str, str]] = [
        ("cfg_scale", "banana", "cfg_scale must be a number"),
        ("cfg_scale", "1e", "cfg_scale must be a number"),
        ("cfg_scale", "0x10", "cfg_scale must be a number"),
        ("cfg_scale", "  1", "cfg_scale must be a number"),
        ("seed", "banana", "seed must be an integer"),
        ("seed", "-1", "seed must be an integer between 0 and 4294967295"),
        ("seed", "4294967296", "seed must be an integer between 0 and 4294967295"),
        ("seed", "99999999999999999999999999", "seed must be an integer between 0 and 4294967295"),
        ("temperature", "nan", "temperature must be a number"),
        ("temperature", "inf", "temperature must be a number"),
        ("temperature", "-1", "temperature must be 0, or greater than 0 and at most 10"),
        ("temperature", "1_0", "temperature must be a number"),
        ("top_k", "banana", "top_k must be an integer"),
        ("top_k", "-5", "top_k must be 0, or an integer between 1 and 10,000"),
        ("top_k", "1.5", "top_k must be an integer"),
        ("top_k", "0x5", "top_k must be an integer"),
        ("top_p", "2", "top_p must be 0, or greater than 0 and at most 1"),
        ("top_p", "banana", "top_p must be a number"),
        ("top_p", "-0.5", "top_p must be 0, or greater than 0 and at most 1"),
        ("top_p", "1,5", "top_p must be a number"),
        ("repetition_penalty", "banana", "repetition_penalty must be a number"),
        ("repetition_penalty", "-1", "repetition_penalty must be 0, or between 0.0001 and 10"),
        ("repetition_penalty", "100000", "repetition_penalty must be 0, or between 0.0001 and 10"),
        ("repetition_penalty", "1e", "repetition_penalty must be a number"),
        ("max_new_tokens", "banana", "max_new_tokens must be an integer"),
        (
            "max_new_tokens",
            "-1",
            f"max_new_tokens must be 0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING:,}",
        ),
        (
            "max_new_tokens",
            "99999",
            f"max_new_tokens must be 0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING:,}",
        ),
        ("max_new_tokens", " 5", "max_new_tokens must be an integer"),
        ("split_chars", "banana", "split_chars must be an integer"),
        ("split_chars", "-1", "split_chars must be an integer between 0 and 10,000"),
        ("split_chars", "100000", "split_chars must be an integer between 0 and 10,000"),
        ("split_chars", "5 ", "split_chars must be an integer"),
    ]
    for field, value, message in bad_number_cases:
        items.append(_numeric(field, value, 400, "invalid_field", message))

    # --- duplicate fields (BC-08): raw urlencoded bodies (review finding 1) ---
    items.append(_urlencoded("duplicate_text_in_body", b"text=one&text=two", 400, "duplicate_field", "text was given more than once"))
    items.append(
        _Item(
            "duplicate_text_body_and_query",
            {"data": {"text": "one"}, "params": {"text": "two"}},
            400,
            "duplicate_field",
            "text was given more than once",
        )
    )
    items.append(_urlencoded("duplicate_cfg_scale", b"text=hi&cfg_scale=1&cfg_scale=2", 400, "duplicate_field", "cfg_scale was given more than once"))
    items.append(_urlencoded("duplicate_voice_id", b"text=hi&voice_id=a&voice_id=b", 400, "duplicate_field", "voice_id was given more than once"))
    items.append(
        _Item(
            "duplicate_seed_body_and_query",
            {"data": {"text": "hi", "seed": "1"}, "params": {"seed": "2"}},
            400,
            "duplicate_field",
            "seed was given more than once",
        )
    )
    items.append(_urlencoded("duplicate_ref_text", b"text=hi&ref_text=a&ref_text=b", 400, "duplicate_field", "ref_text was given more than once"))

    # --- control characters (BC-46): the full field x character grid ---
    control_chars = [("nul", "\x00"), ("bell", "\x07"), ("escape", "\x1b")]
    for field in ("text", "instruction", "ref_text"):
        for char_name, ch in control_chars:
            data = {"text": "hi", field: f"hello{ch}world"}
            items.append(
                _Item(
                    f"{field}_has_{char_name}",
                    {"data": data},
                    400,
                    "invalid_field",
                    f"{field} must be free of control characters",
                )
            )

    # --- reference conflicts (BC-12/BC-13/BC-14) ---
    real_wav = _valid_ref_wav()
    items.append(
        _Item(
            "voice_id_and_ref_audio",
            {
                "data": {"text": "hi", "voice_id": "alice"},
                "files": {"ref_audio": ("r.wav", real_wav, "audio/wav")},
            },
            400,
            "reference_conflict",
            "voice_id and ref_audio cannot be used together",
        )
    )
    items.append(
        _Item(
            "ref_audio_without_ref_text",
            {
                "data": {"text": "hi"},
                "files": {"ref_audio": ("r.wav", real_wav, "audio/wav")},
            },
            400,
            "ref_text_required",
            "ref_text is required with ref_audio",
        )
    )
    items.append(
        _Item(
            "ref_text_without_reference",
            {"data": {"text": "hi", "ref_text": "t"}},
            400,
            "reference_required",
            "ref_text needs ref_audio or voice_id",
        )
    )
    items.append(
        _Item(
            # Distinct from "voice_id_and_ref_audio" above (review finding 3: the two
            # used to be literally the same request): this one also carries ref_text,
            # so it exercises the conflict check winning even when every other field is
            # otherwise satisfiable.
            "voice_id_ref_audio_and_ref_text",
            {
                "data": {"text": "hi", "voice_id": "alice", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", real_wav, "audio/wav")},
            },
            400,
            "reference_conflict",
            "voice_id and ref_audio cannot be used together",
        )
    )
    items.append(
        _Item(
            "voice_id_bad_shape",
            {"data": {"text": "hi", "voice_id": "not a valid id!"}},
            400,
            "invalid_field",
            "voice_id must be a voice name or v_ id",
        )
    )

    # --- corrupt/truncated/oversize audio (BC-11/BC-15/BC-16) ---
    items.append(
        _Item(
            "garbage_ref_audio",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", b"not audio, just noise" * 50, "audio/wav")},
            },
            400,
            "invalid_audio",
            "could not read ref_audio",
        )
    )
    items.append(
        _Item(
            "truncated_wav",
            {
                "data": {"text": "hi", "ref_text": "t"},
                # Just the RIFF/fmt header and a sliver of `data` -- far under the 80 ms
                # minimum even if libsndfile recovers every real byte present (a longer
                # truncation, e.g. half the file, can still leave enough real audio to
                # decode as a short-but-valid clip, which isn't malformed at all).
                "files": {"ref_audio": ("r.wav", real_wav[:100], "audio/wav")},
            },
            400,
            "audio_too_short",
            "ref_audio is too short",
        )
    )
    items.append(
        _Item(
            "empty_ref_audio_part",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", b"", "audio/wav")},
            },
            400,
            "invalid_audio",
            "could not read ref_audio",
        )
    )
    items.append(
        _Item(
            "bogus_data_chunk_length",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", _bogus_data_chunk_length_wav(), "audio/wav")},
            },
            400,
            "audio_too_short",
            "ref_audio is too short",
        )
    )
    items.append(
        _Item(
            "oversize_ref_audio",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {
                    "ref_audio": ("r.wav", b"\x00" * (MAX_AUDIO_BYTES + 1024), "audio/wav")
                },
            },
            400,
            "invalid_audio",
            "could not read ref_audio",
        )
    )

    # --- multipart structural damage ---
    boundary = "xxxxBOUNDARYxxxx"
    items.append(
        _Item(
            "multipart_no_boundary",
            {
                "content": b'Content-Disposition: form-data; name="ref_text"\r\n\r\nx\r\n',
                "headers": {"content-type": "multipart/form-data"},
            },
            400,
            "invalid_field",
            "could not parse the request body",
        )
    )
    items.append(
        _Item(
            "multipart_junk_after_boundary",
            {
                # No `text` field at all, so this must 400 on text_required regardless of
                # whether the trailing junk itself breaks the parse.
                "content": (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="instruction"\r\n\r\n'
                    "hi\r\n"
                    f"--{boundary}--\r\n"
                    "JUNK JUNK JUNK not part of any part"
                ).encode("ascii"),
                "headers": {"content-type": f"multipart/form-data; boundary={boundary}"},
            },
            400,
            "text_required",
            "text is required",
        )
    )
    items.append(
        _Item(
            "multipart_truncated_no_closing_boundary",
            {
                # Cuts off mid-field, before any closing boundary -- and again, no `text`
                # field is ever completed, so the missing-required-field check
                # (text_required) is what answers this, per the task description, not an
                # opaque parser error.
                "content": (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="text"\r\n\r\n'
                    "this never "
                ).encode("ascii"),
                "headers": {"content-type": f"multipart/form-data; boundary={boundary}"},
            },
            400,
            "text_required",
            "text is required",
        )
    )

    # --- bad Content-Type ---
    items.append(_Item("content_type_json", {"content": b'{"text": "hi"}', "headers": {"content-type": "application/json"}}, *_CONTENT_TYPE_ERROR))
    items.append(_Item("content_type_text_plain", {"content": b"text=hi", "headers": {"content-type": "text/plain"}}, *_CONTENT_TYPE_ERROR))
    items.append(_Item("content_type_nonsense", {"content": b"text=hi", "headers": {"content-type": "banana/whatever"}}, *_CONTENT_TYPE_ERROR))

    # --- extra items (review finding 3): field-length limits, a bad charset and an
    # oversized part, none of which the categories above already cover. ---
    items.append(
        _Item(
            "text_too_long",
            {"data": {"text": "a" * (MAX_TEXT_CHARS + 1)}},
            400,
            "text_too_long",
            "text is too long",
        )
    )
    items.append(
        _Item(
            "instruction_too_long",
            {"data": {"text": "hi", "instruction": "a" * (MAX_INSTRUCTION_CHARS + 1)}},
            400,
            "invalid_field",
            f"instruction must be at most {MAX_INSTRUCTION_CHARS:,} characters",
        )
    )
    items.append(
        _Item(
            "ref_text_too_long_alone",
            {"data": {"text": "hi", "ref_text": "a" * (MAX_REF_TEXT_CHARS + 1)}},
            400,
            "invalid_field",
            f"ref_text must be at most {MAX_REF_TEXT_CHARS:,} characters",
        )
    )
    items.append(
        _urlencoded(
            "bad_charset_urlencoded",
            b"text=hi",
            400,
            "invalid_field",
            "request body must be UTF-8 text",
            content_type="application/x-www-form-urlencoded; charset=iso-8859-1",
        )
    )
    items.append(
        _urlencoded(
            "oversized_urlencoded_part",
            b"text=" + b"a" * 300_000,
            400,
            "invalid_field",
            "could not parse the request body",
        )
    )
    items.append(
        _Item(
            "max_new_tokens_over_ceiling",
            {"data": {"text": "hi", "max_new_tokens": str(MAX_NEW_TOKENS_CEILING + 1)}},
            400,
            "invalid_field",
            f"max_new_tokens must be 0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING:,}",
        )
    )

    return items


_CORPUS = _corpus()


def test_corpus_has_at_least_50_items() -> None:
    assert len(_CORPUS) >= 50, len(_CORPUS)


def test_corpus_names_are_unique() -> None:
    names = [item.name for item in _CORPUS]
    assert len(names) == len(set(names)), names


def test_corpus_requests_are_distinct() -> None:
    """review finding 3: the corpus counts *distinct* requests, not just distinct
    names -- two items with the same name would already fail the uniqueness test above,
    but two different names sending the byte-for-byte identical request (as
    "voice_id_and_ref_audio" and "voice_id_ref_audio_and_ref_text" used to) would not."""
    seen: set[tuple[object, ...]] = set()
    for item in _CORPUS:
        # `files` holds real bytes, which aren't hashable as part of a dict repr
        # comparison across different WAV fixtures reliably, so the request is
        # identified by its resolved kwargs' repr -- stable and exact for this
        # corpus's own kwargs shapes (data/params/content/headers/files).
        fingerprint = repr(sorted(item.kwargs.items(), key=lambda kv: kv[0]))
        assert fingerprint not in seen, f"{item.name} duplicates an earlier request"
        seen.add(fingerprint)


@pytest.mark.parametrize("item", _CORPUS, ids=[item.name for item in _CORPUS])
def test_every_malformed_request_gets_its_exact_envelope(
    ready_client: TestClient, item: _Item
) -> None:
    response = ready_client.post(SPEECH_PATH, **item.kwargs)

    assert response.status_code == item.status, (item.name, response.status_code, response.text)
    assert response.json() == {"error": item.message, "code": item.code}, item.name


def test_health_stays_200_throughout_and_after_the_corpus(ready_client: TestClient) -> None:
    """SC-003: no item crashes the server or hangs a connection -- proven by /health
    staying reachable before, interleaved with, and after every corpus item (each also
    checked against its own expected status, review finding 1), and a normal request
    still succeeding afterwards."""
    assert ready_client.get("/health").status_code == 200

    for item in _CORPUS:
        response = ready_client.post(SPEECH_PATH, **item.kwargs)
        assert response.status_code == item.status, item.name
        assert ready_client.get("/health").status_code == 200, f"/health broke after {item.name}"

    normal = ready_client.post(SPEECH_PATH, data={"text": "hello there"})
    assert normal.status_code == 200
    assert len(normal.content) > 0
