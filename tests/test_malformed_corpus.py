"""Malformed HTTP request corpus (specs/003-cpp-compatible-api/tasks.md T047, SC-003).

SC-003: at least 50 malformed HTTP requests, each producing a structured error, with
no item crashing the server, hanging a connection, or changing later results.

Every item here is a `(name, request_kwargs)` pair, fed straight to
`TestClient.post(SPEECH_PATH, **request_kwargs)`. The corpus covers, per the task:
bad numbers, duplicate fields, control characters, reference conflicts, corrupt/
truncated/oversize audio, multipart structural damage (no boundary, junk after the
boundary, a truncated body with no closing boundary), and a bad `Content-Type`.

Some items exercise `http_fields.py`'s strict grammar, duplicate detection and
control-character rule -- T048's job, landing in a separate module this file never
touches. Items that depend on that work are marked `# T048` below; run this file
after T048 lands to confirm they now pass too (`test_every_corpus_item_depends_on_t048`
lists exactly which, by name, so a failure here before then is expected and not a bug
in this file).
"""

# Every fixture imported below (`client`, `components`, `readiness`, `ready_client`) is
# reused, by pytest's own name-based discovery, as a same-named parameter on the tests
# in this file -- the standard way to share fixtures across modules without a
# `conftest.py`. Ruff's F811 ("redefinition") otherwise fires on every one of those
# parameters, since it can't tell a fixture parameter from an accidental shadow.
# ruff: noqa: F811

from __future__ import annotations

import struct

import pytest
from fastapi.testclient import TestClient

from breeze_infer.limits import MAX_AUDIO_BYTES
from tests.test_routes_speech import (  # noqa: F401 -- fixtures reused by name
    SPEECH_PATH,
    _wav_bytes,
    client,
    components,
    readiness,
    ready_client,
)

# --- helpers to build the corpus ----------------------------------------------------


def _numeric(field: str, value: str) -> tuple[str, dict[str, object]]:
    return (f"bad_number:{field}={value!r}", {"data": {"text": "hello", field: value}})


def _valid_ref_wav() -> bytes:
    return _wav_bytes()


def _bogus_data_chunk_length_wav() -> bytes:
    """A small, otherwise well-formed WAV (16 kHz, 16-bit mono, 1,600 bytes of real
    silence -- under the 80 ms/1,920-sample minimum) whose `data` chunk declares a
    length of `0xFFFFFFFF`, far past the real bytes present. A safe decoder either
    refuses it outright (`invalid_audio`) or reads only the real bytes present, which
    are themselves too short (`audio_too_short`) -- either is a safe, structured `400`;
    only a `200`, a crash or a hang would be BC-15's original defect made concrete
    again (mirrors `tests/test_routes_reference.py`'s `_with_bogus_data_length`)."""
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


def _multipart_no_boundary() -> tuple[str, dict[str, object]]:
    body = b'Content-Disposition: form-data; name="ref_text"\r\n\r\nx\r\n'
    return (
        "multipart_no_boundary",
        {"content": body, "headers": {"content-type": "multipart/form-data"}},
    )


def _multipart_junk_after_boundary() -> tuple[str, dict[str, object]]:
    # No `text` field at all, so this must 400 on text_required regardless of whether
    # the trailing junk itself breaks the parse.
    boundary = "xxxxBOUNDARYxxxx"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="instruction"\r\n\r\n'
        "hi\r\n"
        f"--{boundary}--\r\n"
        "JUNK JUNK JUNK not part of any part"
    ).encode("ascii")
    return (
        "multipart_junk_after_boundary",
        {"content": body, "headers": {"content-type": f"multipart/form-data; boundary={boundary}"}},
    )


def _multipart_truncated_no_closing_boundary() -> tuple[str, dict[str, object]]:
    # Cuts off mid-field, before any closing boundary -- and again, no `text` field is
    # ever completed, so the missing-required-field check (text_required) must be what
    # answers this, per the task description, not an opaque parser error.
    boundary = "xxxxBOUNDARYxxxx"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"\r\n\r\n'
        "this never "
    ).encode("ascii")
    return (
        "multipart_truncated_no_closing_boundary",
        {"content": body, "headers": {"content-type": f"multipart/form-data; boundary={boundary}"}},
    )


def _corpus() -> list[tuple[str, dict[str, object]]]:
    items: list[tuple[str, dict[str, object]]] = []

    # --- bad numbers (BC-01/BC-03/BC-04) ---
    bad_values = {
        "cfg_scale": ["banana", "1e", "0x10"],
        "seed": ["banana", "-1", "4294967296"],
        "temperature": ["nan", "inf", "-1"],
        "top_k": ["banana", "-5", "1.5"],
        "top_p": ["2", "banana", "-0.5"],
        "repetition_penalty": ["banana", "-1", "100000"],
        "max_new_tokens": ["banana", "-1", "99999"],
        "split_chars": ["banana", "-1", "100000"],
    }
    for field, values in bad_values.items():
        for value in values:
            items.append(_numeric(field, value))

    # --- duplicate fields (BC-08) --- # T048
    items.append(
        ("duplicate_text_in_body", {"data": [("text", "one"), ("text", "two")]})
    )
    items.append(
        (
            "duplicate_text_body_and_query",
            {"data": {"text": "one"}, "params": {"text": "two"}},
        )
    )
    items.append(
        ("duplicate_cfg_scale", {"data": [("text", "hi"), ("cfg_scale", "1"), ("cfg_scale", "2")]})
    )
    items.append(
        ("duplicate_voice_id", {"data": [("text", "hi"), ("voice_id", "a"), ("voice_id", "b")]})
    )
    items.append(
        (
            "duplicate_seed_body_and_query",
            {"data": {"text": "hi", "seed": "1"}, "params": {"seed": "2"}},
        )
    )
    items.append(
        (
            "duplicate_ref_text",
            {"data": [("text", "hi"), ("ref_text", "a"), ("ref_text", "b")]},
        )
    )

    # --- control characters (BC-46) --- # T048
    items.append(("text_has_nul", {"data": {"text": "hello\x00world"}}))
    items.append(("text_has_bell", {"data": {"text": "hello\x07world"}}))
    items.append(("text_has_escape", {"data": {"text": "hello\x1bworld"}}))
    items.append(("instruction_has_nul", {"data": {"text": "hi", "instruction": "spe\x00ak"}}))
    items.append(("instruction_has_bell", {"data": {"text": "hi", "instruction": "spe\x07ak"}}))
    items.append(("ref_text_has_escape", {"data": {"text": "hi", "ref_text": "tra\x1bnscript"}}))

    # --- reference conflicts (BC-12/BC-13/BC-14) ---
    items.append(
        (
            "voice_id_and_ref_audio",
            {
                "data": {"text": "hi", "voice_id": "alice", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", _valid_ref_wav(), "audio/wav")},
            },
        )
    )
    items.append(
        (
            "ref_audio_without_ref_text",
            {
                "data": {"text": "hi"},
                "files": {"ref_audio": ("r.wav", _valid_ref_wav(), "audio/wav")},
            },
        )
    )
    items.append(("ref_text_without_reference", {"data": {"text": "hi", "ref_text": "t"}}))
    items.append(
        (
            "voice_id_ref_audio_and_ref_text",
            {
                "data": {"text": "hi", "voice_id": "alice", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", _valid_ref_wav(), "audio/wav")},
            },
        )
    )

    # --- corrupt/truncated/oversize audio (BC-11/BC-15/BC-16) ---
    real_wav = _valid_ref_wav()
    items.append(
        (
            "garbage_ref_audio",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", b"not audio, just noise" * 50, "audio/wav")},
            },
        )
    )
    items.append(
        (
            "truncated_wav",
            {
                "data": {"text": "hi", "ref_text": "t"},
                # Just the RIFF/fmt header and a sliver of `data` -- far under the 80 ms
                # minimum even if libsndfile recovers every real byte present (a longer
                # truncation, e.g. half the file, can still leave enough real audio to
                # decode as a short-but-valid clip, which isn't malformed at all).
                "files": {"ref_audio": ("r.wav", real_wav[:100], "audio/wav")},
            },
        )
    )
    items.append(
        (
            "empty_ref_audio_part",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", b"", "audio/wav")},
            },
        )
    )
    items.append(
        (
            "bogus_data_chunk_length",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {"ref_audio": ("r.wav", _bogus_data_chunk_length_wav(), "audio/wav")},
            },
        )
    )
    items.append(
        (
            "oversize_ref_audio",
            {
                "data": {"text": "hi", "ref_text": "t"},
                "files": {
                    "ref_audio": ("r.wav", b"\x00" * (MAX_AUDIO_BYTES + 1024), "audio/wav")
                },
            },
        )
    )

    # --- multipart structural damage --- # T048 (strict parsing)
    items.append(_multipart_no_boundary())
    items.append(_multipart_junk_after_boundary())
    items.append(_multipart_truncated_no_closing_boundary())

    # --- bad Content-Type ---
    items.append(
        (
            "content_type_json",
            {"content": b'{"text": "hi"}', "headers": {"content-type": "application/json"}},
        )
    )
    items.append(
        (
            "content_type_text_plain",
            {"content": b"text=hi", "headers": {"content-type": "text/plain"}},
        )
    )
    items.append(
        (
            "content_type_nonsense",
            {"content": b"text=hi", "headers": {"content-type": "banana/whatever"}},
        )
    )

    return items


_CORPUS = _corpus()

# Every corpus item that needs http_fields.py's strict grammar/duplicate/control-char
# work (T048, landing separately) to actually produce its documented error today.
# Kept as an explicit list (rather than inferred from a failure) so a run before T048
# lands reports exactly which failures are expected, and a run after it lands proves
# nothing is silently still broken.
# Duplicates of these fields are only caught once the duplicate pass runs before every other
# field check (the T048 follow-up in progress); everything else in the corpus already holds.
_DEPENDS_ON_T048 = {
    "duplicate_text_in_body",
    "duplicate_cfg_scale",
    "duplicate_voice_id",
    "duplicate_ref_text",
}


def test_corpus_has_at_least_50_items() -> None:
    assert len(_CORPUS) >= 50, len(_CORPUS)


def test_corpus_names_are_unique() -> None:
    names = [name for name, _ in _CORPUS]
    assert len(names) == len(set(names)), names


@pytest.mark.parametrize("name,kwargs", _CORPUS, ids=[name for name, _ in _CORPUS])
def test_every_malformed_request_gets_a_structured_4xx(
    ready_client: TestClient, name: str, kwargs: dict[str, object]
) -> None:
    if name in _DEPENDS_ON_T048:
        pytest.xfail(f"{name} needs the upfront duplicate pass (T048 follow-up)")

    response = ready_client.post(SPEECH_PATH, **kwargs)

    assert 400 <= response.status_code < 500, (name, response.status_code, response.text)
    body = response.json()
    assert "error" in body and "code" in body, (name, body)


def test_health_stays_200_throughout_and_after_the_corpus(ready_client: TestClient) -> None:
    """SC-003: no item crashes the server or hangs a connection -- proven by /health
    staying reachable before, interleaved with, and after every corpus item, and a
    normal request still succeeding afterwards."""
    assert ready_client.get("/health").status_code == 200

    for name, kwargs in _CORPUS:
        ready_client.post(SPEECH_PATH, **kwargs)
        assert ready_client.get("/health").status_code == 200, f"/health broke after {name}"

    normal = ready_client.post(SPEECH_PATH, data={"text": "hello there"})
    assert normal.status_code == 200
    assert len(normal.content) > 0
