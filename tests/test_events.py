from __future__ import annotations

import io
import json

import pytest

from breeze_infer.events import EVENT_SCHEMA, Emitter


def test_emit_writes_exactly_one_json_line_per_call() -> None:
    sink = io.StringIO()
    ticks = iter([10.0, 11.0])
    events = Emitter(sink=sink, clock=lambda: next(ticks))

    events.emit("voice.used", voice_id="voice_abc")
    events.emit("voice.used", voice_id="voice_def")

    lines = sink.getvalue().splitlines()
    assert len(lines) == 2
    assert sink.getvalue().endswith("\n")
    assert [json.loads(line)["voice_id"] for line in lines] == [
        "voice_abc",
        "voice_def",
    ]


def test_emit_carries_schema_timestamp_and_event_name() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 1234.5)

    record = events.emit("speech.first_audio")

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event_schema"] == EVENT_SCHEMA == 1
    assert parsed["ts"] == 1234.5
    assert parsed["event"] == "speech.first_audio"


def test_emit_preserves_field_types() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    events.emit(
        "voice.prepared",
        warm=True,
        prefix_len=145,
        build_ms=18.25,
        tier="kv",
        reason=None,
    )

    parsed = json.loads(sink.getvalue())
    assert parsed["warm"] is True
    assert parsed["prefix_len"] == 145
    assert parsed["build_ms"] == 18.25
    assert parsed["tier"] == "kv"
    assert parsed["reason"] is None


def test_emit_rejects_non_serialisable_fields() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    with pytest.raises(TypeError):
        events.emit("voice.used", payload=object())
    assert sink.getvalue() == ""


def test_emit_rejects_empty_event_name() -> None:
    events = Emitter(sink=io.StringIO(), clock=lambda: 0.0)

    with pytest.raises(ValueError):
        events.emit("")
