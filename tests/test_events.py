from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from breeze_infer.events import EVENT_SCHEMA, Emitter


class _FlushTrackingSink(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


class _BrokenSink:
    """A sink whose write always fails, like a pipe that's gone away."""

    def write(self, _line: str) -> int:
        raise OSError("broken pipe")

    def flush(self) -> None:
        raise OSError("broken pipe")


def test_emit_writes_exactly_one_json_line_per_call_using_the_injected_clock() -> None:
    sink = io.StringIO()
    ticks = iter([10.0, 11.0])
    events = Emitter(sink=sink, clock=lambda: next(ticks))

    events.emit("voice.used", voice_id="voice_abc")
    events.emit("voice.used", voice_id="voice_def")

    lines = sink.getvalue().splitlines()
    assert len(lines) == 2
    assert sink.getvalue().endswith("\n")
    assert [json.loads(line)["ts"] for line in lines] == [10.0, 11.0]
    assert [json.loads(line)["voice_id"] for line in lines] == [
        "voice_abc",
        "voice_def",
    ]


def test_emit_flushes_after_every_write() -> None:
    sink = _FlushTrackingSink()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    events.emit("voice.used")
    events.emit("voice.used")

    assert sink.flush_calls == 2


def test_emit_carries_schema_timestamp_event_name_and_default_level() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 1234.5)

    record = events.emit("speech.first_audio")

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event_schema"] == EVENT_SCHEMA == 1
    assert parsed["ts"] == 1234.5
    assert parsed["event"] == "speech.first_audio"
    assert parsed["level"] == "info"


def test_emit_accepts_an_explicit_allowed_level() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("speech.failed", level="error")

    assert json.loads(sink.getvalue())["level"] == "error"
    assert record["level"] == "error"


def test_emit_rejects_an_invalid_level() -> None:
    events = Emitter(sink=io.StringIO(), clock=lambda: 0.0)

    with pytest.raises(ValueError):
        events.emit("speech.failed", level="critical")


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


def test_emit_rejects_empty_event_name() -> None:
    events = Emitter(sink=io.StringIO(), clock=lambda: 0.0)

    with pytest.raises(ValueError):
        events.emit("")


@pytest.mark.parametrize("reserved_key", ["ts", "event_schema", "event"])
def test_emit_rejects_a_field_that_collides_with_a_reserved_key(reserved_key: str) -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    with pytest.raises(ValueError):
        events.emit("voice.used", **{reserved_key: "overwrite"})
    assert sink.getvalue() == ""


def test_emit_rejects_a_field_named_event_with_value_error_not_type_error() -> None:
    events = Emitter(sink=io.StringIO(), clock=lambda: 0.0)

    with pytest.raises(ValueError):
        events.emit("voice.used", event="collision")


def test_emit_turns_a_nan_field_into_an_event_invalid_record_instead_of_raising() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("speech.completed", rtf=math.nan)

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event"] == "event.invalid"
    assert parsed["invalid_event"] == "speech.completed"
    assert parsed["error"] == "ValueError"


def test_emit_turns_an_unserialisable_field_into_an_event_invalid_record() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("voice.used", payload=object())

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event"] == "event.invalid"
    assert parsed["invalid_event"] == "voice.used"
    assert parsed["error"] == "TypeError"


def test_emit_serialises_numpy_scalars_via_the_item_hook() -> None:
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    events.emit(
        "speech.completed",
        rtf=np.float32(0.42),
        piece_index=np.int64(3),
    )

    parsed = json.loads(sink.getvalue())
    assert parsed["event"] == "speech.completed"
    assert parsed["rtf"] == pytest.approx(0.42, rel=1e-5)
    assert parsed["piece_index"] == 3


def test_emit_swallows_an_oserror_from_a_gone_sink() -> None:
    events = Emitter(sink=_BrokenSink(), clock=lambda: 0.0)

    # Telemetry must never break a request (Constitution VII): a dead sink is silently dropped.
    events.emit("voice.used")
