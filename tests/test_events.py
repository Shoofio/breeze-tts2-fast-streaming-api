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


def test_emit_round_trips_non_ascii_field_values() -> None:
    """`ensure_ascii=True` escapes non-ASCII as `\\uXXXX` on the wire; this guards that decoding
    still recovers the original string (a regression test for the escaping itself, not just the
    reserved-key/level checks the other tests cover)."""
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("voice.used", voice_id="café_日本語")

    line = sink.getvalue()
    assert line.isascii()  # the raw bytes on the wire are pure ASCII
    assert "\\u00e9" in line  # 'é' (U+00E9), escaped -- not just "not the literal character"
    parsed = json.loads(line)
    assert parsed == record
    assert parsed["voice_id"] == "café_日本語"


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


def test_emit_swallows_a_valueerror_from_a_closed_sink() -> None:
    """`io` raises `ValueError` (not `OSError`) for "I/O operation on closed file"."""
    sink = io.StringIO()
    sink.close()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    events.emit("voice.used")  # must not raise


def test_emit_invalid_fallback_keeps_schema_ts_level_and_request_context() -> None:
    """E1: the `event.invalid` record must still carry `event_schema`/`ts`, be `level=warning`,
    and keep any of `request_id`/`session_id`/`piece_index` that were present -- so a bad field
    never loses the trail back to its request. `fields` lists only the names of fields that
    individually fail to serialise (E3): `other_field` here serialises fine on its own, so it's
    dropped from the top-level record (only `rtf`'s reserved names are kept) but not listed."""
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 42.0)

    record = events.emit(
        "speech.completed",
        rtf=math.nan,
        request_id="req-1",
        session_id="sess-1",
        piece_index=3,
        other_field="dropped",
    )

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event_schema"] == EVENT_SCHEMA
    assert parsed["ts"] == 42.0
    assert parsed["event"] == "event.invalid"
    assert parsed["level"] == "warning"
    assert parsed["invalid_event"] == "speech.completed"
    assert parsed["request_id"] == "req-1"
    assert parsed["session_id"] == "sess-1"
    assert parsed["piece_index"] == 3
    assert "other_field" not in parsed
    assert parsed["fields"] == ["rtf"]


def test_emit_invalid_fallback_drops_a_bad_context_value_but_keeps_the_good_ones() -> None:
    """E1: a context value that itself fails to serialise must not be copied into the fallback
    (that would just fail again) -- it's dropped, and listed under `fields` alongside whatever
    field actually triggered the fallback in the first place."""
    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit(
        "speech.completed",
        rtf=math.nan,
        request_id=object(),  # unserialisable on its own
        session_id="sess-1",
    )

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert "request_id" not in parsed
    assert parsed["session_id"] == "sess-1"
    assert parsed["fields"] == sorted(["rtf", "request_id"])


def test_emit_does_not_unwrap_a_non_numpy_object_with_an_item_method() -> None:
    """E3/E5 (pass 1): `_json_default` only unwraps `np.generic`, not "anything with `.item()`".
    A duck-typed look-alike must still fail serialisation like any other unknown type."""

    class _FakeNumpyLike:
        def item(self) -> int:
            return 42

    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("voice.used", payload=_FakeNumpyLike())

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event"] == "event.invalid"
    assert parsed["error"] == "TypeError"
    assert parsed["fields"] == ["payload"]


def test_emit_turns_an_arbitrary_serialisation_exception_into_event_invalid() -> None:
    """E1/E2: not just `TypeError`/`ValueError` -- literally anything json's encoder raises while
    walking a field's value must be caught, since a bad sink or a hostile value could raise
    anything."""

    class _BoomingDict(dict):
        def items(self):
            raise RuntimeError("boom")

    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("voice.used", payload=_BoomingDict(a=1))

    parsed = json.loads(sink.getvalue())
    assert parsed == record
    assert parsed["event"] == "event.invalid"
    assert parsed["error"] == "RuntimeError"
    assert parsed["fields"] == ["payload"]


def test_emit_falls_back_to_the_fixed_minimal_line_if_the_fallback_itself_cant_serialise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E1: even the enriched `event.invalid` fallback must never raise. This forces that second
    serialisation to fail too (by targeting only the enriched fallback dict, not the per-field
    probes `_serializes` runs, nor the final minimal dict), and checks the line still comes out
    as the fixed `{"event_schema": 1, "event": "event.invalid", "level": "warning"}`."""
    real_dumps = json.dumps

    def flaky_dumps(obj: object, *args: object, **kwargs: object) -> str:
        if isinstance(obj, dict) and obj.get("event") == "event.invalid" and "invalid_event" in obj:
            raise RuntimeError("the fallback serialisation is broken too")
        return real_dumps(obj, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("breeze_infer.events.json.dumps", flaky_dumps)

    sink = io.StringIO()
    events = Emitter(sink=sink, clock=lambda: 0.0)

    record = events.emit("voice.used", payload=object())

    parsed = json.loads(sink.getvalue())
    assert parsed == record == {
        "event_schema": EVENT_SCHEMA,
        "event": "event.invalid",
        "level": "warning",
    }


def test_emit_drops_a_nested_emit_call_from_inside_the_sinks_write() -> None:
    """E2: a sink that itself calls back into `emit` (e.g. logging its own write failure) must
    not recurse -- the thread-local re-entry guard drops the nested call instead of writing it,
    though it still returns a record like any other call."""
    holder: dict[str, Emitter] = {}

    class _ReentrantSink(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.nested_record: dict[str, object] | None = None

        def write(self, line: str) -> int:
            if self.nested_record is None:
                self.nested_record = holder["emitter"].emit("nested.event")
            return super().write(line)

    sink = _ReentrantSink()
    events = Emitter(sink=sink, clock=lambda: 0.0)
    holder["emitter"] = events

    events.emit("outer.event")

    lines = sink.getvalue().splitlines()
    assert len(lines) == 1  # the nested call was dropped, not written
    assert json.loads(lines[0])["event"] == "outer.event"
    assert sink.nested_record is not None
    assert sink.nested_record["event"] == "nested.event"  # still returns a record, just unwritten
