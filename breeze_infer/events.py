"""Structured events: one JSON object per line, nothing else.

Telemetry must never break a request (Constitution VII), so ``emit`` distinguishes two kinds of
failure. A programming error — an empty event name, an invalid ``level``, or a field that
collides with a reserved key — raises, because the caller's code is wrong and should be fixed.
A serialisation problem in the *data* (NaN/inf, or a value ``json`` can't encode) does not raise;
it is replaced with a minimal ``event.invalid`` record so one bad field never takes the request
down with it. An ``OSError`` on write/flush (the sink is gone, e.g. a broken pipe) is swallowed
for the same reason: there is nothing better to do than drop the line.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, TextIO

EVENT_SCHEMA = 1

_ALLOWED_LEVELS = frozenset({"debug", "info", "warning", "error"})
_RESERVED_FIELD_KEYS = frozenset({"ts", "event_schema", "event", "level"})


def _json_default(value: Any) -> Any:
    """Convert values plain ``json`` can't encode, such as numpy scalars.

    Anything with an ``.item()`` method (numpy's scalar types) is unwrapped to its native Python
    value. Anything else is still unencodable, so this raises ``TypeError``, same as the default
    ``json`` behaviour, and ``emit`` falls back to an ``event.invalid`` record.
    """
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class Emitter:
    """Writes structured JSON-line events to an injected sink using an injected clock.

    The write is synchronous: a stalled sink (a slow or blocked pipe) blocks the calling thread.
    That's a known, accepted risk — a background writer isn't built now, since ``emit`` is called
    both from the event loop and from the GPU thread and a stall should be visible, not hidden
    behind a queue. The write and flush are done under a lock so calls from either thread don't
    interleave partial lines.
    """

    def __init__(self, sink: TextIO, clock: Callable[[], float]) -> None:
        self._sink = sink
        self._clock = clock
        self._lock = threading.Lock()

    def emit(self, event: str, /, *, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Write one structured event line and return the record that was written.

        Every record carries ``event_schema``, ``ts`` (seconds since the epoch from the
        injected clock), ``event``, and ``level``. ``fields`` must not use those names — that's
        a programming error and raises ``ValueError``, same as an empty ``event`` or an
        unrecognised ``level``. A field whose *value* can't be serialised (NaN/inf, or a type
        ``json`` doesn't know) does not raise; the line written is an ``event.invalid`` record
        instead, so one bad field never propagates into a request failure.
        """
        if not event:
            raise ValueError("event name must not be empty")
        if level not in _ALLOWED_LEVELS:
            raise ValueError(f"invalid level: {level!r}")
        overwritten = _RESERVED_FIELD_KEYS & fields.keys()
        if overwritten:
            raise ValueError(f"reserved event field(s): {', '.join(sorted(overwritten))}")

        record: dict[str, Any] = {
            "event_schema": EVENT_SCHEMA,
            "ts": float(self._clock()),
            "event": event,
            "level": level,
            **fields,
        }
        try:
            line = json.dumps(record, ensure_ascii=True, allow_nan=False, default=_json_default)
        except (TypeError, ValueError) as exc:
            record = {
                "event": "event.invalid",
                "invalid_event": event,
                "error": type(exc).__name__,
            }
            line = json.dumps(record, ensure_ascii=True, allow_nan=False)

        with self._lock:
            try:
                self._sink.write(line + "\n")
                self._sink.flush()
            except OSError:
                pass  # the sink is gone (e.g. a broken pipe); nothing better to do

        return record
