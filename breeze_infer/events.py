"""Structured events: one JSON object per line, nothing else.

Ported from ``A:breeze_infer/events.py`` (see specs/003-cpp-compatible-api/research.md R10).
The free ``emit(event, *, sink=None, clock=time.time, **fields)`` function from that version
defaulted the sink to ``sys.stdout`` and the clock to ``time.time`` when the caller omitted them.
Constitution III ("explicit dependencies") requires the clock and output stream to be injected,
not defaulted to real I/O inside the module, so those defaults are dropped here: both are
constructor arguments with no fallback. An ``Emitter`` is built once at the composition root
(with the real ``sys.stdout`` and ``time.time``) and passed down to whatever needs to emit
events; nothing in this module holds a global emitter or any other module-level state.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TextIO

EVENT_SCHEMA = 1


class Emitter:
    """Writes structured JSON-line events to an injected sink using an injected clock."""

    def __init__(self, sink: TextIO, clock: Callable[[], float]) -> None:
        self._sink = sink
        self._clock = clock

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one structured event line and return the record that was written.

        Every record carries ``event_schema``, ``ts`` (seconds since the epoch from the
        injected clock), and ``event``. Fields must be JSON serialisable; anything else
        raises ``TypeError`` rather than being coerced to a string.
        """
        if not event:
            raise ValueError("event name must not be empty")
        record: dict[str, Any] = {
            "event_schema": EVENT_SCHEMA,
            "ts": float(self._clock()),
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._sink.write(line + "\n")
        self._sink.flush()
        return record
