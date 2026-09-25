"""Structured events: one JSON object per line, nothing else.

Telemetry must never break a request (Constitution VII), so ``emit`` distinguishes two kinds of
failure. A programming error — an empty event name, an invalid ``level``, or a field that
collides with a reserved key — raises, because the caller's code is wrong and should be fixed.
A serialisation problem in the *data* (NaN/inf, or a value ``json`` can't encode) does not raise;
it is replaced with a minimal ``event.invalid`` record so one bad field never takes the request
down with it. A broken sink on write/flush (a gone pipe, or a stream already closed) raises
``OSError`` or ``ValueError`` respectively, and both are swallowed for the same reason: there is
nothing better to do than drop the line.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, TextIO

import numpy as np

EVENT_SCHEMA = 1

_ALLOWED_LEVELS = frozenset({"debug", "info", "warning", "error"})
# "level" is deliberately not in here: it's keyword-only in `emit`'s signature, so a `**fields`
# entry named "level" can never collide with it in the first place.
_RESERVED_FIELD_KEYS = frozenset({"ts", "event_schema", "event"})
# Carried into the `event.invalid` fallback record when present, so a malformed event can still
# be traced back to the request/session/piece that produced it.
_INVALID_FALLBACK_CONTEXT_KEYS = ("request_id", "session_id", "piece_index")


def _json_default(value: Any) -> Any:
    """Unwrap a numpy scalar (e.g. ``np.float32``, ``np.int64``) to its native Python value.

    Restricted to ``np.generic`` rather than "anything with an ``.item()`` method": that
    duck-typed check also matched unrelated types that happen to define ``.item()``, silently
    coercing values that should instead have failed serialisation (and fallen back to
    ``event.invalid``). numpy is already a hard dependency of this project, so a plain import is
    fine here.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class Emitter:
    """Writes structured JSON-line events to an injected sink using an injected clock.

    The write is synchronous: a stalled sink (a slow or blocked pipe) blocks the calling thread.
    That's a known, accepted risk — a background writer isn't built now, since ``emit`` is called
    both from the event loop and from the GPU thread and a stall should be visible, not hidden
    behind a queue. Everything from reading the clock through writing the line is done under one
    lock, so calls from either thread neither interleave partial lines nor race the clock into
    out-of-order timestamps. It's an ``RLock`` (not a plain ``Lock``) because a broken sink's own
    ``write``/``flush`` could plausibly re-enter ``emit`` (e.g. a sink that logs its own failure);
    a plain lock would deadlock the same thread in that case.
    """

    def __init__(self, sink: TextIO, clock: Callable[[], float]) -> None:
        self._sink = sink
        self._clock = clock
        self._lock = threading.RLock()

    def emit(self, event: str, /, *, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Write one structured event line and return the record that was written.

        Every record carries ``event_schema``, ``ts`` (seconds since the epoch from the
        injected clock), ``event``, and ``level``. ``fields`` must not use those names — that's
        a programming error and raises ``ValueError``, same as an empty ``event`` or an
        unrecognised ``level``. A field whose *value* can't be serialised (NaN/inf, or a type
        ``json`` doesn't know, or anything else that blows up during serialisation) does not
        raise; the line written is an ``event.invalid`` record instead, so one bad field never
        propagates into a request failure.
        """
        if not event:
            raise ValueError("event name must not be empty")
        if level not in _ALLOWED_LEVELS:
            raise ValueError(f"invalid level: {level!r}")
        overwritten = _RESERVED_FIELD_KEYS & fields.keys()
        if overwritten:
            raise ValueError(f"reserved event field(s): {', '.join(sorted(overwritten))}")

        with self._lock:
            # Read inside the lock (not before it): two threads racing to emit must not have
            # their `ts` values assigned in a different order than the lines end up written in.
            ts = float(self._clock())
            record: dict[str, Any] = {
                "event_schema": EVENT_SCHEMA,
                "ts": ts,
                "event": event,
                "level": level,
                **fields,
            }
            try:
                line = json.dumps(
                    record, ensure_ascii=True, allow_nan=False, default=_json_default
                )
            except Exception as exc:  # noqa: BLE001 - any serialisation failure becomes event.invalid
                # Anything at all — not just TypeError/ValueError — becomes event.invalid:
                # a RecursionError from a self-referential field is exactly the kind of bad data
                # this fallback exists for, same as NaN or an unencodable type.
                record = {
                    "event_schema": EVENT_SCHEMA,
                    "ts": ts,
                    "event": "event.invalid",
                    "level": "warning",
                    "invalid_event": event,
                    "error": type(exc).__name__,
                    "fields": sorted(fields.keys()),
                }
                for key in _INVALID_FALLBACK_CONTEXT_KEYS:
                    if key in fields:
                        record[key] = fields[key]
                line = json.dumps(
                    record, ensure_ascii=True, allow_nan=False, default=_json_default
                )

            try:
                self._sink.write(line + "\n")
                self._sink.flush()
            except (OSError, ValueError):
                # OSError: the sink is gone (e.g. a broken pipe). ValueError: the sink is a
                # stream that's already been closed (`io` raises ValueError, not OSError, for
                # "I/O operation on closed file"). Either way there is nothing better to do than
                # drop the line.
                pass

        return record
