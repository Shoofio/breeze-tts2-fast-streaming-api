"""Structured events: one JSON object per line, nothing else.

Telemetry must never break a request (this module's own rule), so ``emit`` distinguishes two
kinds of failure. A programming error — an empty event name, an invalid ``level``, or a field that
collides with a reserved key — raises, because the caller's code is wrong and should be fixed.
A serialisation problem in the *data* (NaN/inf, a circular reference, or a value ``json`` can't
encode) does not raise; the offending field(s) are dropped and replaced with a minimal
``event.invalid`` record so one bad field never takes the request down with it. Nothing past
this point is allowed to raise either: a broken sink on write/flush (a gone pipe, a stream
already closed, or anything else) is swallowed, and a sink that itself calls back into ``emit``
while handling that failure is dropped rather than recursed into.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, TextIO

import numpy as np

EVENT_SCHEMA = 1

_ALLOWED_LEVELS = frozenset({"debug", "info", "warning", "error"})
# "level" is deliberately not in here: it's keyword-only in `emit`'s signature, so a `level=`
# argument -- even one arriving via a `**some_dict` splat at the call site, which is still just
# an ordinary keyword argument as far as Python's binding is concerned -- binds to that
# parameter, never into `**fields`, and so can never collide with a reserved-key check on
# `fields`.
_RESERVED_FIELD_KEYS = frozenset({"ts", "event_schema", "event"})
# Carried into the `event.invalid` fallback record when present *and* individually serialisable,
# so a malformed event can still be traced back to the request/session/piece that produced it.
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


def _serializes(value: Any) -> bool:
    """Whether `value` alone can be JSON-encoded with the exact settings `emit` uses.

    Used to decide, field by field, what a broken record's ``event.invalid`` fallback can
    safely carry: a value that fails this check is left out rather than risking a second
    serialisation failure while building the fallback itself.
    """
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False, default=_json_default)
    except Exception:  # noqa: BLE001 - literally any failure means "can't be serialised"
        return False
    return True


class Emitter:
    """Writes structured JSON-line events to an injected sink using an injected clock.

    The write is synchronous: a stalled sink (a slow or blocked pipe) blocks the calling thread.
    That's a known, accepted risk — a background writer isn't built now, since ``emit`` is called
    both from the event loop and from the GPU thread and a stall should be visible, not hidden
    behind a queue. Everything from reading the clock through writing the line is done under one
    lock, so calls from either thread neither interleave partial lines nor race the clock into
    out-of-order timestamps. The trade-off is that a large or slow-to-serialise payload holds the
    lock for as long as that takes, blocking any other thread's ``emit`` meanwhile; accepted
    because events are small and keeping their on-disk order matching their ``ts`` order is worth
    more than shortening that window. It's an ``RLock`` (not a plain ``Lock``) because a broken
    sink's own ``write``/``flush`` could plausibly re-enter ``emit`` (e.g. a sink that logs its
    own failure); a plain lock would deadlock the same thread in that case. A thread-local flag
    additionally guards against that same re-entrant call recursing into another write: it's
    dropped (still returns a record, just never reaches the sink) instead.
    """

    def __init__(self, sink: TextIO, clock: Callable[[], float]) -> None:
        self._sink = sink
        self._clock = clock
        self._lock = threading.RLock()
        self._reentry = threading.local()

    def emit(self, event: str, /, *, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Write one structured event line and return the record that was written.

        Every record carries ``event_schema``, ``ts`` (seconds since the epoch from the
        injected clock), ``event``, and ``level``. ``fields`` must not use those names — that's
        a programming error and raises ``ValueError``, same as an empty ``event`` or an
        unrecognised ``level``.

        A field whose *value* can't be serialised (NaN/inf, a circular reference, a type
        ``json`` doesn't know, or anything else that blows up during serialisation) does not
        raise. Instead, the line written is a minimal ``event.invalid`` record: ``level`` becomes
        ``"warning"``, ``fields`` lists the names of only the fields that failed to serialise on
        their own, and ``ts``/``request_id``/``session_id``/``piece_index`` are carried over only
        if each is, individually, still serialisable. If even that fallback record somehow can't
        be serialised, the line falls back once more to a fixed minimal
        ``{"event_schema": 1, "event": "event.invalid", "level": "warning"}`` — this method never
        raises past the initial validation above, no matter what ``fields`` contains.
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
            except Exception as exc:  # noqa: BLE001 - any failure at all becomes event.invalid
                # Not just TypeError/ValueError: a circular reference (e.g. a dict containing
                # itself) raises ValueError too ("Circular reference detected"), but anything
                # else json's encoder or `_json_default` might throw is exactly the kind of bad
                # data this fallback exists for.
                record = {
                    "event_schema": EVENT_SCHEMA,
                    "event": "event.invalid",
                    "level": "warning",
                    "invalid_event": event,
                    "error": type(exc).__name__,
                    "fields": sorted(key for key, value in fields.items() if not _serializes(value)),
                }
                if _serializes(ts):
                    record["ts"] = ts
                for key in _INVALID_FALLBACK_CONTEXT_KEYS:
                    if key in fields and _serializes(fields[key]):
                        record[key] = fields[key]
                try:
                    line = json.dumps(
                        record, ensure_ascii=True, allow_nan=False, default=_json_default
                    )
                except Exception:  # noqa: BLE001 - the fallback itself must never raise either
                    record = {
                        "event_schema": EVENT_SCHEMA,
                        "event": "event.invalid",
                        "level": "warning",
                    }
                    line = json.dumps(record, ensure_ascii=True, allow_nan=False)

            if getattr(self._reentry, "writing", False):
                # A nested `emit` call from inside the sink's own write/flush (e.g. a sink that
                # logs its own failure by calling `emit` again): drop it instead of recursing.
                return record
            self._reentry.writing = True
            try:
                self._sink.write(line + "\n")
                self._sink.flush()
            except Exception:  # noqa: BLE001, S110 - a broken sink (gone pipe, closed stream,
                # or anything else) must never propagate; there is nothing better to do than
                # drop the line.
                pass
            finally:
                self._reentry.writing = False

        return record
