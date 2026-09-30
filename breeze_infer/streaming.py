"""The `/v1/audio/speech` response: send, abort, time out, clean up (research.md R2, R3).

The route does everything that can fail with a proper status first (validation, reference
preparation, and producing the first audio chunk), then returns a `SpeechResponse`. From here
on the status is already `200`, so the only honest way to report a failure is to end the
transfer abnormally (BC-17, FR-013).

What this relies on, in Starlette 1.6.0 and uvicorn 0.52.4 (h11, ASGI spec 2.3):

- uvicorn announces spec 2.3 (`uvicorn/protocols/http/h11_impl.py:207`), so
  `StreamingResponse.__call__` runs `stream_response` in a task group next to
  `listen_for_disconnect` (`starlette/responses.py:273-280`). An `http.disconnect` cancels
  `stream_response`, and `__call__` then *returns normally*. The listener is overridden here
  to set a flag, so a disconnect is never mistaken for a completed stream.
- An exception in `stream_response` cancels the listener and is re-raised on its own
  (`create_collapsing_task_group`, `starlette/_utils.py:83-93`). `SpeechResponse` turns it into
  `StreamAborted`. No handler matches that in the exception wrapper, which re-raises it
  (`starlette/_exception_handler.py:52-53`); `ServerErrorMiddleware` then calls the app's
  `Exception` handler but can't send its response, since ours has started, and re-raises
  (`starlette/middleware/errors.py:163-186`). That handler recognises `StreamAborted` and
  stays quiet: the outcome event was already emitted here. uvicorn finally closes the
  transport without the `0\\r\\n\\r\\n` chunked terminator (`h11_impl.py:414-425`). That is
  the abort.
- A client that stays connected but stops reading parks `send()` in `flow.drain()` for ever
  (`h11_impl.py:461-462`); the send timeout and the minimum delivery rate bound it. After a
  disconnect, `send()` returns without writing (`h11_impl.py:464-465`), so a disconnect never
  looks like a send failure.
- On spec 2.4 servers Starlette reports a disconnect as `ClientDisconnect` instead
  (`starlette/responses.py:267-271`); that is handled too, although uvicorn doesn't use it.

Limits, both because ASGI gives no way to reach the transport:

- After a send timeout the connection is *closed*, not aborted: uvicorn's `transport.close()`
  keeps the socket until its write buffer drains, which a stalled reader never does.
- A send timeout on `http.response.start` itself (possible on a keep-alive connection whose
  previous response the client hasn't read) happens before uvicorn marks the response as
  started, so uvicorn answers the exception with its own 500 (`h11_impl.py:422-423`), whose
  `send()` waits in the same drain. Returning without an exception ends the same way
  (`h11_impl.py:431-434`); no ASGI-level choice avoids it.

In both cases the GPU is already released and the outcome already reported. The socket is
evicted by the kernel through `TCP_USER_TIMEOUT` (30 s, set on the listening socket in
`api.bind_http_sockets`), and at shutdown by uvicorn's bounded graceful timeout.

Aborts the client caused (a disconnect, a send timeout, a reader below the delivery floor)
would each log uvicorn's "Exception in ASGI application" traceback. `ClientAbortLogFilter`
drops exactly those records; the composition root installs it on `uvicorn.error`. A
generation failure still logs its traceback.

Never wrap the app in `BaseHTTPMiddleware`: it runs the response in its own task and would
change all of the above. Routes returning a `SpeechResponse` must not use FastAPI `yield`
dependencies either: their exit code runs between building the response and calling it, and
a failure there means `__call__` never runs (the `weakref.finalize` fallback below then
releases the GPU, but only when the response is garbage-collected).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Protocol

from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from breeze_infer.errors import StreamAborted
from breeze_infer.gpu import GpuCloseTimeout, GpuSession
from breeze_infer.limits import (
    HTTP_SEND_TIMEOUT_SECONDS,
    MIN_RATE_GRACE_SECONDS,
    MIN_RATE_REAL_TIME,
)

BYTES_PER_SAMPLE = 2  # s16le mono

# Loop iterations between a peer closing the socket and the disconnect listener having run:
# the selector reports the EOF and the transport schedules `connection_lost`; that sets
# uvicorn's message event; the listener task wakes on it.
_DISCONNECT_HOPS = 3


class Events(Protocol):
    def emit(self, event: str, /, *, level: str = "info", **fields: Any) -> Any: ...


class SendTimeout(Exception):
    """The client isn't reading fast enough. `reason` is `send_timeout` (one send blocked for
    too long) or `too_slow` (below the minimum delivery rate)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class ClientAbortLogFilter(logging.Filter):
    """Drops uvicorn's "Exception in ASGI application" record for a stream the client ended.

    Those aborts are the client's doing and `speech.aborted` already records each one; a
    traceback per stalled or departed client would bury real errors. A `StreamAborted` caused
    by anything else (a generation failure) still logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        client_caused = isinstance(error, StreamAborted) and isinstance(
            error.__cause__, SendTimeout | ClientDisconnect
        )
        return not client_caused


# Cleanup tasks that must finish even if nobody awaits them any more (a cancelled request, or
# a response that was never sent). The event loop only keeps weak references to tasks, so
# without this one could be collected half-way through closing the generator.
_cleanups: set[asyncio.Task[None]] = set()


def _keep_running(task: asyncio.Task[None]) -> asyncio.Task[None]:
    _cleanups.add(task)
    task.add_done_callback(_cleanups.discard)
    return task


class _Outcome:
    """The one outcome event of a response, shared by `__call__` and the finalizer so that
    neither can report a second one. Holds what the event needs, never the response itself
    (the finalizer runs after the response is gone)."""

    def __init__(
        self,
        events: Events,
        request_id: str,
        sample_rate: int,
        clock: Callable[[], float],
        started_at: float,
        event_fields: Mapping[str, str],
    ) -> None:
        self._events = events
        self._request_id = request_id
        self._sample_rate = sample_rate
        self._clock = clock
        self._started_at = started_at
        self._event_fields = event_fields
        self._reported = False
        self.bytes_sent = 0

    def audio_seconds_sent(self) -> float:
        return self.bytes_sent / BYTES_PER_SAMPLE / self._sample_rate

    def report(self, outcome: str, fields: dict[str, str]) -> None:
        if self._reported:
            return
        self._reported = True
        audio_seconds_sent = self.audio_seconds_sent()
        extra: dict[str, Any] = {
            **self._event_fields,
            "audio_seconds_sent": audio_seconds_sent,
            **fields,
        }
        if outcome == "speech.completed" and audio_seconds_sent > 0:
            extra["rtf"] = (self._clock() - self._started_at) / audio_seconds_sent
        level = "error" if outcome == "speech.failed" else "info"
        self._events.emit(outcome, level=level, request_id=self._request_id, **extra)


async def _close_and_report(
    session: GpuSession[Any], outcome: _Outcome, event: str, fields: dict[str, str]
) -> None:
    """Close the session (generator on the GPU thread, then release), then report `event`,
    or `speech.failed` if the close itself failed."""
    try:
        await session.aclose()
    except GpuCloseTimeout as error:
        # The close is still running on the GPU thread, so the gate stays held; see gpu.py.
        event, fields = "speech.failed", {"reason": "gpu_close_timeout", "error": repr(error)}
    except Exception as error:  # noqa: BLE001 - reported as an event, never raised
        # The generator raised while closing; `aclose()` released the gate anyway.
        event, fields = "speech.failed", {"reason": "gpu_close_error", "error": repr(error)}
    outcome.report(event, fields)


def _close_unsent(
    loop: asyncio.AbstractEventLoop, session: GpuSession[Any], outcome: _Outcome
) -> None:
    """`weakref.finalize` callback for a response dropped without being called. It can run on
    any thread (wherever the last reference went), so it only schedules the close."""
    with contextlib.suppress(RuntimeError):  # the loop is closed: nothing can run it any more
        loop.call_soon_threadsafe(
            lambda: _keep_running(
                loop.create_task(
                    _close_and_report(session, outcome, "speech.aborted", {"reason": "not_sent"})
                )
            )
        )


class SpeechResponse(StreamingResponse):
    """A streamed `200` of PCM whose end always closes the generation and releases the GPU.

    - `first_chunk`: PCM the route already produced before deciding on `200` (BC-17).
    - `body`: an async generator of the remaining PCM chunks. It is closed with `aclose()`
      when streaming stops for any reason; Starlette itself never closes it.
    - `session`: the `GpuSession` feeding `body`. `__call__` closes it (generator closed on
      the GPU thread, then the gate released) whatever happens, even when cancelled. If the
      response is dropped without ever being called, a finalizer closes it instead.
    - `sample_rate`: sets the contract headers and turns bytes into audio seconds. `headers`
      adds to the contract headers (and `Content-Type`); it can't replace them, and must not
      carry `Content-Length` (the body is streamed).
    - `media_type`: the `Content-Type` of the stream (`audio/pcm` unless the body carries a
      container, such as WAV).
    - `event_fields`: extra fields for the one outcome event; the outcome's own fields win on
      a clash.
    - `clock` and `started_at`: a monotonic clock and its reading when generation started,
      for the real-time factor in `speech.completed`. The clock also times each audio send
      for the minimum delivery rate (`min_rate_grace`, `min_rate`; see `limits.py`).

    Exactly one of `speech.completed`, `speech.aborted` or `speech.failed` is emitted, after
    the session is closed. Its `audio_seconds_sent` counts audio handed to the server's
    transport while the client was still connected, not audio the client is known to have
    received. Build the response on the event loop that will serve it.
    """

    def __init__(
        self,
        *,
        first_chunk: bytes,
        body: AsyncGenerator[bytes, None],
        session: GpuSession[Any],
        events: Events,
        request_id: str,
        sample_rate: int,
        clock: Callable[[], float],
        started_at: float,
        headers: Mapping[str, str] | None = None,
        media_type: str = "audio/pcm",
        event_fields: Mapping[str, str] | None = None,
        send_timeout: float = HTTP_SEND_TIMEOUT_SECONDS,
        min_rate_grace: float = MIN_RATE_GRACE_SECONDS,
        min_rate: float = MIN_RATE_REAL_TIME,
    ) -> None:
        extra = {name.lower(): value for name, value in (headers or {}).items()}
        if "content-length" in extra:
            raise ValueError("a streamed speech response can't carry Content-Length")
        contract = {
            "content-type": media_type,
            "x-sample-rate": str(sample_rate),
            "x-sample-format": "s16le",
            "cache-control": "no-store",
        }
        super().__init__(body, headers={**extra, **contract}, media_type=media_type)
        self._first_chunk = first_chunk
        self._body = body
        self._session = session
        self._send_timeout = send_timeout
        self._min_rate_grace = min_rate_grace
        self._min_rate = min_rate
        self._outcome = _Outcome(
            events, request_id, sample_rate, clock, started_at, event_fields or {}
        )
        self._clock = clock
        self._send_blocked = 0.0  # seconds spent in audio sends; generation time not included
        self._disconnected = False
        self._finished = False  # the terminator went out to a client still connected
        self._unsent = weakref.finalize(
            self, _close_unsent, asyncio.get_running_loop(), session, self._outcome
        )

    async def listen_for_disconnect(self, receive: Receive) -> None:
        await super().listen_for_disconnect(receive)
        self._disconnected = True

    async def stream_response(self, send: Send) -> None:
        try:
            await self._send(
                send,
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                },
            )
            await self._send_audio(send, self._first_chunk)
            async for chunk in self._body:
                await self._send_audio(send, chunk)
            await self._send(send, {"type": "http.response.body", "body": b"", "more_body": False})
            # Checked here, not after `__call__`'s task group: once the terminator is sent,
            # uvicorn answers the listener with `http.disconnect` too, so a flag set later
            # means nothing.
            self._finished = not self._disconnected
        finally:
            # The body is suspended at a `yield` or already finished, so this can't normally
            # fail; if it did, it must not replace the exception that decides the outcome.
            with contextlib.suppress(Exception):
                await self._body.aclose()

    async def _send_audio(self, send: Send, chunk: bytes) -> None:
        # Only time blocked in audio sends counts against the client: waiting for a slow GPU
        # between sends is the server's own doing and must never trip the floor.
        budget = (
            self._min_rate_grace
            + self._outcome.audio_seconds_sent() / self._min_rate
            - self._send_blocked
        )
        if budget <= 0:
            raise SendTimeout("too_slow", "the client reads below the minimum delivery rate")
        started = self._clock()
        try:
            message = {"type": "http.response.body", "body": chunk, "more_body": True}
            await self._send(send, message, too_slow_after=budget)
        finally:
            self._send_blocked += self._clock() - started
        # After a disconnect uvicorn drops sends without a word, and it takes a few loop
        # iterations for the listener to learn of one: let them run before counting. If the
        # listener does see a disconnect, this task is cancelled right here.
        for _ in range(_DISCONNECT_HOPS):
            await asyncio.sleep(0)
        if not self._disconnected:
            self._outcome.bytes_sent += len(chunk)

    async def _send(
        self, send: Send, message: Message, too_slow_after: float | None = None
    ) -> None:
        timeout, reason = self._send_timeout, "send_timeout"
        if too_slow_after is not None and too_slow_after < timeout:
            timeout, reason = too_slow_after, "too_slow"
        try:
            async with asyncio.timeout(timeout):
                await send(message)
        except TimeoutError as error:
            raise SendTimeout(reason, f"send() blocked for {timeout:g} s ({reason})") from error

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self._unsent.detach()  # from here on, the `finally` below owns the close
        # The outcome if nothing below says otherwise: a disconnect cancels the stream inside
        # Starlette's task group, and `super().__call__` then returns normally.
        event, fields = "speech.aborted", {"reason": "client_disconnect"}
        try:
            await super().__call__(scope, receive, send)
            if self._finished:
                event, fields = "speech.completed", {}
        except asyncio.CancelledError:
            event, fields = "speech.aborted", {"reason": "cancelled"}
            raise
        except SendTimeout as error:
            event, fields = "speech.aborted", {"reason": error.reason}
            raise StreamAborted(error.reason) from error
        except ClientDisconnect as error:
            raise StreamAborted("client_disconnect") from error
        except Exception as error:
            event = "speech.failed"
            fields = {"reason": "generation_error", "error": repr(error)}
            raise StreamAborted("generation_error") from error
        finally:
            # In its own task, so a cancellation of this one can't interrupt it: the gate must
            # be released, and the event emitted, however the request ended.
            closing = _close_and_report(self._session, self._outcome, event, fields)
            await asyncio.shield(_keep_running(asyncio.ensure_future(closing)))
