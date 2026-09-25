"""The `/v1/audio/speech` response: send, abort, time out, clean up (research.md R2, R3).

The route does everything that can fail with a proper status first (validation, reference
preparation, and producing the first audio chunk), then returns a `SpeechResponse`. From here
on the status is already `200`, so the only honest way to report a failure is to end the
transfer abnormally (BC-17, FR-013).

What this relies on, in Starlette 1.6.0 and uvicorn 0.52.4 (h11, ASGI spec 2.3):

- uvicorn announces spec 2.3 (`uvicorn/protocols/http/h11_impl.py:207`), so
  `StreamingResponse.__call__` runs `stream_response` in a task group next to
  `listen_for_disconnect` (`starlette/responses.py:273-280`). An `http.disconnect` cancels
  `stream_response`, and `__call__` then *returns normally*: a disconnect is recognised here
  by the stream not having finished.
- An exception in `stream_response` cancels the listener and is re-raised on its own
  (`create_collapsing_task_group`, `starlette/_utils.py:83-93`). Starlette's exception wrapper
  can't answer once the response has started and raises instead
  (`starlette/_exception_handler.py:55-56`), so it reaches uvicorn, which closes the transport
  without the `0\\r\\n\\r\\n` chunked terminator (`h11_impl.py:413-425`). That is the abort.
- A client that stays connected but stops reading parks `send()` in `flow.drain()` for ever
  (`h11_impl.py:460-461`); the per-send timeout bounds it. After a disconnect, `send()` returns
  without writing (`h11_impl.py:463-464`), so a disconnect never looks like a send failure.
- On spec 2.4 servers Starlette reports a disconnect as `ClientDisconnect` instead
  (`starlette/responses.py:267-271`); that is handled too, although uvicorn doesn't use it.

Never wrap the app in `BaseHTTPMiddleware`: it runs the response in its own task and would
change all of the above.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Protocol

from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from breeze_infer.gpu import GpuCloseTimeout, GpuSession
from breeze_infer.limits import HTTP_SEND_TIMEOUT_SECONDS

BYTES_PER_SAMPLE = 2  # s16le mono


class Events(Protocol):
    def emit(self, event: str, /, *, level: str = "info", **fields: Any) -> Any: ...


class SendTimeout(Exception):
    """One `send()` took longer than the send timeout: the client stopped reading."""


# Cleanup tasks still running after the request task that started them was cancelled. The
# event loop only keeps weak references to tasks, so without this one could be collected
# half-way through closing the generator.
_cleanups: set[asyncio.Task[None]] = set()


class SpeechResponse(StreamingResponse):
    """A streamed `200` of PCM whose end always closes the generation and releases the GPU.

    - `first_chunk`: PCM the route already produced before deciding on `200` (BC-17).
    - `body`: an async generator of the remaining PCM chunks. It is closed with `aclose()`
      when streaming stops for any reason; Starlette itself never closes it.
    - `session`: the `GpuSession` feeding `body`. `__call__` closes it (generator closed on
      the GPU thread, then the gate released) whatever happens, even when cancelled.
    - `clock` and `started_at`: a monotonic clock and its reading when generation started,
      for the real-time factor in `speech.completed`.

    Exactly one of `speech.completed`, `speech.aborted` or `speech.failed` is emitted, after
    the session is closed.
    """

    media_type = "audio/pcm"

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
        send_timeout: float = HTTP_SEND_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(body, headers=headers)
        self._first_chunk = first_chunk
        self._body = body
        self._session = session
        self._events = events
        self._request_id = request_id
        self._sample_rate = sample_rate
        self._clock = clock
        self._started_at = started_at
        self._send_timeout = send_timeout
        self._bytes_sent = 0
        self._finished = False  # set once the final, terminating send went out

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
            await self._send_chunk(send, self._first_chunk)
            async for chunk in self._body:
                await self._send_chunk(send, chunk)
            await self._send(send, {"type": "http.response.body", "body": b"", "more_body": False})
            self._finished = True
        finally:
            await self._body.aclose()

    async def _send_chunk(self, send: Send, chunk: bytes) -> None:
        await self._send(send, {"type": "http.response.body", "body": chunk, "more_body": True})
        self._bytes_sent += len(chunk)

    async def _send(self, send: Send, message: Message) -> None:
        try:
            async with asyncio.timeout(self._send_timeout):
                await send(message)
        except TimeoutError as error:
            raise SendTimeout(f"send() blocked for over {self._send_timeout:g} s") from error

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # The outcome if nothing below says otherwise: a disconnect cancels the stream inside
        # Starlette's task group, and `super().__call__` then returns normally.
        outcome, fields = "speech.aborted", {"reason": "client_disconnect"}
        try:
            await super().__call__(scope, receive, send)
            if self._finished:
                outcome, fields = "speech.completed", {}
        except asyncio.CancelledError:
            outcome, fields = "speech.aborted", {"reason": "cancelled"}
            raise
        except ClientDisconnect:
            raise
        except SendTimeout:
            outcome, fields = "speech.aborted", {"reason": "send_timeout"}
            raise
        except Exception as error:
            # Re-raised so uvicorn drops the connection without the chunked terminator.
            outcome = "speech.failed"
            fields = {"reason": "generation_error", "error": repr(error)}
            raise
        finally:
            # In its own task, so a cancellation of this one can't interrupt it: the gate must
            # be released, and the event emitted, however the request ended.
            cleanup = asyncio.ensure_future(self._close(outcome, fields))
            _cleanups.add(cleanup)
            cleanup.add_done_callback(_cleanups.discard)
            await asyncio.shield(cleanup)

    async def _close(self, outcome: str, fields: dict[str, str]) -> None:
        try:
            await self._session.aclose()
        except GpuCloseTimeout as error:
            # The close is still running on the GPU thread and the gate stays held until it
            # finishes: later requests get `busy` rather than a GPU that is still occupied.
            outcome = "speech.failed"
            fields = {"reason": "gpu_close_timeout", "error": repr(error)}
        except Exception as error:  # noqa: BLE001 - reported as an event, never raised
            # The generator raised while closing; `aclose()` released the gate anyway.
            outcome = "speech.failed"
            fields = {"reason": "gpu_close_error", "error": repr(error)}
        self._emit(outcome, fields)

    def _emit(self, outcome: str, fields: dict[str, str]) -> None:
        audio_seconds = self._bytes_sent / BYTES_PER_SAMPLE / self._sample_rate
        extra: dict[str, Any] = {"audio_seconds": audio_seconds, **fields}
        if outcome == "speech.completed" and audio_seconds > 0:
            extra["rtf"] = (self._clock() - self._started_at) / audio_seconds
        level = "error" if outcome == "speech.failed" else "info"
        self._events.emit(outcome, level=level, request_id=self._request_id, **extra)
