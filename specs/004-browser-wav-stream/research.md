# Research: Browser-Playable WAV Streaming

**Feature**: [spec.md](spec.md) | **Date**: 2026-09-30

No browser spike was run: the user decided to skip it (spec Clarifications). Browser behaviour is
checked at the live gate ([quickstart.md](quickstart.md)). The decisions below come from reading
the code.

## R1. Reuse the POST route's pipeline instead of writing a second one

- **Decision**: `_serve_speech` (`breeze_infer/routes_speech.py`) serves both routes. It gains two
  keyword parameters, one for each step that differs:
  - **How to get the GPU**: `gpu_wait` is `None` for the POST route (`try_acquire`, `409 busy`), and
    60 s for the GET route (a bounded wait, see R3).
  - **The format of the response**: `wav` (a bool) selects the WAV header, `audio/wav` and buffered
    delivery (R4).
- **Rationale**: Everything else is identical: field parsing, splitting, voice lookup, sizing,
  reference resolution, priming the first chunk before `200`, and the cleanup paths. That code has
  been through many review rounds, and a copy would drift from it. There are exactly two callers, so
  parameters are enough; no new abstraction is needed (constitution II).
- **Alternatives considered**:
  - A separate route module: rejected, it would duplicate about 150 lines of subtle lease handling.
  - A strategy object for "acquire" and "respond": rejected, it's an abstraction for two cases that
    two keyword arguments already express.

## R2. Query-string input: the existing parser already copes

- **Decision**: Use no new parsing code. For a GET with no body, `read_fields` reads the query
  string alone. It already rejects `ref_audio` given in a query (`400 invalid_field`) and a field
  repeated in the query (`400 duplicate_field`).
- **Request-line size**: set `h11_max_incomplete_event_size=128 * 1024` in `uvicorn.Config`
  (`breeze_infer/api.py`), next to the existing `http="h11"`. The default is 16,384 bytes (checked:
  h11 0.16.0, uvicorn 0.52.4). 10,000 CJK characters come to 90,000 bytes percent-encoded. Anything
  larger is refused by h11 with `400` before the app runs.
- **Alternative considered**: Raise the limit only for this route. Rejected: h11 applies it per
  connection, before routing, so it can't be per route. The cost of 128 KiB of buffered headers per
  connection is negligible.

## R3. A bounded wait that notices a client disconnect

- **Decision**: The GET route calls `gate.acquire()` inside `asyncio.timeout(60)`. It runs a second
  task alongside, which waits on `http_request.receive()` for `http.disconnect`, and the first of the
  two to finish wins.
  - **Disconnect**: the acquire is cancelled. `GpuGate.acquire` already passes on a lease handed
    over in that same instant. The route ends with `speech.aborted reason=client_disconnect
    queued=true`.
  - **Timeout**: `503 busy_timeout`, and a `speech.queued_timeout` event.
- **Rationale**: For a GET, the body has already been consumed by `read_fields`. So the only message
  `receive()` can still deliver is `http.disconnect`, and uvicorn resumes reading to detect it. That
  is the same mechanism Starlette's `listen_for_disconnect` uses once streaming starts.
- **Wait bound**: 60 s, a constant in `limits.py`, passed in when the route is installed so tests can
  shorten it (constitution III).
- **Alternative considered**: Send `200` and the WAV header at once, then wait. Rejected: a later
  failure (unknown voice after all, no room, timeout) could then only end the body, never return a
  status. The POST route's "hold headers until the first chunk" rule (BC-17) is kept.

## R4. Buffered delivery built from the existing `SpeechResponse`

- **Decision**: No new response class. The GET route builds the existing `SpeechResponse`
  (`breeze_infer/streaming.py`) with the following:
  1. **`body`**: a small async generator, `_buffered(session)`. On its first iteration it starts a
     producer task that runs `session.step()` at full speed, appending each chunk to an unbounded
     `asyncio.Queue`.
     - If `step()` raises (a generation error, or `NoRoomError` on a later piece), the producer
       first closes the session, releasing the GPU at once rather than after the client has
       drained the buffer (Phase 4 review), then puts the exception in the queue. The body generator re-raises it, so it ends the stream as
       `speech.failed`, exactly as the POST route does today.
     - At `DONE`, the producer calls `session.aclose()`. That closes the generator on the GPU thread
       and releases the gate, and the producer then emits `speech.generated`. If that close fails,
       the producer queues the error instead (and emits no `speech.generated`).
     - The body generator yields chunks from the queue.
     - In its `finally`, it cancels and awaits the producer.
  2. **`send_timeout=600`** and **`min_rate_grace=math.inf`**. The minimum rate never trips, and a
     single send blocked for 10 minutes aborts with the existing reason `send_timeout`.
  3. **`media_type="audio/wav"`**: a new constructor parameter, defaulting to `audio/pcm`, so the
     POST route is unchanged. Plus `headers={"Accept-Ranges": "none"}`.
  4. **`first_chunk`**: the 44-byte WAV header followed by the primed first chunk.
- **Why this is enough**:
  - `GpuSession.aclose()` is idempotent: "Repeated and concurrent calls all wait for the same
    close". So `SpeechResponse`'s own `finally` can still close the session and report exactly one
    outcome.
  - A close failure at `DONE` goes through the stream rather than being left to that final close.
    After a close timeout, the final close waits afresh and may succeed, which would report a
    success on a poisoned gate. The request ends as `speech.failed` with
    `reason=gpu_close_timeout`, as on the POST route (Phase 4 review).
  - When the client disconnects mid-generation, cancelling the producer cancels a `session.step()`.
    That is exactly what happens on the POST route today, so "GPU freed within one chunk" holds by
    the same path.
- **Accepted imprecision**: the 44 header bytes count toward `audio_seconds_sent`. That is 22
  samples, under 1 ms, and not worth a second send path.
- **Memory**: an unbounded queue per stream. The sizes are in the spec's Assumptions (worst case
  about 100–115 MB for 10,000 CJK characters). No cap, as decided.
  - The 600 s send timeout ends a reader that *stops*, not one that keeps reading slowly. A client
    that reads a few bytes every 10 minutes can hold its buffer indefinitely (Phase 4 review,
    finding 3).
  - This is accepted with the no-cap decision. Only a deliberately trickling client does it, and
    it spends memory, never the GPU.
  - If it matters later, the cheapest bound is a total response deadline, e.g. a multiple of the
    audio's duration.
- **Alternative considered**: A new response class that owns its own buffer and timers. Rejected:
  it would duplicate `SpeechResponse`'s outcome bookkeeping, finalizer and cancellation shielding,
  the most carefully reviewed code in the server.

## R5. WAV header

- **Decision**: a standard 44-byte `RIFF`/`WAVE` header:
  - `fmt ` chunk: PCM (1), 1 channel, the loaded model's rate, 16 bits.
  - The RIFF size and the `data` size are both `0xFFFFFFFF`.
  - Built with `struct.pack` in a small pure function next to the route, and unit-tested
    byte-for-byte.
- **Rationale**: `0xFFFFFFFF` is the usual "unknown length" convention for streamed WAV. Whether
  browsers accept it is checked at the live gate (spec Assumptions).

## R6. Version and rollout

- **Decision**: Development builds are `2.1.0.devN`, and the release is `2.1.0`.
  - `breeze_infer/__init__.py` holds the version.
  - The changes are recorded in the CHANGELOG under "Unreleased → Added".
  - The route is additive, so there is no feature flag.
- **Rollback**: restart on the `v2.0.0` tag. The extension sees `X-Breeze-Version` `2.0.0` and
  falls back to the WebSocket.

## R7. The kernel's TCP_USER_TIMEOUT must outlast the WAV send timeout (found in T007)

- **Finding**: Linux applies `TCP_USER_TIMEOUT` to a peer that keeps a zero receive window, not
  only to one that stopped acknowledging. With 003's 30 s (R5 there), a browser that stops reading
  once it is far enough ahead is reset about 30 s after the socket buffers fill, long before
  FR-016's 600 s. The T007 harness showed it at its 2 s setting as
  `httpx.ReadError: [Errno 104] Connection reset by peer`.
- **Decision (user, 2026-09-30)**: set `TCP_USER_TIMEOUT_MS = WAV_SEND_TIMEOUT_SECONDS * 1000`
  (600 s) for every route.
- **Rationale**:
  - The option only frees a socket whose peer is gone. The GPU is released by application
    timeouts: the POST route's 30 s send timeout, the WebSocket stall timer, and the WAV
    producer.
  - Windows production has no such option at all.
  - The cost is that a vanished peer's socket lingers up to 10 minutes on Linux.
- **Rejected alternatives**:
  - Keeping 30 s and amending FR-016: this breaks the feature on WSL and in Docker.
  - A per-connection timeout: ASGI doesn't expose the socket.
