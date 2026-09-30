# Feature Specification: Browser-Playable WAV Streaming

**Feature Branch**: `004-browser-wav-stream`

**Created**: 2026-09-30

**Status**: Draft

**Input**: User description: "Feature 004-browser-wav-stream. A single GET `/v1/audio/speech.wav`
route with query-string fields (same parsing and validation as `POST /v1/audio/speech`, `voice_id`
or no reference), a progressive WAV (44-byte header with `0xFFFFFFFF` sizes, then s16le PCM), a
raised request-line limit (128 KiB), a 60 s bounded GPU queue wait then `503 busy_timeout`,
buffered delivery that releases the GPU when generation ends and has a 10-min idle timeout (this
route only), no cross-site guard (risk accepted and documented), feature detection via
`X-Breeze-Version >= 2.1.0`, `/health` unchanged, POST route and WS unchanged, version 2.1.0.
Record that the browser spike was skipped by user decision, and include the live acceptance checks
(Chrome and Firefox, >3 min message at 0.5x/0.9x/1x, WS preview during a long drain)."

**Origin**: a draft from the SillyTavern extension's agent (`st-agent`, 2026-09-30). It proposed a
two-step design: a POST creates a job, then a GET streams it. The user chose a single GET instead,
and `st-agent` reviewed the revised design with no objection.

## Clarifications

### Session 2026-09-30

- Q: Single GET or two steps (POST a job, then GET it)? → A: **Single GET**.
  - It needs no server-side job state, and a URL never expires. SillyTavern creates every segment's
    URL before playback starts, so a job expiry would drop later segments.
  - A browser re-request simply restarts synthesis.
  - The draft's arguments for two steps are weaker here than stated: access logging is off, and
    media requests don't enter browser history.
  - Costs accepted: the text travels in the URL, and invalid input reaches an `<audio>` element only
    as a bare media error.
- Q: Block other websites from triggering synthesis through this route? → A: **No guard**.
  - An `<audio>` GET carries no `Origin`, and SillyTavern sends `Referrer-Policy: no-referrer`, so
    no reliable guard can admit SillyTavern in every host setup.
  - The risk, that any page the user visits can spend GPU time, is accepted and documented.
- Q: A browser spike before the spec (re-requests, holding headers, read-ahead stalls)? → A:
  **Skipped by user decision**. The live acceptance checks cover these instead (SC-002 to SC-005).
- Q: How should the route deal with a browser that reads slowly or pauses reading? → A:
  **Separate generation from delivery** (`st-agent`'s proposal).
  - Generation runs at full speed, and the GPU is released as soon as it ends.
  - The client drains the buffered audio at its own pace.
  - SillyTavern's playback speed goes from 0× to 3×, and browsers stop reading while they are far
    ahead. The existing 30 s send timeout and 0.5× minimum read rate would cut normal playback.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Gapless streamed playback in the browser (Priority: P1)

A SillyTavern user has streaming TTS turned on. The extension builds a URL for each segment and
hands it to SillyTavern, which sets it as the source of its `<audio>` element. Playback starts
while synthesis is still running and plays through without gaps or clicks. SillyTavern's queue,
Stop button and playback speed all keep working.

**Why this priority**: This is the whole feature. Today's blob-per-2-s approach glitches at every
join.

**Independent Test**: Set a route URL as `src` on an `<audio>` element in Chrome and in Firefox.
Check four things: audio starts before synthesis ends, there are no audible gaps, the server records
exactly one synthesis per URL, and the stream ends cleanly.

**Acceptance Scenarios**:

1. **Given** a running server and a saved voice, **When** a client GETs
   `/v1/audio/speech.wav?text=…&voice_id=…`, **Then** it receives:
   - `200` with `Content-Type: audio/wav`, a chunked body, `Cache-Control: no-store`,
     `Accept-Ranges: none`, `X-Sample-Rate` and `X-Breeze-Version`;
   - a 44-byte mono 16-bit WAV header at the model's sample rate, with the RIFF and `data` sizes
     set to `0xFFFFFFFF`;
   - s16le PCM that starts arriving within the POST route's time to first audio for the same text.
2. **Given** the same text, voice and seed, **When** the audio from this route is compared with
   the audio from `POST /v1/audio/speech`, **Then** the PCM after the WAV header is the same
   synthesis:
   - With a deterministic test runtime, it is byte-identical.
   - On the GPU, it is within the run-to-run tolerance already used by the long-text GPU test. GPU
     kernels are not bit-reproducible, even with a fixed seed.
3. **Given** a request with a `Range: bytes=0-` header, **When** it arrives, **Then** the header is
   ignored and the full stream is served with `200` (never `206` or `416`).
4. **Given** playback in progress, **When** SillyTavern changes or clears `audio.src`, **Then** the
   server records a client-disconnect abort. If synthesis was still running, the GPU is released
   within one chunk, and the next request starts at once.

---

### User Story 2 - Slow or paused readers never hold the GPU (Priority: P1)

The user plays a long message at a slow playback speed, or the browser stops reading because it is
far ahead of playback. Synthesis still finishes at full speed and frees the GPU. The browser keeps
receiving the rest of the audio at its own pace, and nothing is cut short.

**Why this priority**: Without it, long messages are cut in normal use (speed slider below 0.5×,
browser read-ahead limits). A slow tab would also block WebSocket previews and the next segment.

**Independent Test**: Stream a message with more than 3 minutes of audio to a client that reads
slowly or not at all. Check that generation completes and releases the GPU. Check that the client
can still read every byte afterwards, and that a WebSocket preview started meanwhile runs at once.

**Acceptance Scenarios**:

1. **Given** a client that has stopped reading, **When** synthesis finishes, **Then** the GPU is
   released and a `speech.generated` event is recorded, while the response remains open.
2. **Given** a client that reads at a quarter of real time, **When** the stream runs, **Then** it
   is not aborted for reading slowly, and every generated byte is delivered.
3. **Given** a client that reads no bytes for 10 minutes, **When** that time passes, **Then** the
   stream is aborted with reason `send_timeout` and its buffered audio is freed.
4. **Given** a long drain in progress after generation has finished, **When** another client
   starts a WebSocket session, **Then** that session is not queued behind the drain.

---

### User Story 3 - Waiting for a busy GPU instead of failing (Priority: P2)

A WebSocket session or another stream is using the GPU when the browser requests a URL. The request
waits its turn, then streams. It does not fail right away, because an `<audio>` element can't
retry.

**Why this priority**: SillyTavern would hang on an immediate failure. The draft's acceptance check
5 requires this.

**Independent Test**: Hold the GPU with a WebSocket session, request a URL, then release the GPU.
The GET streams after the release. If the GPU is held beyond the bound, the GET gets `503`.

**Acceptance Scenarios**:

1. **Given** the GPU is busy, **When** a GET arrives, **Then** it waits in the same first-in,
   first-out order as other waiters and streams once it gets the GPU.
2. **Given** the GPU stays busy for 60 s after the GET arrives, **When** the bound passes, **Then**
   the GET gets `503` with code `busy_timeout`, and a `speech.queued_timeout` event is recorded.
3. **Given** a GET waiting for the GPU, **When** its client disconnects, **Then** it leaves the
   queue without ever taking the GPU.

---

### User Story 4 - Long and non-ASCII text works in a URL (Priority: P2)

A message of up to the 10,000-character text limit, in any script, fits in the request URL.

**Why this priority**: Chinese text grows about 9× when percent-encoded. The server's default
request-line limit (16 KiB) would reject it.

**Independent Test**: GET a 10,000-character Chinese text and a 10,000-character English text.
Both stream to completion.

**Acceptance Scenarios**:

1. **Given** a 10,000-character Chinese text, percent-encoded (about 90 KB), **When** it is
   requested, **Then** it streams to completion.
2. **Given** a request line longer than 128 KiB, **When** it arrives, **Then** it is rejected
   before any work is done.

---

### User Story 5 - Clients can detect the feature and see input errors (Priority: P3)

The extension decides whether to use this route or the WebSocket path. From `curl`, or from its own
checks, it can tell why a request was refused.

**Why this priority**: It provides the fallback when the server is older, or when the C++ server
has been rolled back in.

**Independent Test**: Read `X-Breeze-Version` from any response. Send invalid fields to the route
and compare the status and body with the POST route.

**Acceptance Scenarios**:

1. **Given** this server, **When** any response is read, **Then** `X-Breeze-Version` is `2.1.0` or
   later. `/health` is unchanged.
2. **Given** an unknown `voice_id`, empty text, over-long text or an out-of-range value, **When** it
   is sent to the route, **Then** the status code and `{"error","code"}` body match what
   `POST /v1/audio/speech` returns for the same fields, before any audio is sent.
3. **Given** a `ref_audio` query parameter, **When** it is sent to the route, **Then** it is
   rejected with `400 invalid_field`, as the POST route already rejects a query-string
   `ref_audio`.

---

### Edge Cases

- **Voice deleted after the extension checked it**: the GET gets `404 unknown_voice`, and the
  browser fires a media `error`.
- **The model cannot finish the text mid-stream** (for example, no context room for a later piece):
  the body ends early, as on the POST route. The browser plays what it received and then fires
  `ended` or `error`.
- **The same URL is requested twice** (browser re-request): each request is an independent
  synthesis. With the same seed it produces the same audio, and it costs GPU time again.
- **A request while the model is loading**: `503 loading`, as on every other route.
- **A client disconnects after generation has finished**: the buffered audio is freed. Nothing else
  happens.
- **Playback speed 0×**: the client reads nothing, and the 600 s send timeout ends the stream after 10
  minutes.
- **A field given twice, or in both the query string and a body**: `400`, as on the POST route.
- **A GET with a request body**: the body is not read for fields. Fields come from the query string
  only.

## Requirements *(mandatory)*

### Functional Requirements

**Route and input**

- **FR-001**: The server MUST expose `GET /v1/audio/speech.wav`. Every other endpoint, including
  `POST /v1/audio/speech`, the WebSocket API and `/health`, MUST keep its current contract.
- **FR-002**: The route MUST accept the same fields as `POST /v1/audio/speech`, from the query
  string only: `text`, `voice_id`, `instruction`, `cfg_scale`, `seed`, `temperature`, `top_k`,
  `top_p`, `repetition_penalty`, `max_new_tokens`, `split_chars`. The same parsing, defaults, limits,
  validation order, status codes and `{"error","code"}` error body apply.
- **FR-003**: The reference MUST be a saved `voice_id` or none. `ref_audio` must be a file part,
  so it can't arrive in a query string; the existing check rejects it with `400 invalid_field`.
  `ref_text` follows the POST route's rules: with `voice_id` it overrides the stored transcript,
  and alone it is `400 reference_required`.
- **FR-004**: The server MUST accept request lines of up to 128 KiB, so that 10,000 characters of
  any script fit percent-encoded, and MUST reject longer ones without doing any work.
- **FR-005**: All validation that the POST route does before streaming MUST happen before this
  route sends a status line, so every rejection is a real non-2xx response.

**Queueing**

- **FR-006**: When the GPU is busy, the route MUST wait for it in the same first-in, first-out
  order as other waiters, for at most 60 s. After that it MUST return `503` with code
  `busy_timeout`.
- **FR-007**: A waiting request whose client disconnects MUST leave the queue and never take the
  GPU.
- **FR-008**: The status line and headers MUST be held until the first audio chunk exists, as on
  the POST route. A failure before then is reported as an error status.

**Response**

- **FR-009**: A successful response MUST be `200` with `Content-Type: audio/wav`, chunked transfer
  with no `Content-Length`, and the headers `Cache-Control: no-store`, `Accept-Ranges: none`,
  `X-Sample-Rate` and `X-Breeze-Version`.
- **FR-010**: The body MUST start with a 44-byte canonical PCM WAV header:
  - mono, 16-bit;
  - the loaded model's sample rate;
  - the RIFF chunk size and the `data` chunk size both set to `0xFFFFFFFF`.

  It MUST continue with s16le PCM, produced by the same chunk ramp as the POST route.
- **FR-011**: The route MUST ignore `Range` request headers and always serve the full stream with
  `200`.
- **FR-012**: A failure after the headers MUST end the body early (no chunked terminator), as on
  the POST route, and MUST be recorded as `speech.failed`.

**Buffered delivery (this route only)**

- **FR-013**: Generation MUST NOT depend on how fast the client reads. Audio is generated at full
  speed into a buffer for that request, and each chunk is offered to the client as soon as it
  exists.
- **FR-014**: The GPU MUST be released as soon as generation ends, even while buffered audio is
  still being delivered.
- **FR-015**: If the client disconnects while generation is still running, generation MUST stop,
  and the GPU MUST be released within one chunk. The event is `speech.aborted` with
  `reason=client_disconnect`.
- **FR-016**: The route MUST NOT apply the POST route's 30 s send timeout or its 0.5× minimum read
  rate. Instead, a stream whose client reads no bytes for 10 minutes (a single send blocked for
  600 s) MUST be aborted with the existing reason `send_timeout`, and its buffer freed.
- **FR-017**: The POST route MUST keep its current coupled delivery, send timeout and minimum rate
  unchanged.

**Telemetry**

- **FR-018**: The route MUST emit these structured events, each with the request id and a
  `format=wav` field:
  - exactly one of `speech.completed`, `speech.aborted` or `speech.failed` per request that reached
    the GPU;
  - `speech.generated` when the GPU is released while delivery continues;
  - `speech.queued_timeout` for the `503`.

**Versioning and documentation**

- **FR-019**: The version MUST become `2.1.0`, reported in `X-Breeze-Version`. Clients detect the
  feature by `X-Breeze-Version >= 2.1.0`.
- **FR-020**: The README, the HTTP contract and the CHANGELOG MUST describe:
  - the route, its fields, headers, WAV framing, queueing and timeouts;
  - that the route has no cross-site protection, so any web page can trigger synthesis;
  - that it reverses 003's kept "no WAV output" behaviour for this route only.

### Key Entities

- **Stream buffer**: holds the audio already generated for one request but not yet delivered. It is
  created when streaming starts and freed when delivery completes, the client disconnects, or the
  600 s send timeout fires. Its size is about 48 KB per second of audio.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For the same text, voice and seed, the route and `POST /v1/audio/speech` deliver the
  same synthesis. It is byte-identical on a deterministic test runtime, and within the existing
  GPU run-to-run tolerance on the GPU. Time to first audio is within 10% between the two.
- **SC-002**: In Chrome and Firefox, a message with more than 3 minutes of audio plays to the end
  without audible gaps and without being cut. This must hold at playback speeds 0.5×, 0.9× and 1×.
  The server records exactly one synthesis per URL.
- **SC-003**: Changing `audio.src` during synthesis releases the GPU within one chunk, and the next
  segment's audio starts without waiting on the abandoned one.
- **SC-004**: While a long message is still draining after generation, a WebSocket preview started
  from SillyTavern produces audio as fast as it does on an idle server.
- **SC-005**: A GET made during an active WebSocket session streams once the session ends, and
  never returns `409`.
- **SC-006**: A 10,000-character Chinese text and a 10,000-character English text each stream to
  completion.
- **SC-007**: A representative set of invalid inputs gets the same status and error body on this
  route as on the POST route. The set is: unknown `voice_id`, empty `text`, an out-of-range value,
  and `ref_audio` in the query.
- **SC-008**: Time to first audio and real-time factor on `POST /v1/audio/speech` and the WebSocket
  stay within 10% of the 2.0.0 baseline.

## Assumptions

- **Primary client** is the SillyTavern Breeze TTS extension.
  - It pre-checks `voice_id` against `/v1/voices` and the 10,000-character limit.
  - It listens for the media `error` event, because SillyTavern itself doesn't.
  - It always puts a seed in the URL.
  - It builds the URL from its configured server base URL.
  - It keeps the WebSocket path for previews, and as a fallback when `X-Breeze-Version` is missing
    or older than 2.1.0.
- **SillyTavern behaviour** (checked by `st-agent`):
  - Stop and moving to the next clip both replace `audio.src`, which drops the connection.
    SillyTavern has no pause.
  - It creates every segment's URL up front, but GETs a URL only when that clip starts playing, so
    each tab has at most one GET in flight.
- **Trade-off owned by the extension**: in URL mode, SillyTavern's RVC voice conversion and VRM
  lip-sync don't run, because they only accept Blob audio.
- **Memory**: no separate cap on stream buffers.
  - Only one request generates at a time (single GPU).
  - Buffers still draining are limited by one GET per SillyTavern tab and by the 600 s send
    timeout.
  - The worst case for one request is roughly 100–115 MB, for 10,000 CJK characters (about 35–40
    min of audio). 10,000 English characters is roughly 29 MB.
- **Sample rate** is 24,000 Hz for current checkpoints. The header always carries the loaded
  model's rate.
- **Browsers** tolerate `0xFFFFFFFF` WAV sizes, headers held for up to 60 s, and a stream that ends
  before the declared size. These are not tested in advance (the spike was skipped). SC-002 to
  SC-005 are the checks. If a browser re-requests a URL, it is served again as an independent
  synthesis.
- **Deployment**: the same hand-started launcher as 2.0.0. Rollback is to 2.0.0, or to the C++
  server as rehearsed in T085. The extension falls back to the WebSocket when the header is absent.
