# Feature Specification: C++-Compatible API (Fixed)

**Feature Branch**: `perf-and-fixes`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "Replace this API server's endpoints with ones compatible with the
endpoints offered by HoppouAI/Breeze-TTS-2.cpp, so it can replace that server. No voice
conversion, web UI, or command-line compatibility. Include the CORS support from the local
`server-cors` branch. Build on `perf-and-fixes`. No backwards compatibility with the current Python
API; this becomes the new default. Do not reimplement the C++ server's bugs or bad designs: fix
them, but list every breaking change to the API. Reuse earlier branch work where sound,
revalidated."

**Reference implementation ("the C++ server")**: `Breeze-TTS-2.cpp` branch `server-cors`, which is
upstream `main` `a543664` (identical to `HoppouAI/Breeze-TTS-2.cpp` HEAD on 2026-09-24), plus
`edb927c` (CORS), plus the uncommitted WebSocket forced-drain fix in `apps/server/ws_api.cpp`.
Its documented contract lives in `docs/server.md`, `docs/voices.md` and `docs/websocket.md`.

## Compatibility Policy

"Compatible" means a client written against the C++ server's documented contract works against
this server by changing only the host and port, **except** where the client depends on behavior
listed under [Breaking Changes](#breaking-changes-from-the-c-server). The same paths, field names,
defaults, success response shapes, audio format, WebSocket message types, and error body key are
kept.

A C++ behavior is changed (and listed as breaking) when it is a **defect**: it crashes or stalls
the server, silently discards or reinterprets what the client asked for, corrupts or loses data,
creates a security exposure, or contradicts the C++ server's own documentation. A C++ behavior that
is merely unconventional but works is **kept** for compatibility (clarified 2026-09-24); see
[Intentionally Kept C++ Behaviors](#intentionally-kept-c-behaviors).

## Clarifications

### Session 2026-09-24

- Q: Should C++ choices that are unconventional but work (409 busy, 200 on create, `{"error":
  string}`, `audio/pcm`, query-string fields, `0` = default) also change? → A: No. Keep them; fix
  only defects.
- Q: Saved-voice overwrite and delete? → A: `DELETE` removes the file; `POST` with an existing name
  (ignoring case) gets `409 voice_exists`.
- Q: Share `.breeze` voice files with the C++ server? → A: No. Use this server's own versioned
  format and ignore `.breeze` files.
- Q: SillyTavern's Breeze provider handles "Replace voice?" by posting the same name again. How
  should it work with `409`? → A: The server keeps `409 voice_exists`. SillyTavern changes to
  DELETE then POST on a confirmed replace, and updates its delete dialog wording. That change is
  coordinated with the SillyTavern extension's owner.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A C++-server HTTP client works against this server (Priority: P1)

An integrator has a client built against the C++ server's HTTP contract: `GET /health`,
`POST /v1/audio/speech`, and the `/v1/voices` routes. They point it at this server and it works
without code changes, as long as it doesn't rely on a documented breaking change.

**Why this priority**: It is the purpose of the feature. Without it, nothing else matters.

**Independent Test**: Run every HTTP example from the C++ server's docs against this server.
Compare status, headers and body shape with the documented contract, allowing only the listed
breaking changes.

**Acceptance Scenarios**:

1. **Given** a running server, **When** a client calls `GET /health`, **Then** it gets
   `{"status":"ok","sample_rate":24000,"ws_port":N}`. `N` is the WebSocket port that is actually
   listening, or `0` when there is none.
2. **Given** a running server, **When** a client posts `text` (plus any of `instruction`,
   `ref_audio`, `ref_text`, `voice_id`, `cfg_scale`, `seed`, `temperature`, `top_k`, `top_p`,
   `repetition_penalty`, `max_new_tokens`, `split_chars`) as multipart or url-encoded form data to
   `POST /v1/audio/speech`, **Then** it gets a streamed `200` response of headerless mono s16le
   PCM, with `Content-Type: audio/pcm`, `X-Sample-Rate`, `X-Sample-Format: s16le` and
   `Cache-Control: no-store`.
3. **Given** a running server, **When** a client registers a voice with `POST /v1/voices`, lists
   voices with `GET /v1/voices` and deletes one with `DELETE /v1/voices/{id}`, **Then** the
   request fields, response objects (`id`, `frames`, `seconds`, `encode_ms`, `saved`, `ref_text`)
   and id rules match the C++ contract.
4. **Given** a registered voice, **When** a client passes its `voice_id` to
   `POST /v1/audio/speech`, **Then** the speech uses that voice without re-uploading audio.
5. **Given** the server is generating for another request, **When** a new speech or voice-encode
   request arrives, **Then** that request is rejected at once with `409` and
   `{"error":"busy",...}`.

---

### User Story 2 - Invalid input is rejected with a clear error (Priority: P1)

An integrator makes a mistake: a typo in a number, a reference clip with no transcript, an
unreadable WAV, a `voice_id` combined with an upload. Instead of silently getting different speech
(or crashing the server, as the C++ server can), they get an error saying what is wrong.

**Why this priority**: The C++ server silently turns several of these mistakes into voice-design
speech in a random voice. Some inputs can crash the server for everyone. A replacement that copied
this would inherit production incidents.

**Independent Test**: Send a fixed corpus of malformed requests (bad numbers, out-of-range values,
unpaired reference fields, corrupt/truncated/zero-channel WAVs, oversize bodies). Each one returns
a `4xx` with the error envelope, and the server keeps serving afterwards.

**Acceptance Scenarios**:

1. **Given** `cfg_scale=banana`, **When** the request arrives, **Then** it gets `400`, not
   generation at `cfg_scale=0`.
2. **Given** `ref_audio` without `ref_text`, or `ref_text` without either `ref_audio` or `voice_id`,
   **When** the request arrives, **Then** it gets `400`, not voice-design speech.
3. **Given** both `voice_id` and `ref_audio`, **When** the request arrives, **Then** it gets `400`
   (the two are mutually exclusive).
4. **Given** a `ref_audio` that can't be decoded, is empty, is longer than the reference-duration
   limit, or is too short to produce any reference frames, **When** the request arrives, **Then**
   it gets `400`.
5. **Given** a malformed WAV built to over-read or divide by zero, **When** it is uploaded to either
   route, **Then** it gets `400` and the server process is unaffected.
6. **Given** an invalid request while another request is generating, **When** it arrives, **Then**
   it gets its validation error (`400`/`404`), not `409 busy`.

---

### User Story 3 - Long text completes (Priority: P1)

An integrator sends a long passage (about 3000 characters) with no reference. It streams to the end
instead of failing or truncating.

**Why this priority**: The C++ server handles this routinely. It splits the text, and the first
piece's generated audio anchors the voice for the rest. A replacement must match this.

**Independent Test**: Send a fixed 3000-character passage with no reference. The stream completes,
covers the whole text, and the same speaker is heard throughout.

**Acceptance Scenarios**:

1. **Given** text longer than `split_chars`, **When** there is no reference, **Then** the text is
   split at sentence and clause boundaries. The first piece is packed against a 200-weighted-character
   opening budget (or `split_chars`, if that is smaller). The budget is soft, as in C++: a single
   clause longer than 200 is kept whole. Every later piece uses the first piece's audio and text
   as its reference. Anchoring is skipped (later pieces stay voice design) when the first piece
   was truncated, or when the anchor would cut a later piece below its token cap while that piece
   would get its full cap without it (a piece limited by the context either way doesn't count).
2. **Given** text longer than `split_chars`, **When** a reference (`voice_id` or inline) is given,
   **Then** every piece uses that reference.
3. **Given** a request seed `s`, **When** piece `i` is generated, **Then** it uses seed
   `(s + i) mod 2^32`.
4. **Given** `split_chars=0`, **When** the request arrives, **Then** the text is generated as one
   piece.

---

### User Story 4 - Incremental text over a WebSocket session (Priority: P2)

A conversational client streams LLM output into a WebSocket session. It hears each sentence as soon
as the sentence is complete, can change the delivery instruction between sentences, and can cancel
or end the session. The server never hangs the client or the other users.

**Why this priority**: This is the C++ server's second interface and the lowest-latency path for
conversational agents. It depends on the generation core from Stories 1 and 3.

**Independent Test**: Run a scripted client through: `start`, token-by-token `text`, `instruction`,
`flush`, `cancel` at random moments, and `end`. Check the order of events and that every `end`
gets exactly one `done` and every `cancel` gets exactly one `cancelled`.

**Acceptance Scenarios**:

1. **Given** a WebSocket connection to `ws_port`, **When** it opens, **Then** the server sends
   `ready` with the real model sample rate and `format: "s16le"`.
2. **Given** a `start` (with optional `voice_id`, `instruction`, `ref_text`, `cfg_scale`, `seed`,
   `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`, `split_chars`),
   **When** it is valid, **Then** the server replies `started` with the `voice_id`.
3. **Given** a started session, **When** `text` messages arrive token by token, **Then** each
   complete sentence is spoken as a piece: `speaking` with the piece's exact text, then its binary
   PCM frames. A sentence that ends at the very end of the buffer waits for the next character (or
   `flush`/`end`) before it is cut, so a token boundary such as `3.` followed by `14` does not
   split the number. (Abbreviations followed by a space, such as `Dr. Smith`, are cut as in C++.)
4. **Given** a `flush` or `end`, **When** text is buffered, **Then** all of it is spoken.
5. **Given** an `end`, **When** there is nothing left to speak (including when the last piece has
   already finished), **Then** exactly one `done` is still sent, after all earlier pieces.
6. **Given** a `cancel` at any moment (idle, queued, mid-piece, or between pieces), **When** it
   arrives, **Then** exactly one `cancelled` is sent. Audio for the cancelled work stops before
   `cancelled`, and no later piece is dropped.
7. **Given** text in any language, **When** the client JSON-escapes non-ASCII characters (as
   Python's `json.dumps` does by default), **Then** the text is spoken correctly.
8. **Given** a client that stops reading from its socket, **When** other clients use HTTP or their
   own WebSocket sessions, **Then** they are not blocked beyond the piece currently being
   generated.
9. **Given** generation fails for one piece, **When** that happens, **Then** the session gets an
   `error` event and the server process keeps running.

---

### User Story 5 - Browser clients and cross-origin safety (Priority: P3)

A browser page on an allowed origin calls the HTTP API and the WebSocket. Pages on other origins
can't use the server to spend GPU time or write voice files.

**Why this priority**: The `server-cors` branch added this for browser integrations. The C++ version
also leaves cross-site write and WebSocket-hijack holes, which this fixes.

**Independent Test**: With the CORS allowlist enabled, check preflight and response headers from an
allowed origin, a disallowed origin and no origin, for every route and the WebSocket handshake.

**Acceptance Scenarios**:

1. **Given** CORS is enabled with `*`, **When** any origin calls an API route, **Then** it gets
   `Access-Control-Allow-Origin: *` and
   `Access-Control-Expose-Headers: X-Sample-Rate, X-Sample-Format`.
2. **Given** CORS is enabled with an allowlist, **When** an allowed origin calls, **Then** its
   origin is echoed back with `Vary: Origin`. Every response in allowlist mode carries
   `Vary: Origin`.
3. **Given** a preflight `OPTIONS` for an existing route, **When** it arrives, **Then** it gets
   `204` with `Access-Control-Allow-Methods` listing that route's methods,
   `Access-Control-Allow-Headers` echoing the requested headers, and
   `Access-Control-Max-Age: 86400`. A preflight for a path that doesn't exist gets `404`.
4. **Given** a browser request carrying an `Origin` that is not allowed (including every origin
   when CORS is off), **When** it is a `POST` or `DELETE` or a WebSocket handshake, **Then** it is
   rejected with `403` before any work or file write happens.
5. **Given** a non-browser client that sends no `Origin`, **When** it calls any route or opens the
   WebSocket, **Then** it is served normally.

---

### Edge Cases

- **Empty or whitespace-only `text`**: rejected with `400 text is required`. The C++ server accepts
  whitespace-only text.
- **`text` longer than the maximum text length**: `400`. **WebSocket buffer that would exceed the
  same limit**: an `error` event; the offending text is not appended.
- **Text that splits into more pieces than fit the context**: a later piece that has no room to
  generate ends the stream as a failure (see FR-013). It must not be passed off as success.
- **A field sent both in the query string and the body, or twice**: `400`. There is no silent
  precedence rule.
- **A field sent with an empty value (`seed=`)**: treated as absent, so the default applies.
- **Client disconnects mid-stream**: generation stops and the GPU is released within one chunk.
- **WebSocket `start` while a session is active**: the in-flight work is cancelled (one
  `cancelled`), then the new session starts (`started`). There is no concurrent mutation.
- **WebSocket `start` with an unknown `voice_id`**: an `error` event; the previous session (if any)
  continues unchanged.
- **WebSocket message before `start`**: an `error` event, as in C++.
- **Unknown message type, invalid JSON, wrong field type**: an `error` event naming the problem;
  the connection stays open.
- **CJK text with no punctuation streamed over WebSocket**: it is cut at a clause mark once a
  clause reaches the budget; with no clause mark at all, it is hard-cut into budget-sized chunks
  once it passes twice the budget. It never grows without bound.
- **WebSocket port in use at startup**: the HTTP server still starts, `/health` reports
  `ws_port: 0`, and a structured startup event records why.
- **Voice file on disk with an invalid name or corrupt content**: skipped at startup with a
  structured event; it never breaks `GET /v1/voices`.
- **Two saved voice names that differ only by case**: the second is rejected, because the voices
  directory may be on a case-insensitive filesystem.
- **Unnamed voice registered again with identical audio and transcript**: returns the cached entry
  without encoding and without needing the GPU (as in C++).

## Requirements *(mandatory)*

### Functional Requirements

**Scope and surface**

- **FR-001**: The server MUST expose exactly these endpoints: `GET /health`,
  `POST /v1/audio/speech`, `POST /v1/voices`, `GET /v1/voices`, `DELETE /v1/voices/{id}`, and a
  WebSocket session endpoint on a separate port advertised as `ws_port`. It MUST NOT expose voice
  conversion (`/v1/audio/convert`), web-UI routes, or the current Python API's request shapes.
- **FR-002**: The default HTTP port MUST be `8080`. By default the WebSocket port MUST be the HTTP
  port + 1, and it MUST be possible to disable it. Both MUST bind only to the configured host
  address. The WebSocket MUST never widen to all interfaces.
- **FR-003**: Launch options MUST let an operator set: host, HTTP port, WebSocket port (or
  disable it), CORS mode/allowlist, default `split_chars`, stream chunk sizes (first/max), and the
  voices directory. Option names need not match the C++ command line.

**Request parsing and validation (HTTP)**

- **FR-004**: `POST /v1/audio/speech` and `POST /v1/voices` MUST accept `multipart/form-data` and
  `application/x-www-form-urlencoded` bodies. They MUST also accept fields in the query string. A
  field given more than once (across or within sources) MUST be rejected with `400`.
- **FR-005**: A field that is absent or has an empty value MUST take its documented default.
- **FR-006**: Numeric fields MUST parse strictly. Non-numeric, non-finite and out-of-range values
  MUST be rejected with `400` naming the field. Ranges, documented in the contract: `cfg_scale`
  finite and ≥ 0 (1.0 = off, 0 = unconditional-only); `seed` 0 to 4294967295; `temperature` ≥ 0;
  `top_k` ≥ 0; `top_p` 0 to 1; `repetition_penalty` ≥ 0; `max_new_tokens` 0 to the server maximum;
  `split_chars` ≥ 0. Upper bounds are set in the contract. For the sampling fields and
  `max_new_tokens`, `0` MUST mean "model default", as in C++.
- **FR-007**: Validation MUST happen in this order, and all of it before the busy check:
  1. request size limits (`413`);
  2. field syntax and ranges (`400`);
  3. reference consistency (`400`);
  4. unknown `voice_id` (`404`);
  5. `ref_audio` decode and limits (`400`);
  6. busy (`409`).
- **FR-008**: Reference rules:
  - `ref_audio` requires `ref_text`.
  - `ref_text` requires `ref_audio` or `voice_id`.
  - `voice_id` and `ref_audio` are mutually exclusive.
  - `ref_text` given with `voice_id` overrides the stored transcript, as documented for the C++
    server.
  - Each violation MUST be a `400`.
- **FR-009**: Reference audio MUST be decoded by a bounds-safe decoder. Anything it can't decode,
  that is empty, over 30 seconds, over the upload size limit (25 MiB), or shorter than one full
  codec frame (80 ms, 1,920 samples at 24 kHz) MUST be rejected with `400`. Any sample rate and channel count MUST be
  accepted and converted.
- **FR-010**: An empty or whitespace-only `instruction` MUST mean the default instruction
  (`"Speak clearly and naturally."`) on every interface.
- **FR-011**: `text` MUST be non-blank and at most the maximum text length (default 10,000
  characters). `ref_text` and `instruction` MUST each be at most 2,000 characters. `text`, `ref_text` and `instruction`
  MUST NOT contain control characters other than tab, carriage return and line feed.

**Speech generation (HTTP)**

- **FR-012**: The speech response MUST:
  - be a streamed `200` of headerless mono signed 16-bit little-endian PCM at the model sample
    rate;
  - carry the headers from User Story 1;
  - send chunks that grow from the configured first chunk size to the maximum chunk size.
- **FR-013**: All validation and reference preparation MUST finish before the `200` is sent. A
  failure after streaming has started MUST end the response abnormally, so the client sees an
  incomplete transfer. It MUST NOT end the response like a success. Reaching `max_new_tokens` is a
  normal end.
- **FR-014**: Long text MUST be split and anchored as in User Story 3, with one text segmenter
  shared by HTTP and WebSocket. `split_chars=0` MUST mean "no length splitting" on both interfaces.
- **FR-015**: A client disconnect MUST stop generation and release the GPU.
- **FR-016**: Only one generation runs at a time. HTTP requests that find the GPU busy MUST get
  `409` at once. WebSocket pieces MUST wait their turn, announced with `queued`.

**Voices**

- **FR-017**: `POST /v1/voices` MUST take `ref_audio` (required), `ref_text` (required) and an
  optional `name` matching `[A-Za-z0-9_-]{1,64}` that doesn't start with `v_`. It MUST return
  `200` with the voice object.
  - An unnamed voice MUST get the id `v_` followed by 16 lowercase hex characters, derived
    deterministically from the audio and transcript. It is held in memory and may be evicted
    (at most 64 unnamed voices; the oldest unnamed voice is evicted first). Saved voices do not
    count toward this cap.
  - An unnamed voice identical to an existing one MUST return the existing entry without encoding
    and without the busy check.
- **FR-018**: A named voice MUST be saved to the voices directory and survive restarts. Saved
  voices are never evicted.
  - **Existing name:** a `POST` with a name that is already saved MUST be rejected with `409`
    and code `voice_exists`; nothing is encoded or overwritten. To replace a voice, the client
    deletes it and registers it again.
  - **Delete:** `DELETE` of a saved voice MUST remove its file, so it does not return after a
    restart, and respond `200 {"deleted":"<id>","file_kept":false}`. `file_kept` is always
    `false`.
- **FR-019**: A name that matches an existing saved voice when case is ignored MUST be rejected the
  same way as an existing name (`409 voice_exists`).
- **FR-020**: Saved voices MUST use this server's own file format, with a format version field
  (Constitution IV). The server MUST NOT read or write the C++ server's `.breeze` files. `.breeze`
  files in the voices directory MUST be ignored, and the server MUST record one structured startup
  event listing how many were ignored.
- **FR-021**: `GET /v1/voices` MUST return an array of voice objects in a deterministic order:
  saved voices sorted by id, then unnamed voices in registration order. Voice files with invalid
  names or corrupt content MUST be skipped at startup and never appear in the list.
- **FR-022**: An unknown id on `DELETE` MUST return `404 {"error":"unknown voice_id",...}`.

**Errors**

- **FR-023**: Every error response on every route (including unknown route `404`, wrong method
  `405`, oversize `413`, and unhandled failures `500`) MUST be JSON `{"error": "<message>",
  "code": "<machine code>"}`. `error` stays the human-readable string the C++ server's clients
  read.
- **FR-024**: Internal failure details MUST NOT leak into error messages. Every failure MUST be
  recorded as a structured event with a request id.

**WebSocket session**

- **FR-025**: The WebSocket MUST follow the WebSocket protocol standard. That includes answering a
  client close with a close frame, and closing on protocol violations with the right close code.
- **FR-026**: Client messages MUST be parsed as real JSON against a strict schema.
  - **Message types:** `start`, `text`, `flush`, `end`, `instruction`, `cancel`, with the C++
    fields.
  - **`start`:** it additionally accepts `top_p`, `repetition_penalty` and `max_new_tokens`, with
    the same validation as HTTP.
  - **Invalid messages:** invalid JSON, an unknown type, a wrong field type or an out-of-range
    value MUST produce an `error` event with `message`, `code` and `request_type` (the offending
    client message type), and the
    connection stays open. Binary frames from the client MUST produce an `error` event.
- **FR-027**: Server events MUST keep the C++ types and fields:
  - `ready` (with the real sample rate)
  - `started`
  - `queued`
  - `speaking` (with the exact piece text)
  - binary PCM
  - `instruction_set`
  - `cancelled`
  - `done`
  - `error`
- **FR-028**: Every `end` MUST produce exactly one `done`. It comes after every piece queued before
  it, including when nothing remains. A later `cancel` supersedes a pending `end`: the client gets
  `cancelled` instead of `done`.
- **FR-029**: Every `cancel` MUST produce exactly one `cancelled`. It comes after the last audio
  frame of any cancelled piece. It MUST discard buffered text and queued pieces, and it MUST NOT
  affect pieces submitted after it.
- **FR-030**: A `start` while the session has buffered text, queued pieces or a piece in flight
  MUST cancel that work as FR-029 describes (one `cancelled`) before the new session begins. A
  `start` on an idle session produces only `started`. A `start` with `ref_text` but no `voice_id`
  MUST produce an `error` event, and the previous session stays unchanged.
- **FR-031**: Sentence draining MUST use the shared segmenter (FR-014).
  - **End of buffer:** sentence-ending punctuation at the very end of the buffer MUST NOT trigger a
    cut until more text, `flush` or `end` arrives.
  - **Opening budget:** the 200-character opening budget applies only to the first piece of a
    session without a reference.
  - **Unpunctuated text:** a sentence over the budget MUST be cut into clauses at a clause or
    space boundary once a clause reaches the budget; a run with no boundary MUST be hard-cut once
    it exceeds twice the budget, so the buffer stays bounded (contracts/ws-api.md gives the exact
    rule).
- **FR-032**: A generation failure in a session MUST be reported as an `error` event. It MUST NOT
  end other sessions or the process.
- **FR-033**: Audio for a WebSocket client MUST go through a bounded per-connection outgoing
  buffer, never written while the GPU is held.
  - **Slow client:** a client whose backlog exceeds the bound MUST be disconnected with a
    documented close code.
  - **Connection limits:** the number of concurrent WebSocket connections and the handshake time
    MUST be limited. Connections over the limit are refused.

**CORS and cross-origin protection**

- **FR-034**: CORS MUST be off by default. When enabled, it takes either `*` or a comma-separated
  allowlist; entries are whitespace-trimmed, validated and canonicalized (lowercase scheme and
  host, IDN hosts in punycode, IP literals in canonical form, the scheme's default port dropped),
  and deduplicated after canonicalizing. `*` mixed with other entries MUST be a startup error. An
  incoming `Origin` header is canonicalized the same way before matching against the allowlist.
  Response and preflight headers MUST behave as in User Story 5, including on error responses and
  streamed responses.
- **FR-035**: A `POST`, `DELETE` or WebSocket handshake whose `Origin` header is present and not
  allowed MUST be rejected with `403` before any work. Requests without `Origin` are unaffected.

**Health, docs, versioning**

- **FR-036**: `GET /health` MUST return `503 {"status":"loading","error":"model is loading","code":"loading"}` until the model is ready. After
  that it returns the body in User Story 1, with `ws_port` reporting the port that is actually
  listening. While loading, every other HTTP route MUST return the same `503` body, and WebSocket
  handshakes MUST be refused with `503`.
- **FR-036a**: When a piece has room to start but less room than its token cap, it MUST be
  generated up to the room available and end normally, like reaching `max_new_tokens`. The server
  MUST record a structured event. A first piece with no room at all MUST be rejected with
  `400 text is too long` before streaming, and a later piece with no room MUST abort the stream
  (FR-013).
- **FR-037**: The project MUST gain a version number, set to `2.0.0`, and a changelog. The changelog
  entry MUST reproduce the [Breaking Changes](#breaking-changes-from-the-c-server) list and state
  that the previous Python API is removed.
- **FR-037a**: Every HTTP response (including errors, preflights and streamed speech) and every
  WebSocket handshake response MUST carry `X-Breeze-Version: <server version>`, so consumers can
  pin the contract version (Constitution IV). It MUST be listed in
  `Access-Control-Expose-Headers` when CORS is enabled.
- **FR-038**: The README MUST document the new API for users: every endpoint, field, error code,
  launch option and WebSocket message, plus the breaking-changes list and a link to the C++ docs.
  The launch scripts and Docker run script MUST be updated to the new port and options.

### Breaking Changes from the C++ Server

Every change below is a deliberate break with the C++ server's behavior. "Affected" says which
clients see a difference. IDs are referenced by tests and the changelog. IDs are stable and grouped
by area, not listed in numeric order (BC-46 to BC-48 were added later).

**HTTP: parsing and validation**

| ID | C++ behavior | New behavior | Affected |
|---|---|---|---|
| BC-01 | Unparseable numbers become `0` (`cfg_scale=banana` → CFG 0, `seed=x` → 0) | `400` naming the field | Clients sending malformed numbers |
| BC-02 | Empty field value (`seed=`) is used as `""`/`0` | Treated as absent; default applies | Clients sending empty fields |
| BC-03 | Negative, NaN or out-of-range sampling values silently mean "default" or pass through unchecked (`top_p` > 1, negative `cfg_scale`) | `400`; only `0` means default | Clients sending such values |
| BC-04 | `max_new_tokens` unbounded (can exhaust memory and kill the process) | `400` above the server maximum | Clients asking for more than the maximum |
| BC-05 | `text` and `instruction` unbounded (url-encoded bodies over 8 KB instead get an empty `413`) | `400` above the maximum text or instruction length; url-encoded bodies accepted up to the request limit | Clients sending very long text |
| BC-06 | Multipart bodies unbounded | `413` with the error envelope above the upload limit | Clients uploading very large files |
| BC-07 | Busy check runs before validation | Validation first, so an invalid request gets `400`/`404` even while busy | Clients sending invalid requests during generation |
| BC-08 | A field repeated or present in both query and body resolves silently | `400` | Clients sending duplicate fields |
| BC-09 | Empty `instruction` on HTTP is used literally | The default instruction is used | Clients sending `instruction=` |
| BC-10 | Whitespace-only `text` accepted | `400 text is required` | Clients sending blank text |

**HTTP: reference audio**

| ID | C++ behavior | New behavior | Affected |
|---|---|---|---|
| BC-11 | Undecodable or empty `ref_audio` ignored; request silently becomes voice design | `400` | Clients uploading bad audio |
| BC-12 | `ref_audio` without `ref_text` ignored silently | `400` | Clients omitting the transcript |
| BC-13 | `ref_text` without `ref_audio`/`voice_id` ignored silently (HTTP and WebSocket `start`) | `400` on HTTP; `error` event on WebSocket | Clients sending a stray transcript |
| BC-14 | `voice_id` plus `ref_audio`: the upload is ignored | `400` (mutually exclusive) | Clients sending both |
| BC-15 | Malformed WAV can over-read memory or divide by zero; 8/24-bit PCM decodes as silence | Safe decode of all common PCM/float formats; undecodable input gets `400` | Clients with such files (now work or get a clear error) |
| BC-16 | Reference clip of unlimited length; a clip too short for any frame is silently ignored | `400` over 30 s or under one full codec frame (80 ms) | Clients sending such clips |

**HTTP: responses, errors and CORS**

| ID | C++ behavior | New behavior | Affected |
|---|---|---|---|
| BC-46 | Control characters accepted in text; NUL counts as sentence-closing punctuation | `400` (`error` event on WebSocket) for control characters other than tab, CR and LF | Clients sending control characters |
| BC-47 | No fixed context limit: a single piece of any length can generate (the cache is sized per piece) | A piece that cannot fit the 2,048-token context gets `400 text is too long` (first piece) or an aborted stream (later piece) | Clients sending `split_chars=0` or very large `split_chars` with long text, or long references with long transcripts |
| BC-17 | Failure mid-stream ends the stream like success; failure before audio gives `200` with an empty body | Failures before streaming get a proper error status; failures after streaming starts abort the response | Clients that treated a truncated stream as complete |
| BC-18 | Unknown route `404`, wrong method `400`/`404`, oversize `413`: empty non-JSON bodies; `OPTIONS` without CORS gets `404` | JSON error envelope; wrong method (including `OPTIONS` without CORS) is `405` with `Allow` | Clients parsing these responses |
| BC-19 | CORS allowlist entries not trimmed (`"a, b"` never matches `b`) | Entries trimmed | Operators with spaced lists (now works) |
| BC-20 | `Vary: Origin` only on matching responses in allowlist mode | On every response in allowlist mode | Caches (fixes mixing) |
| BC-21 | Preflight returns `204` for any path, advertising every method | Existing routes only, with that route's methods; unknown paths get `404` | Clients preflighting non-existent paths |
| BC-22 | `*` mixed into an allowlist is a dead entry | Startup error | Operators with such configs |
| BC-23 | Cross-origin browser `POST`/`DELETE` runs (and can write voice files) even when CORS is off | `403` when `Origin` is present and not allowed | Browser pages on disallowed origins |
| BC-24 | `/health` reports `ws_port` even when the WebSocket failed to bind | Reports `0` in that case | Clients discovering the WebSocket |

**Voices**

| ID | C++ behavior | New behavior | Affected |
|---|---|---|---|
| BC-25 | Voice files with unsafe names load and break `GET /v1/voices` JSON; list order is unspecified | Invalid files skipped; deterministic order | Operators with hand-placed files |
| BC-26 | Names may start with `v_`, colliding with generated ids; names differing only by case collide on case-insensitive filesystems | `v_` prefix reserved (`400`); case-insensitive duplicates rejected | Clients using such names |
| BC-27 | Named `POST` for an existing name silently overwrites it | `409` with code `voice_exists`; delete then register to replace | Clients that re-POST a name to update it |
| BC-28 | `DELETE` keeps the file (`file_kept: true`), so the voice comes back on restart | The file is removed; `file_kept` is always `false` | Clients relying on deleted voices returning |
| BC-48 | The 64-voice cap counts saved voices too, so many saved voices leave little room for unnamed ones | The cap counts only unnamed voices | Clients registering many unnamed voices alongside saved ones |
| BC-29 | Voices stored as `.breeze` files | Own versioned format; `.breeze` files are ignored, so C++ voices must be registered again | Operators switching from the C++ server |

**WebSocket**

| ID | C++ behavior | New behavior | Affected |
|---|---|---|---|
| BC-30 | WebSocket binds `0.0.0.0` when the host is not an IPv4 literal | Binds only the configured host | Deployments using `localhost`/IPv6 |
| BC-31 | No `Origin` check on the handshake (cross-site hijacking) | `403` for disallowed browser origins | Browser pages on disallowed origins |
| BC-32 | Hand-rolled JSON reader: `\uXXXX` escapes deleted, keys matched inside values, `null` misread, invalid JSON accepted, wrong types become defaults | Real JSON with schema validation; an `error` event on invalid input | All clients escaping non-ASCII (now works); clients sending bad messages |
| BC-33 | `ready.sample_rate` hard-coded to 24000 | Real model rate (24000 for current models) | None in practice |
| BC-34 | `done` missing if `end` arrives with nothing left to speak | Exactly one `done` per `end` | Clients that worked around the hang |
| BC-35 | `cancel` sometimes unacknowledged, sometimes spurious, and can swallow the next piece | Exactly one `cancelled` per `cancel`; later pieces never dropped | Clients counting or ignoring `cancelled` |
| BC-36 | `start` mid-speech mutates the running session (a data race) | Cancels in-flight work (`cancelled`), then `started` | Clients restarting mid-speech |
| BC-37 | Empty `instruction` message stores `""` | Resets to the default instruction | Clients sending an empty instruction |
| BC-38 | `split_chars ≤ 0` at `start` means 600; on HTTP `0` means no splitting | `0` means no length splitting on both; negative gets an `error` | Clients sending `split_chars: 0` |
| BC-39 | Period at the end of the buffer cuts immediately (splits `Dr.`, `3.` mid-stream); CJK without spaces never drains; the 200-char opening budget applies to every piece of that drain; different stop set from HTTP | Shared segmenter; end-of-buffer punctuation waits for the next character; CJK fallback; opening budget on the first piece only | All clients: streaming piece boundaries differ, and so do HTTP piece boundaries for long text (over-budget sentences are now cut at spaces; pieces with no letter or digit, such as emoji-only ones, are dropped) |
| BC-40 | Buffered text unbounded | `error` event above the maximum text length | Clients buffering very long text without punctuation |
| BC-41 | Generation error in a session terminates the whole server process | `error` event; session and server continue | All |
| BC-42 | A client that stops reading holds the GPU and blocks every other client; unbounded connection threads; no handshake timeout | Bounded outgoing buffer; slow client disconnected; connection and handshake limits | Very slow clients; connection floods |
| BC-43 | Client close not answered with a close frame (browser sees 1006); unmasked frames, bad UTF-8 and malformed control frames accepted | Standard-conformant WebSocket | Non-conformant clients |
| BC-44 | `speaking.text` drops tabs and carriage returns | Exact piece text | Clients comparing piece text |
| BC-45 | Binary frames from the client silently ignored | `error` event | Clients sending binary frames |

**Additive changes (not breaking)**: an `X-Breeze-Version` header on every HTTP response and
WebSocket handshake (FR-037a); an error `code` field alongside `error` on HTTP and
WebSocket errors; `type` on WebSocket error events; `top_p`, `repetition_penalty`,
`max_new_tokens` on WebSocket `start`; any sample rate and channel count, 8/24-bit PCM and other
common audio containers accepted as `ref_audio`.

### Known Differences Outside the API Contract

- **Repetition penalty:** this server applies it once per distinct generated token (the reference
  model implementation's semantics). The C++ server compounds it once per occurrence. The field,
  its range and its default are the same; only the resulting audio differs.
- **First chunk size:** streams start at 1 codec frame (80 ms) by default and ramp to 25, where
  C++ starts at 4. This lowers time to first audio. It changes chunk boundaries, not the audio
  format, and can be set back with the chunk-size launch options.
- **Default length:** with `max_new_tokens` absent or `0`, a piece is capped at the model default
  of 750 frames (60 s), as in C++. The previous Python API used 1,500.

### Intentionally Kept C++ Behaviors

These C++ choices are unconventional but are kept for compatibility:

- `409` with `{"error":"busy"}` when the GPU is busy on HTTP. `503` with `Retry-After` would be
  more conventional.
- `200` rather than `201` on voice creation.
- `{"error": "<string>"}` as the error body key (a `code` is added alongside).
- `Content-Type: audio/pcm` with rate and format in headers; no WAV or `response_format` option.
- Fields accepted from the query string.
- `0` meaning "model default" for sampling fields and `max_new_tokens` (greedy decoding cannot be
  requested).
- Per-piece seed `seed + i`, `ref_text` overriding a voice's stored transcript, the unnamed voice
  id format, the 64-entry unnamed voice cap, WebSocket on a separate port, and no authentication.

### Key Entities

- **Speech request**: text, delivery instruction, an optional reference (a saved voice id or an
  inline clip with its transcript), sampling settings, seed, and split budget.
- **Reference**: audio codes plus transcript that fix the speaker. It comes from a saved voice, an
  inline clip, or (for long text without one) the request's own first piece.
- **Voice**: id, optional name, transcript, reference codes, frame count, duration, encode time,
  and whether it is saved. Unnamed voices live in memory; named voices persist in the voices
  directory.
- **Session (WebSocket)**: per-connection settings from `start`, text buffer, queue of pieces,
  pending `end`/`cancel` state, and the reference used for the session.
- **Piece**: a segment of text generated as one unit, with its own seed and audio frames.
- **Error**: HTTP status or WebSocket event, human message, and machine code.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the HTTP and WebSocket examples in the C++ server's documentation produce the
  documented status, headers and body or event shapes against this server. The only exceptions
  are cases that match a listed breaking change, and each exception is traceable to its BC id.
- **SC-002**: Each breaking change BC-01 to BC-48 has an automated check that shows the new
  behavior. For changes that fix a defect, that check fails against the defective behavior.
- **SC-003**: A corpus of at least 50 malformed HTTP requests and 50 malformed WebSocket messages
  produces a structured error for every item. No item crashes the server, hangs a connection, or
  changes later results.
- **SC-004**: A 3000-character request with no reference completes with audio covering the whole
  text in 100% of 5 runs.
- **SC-005**: In 1,000 scripted WebSocket sequences with randomized timing of `text`, `flush`,
  `cancel` and `end`: every `end` gets exactly one `done` unless a later `cancel` supersedes it;
  every `cancel` gets exactly one `cancelled`; and no uncancelled piece is lost.
- **SC-006**: While one WebSocket client has stopped reading, another client's HTTP speech request
  starts streaming within the time it takes to finish the piece in progress plus 1 second.
- **SC-007**: Time to first audio and real-time factor for short and medium requests stay within
  10% of the pre-change baseline measured on the same machine and branch.
- **SC-008**: An integrator can find every endpoint, field, error code, launch option and breaking
  change in the README within 5 minutes.

## Assumptions

- **Primary live client** is the SillyTavern Breeze TTS extension
  (`<SillyTavern checkout>/extensions/SillyTavern-BreezeTTS`). It synthesizes only over the WebSocket
  (`start` with `voice_id` and `cfg_scale` 1 to 10 in 0.5 steps, then a single `end` with the whole
  text). It uses HTTP for `/health` and voice upload, list and delete, and it reads HTTP `error`
  and WebSocket `message` keys.
- **Target clients** are the C++ server's documented HTTP and WebSocket clients, plus Liveva and
  other internal integrations. They will switch at the same time. No client needs the old Python
  API, so it is removed rather than versioned. This is an explicit waiver of Constitution IV's
  "new version plus migration path" rule, requested by the user and recorded here.
- **The contract source** is the C++ server's code at the reference commit. Where its docs and code
  disagree, the docs' intent wins when the code's behavior is a defect. Every such case is a listed
  breaking change.
- **Single GPU, single process**: one generation at a time, as today and in C++. The WebSocket
  listener runs in the same process as HTTP.
- **Model sample rate** is 24000 Hz for current checkpoints. The server reports whatever the loaded
  model uses.
- **Default limits** follow the earlier Python server where one existed: 30 s reference, 25 MiB
  upload, 2,000-character transcript. New limits: 10,000-character text, a server maximum of 1,500
  for `max_new_tokens`, and a small fixed WebSocket connection cap. These are adjustable in the
  plan if measurements say so.
- **Reuse** of earlier branch work is expected but must be revalidated against this spec, not the
  earlier parity spec (`api-alignment:specs/002-cpp-server-compat`):
  - the text splitter port and its C++ golden tests, with the WebSocket drain fixed;
  - saved-voice storage from feature 001;
  - the error envelope and CORS setup;
  - per-request sampling overrides and context-room checks;
  - long-text anchoring and the stream chunk ramp;
  - the benchmark script.

  Work that exists only to copy C++ quirks is not reused: the C-style number coercion and the
  tests asserting silent fallbacks.
- **Out of scope**: voice conversion, web UI, command-line compatibility with the C++ binary,
  authentication, OpenAI-style request fields, and non-PCM response formats.
