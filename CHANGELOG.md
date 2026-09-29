# Changelog

All notable changes to this project are documented here. See
`specs/003-cpp-compatible-api/spec.md` for the full breaking-changes list and rationale (IDs
`BC-nn`).

## 2.0.0 — 2026-09-29

The server now exposes a C++-server-compatible HTTP and WebSocket API; see the README and
`specs/003-cpp-compatible-api/contracts/` for the full contract.

### Removed

- **The old Python API is removed.** The pre-2.0 request and response shapes and the port 7860
  default are gone; there is no compatibility mode. HTTP is on 8080 and the WebSocket on 8081.

### Added

These are additive and don't break C++ clients:

- An `X-Breeze-Version` header on every HTTP response and WebSocket handshake.
- A machine-readable `code` next to `error` on every HTTP and WebSocket error.
- `request_type` on WebSocket `error` events: the client message type that caused the error.
- `top_p`, `repetition_penalty` and `max_new_tokens` on WebSocket `start`.
- `ref_audio` accepts any sample rate and channel count, 8- and 24-bit PCM, and other common
  audio containers.

### Breaking changes

- **BC-18**: Every error response now uses the JSON envelope `{"error", "code"}`. A method the
  route doesn't support (including `OPTIONS` without CORS enabled) now returns `405` with an
  `Allow` header, instead of a bare `404`.
- **BC-19**: CORS allowlist entries are trimmed of surrounding whitespace before matching, so a
  spaced list (`"a, b"`) now matches `b`.
- **BC-20**: In CORS allowlist mode, every response carries `Vary: Origin`, not only responses
  that matched an allowed origin.
- **BC-21**: A CORS preflight now returns `204` only for a route that actually exists, advertising
  only that route's methods; a preflight for an unknown path gets `404`.
- **BC-22**: Mixing `*` into a CORS allowlist is a startup error instead of a silently dead entry.
- **BC-23**: A cross-origin browser `POST`/`DELETE` from a disallowed (or, with CORS off, any)
  origin now gets `403` before any work is done, instead of running and possibly writing a voice
  file.
- **BC-24**: `GET /health` reports `ws_port: 0` when the WebSocket failed to bind, instead of
  reporting a port that isn't actually listening.

- **BC-01**: A number that doesn't parse (`cfg_scale=banana`, `seed=x`, `1e`, `0x10`, `inf`) is
  `400` naming the field, instead of silently becoming `0`.
- **BC-02**: An empty field value (`seed=`) counts as absent, so the default applies.
- **BC-03**: Out-of-range, negative or NaN sampling values are `400`; only `0` means the model
  default.
- **BC-04**: `max_new_tokens` above 1,500 is `400`.
- **BC-05**: `text` over 10,000 characters and `instruction` or `ref_text` over 2,000 are `400`;
  url-encoded bodies are accepted up to the request limit.
- **BC-06**: A body over 26 MiB is `413` with the error envelope, with or without
  `Content-Length`.
- **BC-07**: Validation runs before the busy check, so an invalid request gets its `400`/`404` even
  while the GPU is busy.
- **BC-08**: A field given more than once (in the body, the query string or both) is `400
  duplicate_field`.
- **BC-09**: A blank `instruction` uses the default instruction.
- **BC-10**: Blank or whitespace-only `text`, or text with nothing speakable in it, is `400 text is
  required`.
- **BC-11**: An undecodable or empty `ref_audio` is `400` instead of silently becoming voice design.
- **BC-12**: `ref_audio` without `ref_text` is `400`.
- **BC-13**: `ref_text` without `ref_audio` or `voice_id` is `400`.
- **BC-14**: `voice_id` together with `ref_audio` is `400` (they are mutually exclusive).
- **BC-15**: Reference audio is decoded safely (WAV, WAVEX, FLAC, OGG; any rate from 8 to 192 kHz;
  up to 8 channels; 8/24-bit, float); malformed files are `400`, never a crash.
- **BC-16**: A reference over 30 s or under 80 ms (one full codec frame) is `400`.
- **BC-17**: A failure before the first audio chunk gets a proper error status; a failure after
  streaming starts aborts the response (no chunked terminator), so clients see an incomplete
  transfer instead of a short "successful" stream.
- **BC-46**: Control characters other than tab, CR and LF in `text`, `instruction` or `ref_text`
  are `400`.
- **BC-47**: Text is split into pieces that must fit the 2,048-token context: a first piece that
  can't is `400 text is too long`, and a later piece that can't aborts the stream. Long text with
  no reference is anchored on its first piece so one speaker is heard throughout. The anchor is
  skipped (`speech.anchor_skipped`) when the first piece was cut off at its frame limit, when
  the anchor would shorten any later piece, when measuring the later pieces failed or timed out,
  or when the server is shutting down; the later pieces then use voice design, so the speaker
  can change after the first piece. The opening piece is at most 200 characters, or
  `split_chars` if that is smaller.
- **BC-25**: Voice files with invalid names or contents are skipped at startup with a
  `voice.skipped` event instead of breaking `GET /v1/voices`, and the list order is fixed: saved
  voices sorted by id, then unnamed voices in registration order.
- **BC-26**: Voice names may not start with `v_` (any case), which is reserved for generated ids,
  and names differing only by case count as the same name.
- **BC-27**: `POST /v1/voices` with a name that already exists (in any case) is
  `409 voice_exists` instead of silently overwriting it; delete the voice first to replace it.
- **BC-28**: `DELETE /v1/voices/{id}` removes the file, so the voice stays gone after a restart;
  `file_kept` is always `false`. The id must match the listed id exactly.
- **BC-29**: Voices are stored in the server's own versioned `.voice.json` format. C++ `.breeze`
  files are ignored (and counted), so voices from the C++ server must be registered again.
- **BC-48**: The 64-voice cap counts only unnamed voices, so saved voices no longer use up room.
- **New voice and speech errors**: `400 voice_too_long` when a voice's reference plus transcript
  wouldn't leave the model room to speak, and `503 gpu_out_of_memory` when a voice's prompt can't
  be built for lack of GPU memory and the fallback doesn't fit either.
- **BC-30**: The WebSocket binds only the configured host (every address it resolves to), never
  `0.0.0.0`.
- **BC-31**: A WebSocket handshake from a disallowed browser `Origin` is `403`; with CORS off, no
  browser origin is allowed. Clients that send no `Origin` are unaffected.
- **BC-32**: WebSocket messages are parsed as real JSON: `\uXXXX` escapes work, and invalid JSON,
  unknown types and wrong or out-of-range fields get an `error` event (`invalid_json`,
  `unknown_type`, `invalid_field`) with `request_type`, using the same ranges as HTTP.
- **BC-33**: `ready.sample_rate` is the loaded model's rate.
- **BC-34**: Every `end` gets exactly one `done`, even when nothing was left to speak.
- **BC-35**: Every `cancel` gets exactly one `cancelled`, even when idle, and never swallows a later
  piece. A cancel replaces only its own session's pending `done`.
- **BC-36**: A `start` while a piece is queued or speaking cancels it (`cancelled`), then sends
  `started`.
- **BC-37**: A blank `instruction` message resets to the default instruction.
- **BC-38**: `split_chars: 0` means no length splitting (as on HTTP); a negative value is an error.
- **BC-39**: WebSocket text is cut by the same segmenter as HTTP: punctuation at the very end of
  the buffer waits for more text, CJK without punctuation still drains, and the 200-character
  opening budget applies to the first piece only.
- **BC-40**: Buffered text over 10,000 characters is `text_too_long` and is not appended.
- **BC-41**: A failed piece is an `error` event (`generation_failed`); the session and the server
  keep running.
- **BC-42**: A client that stops reading is disconnected (`1008 client too slow`) once 2 MiB of
  output is waiting or nothing drains for 30 s, and the piece in flight is cancelled so the GPU is
  freed; a client that doesn't read the close frame within 2 s is dropped. At most 16
  connections, and the handshake must complete within 10 s.
- **BC-43**: The WebSocket follows RFC 6455: a client close is echoed (1000), and protocol errors
  close with 1002, bad UTF-8 with 1007 and oversized messages (over 1 MiB) with 1009.
- **BC-44**: `speaking.text` is the exact piece text, tabs and carriage returns included.
- **BC-45**: A binary frame from the client gets an `unsupported_binary` error.
- **New WebSocket behaviour**: handshake refusals carry the JSON envelope and `X-Breeze-Version`
  (`503 loading`, `503 gpu_unavailable`, `503 too_many_connections`, `503 shutting_down`,
  `426 upgrade_required`, `400 bad_handshake`); server shutdown closes sessions with 1001. Without a
  voice, the first piece anchors the later ones, unless the anchor would shorten a given piece,
  which is then spoken without it; a piece that doesn't fit the context gets `text_too_long` and
  the session continues.

### Known differences outside the API contract

- **Repetition penalty**: applied once per distinct generated token (the reference model
  implementation's semantics); the C++ server compounds it once per occurrence. The field, its
  range and its default are the same; only the resulting audio differs.
- **First chunk size**: streams start at 1 codec frame (80 ms) and ramp to 25, where C++ starts
  at 4. This lowers time to first audio; `--chunk-first`/`--chunk-max` set it back.
- **Default length**: with `max_new_tokens` absent or `0`, a piece is capped at the model default
  of 750 frames (60 s), as in C++. The old Python API used 1,500.

### Fixed

- An inline reference wav now encodes to the same codec codes regardless of which server process
  handled the request, on the same GPU and software stack (GPU, driver, CUDA, cuDNN, torch), so
  the same reference, text and seed give the same audio across a restart there. (Previously,
  `--fast-all`'s process-wide `cudnn.benchmark = True` let cuDNN settle on a different,
  differently-rounding conv algorithm per process for the reference encode step.) A different
  stack, or cuDNN falling back to another deterministic engine under workspace pressure, may
  still give different codes; see research.md R18.
