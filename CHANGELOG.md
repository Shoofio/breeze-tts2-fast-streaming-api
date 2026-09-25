# Changelog

All notable changes to this project are documented here. See
`specs/003-cpp-compatible-api/spec.md` for the full breaking-changes list and rationale (IDs
`BC-nn`).

## Unreleased — 2.0.0

The previous Python HTTP API (the pre-2.0 request/response shapes) is removed. The server now
exposes a C++-server-compatible HTTP and WebSocket API; see the README and
`specs/003-cpp-compatible-api/contracts/http-api.md` for the full contract.

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
  skipped, and the server logs `speech.anchor_skipped`, when the first piece was cut off at its
  frame limit or when the anchor would shorten any later piece; the later pieces then use voice
  design. The opening piece is at most 200 characters, or `split_chars` if that is smaller.

Later phases append their own `BC-nn` entries to this section as they land.
