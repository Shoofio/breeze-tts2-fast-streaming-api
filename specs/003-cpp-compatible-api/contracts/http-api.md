# HTTP API Contract — v2.0.0

**Contract version**: `2.0.0`, the server version in `breeze_infer.__version__`. Paths keep `/v1`
for C++ client compatibility. Breaking changes from the C++ server are tagged `BC-nn` (see
[spec.md](../spec.md#breaking-changes-from-the-c-server)). Anything not marked is the same as C++
(`docs/server.md`, `docs/voices.md` at the reference commit).

## Common rules

**Transport and body**
- HTTP/1.1 on `--host:--port` (default `127.0.0.1:8080`).
- Request bodies: `multipart/form-data` or `application/x-www-form-urlencoded`. JSON bodies are not
  accepted.
- Fields may also come from the query string. A field present more than once, anywhere, gets
  `400 duplicate_field` (BC-08).
- An empty field value means absent (BC-02).
- Body larger than 26 MiB: `413 payload_too_large`, whether or not `Content-Length` is set (BC-06).

**Version header**: every response, including errors, preflights and streamed speech, carries
`X-Breeze-Version: 2.0.0` (the server version; `2.0.0.devN` during development). This is additive
and lets consumers pin the contract version (FR-037a).

**Errors**
- Every error has body `{"error": "<message>", "code": "<code>"}` and
  `Content-Type: application/json`. This includes `404`, `405` (with an `Allow` header), `413` and
  `500` (BC-18).
- Every `500` closes the connection.

**Validation order** (BC-07):
1. body size (`413`);
2. field syntax and ranges (`400`);
3. reference consistency (`400`);
4. unknown `voice_id` (`404`);
5. audio decode and limits (`400`);
6. busy (`409`).

**Busy and loading**
- Only one generation or encode runs at a time. HTTP never waits: when busy it returns
  `409 {"error":"busy","code":"busy"}`. HTTP also returns `409` while a WebSocket piece is
  running or queued.
- Until the model is ready, every route returns
  `503 {"status":"loading","error":"model is loading","code":"loading"}`.
- Once the GPU has stopped responding (a generation's cleanup ran past 30 s), every route
  returns `503 {"status":"error","error":"gpu is not responding","code":"gpu_unavailable"}`
  until the server is restarted.

## CORS (launch option; off by default)

Off by default, which means no CORS headers are sent and every `OPTIONS` request gets
`405 method_not_allowed`.

When CORS is enabled with `*` or an allowlist (entries trimmed, matched exactly, BC-19):

| Request | Response |
|---|---|
| Any response to an allowed `Origin` | `Access-Control-Allow-Origin: <origin or *>` and `Access-Control-Expose-Headers: X-Sample-Rate, X-Sample-Format, X-Breeze-Version`. Applies to errors, `500`s and streams |
| Any response in allowlist mode | `Vary: Origin` (BC-20) |
| `OPTIONS` preflight on an existing path | `204`, `Access-Control-Allow-Methods: <that route's methods>, OPTIONS`, `Access-Control-Allow-Headers: <echo of the request>`, `Access-Control-Max-Age: 86400` (BC-21) |
| `OPTIONS` preflight on an unknown path | `404 not_found` |
| `OPTIONS` preflight asking for a method the route lacks | `405 method_not_allowed` |
| `OPTIONS` preflight from a disallowed origin | `403 origin_not_allowed` |

Independent of the CORS setting: a `POST` or `DELETE` whose `Origin` header is present and not
allowed gets `403 origin_not_allowed` before any work is done. With CORS off, no origin is
allowed (BC-23). Requests without `Origin` are unaffected.

## `GET /health`

**`200`** with exactly `{"status":"ok","sample_rate":24000,"ws_port":8081}` and no other fields.
- `sample_rate` is the loaded model's rate.
- `ws_port` is the port that is actually listening, or `0` when the WebSocket is disabled or
  failed to bind (BC-24).

**`503`** with the loading body while the model loads, and with the `gpu_unavailable` body
once the GPU has stopped responding (it stays so until restart, so a supervisor can restart
the server). `HEAD` is supported.

## `POST /v1/audio/speech`

### Fields

| Field | Type | Default | Valid |
|---|---|---|---|
| `text` | string | required | Non-blank (BC-10); ≤ 10,000 characters (BC-05); no control characters except TAB, CR and LF (BC-46) |
| `instruction` | string | `Speak clearly and naturally.` | ≤ 2,000 characters; blank → default (BC-09); control-character rule |
| `voice_id` | string | — | A registered id |
| `ref_audio` | file part | — | WAV, WAVEX, FLAC or OGG; 1–8 channels; 8–192 kHz; ≤ 25 MiB; ≤ 30 s; at least 80 ms, one full codec frame (BC-11, BC-15, BC-16) |
| `ref_text` | string | — | ≤ 2,000 characters; control-character rule |
| `cfg_scale` | decimal | `1.0` | Finite, 0–100. `1` = no CFG; `0` = unconditional only |
| `seed` | integer | `42` | 0–4294967295 |
| `temperature` | decimal | `0` = model default | 0 or (0, 10] (BC-03) |
| `top_k` | integer | `0` = model default | 0–10,000 |
| `top_p` | decimal | `0` = model default | 0 or (0, 1] |
| `repetition_penalty` | decimal | `0` = model default | 0 or [0.0001, 10] (below 1e-4 the logits overflow) |
| `max_new_tokens` | integer | `0` = model default (750) | 0–1,500 (BC-04) |
| `split_chars` | integer | server `--split-chars` (600) | 0–10,000. `0` = no length splitting |

Number grammar: integers match `^[+-]?\d+$`; decimals match `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`.
Anything else is `400 invalid_field` (BC-01).

### Reference rules (BC-12 to BC-14)

| Fields sent | Result |
|---|---|
| `voice_id` and `ref_audio` | `400 reference_conflict` |
| `ref_audio` without `ref_text` | `400 ref_text_required` |
| `ref_text` without `ref_audio` or `voice_id` | `400 reference_required` |
| `voice_id` and `ref_text` | The given text overrides the voice's stored transcript for this request |

### Splitting

Long text is split with the shared segmenter, using `budget = split_chars`.
- With no reference, the first piece gets a soft opening budget of 200. Its generated audio and
  text then become the reference for every later piece.
- Piece `i` is generated with seed `(seed + i) mod 2^32`.

### `200` response

- Headers: `Content-Type: audio/pcm`, `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`,
  `Cache-Control: no-store`, `Transfer-Encoding: chunked`.
- Body: headerless mono signed 16-bit little-endian PCM.
- Chunks grow from `--chunk-first` to `--chunk-max` codec frames (1,920 samples per frame).
- The response is sent only after validation, reference preparation and the first audio chunk have
  all succeeded (BC-17).

### Failure after streaming starts

The connection is closed without the chunked terminator, so clients see an incomplete transfer
(BC-17). This covers a later piece with no room left (BC-47) and generation errors.

Normal endings:
- reaching `max_new_tokens`;
- a piece clamped to the room left (the server records an event).

### Client disconnect

Generation stops within one chunk, and the GPU is released.

### Errors

| Status | Code | Message |
|---|---|---|
| 400 | `invalid_field` | `<field> must be <rule>` |
| 400 | `invalid_field` | `<rule>` per field: numbers "an integer" / "a number"; control characters "free of control characters"; `cfg_scale` "finite and between 0 and 100"; `seed` "an integer between 0 and 4294967295"; `temperature` "0, or greater than 0 and at most 10"; `top_k` "0, or an integer between 1 and 10,000"; `top_p` "0, or greater than 0 and at most 1"; `repetition_penalty` "0, or between 0.0001 and 10"; `max_new_tokens` "0, or an integer between 1 and 1,500"; `split_chars` "an integer between 0 and 10,000"; `instruction`/`ref_text` "at most 2,000 characters"; `voice_id` "a voice name or v_ id" |
| 400 | `invalid_field` | `content type must be multipart/form-data or application/x-www-form-urlencoded` |
| 400 | `duplicate_field` | `<field> was given more than once` |
| 400 | `text_required` | `text is required` |
| 400 | `text_too_long` | `text is too long` (over the limit, or the first piece doesn't fit the context, BC-47) |
| 400 | `reference_conflict` | `voice_id and ref_audio cannot be used together` |
| 400 | `ref_text_required` | `ref_text is required with ref_audio` |
| 400 | `reference_required` | `ref_text needs ref_audio or voice_id` |
| 400 | `invalid_audio` | `could not read ref_audio` |
| 400 | `audio_too_long` | `ref_audio is longer than 30 seconds` |
| 400 | `audio_too_short` | `ref_audio is too short` |
| 404 | `unknown_voice` | `unknown voice_id` |
| 409 | `busy` | `busy` |
| 413 | `payload_too_large` | `request body is too large` |

## `POST /v1/voices`

### Fields

| Field | Type | Valid |
|---|---|---|
| `ref_audio` | file part, required | Same rules as speech |
| `ref_text` | string, required | Non-blank; ≤ 2,000 characters; control-character rule |
| `name` | string, optional | `[A-Za-z0-9_-]{1,64}`; must not start with `v_` (BC-26) |

### Order of checks

1. body size;
2. fields (missing `ref_audio` or `ref_text` → `400 voice_fields_required`; bad name →
   `400 invalid_name`);
3. named voice: a name already taken, ignoring case (including a name held by a skipped file)
   → `409 voice_exists` (BC-27, BC-26);
4. unnamed voice: compute the id; if it already exists → `200` with the existing entry, without
   checking busy;
5. decode and limits (`400`);
6. busy (`409`);
7. encode;
8. named voice: write the file (a name clash found at commit → `409 voice_exists`; write failure →
   `500 voice_write_failed`).

Unnamed ids are `v_` followed by 16 lowercase hex characters, derived deterministically from the
audio bytes and the transcript. Unnamed voices live in memory only, at most 64 of them, and the
oldest unnamed voice is evicted first; saved voices don't count toward the cap (BC-48).

### `200` response

`{"id":"alice","frames":375,"seconds":30.00,"encode_ms":812,"saved":true,"ref_text":"…"}`

- `seconds` has 2 decimals.
- `encode_ms` is an integer.
- `200`, not `201`, is kept from C++.

### Errors

| Status | Code | Message |
|---|---|---|
| 400 | `voice_fields_required` | `ref_audio and ref_text are required` |
| 400 | `invalid_name` | `name can only use letters, digits, dash and underscore` (also used for the `v_` prefix) |
| 400 | `invalid_audio`, `audio_too_long`, `audio_too_short` | as speech |
| 409 | `voice_exists` | `voice already exists` |
| 409 | `busy` | `busy` |
| 500 | `voice_write_failed` | `could not write the voice file` |

## `GET /v1/voices`

`200` with an array of voice objects: saved voices sorted by id, then unnamed voices in
registration order (BC-25).

Files that failed validation at startup are not listed.

## `DELETE /v1/voices/{id}`

**`200`** with `{"deleted":"<id>","file_kept":false}`.
- A saved voice's file is removed, so the voice does not come back after a restart (BC-28).
- An unnamed voice is removed from memory.
- An id reserved by a skipped file has that file removed, and the name is released.
- The cached reference prefix is dropped.
- DELETE never checks busy.

| Status | Code | Message |
|---|---|---|
| 404 | `unknown_voice` | `unknown voice_id` |
| 500 | `voice_delete_failed` | `could not delete the voice file`. The voice stays registered |

## Other routes

| Case | Response |
|---|---|
| Unknown path | `404 not_found` |
| Known path, wrong method | `405 method_not_allowed`, with an `Allow` header |
| `/v1/audio/convert` | `404` (voice conversion is out of scope) |
| Web UI routes (`/`, `/app.js`, `/style.css`) | `404` |

## Error codes

`invalid_field`, `duplicate_field`, `text_required`, `text_too_long`, `reference_conflict`,
`ref_text_required`, `reference_required`, `invalid_audio`, `audio_too_long`, `audio_too_short`,
`voice_fields_required`, `invalid_name`, `unknown_voice`, `voice_exists`, `busy`, `loading`,
`gpu_unavailable`,
`not_found`, `method_not_allowed`, `payload_too_large`, `origin_not_allowed`,
`voice_write_failed`, `voice_delete_failed`, `internal_error`.

Codes are stable identifiers. Messages may change, except where they match the C++ server's
strings.
