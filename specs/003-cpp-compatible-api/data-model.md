# Data Model: C++-Compatible API (Fixed)

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

These are the entities that the boundary modules own (Constitution IV). Wire shapes are in
[contracts/](contracts/). This file covers fields, validation and state.

## Settings (launch configuration)

A frozen value object, built once at the composition root from the command line.

| Field | Default | Rule |
|---|---|---|
| `model_path` | required | Existing directory |
| `host` | `127.0.0.1` | HTTP and WebSocket both bind here only |
| `port` | `8080` | 1–65535 |
| `ws_port` | `port + 1` | 1–65535, or `disabled` |
| `cors` | off | `*` or a comma-separated allowlist: entries are trimmed, validated and canonicalized (lowercase scheme/host, IDN to punycode, IP literals canonicalized, default port dropped), then deduplicated; an incoming `Origin` is canonicalized the same way before matching. Startup errors: `*` mixed with other entries, an empty list, or an entry that isn't a bare `scheme://host[:port]` origin (it could never match a browser `Origin`) |
| `split_chars` | `600` | ≥ 0 |
| `chunk_first` / `chunk_max` | `1` / `25` frames | ≥ 1; `chunk_first` is clamped to `chunk_max` |
| `voices_dir` | `voices` | Created if missing |
| `fast_*`, `attn_implementation`, `compile_cache_dir` | as today | Unchanged runtime flags |

`chunk_first` defaults to 1, not C++'s 4, to keep today's time to first audio. This is a latency
choice and doesn't change the contract. Limits are constants in `limits.py` rather than options
(Principle II: no option without a caller).

## Limits (constants)

| Name | Value | Used by |
|---|---|---|
| `MAX_BODY_BYTES` | 26 MiB | HTTP body limit (`413`) |
| `MAX_AUDIO_BYTES` | 25 MiB | `ref_audio` part |
| `MAX_REF_SECONDS` | 30 | Reference duration |
| `MAX_TEXT_CHARS` | 10,000 | `text`; WebSocket buffer plus incoming text |
| `MAX_REF_TEXT_CHARS` | 2,000 | `ref_text` |
| `MAX_INSTRUCTION_CHARS` | 2,000 | `instruction` |
| `MAX_NEW_TOKENS_CEILING` | 1,500 | `max_new_tokens` upper bound |
| `ANCHOR_CHARS` | 200 | Opening budget with no reference |
| `UNNAMED_VOICE_CAP` | 64 | Unnamed voices in memory |
| `WS_MAX_MESSAGE_BYTES` | 1 MiB | Inbound WebSocket message |
| `WS_MAX_CONNECTIONS` | 16 | Concurrent WebSocket connections |
| `WS_HANDSHAKE_SECONDS` | 10 | `open_timeout` |
| `WS_OUTBOX_BYTES` | 2 MiB | Outgoing backlog per connection |
| `HTTP_SEND_TIMEOUT_SECONDS` | 30 | Per-chunk send on streamed speech |
| `TCP_USER_TIMEOUT_MS` | 30,000 | Listening sockets |

## SpeechRequest (HTTP boundary → synthesis)

Produced by `http_fields.parse_speech()`. Every field is already validated.

| Field | Type | Source field and rule |
|---|---|---|
| `text` | str | `text`: required, non-blank, ≤ `MAX_TEXT_CHARS`, no control characters except `\t\r\n` |
| `instruction` | str | `instruction`: absent or blank → `"Speak clearly and naturally."`; ≤ 2,000; control-character rule |
| `reference` | `ReferenceSpec` | See below |
| `cfg_scale` | float | Default 1.0; finite, 0–100 |
| `seed` | int | Default 42; 0–4294967295 |
| `temperature` | float or None | `0` or absent → None (model default); otherwise 0 < x ≤ 10 |
| `top_k` | int or None | `0` or absent → None; otherwise 1–10,000 |
| `top_p` | float or None | `0` or absent → None; otherwise 0 < x ≤ 1 |
| `repetition_penalty` | float or None | `0` or absent → None; otherwise 1e-4 ≤ x ≤ 10 |
| `max_new_tokens` | int or None | `0` or absent → None (750); otherwise 1–1,500 |
| `split_chars` | int | Absent → `settings.split_chars`; otherwise 0–10,000 (`0` = no length splitting) |

Rules that apply to every field:
- An empty value counts as absent.
- A field that appears more than once (within the query string, within the body, or in both)
  gets `400 duplicate_field`.
- Numbers must match a strict decimal grammar (integers match `^[+-]?\d+$`, with ASCII digits
  only, matching contracts/http-api.md). Non-finite values are rejected.

### ReferenceSpec

A tagged union:

| Variant | Fields | When |
|---|---|---|
| `NoReference` | — | no `voice_id` and no `ref_audio` |
| `VoiceRef` | `voice_id`, `ref_text_override` or None | `voice_id` present, no `ref_audio` |
| `InlineRef` | `audio: DecodedAudio`, `ref_text` | `ref_audio` and `ref_text` present, no `voice_id` |

Validation, in this order (all `400`):
1. `voice_id` together with `ref_audio` gives `reference_conflict`.
2. `ref_audio` without `ref_text` gives `ref_text_required`.
3. `ref_text` with neither gives `reference_required`.

## DecodedAudio

Produced by `reference_audio.decode()`: mono float32 samples, a sample rate, a duration, and a
predicted frame count.

The header checks run before anything is decoded:
- format is WAV, WAVEX, FLAC or OGG;
- 1–8 channels;
- 8,000–192,000 Hz.

The header's own frame count decides duration up front only when it's trustworthy: a real file
never reports anywhere near libsndfile's own "unknown length" sentinel (observed as `INT64_MAX`
for a FLAC whose STREAMINFO `total_samples` was left at 0 -- what a streaming encoder emits, e.g.
one piped through `ffmpeg`). A trustworthy count over 30 s is rejected without decoding.
Otherwise the actual decoded length decides instead: the decode is bounded at one sample past
30 s, so an untrustworthy-length file still can't allocate or read past that regardless of what
its header claims. Ending early there is expected rather than an error, once the underlying bytes
are almost fully consumed -- a check skipped when the recovered length is already going to be
rejected as too short, below. **Known limit**: for a FLAC with unknown length (piped), a stream cut
short -- between frames or inside one -- can be accepted with fewer samples than the original;
nothing here can tell a deliberately or accidentally shortened but well-formed stream apart from a
genuinely short recording without an independently known length. A size-based plausibility check
(an upper bound on compressed bytes per sample) was tried and removed: it was both too strict --
it rejected valid `ffmpeg` output, a short final frame at the end of a `-frame_size`-encoded
stream -- and too loose, since a cut landing inside a frame could still look plausible. Genuinely
corrupted data is still rejected reliably: libsndfile itself raises a distinct, non-tolerated
decode failure for that, separate from the one signalling a stream's real end.

Failures:
- empty upload gives `invalid_audio`;
- any of the header checks above failing gives `invalid_audio`;
- unreadable audio gives `invalid_audio` (a genuine decode failure, distinct from a stream that
  legitimately ends early because its length was never known);
- non-finite or absurdly large samples (`|sample| > 8.0` after the downmix) give `invalid_audio`;
- more than 30 s -- whether known from a trustworthy header or from the actual decoded length --
  gives `audio_too_long`;
- shorter than one full codec frame gives `audio_too_short`: exactly 80 ms, checked as
  `n_samples * 24000 < 1920 * sample_rate` in integer arithmetic on the native sample count. The
  codec itself rounds a partial frame up, so without this minimum a 1-sample clip would become
  one frame of mostly padding (decided with the user, 2026-09-25).

## Reference (synthesis-internal)

The resolved reference for a request or session. It is never exposed on the wire.

| Variant | Content |
|---|---|
| none | voice design |
| codes | `codes: int tensor [frames, codebooks]`, `ref_text` (inline, a voice with an overridden transcript, or an anchor) |
| prefix | `ReferencePrefix` (cached KV) plus the stored `ref_text`, for a saved or unnamed voice with no override |

Transitions: `none` becomes `codes(anchor)` after piece 0 succeeds with at least one non-pad
frame. It never happens on cancel, on failure, or when piece 0 produced zero frames.

## Piece

`(index, text, seed = (request_seed + index) & 0xFFFFFFFF)`.
- Its token cap is `min(max_new_tokens or 750, room)`.
- A room of 0 fails: `400 text_too_long` for piece 0 before streaming, or an aborted stream for a
  later piece.
- A room that is smaller than the cap clamps the piece and emits `speech.piece_clamped`.

## Voice

| Field | Type | Notes |
|---|---|---|
| `id` | str | Saved: the name, `[A-Za-z0-9_-]{1,64}` and not `v_*`. Unnamed: `v_` + 16 lowercase hex characters |
| `ref_text` | str | ≤ 2,000 |
| `codes` | int16 `[frames, codebooks]` | Every value within `[0, codebook_size)` |
| `codes_sha256` | hex | Part of the prefix-cache key |
| `frames` | int | ≥ 1 |
| `seconds` | float | `frames × samples_per_frame / sample_rate` |
| `encode_ms` | float | Measured at creation |
| `saved` | bool | True when named |

**Wire shape** (unchanged from C++):
`{"id", "frames", "seconds" (2 decimals), "encode_ms" (integer ms), "saved", "ref_text"}`.

**Registry rules**:
- Names are unique ignoring case, including names reserved by skipped files.
- Unnamed voices are capped at `UNNAMED_VOICE_CAP`, and the oldest unnamed voice is evicted first.
- An identical unnamed registration (same id) returns the existing entry.
- List order: saved voices sorted by id, then unnamed voices in registration order.

**Lifecycle**:

```text
POST (named)   → [validate] → [409 if name taken, ignoring case] → [decode] → [gate] → [encode]
               → [write file under lock, re-checking the name] → registered (saved)
POST (unnamed) → [validate] → [hash → existing? return it] → [decode] → [gate] → [encode]
               → registered (unnamed; may evict the oldest unnamed voice)
DELETE         → saved: rename to .del-*, then unlink → unregistered; prefix cache entry dropped
               → unnamed: unregistered; prefix cache entry dropped
               → skipped-file name: file removed, name released
restart        → saved voices reloaded from files; unnamed voices gone
```

## Voice file v1

Path: `<voices_dir>/<id>.voice.json`, UTF-8 JSON. A consumer rejects any file whose `format` or
`version` it doesn't know.

```json
{
  "format": "breeze-tts-voice",
  "version": 1,
  "id": "alice",
  "ref_text": "…",
  "frames": 375,
  "codebooks": 16,
  "codes": "<base64 of int16 little-endian, row-major [frames][codebooks]>",
  "codes_sha256": "<hex sha256 of the decoded bytes>",
  "codec_fingerprint": "<hex sha256 of the codec config's required identity fields (sample rates, quantizer counts, codebook sizes/dims, and the decoder's upsample_rates AND upsampling_ratios -- two distinct required fields, not aliases -- from both the encoder and decoder blocks of audio_tokenizer/config.json) ‖ the safetensors header ({tensor: [dtype, shape]} only, no tensor data) of the ONE weight file the codec loader actually reads (model.safetensors, else the sharded index's file list, in that order -- transformers' own resolution order) -- detects an architecture/shape/dtype change, not a retrain of same-shaped weights>",
  "encode_ms": 812,
  "created_at": "2026-09-24T20:15:00Z"
}
```

A file is skipped at startup, with a `voice.skipped{file, reason}` event, when any of these holds:
- JSON or schema error;
- the file stem differs from `id`;
- invalid name or a `v_` prefix;
- a case-duplicate of a file that sorts earlier;
- the byte length or sha256 of `codes` doesn't match;
- a code is out of range;
- the fingerprint doesn't match.

Other files in the directory:
- Leftover `.del-*` and `.tmp-*` files are removed.
- `*.breeze` files are counted and reported in `voices.loaded{loaded, skipped, breeze_ignored}`.

## WebSocket Session

| Field | Notes |
|---|---|
| `epoch` | Incremented by `cancel`, by `start` with pending work, and on disconnect |
| `config` | Snapshot from `start`: reference, instruction, cfg, seed, sampling, `split_chars`. `instruction` is mutable via the `instruction` message and read when each piece starts |
| `buffer` | Pending text: the unfinished clause after the last complete sentence or closed clause, weighing at most max(2 × budget, 93), where 93 is one capped grapheme cluster of 31 non-ASCII code points (with `split_chars` 0, unbounded except by the character limit); total characters ≤ 10,000 |
| `opening_pending` | True while there is no reference and no anchor yet (and none queued) |
| `anchor` | Reference built from piece 0 |
| `piece_index` | Resets at `start` |
| `work` | Deque of `Piece(epoch)`, `EndMark(epoch)`, `CancelMark`, `StartMark` |

**States**: `connected` (after `ready`, before a valid `start`), then `started` (which is idle or
busy depending on `work`), then `closed`.
- Any message other than `start` before `started` gets `error{code: not_started}`.
- A `start` with an unknown `voice_id`, or with `ref_text` and no `voice_id`, gets an `error`, and
  the state is unchanged.

**Worker loop**: one work item at a time.

| Item | Action |
|---|---|
| `CancelMark` | Send `cancelled` |
| `StartMark` | Send `started` |
| `EndMark(e)` | Send `done` if `e` is the current epoch |
| `Piece` | Acquire the gate (its `on_wait` callback enqueues `queued` before blocking), skip if its epoch is stale, send `speaking`, stream PCM, anchor on success; on failure send `error{code: generation_failed}` and keep the session |

## Error

`{error: str, code: str}` on HTTP. On WebSocket:
`{"type": "error", "message": str, "code": str, "request_type": str | null}`, where `request_type`
is the client message type that caused it. The full code catalog is in
[contracts/http-api.md](contracts/http-api.md#error-codes) and
[contracts/ws-api.md](contracts/ws-api.md#error-codes).

## Events (structured telemetry)

JSON lines produced by `Emitter(sink, clock).emit(name, *, level="info", **fields)`. The sink and
clock are injected at construction (no module-level global emitter); `level` is one of `debug`,
`info`, `warning`, `error`. A field that collides with a reserved key (`ts`, `event_schema`,
`event`) or an invalid `level` raises `ValueError`; `level` itself is keyword-only, so a field
named `level` can never collide with it. A field value that can't be serialised (NaN/inf, an
unencodable type, or anything else that blows up mid-serialisation) does not raise — the line
written is an `event.invalid` record instead (`level="warning"`, keeping `event_schema`, `ts`,
any of `request_id`/`session_id`/`piece_index` that were present, and a `fields` list of the
offending record's field names), so one bad field never breaks the request emitting it.

- Every request event carries `request_id`; WebSocket events also carry `session_id` and
  `piece_index`.
- Event names:
  - server: `server.started` (with every bound `addresses`), `server.bind_failed`,
    `server.stopping` (exactly one when the process hard-exits or `serve()` crashes; none on
    a clean stop. `reason`: the hard-exit reason, `load in progress`, `signal during drain`,
    `gpu drain timed out` (exits 70), `gpu stop cancelled` or `gpu stop failed` (exits 70),
    else `serve raised`; if `serve()` crashed, the exit code is the crash's own; optional
    `crash` and `stop_error`, each a formatted traceback; `level` is `error` when either is
    present, else `warning`. A cancellation is never a crash or a stop failure), `ws.bind_failed`,
    `model.loaded`,
    `model.load_failed` (the process then exits non-zero), `gpu.close_failed`,
    `gpu.close_timeout` (a `gen.close()` ran past 30 s: the GPU gate is poisoned and `/health`
    answers `503 gpu_unavailable` until restart);
  - voices: `voices.loaded`, `voice.skipped`, `voice.created`, `voice.deleted`;
  - speech: `speech.accepted`, `speech.first_audio` (`ttfa_ms`), `speech.piece_clamped`,
    `speech.piece_done` (`piece_index`, `frames`), `speech.completed` (`rtf`),
    `speech.failed`, `speech.aborted`, `speech.frame_prediction_mismatch`
    (`predicted_frames`, `actual_frames`);
  - WebSocket: `ws.connected`, `ws.rejected` (`reason`), `ws.closed` (`code`), `ws.piece`;
  - errors: `request.failed`.
