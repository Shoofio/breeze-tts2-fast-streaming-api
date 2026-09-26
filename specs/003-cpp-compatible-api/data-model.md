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
| `VOICE_PREFIX_CACHE_BYTES` | 1 GiB | Estimated KV the voice prefix cache holds (about the old server's 16 prefixes × 548 tokens × 114,688 B) |
| `MAX_VOICE_FILE_BYTES` | 256 KiB | Largest `*.voice.json` the startup scan reads (about six times the largest valid file) |
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

A voice with an overridden transcript uses the codes path: its stored codes with the given
`ref_text`. The room check before the gate sizes a voice on the path it will take: without an
override, the prefix path, by its stored `prefix_len` (a stored prefix longer than the model builds,
`prefix_len > max_seq_len - 1 - MIN_SUFFIX_ROOM`, is `400 text_too_long` there); with one, the codes
path, by its whole prompt.

A prefix build that runs out of GPU memory, while assembling its inputs or while building, is not
cached and frees what it held on the GPU thread before anything else runs there; it is never a
`500` (T066). The request then evicts every cached prefix (`voice.prefix_evicted`, reason `oom`),
since the codes path needs more memory than the build did, and empties PyTorch's cache on the GPU
thread. Whether it may fall back to the codes path is decided only then (decided with the user on
2026-09-26, reversing review 42's rule that piece 0 must fit both paths before the gate, which
refused a voice near the limit on every request because of a rare out-of-memory): piece 0's room
on the codes path is measured on the CPU tokenizer's worker. With room, the request uses the codes
path with the stored `ref_text` (`speech.prefix_fallback`, reason `out_of_memory`); without, it is
refused before the `200` with `503 gpu_out_of_memory` (`speech.prefix_fallback`, reason
`no_room`). Only piece 0 is sized for the codes path then, so after a fallback a later piece can
still find no room and abort the stream after the `200` (BC-47): a case that only follows an
out-of-memory. Later pieces are always sized when they run, on the path the request took.

Transitions: `none` becomes `codes(anchor)` after piece 0 succeeds with at least one non-pad
frame. It never happens on cancel, on failure, or when piece 0 produced zero frames. It is also
skipped, with `speech.anchor_skipped` (`piece_index` 0, `reason`), when:
- `piece_truncated`: piece 0 used its whole frame limit (its cap or its room), so it stopped
  there rather than at EOS;
- `no_room`: any later piece would end up with a smaller effective frame limit --
  `min(cap, room)` -- with the anchor than without it, whether or not either room already
  clamped it below its cap (decided with the user, 2026-09-25). The anchor is never trimmed
  to fit: its codes must stay paired with its text.
  Every later piece's prompt length without an anchor, and what the anchor's text and each of
  its frames add, are measured on the CPU right after the GPU gate is taken, alongside piece 0's
  own preparation and generation on the GPU thread; no prompt is kept. Once piece 0 has finished,
  and before piece 1 starts, the decision is arithmetic on those lengths plus the anchor's frame
  count, with each room taken from the runtime (CFG rows and prefill bucket padding included).
- `sizing_failed`: measuring the later pieces raised (a template or tokenizer error). The error
  is also reported as `speech.anchor_sizing_failed`, not `request.failed`: the stream goes on,
  and the request can still complete.
- `sizing_timeout`: the measurement was still unfinished `ANCHOR_SIZING_TIMEOUT_SECONDS` (5 s)
  after piece 0 finished. A backstop only, since the measurement has a CPU worker of its own.
  The GPU thread does not wait longer: a disconnect's close would queue behind it, and a close
  past 30 s poisons the GPU gate.
- `shutdown`: server shutdown cancelled the measurement before it ran.

A skipped anchor leaves the later pieces as voice design. The measurement runs on a CPU worker
of its own, with its own tokenizer copy, not the one every request's room check before the gate
queues on, so a burst of requests can't delay it. It is cancelled, if still queued, as soon as
nothing will read it: when the anchor is decided without it (`piece_truncated`, zero non-pad
frames, `sizing_timeout`), and when the request ends before the anchor decision (an error, a
`400`, or a disconnect before or after the `200`). A cancel by the request as it ends emits no
`speech.anchor_skipped`: the stream is over anyway.

## Piece

`(index, text, seed = (request_seed + index) & 0xFFFFFFFF)`.
- Its model inputs are built on the GPU thread when the stream reaches it, not ahead of time, and
  are not kept once it has been generated. For a later piece with no reference, only its prompt
  length is measured ahead, for the anchor decision ("Reference"): after the GPU lease is taken,
  on the anchor-sizing worker, while piece 0 generates.
- Its token cap is `min(max_new_tokens or 750, room)`.
- A room of 0 fails: `400 text_too_long` for piece 0 before streaming, or an aborted stream for a
  later piece.
- A room that is smaller than the cap clamps the piece and emits `speech.piece_clamped`
  (`piece_index`, `requested`: the client's `max_new_tokens` or null for the default, `cap`:
  the server-clamped value, `room`).

## Voice

| Field | Type | Notes |
|---|---|---|
| `id` | str | Saved: the name, `[A-Za-z0-9_-]{1,64}` and not `v_*`. Unnamed: `v_` + 16 lowercase hex characters |
| `ref_text` | str | ≤ 2,000 |
| `codes` | int16 `[frames, codebooks]` | Every value within `[0, codebook_size)`. A new voice's codes pass the same checks a voice file does at startup (frames, codebooks, range, `encode_ms`), then are kept as int16 |
| `codes_sha256` | hex | Checksum of `codes` in the voice file |
| `frames` | int | ≥ 1 |
| `seconds` | float | `frames × samples_per_frame / sample_rate` |
| `encode_ms` | float | Measured at creation |
| `saved` | bool | True when named |
| `prefix_len` | int | The length its reference prefix builds to, measured once (at registration or the startup scan) with the same prefix-input assembly the build uses; never stored in the file |
| `prefix_key` | `(id, content_hash)` | Its prefix-cache key (below), computed once with `prefix_len` |

**Prefix-cache key**: `(id, content_hash)` from `voice_file.prefix_key`, where `content_hash` is
a sha256 over the length-prefixed `ref_text`, the codes' shape and the codes' bytes. The KV
depends on the transcript as well as the audio, so the same codes with another `ref_text` are a
different key. Deletes are ordered against builds by one counter: a request reads the cache's
token when it resolves the voice, and a build by a request that resolved the voice before its
delete is never cached, even if the voice is re-registered with the same audio and text.

**Wire shape** (unchanged from C++):
`{"id", "frames", "seconds" (2 decimals), "encode_ms" (integer ms), "saved", "ref_text"}`.

**Registry rules**:
- Names are unique ignoring case, including names reserved by skipped files. Case is ignored
  only at create: `DELETE` matches an id exactly, as `GET /v1/voices` lists it, or a skipped
  file's exact file stem.
- Unnamed voices are capped at `UNNAMED_VOICE_CAP`, and the oldest unnamed voice is evicted first.
- An identical unnamed registration (same id) returns the existing entry.
- List order: saved voices sorted by id, then unnamed voices in registration order.

**Lifecycle**:

```text
POST (named)   → [validate] → [409 if name taken, ignoring case] → [decode] → [prefix fits?
                 else 400 voice_too_long] → [gate] → [encode] → [codes valid? else 500]
               → [write file under lock, re-checking the name; a registry refusal after the
                 write removes the file again → 409] → registered (saved)
POST (unnamed) → [validate] → [hash → existing? return it] → [decode] → [prefix fits? else
                 400 voice_too_long] → [existing now? return it] → [gate] → [encode] → [codes
                 valid? else 500] → registered (unnamed; may evict the oldest unnamed voice),
                 unless an identical request registered it first: its entry is returned
DELETE         → saved: rename to .del-* and fsync the directory, then unlink (best-effort: a
                 failure leaves the .del-* file for the next startup's sweep) → unregistered;
                 prefix cache entry dropped
               → unnamed: unregistered; prefix cache entry dropped
               → skipped-file name: file removed, name released; a skipped entry that is a
                 directory is refused (`500 voice_delete_failed`) and stays reserved
restart        → saved voices reloaded from files; unnamed voices gone
```

**Startup order**: model load (on the GPU thread) and sizing the prefix cache from the model's
config, then the voice scan (on a worker thread: checking a file needs the loaded model's codebook
facts and codec fingerprint, and each loaded voice's `prefix_len` is measured with the load's CPU
tokenizer copy), then mark ready. Every route answers `503 loading` until the scan has finished,
and `voices.loaded`/`voice.skipped` come before `model.loaded`. A model config the prefix cache
can't be sized from fails startup with `stage: model`. A voices directory that can't be created,
listed or fingerprinted (the codec's `audio_tokenizer` files) fails startup: `model.load_failed`
with `stage: voices`, exit non-zero, never ready (decided by the user on 2026-09-26). A bad single
file is still only skipped (BC-25). A voice whose prefix no longer fits the context (a file from a
larger `max_seq_len`) still loads, and a request for it is `400 text_too_long` before the gate.

**Changes**: every store and registry change (create, register, delete) runs on one worker thread
of the voice services' own, never the event loop or asyncio's shared pool, one at a time. What
goes with a change on the event loop (dropping a cached prefix, the `voice.*` events) is scheduled
by that thread right after the change, so a request cancelled mid-change still gets them. A
change that arrives after server stop has shut that thread down, or that was still queued when it
did, answers `503 gpu_unavailable`, as the CPU tokenizer's room check does in the same race. A
change that finishes after the event loop has closed keeps its result; what goes with it is
dropped.

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
- it is larger than `MAX_VOICE_FILE_BYTES` (only that much is read);
- it can't be stat'ed (the reason is the OS error), or isn't a regular file (a directory, say:
  reason `not a regular file`);
- JSON or schema error: not UTF-8, a missing field, a field of the wrong JSON type (integers
  must be integers, not floats or booleans), a `ref_text` that is blank, over 2,000
  characters, holds control characters (BC-46, the same rule as `POST`) or a lone surrogate,
  `frames` outside 1 to `reference_audio.MAX_REF_FRAMES` (376: the most a 30 s reference
  encodes to at any accepted sample rate), `codebooks` other than the model's codebook count,
  `encode_ms` outside 0 to 3,600,000 (an hour), or a `created_at` not exactly
  `YYYY-MM-DDTHH:MM:SSZ` and a real date;
- the file stem differs from `id`;
- invalid name or a `v_` prefix;
- a case-duplicate of a file that sorts earlier, whether that file loaded or was skipped;
- the byte length or sha256 of `codes` doesn't match;
- a code is out of range;
- the fingerprint doesn't match;
- the voice's prefix can't be measured (its `ref_text` breaks the prompt template, say): the
  reason is the error. This check runs after the scan, so this `voice.skipped` follows
  `voices.loaded`, which counted the file as loaded; the name is reserved as for any other
  skipped file, and a `DELETE` of it removes the file.

Other files in the directory:
- Leftover `.del-*` and `.tmp-*` files are removed. One that can't be removed or stat'ed, or a
  directory with such a name, is reported with `voice.cleanup_failed` and left.
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
    a clean stop. `reason` is the hard-exit reason if there is one, else `serve raised`. The
    hard-exit reasons: `load in progress`, `signal during drain` and `gpu stop cancelled`
    (the GPU may be busy, but someone asked to stop: not a GPU failure), and
    `gpu drain timed out` and `gpu stop failed` (the GPU failures, which exit 70 unless
    `serve()` crashed with a non-zero code, in which case the crash's code wins; a cancel
    landing after the drain had already timed out or raised still reports that failure, with
    the drain's own traceback as `stop_error`). A crash exits with its own code (1,
    `SystemExit`'s code, 130 for Ctrl+C), normalised to what the process can really exit
    with: on POSIX its low 8 bits (257 exits 1, 256 exits 0); on Windows the code itself, or
    1 if it doesn't fit in a C int. After a GPU failure, a normalised code of 0 becomes 70: a
    GPU failure never exits 0. Optional `crash` and `stop_error`, each a formatted
    traceback; `level` is `error` when either is present, else `warning`. A cancellation is
    never a crash or a stop failure), `ws.bind_failed`,
    `model.loaded`,
    `model.load_failed` (the process then exits non-zero; `stage`: `model` (the load itself, or a
    model config the voice prefix cache can't be sized from),
    `voices` (the startup voice scan) or `ready` (marking the server ready or reporting it)),
    `gpu.close_failed`,
    `gpu.close_timeout` (a `gen.close()` ran past 30 s: the GPU gate is poisoned and `/health`
    answers `503 gpu_unavailable` until restart);
  - voices: `voices.loaded`, `voice.skipped`,
    `voice.created` (`request_id`, `voice_id`, `saved`, `frames`, `encode_ms`; only when this
    request created the entry: none for an unnamed `POST` answered with an existing entry, even
    one that encoded before finding it), `voice.deleted` (`request_id`, `voice_id`, `kind`:
    `saved`, `unnamed` or `reserved`), `voice.store_mismatch` (level `warning`; `request_id`,
    `voice_id`, `file_removed`, `kind`: a `DELETE` where the store and the registry disagreed
    about whether the voice has a file, e.g. a file removed for an id the registry didn't hold
    (`kind` null, no `voice.deleted`); the delete still answers `200`. Also a named `POST` whose
    name the registry refused after the file was written, when taking the file back out failed:
    `file_removed` false, `kind` null, plus `error`; the file stays on disk and the `POST` answers
    `409 voice_exists`, since the name is taken either way),
    `voice.frame_prediction_mismatch` (level `warning`; `request_id`, `predicted_frames`,
    `actual_frames`: a `POST /v1/voices` encode, as `speech.frame_prediction_mismatch`),
    `voice.cleanup_failed` (`file`, `op`: `sweep`, `stat`, `unlink_tmp` or `unlink`, `error`: a
    leftover the store couldn't remove or examine; it is left for the next startup's sweep),
    `voice.prefix_built` (`voice_id`, `tokens`: the prefix length, `bytes`: its estimated KV
    size, `request_id`), `voice.prefix_evicted` (`voice_id`, `reason`: `budget` (dropped, least
    recently used first, to make room), `deleted` (the voice was deleted, including a build by a
    request that resolved the voice before its delete, which is returned to that request but not
    cached), `too_large` (a prefix bigger than the whole budget, returned to its request but
    never cached) or `oom` (every cached prefix, dropped after a prefix build ran out of GPU
    memory, "Reference"); `request_id`: the request whose build caused it, or the DELETE's), and
    `voice.prefix_build_failed` (`voice_id`, `error`, `request_id`: a build that raised after
    its request was cancelled, so no caller received the error). All three are emitted only
    after the cache's state is final, and an `on_event` error is logged through the event
    loop's exception handler without failing the request or the delete. The prefix cache holds
    at most `VOICE_PREFIX_CACHE_BYTES` (1 GiB) of estimated KV, where an entry costs
    `prefix_len` × 2 (key and value) × layers × KV heads × `head_dim` × dtype size (114,688 B per
    token for this checkpoint). It isn't warmed at startup: a voice's first request builds its prefix;
  - speech: `speech.accepted` (`pieces`, `reference`: `none`, `inline`, `voice_prefix` (with
    `warm`: whether the prefix was already cached) or `voice_codes`), `speech.prefix_fallback`
    (level `warning`; `voice_id`, `error`, `reason`: a voice's prefix build ran out of GPU memory,
    and the request either used the codes path (`out_of_memory`) or, with no room for piece 0
    there, was refused with `503 gpu_out_of_memory` (`no_room`); T066, "Reference"),
    `speech.first_audio` (`ttfa_ms`), `speech.piece_clamped`
    (`piece_index`, `requested`, `cap`, `room`), `speech.anchor_skipped` (`piece_index`,
    `reason`: `piece_truncated`, `no_room`, `sizing_failed`, `sizing_timeout` or `shutdown`),
    `speech.anchor_sizing_failed` (level `error`; `error`, `traceback`: measuring the later
    pieces for the anchor raised, and the request goes on without the anchor),
    `speech.piece_done` (`piece_index`, `frames`), `speech.completed` (`rtf`),
    `speech.failed`, `speech.aborted`, `speech.frame_prediction_mismatch`
    (`predicted_frames`, `actual_frames`);
  - WebSocket: `ws.connected`, `ws.rejected` (`reason`), `ws.closed` (`code`), `ws.piece`;
  - errors: `request.failed`.
