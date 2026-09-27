# Research: C++-Compatible API (Fixed)

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-09-24

Sources: an audit of the C++ server (`Breeze-TTS-2.cpp` `server-cors` plus the uncommitted
`ws_api.cpp` fix), an inventory of the current Python API on `perf-and-fixes` (`ee4970f`), a
review of the `api-alignment` branch's reusable work, web-stack experiments run in the repo's venv
(scripts in the session scratchpad, `webstack/q1..q7`), and the SillyTavern extension owner's
answers (`sillytavern-agent`). Citation shorthand: `P:` = `perf-and-fixes` at `ee4970f`, `A:` =
`api-alignment` at `59a3528`, `C++:` = the reference implementation.

## R1. HTTP server stack

**Decision**: Keep FastAPI/Starlette on uvicorn (h11) for HTTP, with fastapi 0.141.1,
starlette 1.6.0 and uvicorn 0.52.4 pinned. All middleware is pure ASGI. Endpoints take `Request`
and parse forms themselves; they do not use `Form()` parameters.

**Rationale**:
- The stack is already a dependency, and the experiments confirmed each behaviour the contract
  needs on it.
- **`BaseHTTPMiddleware` (`@app.middleware("http")`) is banned.** It turns an exception raised
  mid-stream into a clean chunked terminator (`starlette/middleware/base.py:139-146, 236-241`),
  which is exactly C++ defect BC-17.
- **FastAPI `Form()` can't meet the contract.** It silently keeps the last of duplicate fields,
  and its parser limits can't be changed.

**Alternatives considered**: a different framework (unneeded churn); FastAPI `Form()` parameters
(rejected for the reasons above).

## R2. Mid-stream failure and failing before the `200`

**Decision**: Prepare everything, then *prime* the first audio chunk inside the endpoint before
returning the response. A failure during priming becomes a normal JSON `4xx`/`5xx`. After the
headers are sent, exceptions propagate out of the body iterator, and uvicorn closes the connection
without sending a `0\r\n\r\n` terminator.

**Rationale**: This was measured on h11 and on httptools. curl exits with code 18 ("transfer closed
with outstanding read data"), and httpx raises `RemoteProtocolError`. The behaviour comes from
`uvicorn/protocols/http/h11_impl.py:414-438`.

**Alternatives considered**: a trailer or an in-band error marker (C++-contract clients wouldn't
read it); forcing a TCP reset (plain ASGI can't reach the transport, and it isn't needed).

## R3. Streaming response lifecycle: disconnect, slow readers, lock release

**Decision**:
- A `StreamingResponse` subclass owns cleanup.
- Its `stream_response` wraps each `send` in `asyncio.timeout(SEND_TIMEOUT = 30 s)` and
  `aclose()`s the body iterator in `finally`.
- Its `__call__` releases the GPU gate in a shielded `finally`.
- The body is an async generator that awaits `gpu.step(gen)` for each chunk.

**Rationale**:
- Under ASGI spec 2.3, Starlette cancels the body on `http.disconnect`. The measured delay was
  about 20 ms, and the producer stopped within one chunk.
- A client that stays connected but stops reading blocks `send()` forever (`h11_impl.py:461`).
  That would hold the GPU gate indefinitely, and the send timeout bounds it.
- Starlette never calls `aclose()` on the iterator, so its `finally` would otherwise run only at
  garbage collection.
- `BackgroundTask` doesn't run on cancellation.
- `A:`'s `_SpeechResponse` (`api.py:1215-1242`) already had the right shape: the response's
  `finally` owns cleanup. The fixes are listed in R10.

**Minimum delivery rate**: the send timeout only catches a client that stops reading entirely.
A client that trickles (draining a few bytes just before each timeout) would keep every send
under 30 s and hold the single GPU for hours. So the total time the stream has spent blocked in
`send()` may not exceed a 30 s grace period plus twice the audio delivered so far:
`send_blocked ≤ 30 s + audio_seconds_sent / 0.5`, i.e. past the grace period the client must read
at least at half of real time. A stream past that is aborted like a send timeout, as
`speech.aborted` with reason `too_slow`. Each send's timeout is the shorter of the send timeout
and the budget left, so a trickle is caught mid-send too. Only time blocked in `send()` counts:
waiting for generation between sends is the server's own doing, so a slow GPU never trips the
rule and only a slow client does. The grace period absorbs slow starts and hiccups, and any
client that plays the audio reads at least at real time. The constants are
`MIN_RATE_GRACE_SECONDS` and `MIN_RATE_REAL_TIME` in `limits.py`.

**Alternatives considered**: releasing the gate in the generator's `finally` (skipped when the
body never starts); polling `request.is_disconnected()` (redundant under spec 2.3).

## R4. WebSocket transport

**Decision**: Add the **`websockets`** library (>= 17.1, < 18; 17.1 tested) and serve the WebSocket port
with its native asyncio server (`websockets.asyncio.server.serve`) on the same event loop as
uvicorn. Configuration:
- a socket pre-bound on the configured host only (BC-30);
- `process_request` performs the Origin check (`403`, BC-31), the connection cap (`503`) and the
  loading check (`503`), each with the JSON error envelope;
- `open_timeout=10` (handshake limit);
- `max_size=1 MiB`;
- `ping_interval=20` and `ping_timeout=20`;
- `close_timeout=2`.

**Rationale**:
- **Without a WebSocket library, WebSocket routes return `404`.** The venv has neither
  `websockets` nor `wsproto`, so uvicorn treats an upgrade request as plain HTTP.
- **A client that stops reading can't be evicted through uvicorn.** Its WebSocket protocols close
  with `transport.close()` and never call `abort()`, so neither a ping timeout nor an application
  `close()` frees a stalled socket (measured). The `websockets` server aborts the transport when
  `close_timeout` expires.
- **It checks standards conformance.** In the probes (uvicorn's sansio adapter uses the same
  protocol core), an unmasked frame closes with 1002, bad UTF-8 with 1007, and an oversized
  message with 1009, and a client close is echoed (BC-43).
- **It avoids uvicorn's startup problems.** uvicorn calls `sys.exit(3)` when a port fails to bind,
  which kills every server in the process (`uvicorn/server.py:171-183`), and a second uvicorn
  instance takes over the signal handlers.

**Plan check (first WebSocket task)**: before building on it, a prototype must confirm that
`close_timeout` aborts a stalled peer, and that `process_request` can return a JSON `403`/`503`.

**Amendment after the plan check (T070, 2026-09-26)**: `close_timeout` does *not* evict a peer
that stopped reading while we were sending: `ws.close()` sets its deadline but awaits `drain()`
before enforcing it, so it blocks forever, and so do the keepalive ping and `Server.close()`
(`research/ws-prototype.md`). The rationale bullet above holds only for an idle peer. Decision
(user): keep `websockets` and bound every server-initiated close ourselves:
`asyncio.timeout(WS_CLOSE_TIMEOUT_SECONDS)` around `ws.close()`, then `SO_LINGER(1, 0)` and
`ws.transport.abort()`, on every close path (outbox overflow, a send blocked for
`WS_SEND_TIMEOUT_SECONDS`, shutdown, every handler exit), with our own connection set and
`TCP_USER_TIMEOUT` (30 s) on the WebSocket listening socket as a Linux-only backstop. Other
libraries were not adopted: uvicorn never aborts, `wsproto` means hand-rolled I/O, and the rest
are untested and would need the same bounded close.

**Alternatives considered**:
- uvicorn with `websockets-sansio` on a second `uvicorn.Server`: works and conforms, but needs a
  `TCP_USER_TIMEOUT` kernel option to evict stalled peers, a signal-handling override, and a
  `sys.exit` workaround.
- `wsproto`: drops the TCP connection on protocol violations without a close frame.
- A hand-rolled implementation: that is where C++ got BC-41 and BC-43 wrong.

## R5. Evicting stalled HTTP streams at the kernel level

**Decision**: Set `TCP_USER_TIMEOUT = 30 s` on the pre-bound HTTP listening socket; accepted
sockets inherit it. uvicorn is served with `sockets=[sock]`.

**Rationale**: An asyncio close waits for the send buffer to flush. The kernel timeout is what
frees a socket whose peer vanished, and it was the only mechanism that did so in the experiments.

**Alternatives considered**: application timeouts alone (the socket and its send buffer linger).

## R6. Body limits, form parsing, duplicates

**Decision**:
- **Body limit:** a pure-ASGI `BodyLimitMiddleware` rejects up front using `Content-Length`, then
  wraps `receive` to count bytes. It returns `413` with the envelope above **26 MiB** (25 MiB audio
  plus room for fields).
- **Form parsing:** `breeze_infer/http_fields.py`'s `read_fields` parses the body itself rather
  than calling `request.form()` (T037 review 1): Starlette's own parser is silently lossy on
  invalid UTF-8 (its urlencoded parser calls `unquote_plus` with the default `errors="replace"`,
  and its multipart parser's `_user_safe_decode` falls back to latin-1 on a decode failure), so
  neither path can tell a corrupt body from a valid one. `multipart/form-data` goes through
  `_StrictMultiPartParser`, a subclass overriding only the one callback that decodes a text part,
  with `errors="strict"` and the declared charset pinned to `utf-8`;
  `application/x-www-form-urlencoded` is hand-parsed with `urllib.parse.unquote_plus(...,
  errors="strict")`. Limits: `max_files=1`, `max_fields=32`,
  `max_part_size = MAX_TEXT_CHARS * 12 = 120,000` bytes -- derived, not copied from this
  document's earlier 64 KiB example: a 4-byte UTF-8 code point (an emoji, or a CJK Extension-B
  character) percent-encodes to 12 ASCII bytes (`%XX` x 4), so `MAX_TEXT_CHARS` (10,000) of them
  needs up to 120,000 bytes on the wire. At 64 KiB that text would have been cut off as a generic
  parser error before the field-level length check (T048's `text_too_long`) ever got to run.
- **Duplicates:** a field is a duplicate when `getlist(k)` has more than one value in the form or
  the query string, or when the same key appears in both (`400 duplicate_field`).
- **Parser errors:** python-multipart's `FormParserError` maps to `400`. Without that mapping it
  becomes a `500`.
- **Required parts:** a truncated multipart body parses as an empty form with status `200`, so
  required fields are checked explicitly.

**Rationale**: Every path was exercised, and memory stayed flat for a 30 MiB chunked upload.
`max_part_size` doesn't cover file parts, so the total limit has to come from middleware.

**Alternatives considered**: a reverse-proxy limit (there is no proxy).

## R7. Error envelope

**Decision**: One envelope, `{"error": "<message>", "code": "<code>"}`, on every status:
- `StarletteHTTPException` handling passes `exc.headers` through, so `405` keeps its `Allow`
  header.
- An `Exception` handler returns `500` with `internal_error` and records a structured event. The
  exception text is never included.
- There is no `422`: every validation failure is raised as a `400` by the fields module.

Where the C++ server has a message string, it is reused verbatim (`busy`, `unknown voice_id`,
`text is required`, and so on); see [contracts/http-api.md](contracts/http-api.md).

**Rationale**: FR-023/024. `error` stays the string that clients such as SillyTavern read
(`breeze-http.js:30-37`). Every `500` closes the keep-alive connection (Starlette re-raises after
sending). That is acceptable for real failures, and it is documented.

**Alternatives considered**: `{"error":{"code","message"}}` (breaks every C++ client).

## R8. CORS and cross-origin protection

**Decision**: A custom pure-ASGI CORS middleware (about 80 lines) wraps the whole app *outside*
`ServerErrorMiddleware`. It:
- trims the allowlist;
- sends `Vary: Origin` on every response in allowlist mode;
- answers preflight with `204` when the route exists (its methods, echoed request headers,
  `Max-Age 86400`), `404` for an unknown path, and `405` for a disallowed method;
- sends expose-headers;
- adds CORS headers to `500` responses;
- returns `403` for a `POST`/`DELETE` whose `Origin` is present and not allowed, before the
  endpoint runs.

The WebSocket handshake reuses the same pure `origin_allowed()` function.

**Rationale**: Starlette's `CORSMiddleware` fails each of the following (all demonstrated):
- it doesn't trim the allowlist;
- it sends no `Vary` for disallowed origins;
- it adds no CORS headers to `500`s;
- it answers preflight with `200` for any path;
- it lets a `POST` from a disallowed origin run.

`A:`'s `_parse_cors_origins` (`api.py:1384-1431`) is reused for parsing, since it already trims
and rejects `*` mixed with other entries.

**Alternatives considered**: `CORSMiddleware` plus two extra layers (three layers, and still no
headers on `500`s).

## R9. Reference audio decoding

**Decision**: `soundfile` (libsndfile 1.2.2) on a `BytesIO`:
1. Enforce the size cap.
2. Open with `sf.SoundFile` and check the header: the format is in {WAV, WAVEX, FLAC, OGG},
   channels are 1–8, and the sample rate is 8,000–192,000 Hz. The header's own frame count
   rejects an over-30 s file immediately, without decoding it -- but only when that count is
   trustworthy. A FLAC written by a streaming encoder with no known length upfront (STREAMINFO
   `total_samples` left at 0, e.g. one piped from `ffmpeg`) makes libsndfile report `frames` as
   `INT64_MAX` instead of raising; a real file never comes anywhere near that, so any frame count
   above a generous, clearly-implausible ceiling is treated as unknown rather than trusted.
3. For a trustworthy, in-bounds count, decode directly. For an untrustworthy one, decode in
   small, bounded blocks straight into a preallocated mono buffer -- downmixing each block in
   float64, since two channels near float32's max would overflow a float32 sum -- and stop at one
   sample past the 30 s cap, so neither the length claim nor an enormous real file can force
   reading or allocating past it. Ending early there is expected for such a file (the encoder
   never promised a length); libsndfile raises a specific internal error once such a stream's
   real end is reached, tolerated once at least one block has already been read successfully. A
   genuine decode failure still raises normally either way.
4. Reject non-finite or absurdly large (`|sample| > 8.0`) samples, then hand the mono float32
   result to the codec's own resampler -- except a clip under 80 ms (`n_samples * 24000 <
   1920 * sample_rate`, exact integer arithmetic on the native sample count), which is rejected
   outright rather than handed to the codec: the codec rounds a partial frame up, so without this
   minimum a 1-sample clip would become one frame of mostly padding.

A `RuntimeError` (the base of `LibsndfileError`) maps to `400 invalid_audio`.

**Rationale**: Every malformed case was rejected cleanly:
- a data length of `0xFFFFFFFF` was clamped to the real size;
- a truncated `fmt` chunk, zero channels, zero bits or a zero sample rate each raised an error;
- 18,000 random mutations produced no crash, and memory peaked at 41 MiB; a later, more targeted
  fuzzing pass (4,200 files) found the streaming-FLAC-length and float-overflow cases above, both
  fixed the same way -- nothing escaped after.

libsndfile does accept a sample rate of 1, which is why the header checks are ours. MP3 is
excluded: it isn't needed, and libmpg123 writes to stderr on bad input. Tests use soundfile for
real, since it is an external dependency.

**Alternatives considered**: a hand-written WAV parser (the source of C++ defect BC-15);
torchaudio/ffmpeg (heavier, with a bigger attack surface).

## R10. Porting `api-alignment` work (revalidated)

**Decision**: Port selectively. `A:api.py` does not merge with `P:api.py` (it is built on
feature 001's layout), so `api.py` is rewritten as a composition root and logic moves into small
modules; see [plan.md](plan.md) Project Structure.

| Earlier work | Verdict |
|---|---|
| `text_split.py`, goldens (`72d9bdd`, `6c9bc4f`, `0e882ab`) | Port with the fixes in R11. |
| `events.py` (`6fb3736`) | Port unchanged, adding `request_id` at every call site. |
| Error envelope and CORS parsing (`45295d2`, `1c90c6d`, `ec1e2b5`) | Port the handlers and `_parse_cors_origins`; add `code`; replace `CORSMiddleware` (R8). |
| `form_fields.py` (C `atoi`/`atof` coercion) | **Drop.** It exists only to copy BC-01/BC-03. |
| Runtime sampling overrides and room checks (`86647b8`, `220b3ca`) | Port onto 81a5ca7 (R12). |
| `templates.py` codes path, prefix/suffix split, `audio.encode_prompt_waveform` (`6fb3736`) | Port. The fingerprint must not hash the absolute checkpoint path (`A:audio.py:43`). |
| `voice_prefix.py` LRU (`6fb3736`) | Port, re-keyed by `(voice_id, codes_sha256)` so a delete-then-re-register can't serve a stale KV. |
| `voices.py` / `voice_index.py` storage (`6fb3736`, `b903011`) | Rewrite for the new format (R13), borrowing the atomic-write, rename-retry and eviction code. |
| Speech route (`59a3528`, unreviewed WIP) | Port the ideas: anchoring, chunk ramp, per-piece seed, room check, response-owned cleanup. Fix everything it got wrong: busy check before validation, preparation on the event loop, a module-global lock, and silent truncation on failure or when out of room. |
| `bench_api.py` | Port, pointed at the new contract; record the baseline on `P:` first. |
| stash@{0} | Nothing to recover (line-ending changes only). |

**Rationale**:
- Several `A:` commits never got their second review pass: 86647b8 had one pass, and b903011 and
  59a3528 had none. Everything ported is therefore re-reviewed under this feature's two-pass loop.
- `A:` never had a WebSocket server. `/health` hard-coded `ws_port: 0` (`A:api.py:672-674`), so
  all WebSocket code is new.

**Alternatives considered**: cherry-picking `A:` commits (large conflicts in `api.py` and
`fast_streaming.py`, and it would bring back parity quirks the spec removes).

## R11. Shared text segmenter

**Decision**: One pure module, `text_split.py`:

```text
weigh(text) -> int                      # ASCII 1, other code points 3 (C++ text_split.cpp:16-24)
segment(buffer, *, budget, first_budget=0, final) -> (pieces, remaining)
split_text(text, *, budget, first_budget=0) -> pieces   # == segment(text, ..., final=True)[0]
```

What it keeps from C++: sentence merging and clause splitting, closing-quote absorption (now on
both interfaces), and the soft opening budget.

What it fixes (BC-39, BC-46):
- the NUL-as-closer quirk;
- the byte-length drain fallback (now weighted);
- two different stop sets (now one);
- an end-of-buffer stop cutting at once (it waits unless `final`);
- unbounded CJK runs without punctuation (clause-mark or space fallback, then a hard cut at
  2 × budget);
- the opening budget applying to a whole drain (now the first piece only, with the caller owning
  the flag).

It also changes three things:
- `budget == 0` means no length splitting on both interfaces.
- Pieces are stripped of surrounding whitespace, and empty pieces are dropped.
- Tabs and carriage returns inside a piece are preserved (BC-44).

**Golden tests**: `gen_goldens.py` no longer rewrites the test file. The test compares against
normalized C++ output and keeps a table of intentional differences, each tagged with its BC id;
that table is SC-002's evidence.
- **Change after normalization (stripped, empties dropped):** 6 split goldens and 6 drain
  goldens, after the T019/T020 review fixes: NUL (BC-46); the undrained byte-vs-weight case, quote
  absorption and the fullwidth period (BC-39, from the port itself); four split goldens from
  removing C++'s quarter-budget space guard, which let spaced text with no punctuation grow past the
  context; one from dropping emoji-only pieces; and three drain goldens from the streaming clause
  rules (the over-budget clause rule and bounded overflow). The planning estimate
  here was 5 and 5, and the port alone changed 1 and 3. The rest match once normalized (empty,
  whitespace-only) or were already right in C++ `split_text`. The
  table is `INTENTIONAL_DIFFERENCES` in `tests/test_text_split.py`.

**Rationale**: FR-014/FR-031. The same text gives the same pieces over HTTP and WebSocket.

**Alternatives considered**: keeping two segmenters (the source of C++'s inconsistencies).

## R12. Runtime changes on `perf-and-fixes`

**Decision**:
1. **Per-request sampling overrides** (temperature, top_k, top_p, repetition_penalty,
   max_new_tokens) on `iter_audio_chunks`, applied to the backbone only (as C++
   `generation.cpp:123-127`).
   - `None` means the model default.
   - The runtime raises `ValueError` on anything else invalid; validation belongs to the boundary.
   - The temperature floor of 1e-5 stays, since float32 overflow gives NaN.
   - Warmup exercises the sampling kernels, so they don't compile lazily inside a request.
2. **Default length:** `max_new_tokens` defaults to the model default (750). 1,500 is the server
   maximum.
3. **`_prefill_plan` returns `(use_graph, prefill_len)`.** Adopt `A:`'s tuple and update
   81a5ca7's two call sites. `max_new_tokens_room(requested, inputs, prefix_len)` is the third
   caller.
4. **Prefix guard:** `build_reference_prefix` rejects a prefix that leaves less than
   `MIN_SUFFIX_ROOM` (22) slots, i.e. `prefix_len > max_seq_len - 1 - MIN_SUFFIX_ROOM`, so such a
   voice is refused when it is registered. `MIN_SUFFIX_ROOM` is the suffix of a one-word text
   with the default instruction on the guided branch (10 tokens, measured with
   `prepare_suffix_inputs` and the real tokenizer) plus `MIN_SUFFIX_FRAMES` (12 frames, about
   1 s). It holds at the exact prefill length; graph bucket padding can take up to 31 more
   slots, and each request's own room is checked by `max_new_tokens_room`. The old guard
   rejected any voice prefix over 548 tokens (`P:fast_streaming.py:864-867`).
5. **Room semantics:** a partial room clamps and emits `speech.piece_clamped`. Piece 0 with no
   room gets `400 text_too_long`. A later piece with no room aborts the stream (FR-036a, BC-47).
6. **Other `cfg_scale` values already work on the warmed graphs.** The guidance scale is a runtime
   tensor (`backbone_graph.py:50-51,107,302`). The only thing in the way today is the API's
   `cfg_scale <= 0` check (`P:api.py:215-218`), and SillyTavern sends 1–10 in 0.5 steps. A GPU
   test covers 2.5, 7.5 and 0.
7. **Seeds:** piece seed = `(s + i) & 0xFFFFFFFF`; the runtime already reseeds after lazy setup.

**Rationale**: FR-006, FR-012–FR-014 and the spec's Known Differences. These are the minimal
runtime changes; everything else is orchestration outside `models/`.

**Alternatives considered**: raising `max_seq_len` above 2048. It would grow the static cache and
the warmup set, and changes memory and TTFA. It stays deferred until measurements justify it, and
BC-47 documents the limit.

## R13. Voice storage

**Decision**: One file per saved voice, `<voices_dir>/<id>.voice.json`, format
`breeze-tts-voice` version 1, storing codes only (details in
[data-model.md](data-model.md#voice-file-v1)).
- **Create:** atomic (temporary file, fsync, `os.replace`, directory fsync), under a write lock.
  It never overwrites.
- **Delete:** rename to `.del-<id>-<nonce>` with a `PermissionError` retry (5 × 50 ms, `sleep`
  injected), then unlink. Leftover `.del-*` files are swept at startup.
- **Startup validation:** stem equals id, the name pattern, not `v_`, no case duplicate, codes
  length and sha256, every code within `[0, codebook_size)`, and the codec fingerprint.
- **Skipped files:** each emits a `voice.skipped` event. A skipped file with a valid name still
  reserves that name (`409`) and can be removed with DELETE.
- **`.breeze` files:** counted in `voices.loaded`.
- **Unnamed ids:** `v_` followed by 16 hex characters of `blake2b(len(wav) ‖ wav ‖ text,
  digest_size=8)`.
- **Codec fingerprint:** sha256 of (a) the canonical JSON of `config.json`'s required identity
  fields -- input/output sample rates, `encoder_valid_num_quantizers`, encode/decode
  frame/upsample rates at the top level; codebook size/dim and quantizer count from both
  `encoder_config` and `decoder_config` (which differ between the two); plus `decoder_config`'s
  semantic-codebook fields and upsample schedule -- every field required, none defaulted; and
  (b) the tensor name/dtype/shape map (not the trained values) of the safetensors header of the
  one weight file (or, sharded, every shard the loader's own index lists) the codec loader
  actually reads. No path is hashed. This detects an architecture, shape or dtype change, not a
  retrain of weights with identical shapes -- an accepted limit of a fingerprint cheap enough to
  compute at every startup.

**Rationale**: FR-017–FR-022 and BC-25–BC-29/48.
- All saved codes live in memory (about 12 KB per 30 s voice), which removes 001's lazy-load and
  delete races.
- An out-of-range code would trigger a CUDA device assert that poisons the whole process, so
  validating codes matters.
- `blake2b` releases the GIL; pure-Python FNV costs about 1 s of GIL time per 25 MiB upload.
- The spec requires only the id format, not C++ id parity.

**Alternatives considered**: 001's directory-per-voice layout with the WAV kept (Q3 chose codes
only in our own format; re-encoding isn't required); C++ `.breeze` (rejected by Q3).

## R14. Concurrency model

**Decision**:
- **`GpuGate`:** an asyncio gate used only on the event loop.
  - `try_acquire() -> GpuLease | None` is for HTTP. It fails if the gate is held *or* a WebSocket
    piece is waiting; the caller then answers `409 busy`.
  - `await acquire(on_wait=...)` is for WebSocket pieces. `on_wait` runs synchronously just before
    the call blocks, and only if it will block, so the worker can enqueue `queued` ahead of the
    wait. A returned flag would arrive too late: after the wait.
  - `lease.release()` hands the gate directly to the next waiter; only the current lease can
    release. `GpuSession(lease, gpu, gen)` holds the lease for a generator's whole life and releases
    it only after the generator has been closed on the GPU thread, so a cancelled step can't leave
    the GPU busy behind a free gate. HTTP and WebSocket both use it.
- **`GpuThread`:** a single-thread executor that runs *all* CUDA work: model load and warmup
  (in the background, so `/health` shows `503 loading`), reference encode, prefix builds,
  `prepare_inputs`, every `next(gen)` and every `gen.close()`. It calls `torch.cuda.set_device`
  once.

**Rationale**:
- Model load becomes observable (FR-036). Today the lifespan blocks until the model is loaded
  (`P:api.py:181-186`).
- RNG reseeding stays on one thread.
- Device selection becomes correct for `LOCAL_RANK != 0`, which today's anyio worker threads get
  wrong.
- A `close()` queues behind an in-flight `next()`.
- `asyncio.Lock` isn't enough: `locked()` can report free while a waiter is being woken.

**Alternatives considered**: `threading.Lock` (today's global, `P:api.py:59`: a Principle III
violation, and it can't express WebSocket queueing); anyio worker threads (arbitrary threads touch
CUDA).

## R15. WebSocket session design

**Decision**:
- **Session state:** a pure `ws_session.py` holding the epoch, the config snapshot, the buffer,
  the opening-budget flag, the anchor, the piece index, and a work deque of
  `Piece(epoch, index, text)`, `EndMark(epoch)`, `CancelMark` and `StartMark(config)`.
- **I/O shell:** `ws_server.py` runs a reader task, one worker coroutine, and one ordered outgoing
  queue (bounded at 2 MiB) with a sender task.
- **Message semantics:**
  - `cancel` drops the queued pieces and the current epoch's end markers (an end marker an
    earlier session left queued keeps its `done`), bumps the epoch, which signals the piece in
    flight, and enqueues exactly one `CancelMark`.
  - `end` enqueues exactly one `EndMark`.
  - `start` validates first, cancels only when a piece is queued or in flight, begins a new epoch
    (every accepted `start` does), and then enqueues `StartMark`.
  - Disconnect: `Session.close()` bumps the epoch and clears the whole deque; nothing more is sent.
- **Slow client:** if the outgoing queue would overflow, or the connection's write buffer makes no
  progress for `WS_SEND_TIMEOUT_SECONDS` (the stall watchdog, R4 amendment), the piece in flight is cancelled (the GPU
  is freed within one chunk) and the connection closes with 1008 and reason `client too slow`.

**Rationale**:
- **Exactly one `done` per `end` and one `cancelled` per `cancel`:** each message places exactly
  one marker in the work queue. Markers are processed in order, only after the in-flight piece's
  generator has closed, and audio shares the same queue as events, so `cancelled` comes after the
  last frame. This fixes C++ defects BC-32–BC-36.
- **SC-005 is cheap to test:** the session module is pure, so 1,000 randomized sequences run fast
  and deterministically.

**Alternatives considered**: C++'s latched `cancel` flag (the source of the races).

## R16. SillyTavern as the live test client

**Findings** (from `sillytavern-agent`, extension v0.1.2, `<SillyTavern checkout>/extensions/
SillyTavern-BreezeTTS`):
- **Synthesis is WebSocket only.** One connection per segment on `ws://127.0.0.1:8081`: it waits
  for `ready`, then sends `start{voice_id, cfg_scale 1–10 in 0.5 steps, [instruction], [seed]}`,
  then a single `end{text}` (whose text always ends with a terminator). It may send `cancel`.
- **HTTP** is used for `GET /health`, `GET/POST /v1/voices` (multipart with `ref_audio`,
  `ref_text` and `name`) and `DELETE /v1/voices/{id}`. Errors come from the `error` key over HTTP
  and the `message` key over WebSocket.
- **Timeouts:** connect 5 s, `ready` to `started` 10 s, `started` to first audio 10 s (60 s after
  `queued`), 60 s between frames.
- **The browser talks to Breeze directly,** so the server must run with CORS enabled for
  `http://127.0.0.1:8000` (or `*`).
- **Playwright scripts** to copy: `<SillyTavern checkout>/specs/001-breeze-tts-provider/
  us*-browser-validation.mjs` (playwright-core from the npx cache, headless chromium). Every
  provider step logs `console.debug('breeze', {event,...})`.
- **Chat and voice rules:**
  - Always use the "Breeze validation" chat under Seraphina. Never use the user's own chat.
  - Never delete the `eric` or `vale` voices; test uploads use a throwaway name.
- **The extension's own node live tests** (`npm run test:live`) assert that `/health` deep-equals
  `{status, sample_rate, ws_port}`, so `/health` must not grow.

**Decisions**:
- **Replace flow** (resolved, spec Clarifications): the server keeps `409 voice_exists`.
  SillyTavern changes its replace flow to DELETE then POST; the extension owner is making that
  change with their user's approval.
- **Loading `503`:** the body carries `error`, so SillyTavern shows a readable message.
- **Voices:** the C++ `.breeze` voices aren't read (Q3), so `eric` and `vale` are registered once
  on the new server from `$REFERENCE_VOICES_DIR/{eric,vale}`.

## R17. Performance baseline (SC-007)

**Decision**: Before any code change, record time to first audio (TTFA) and real-time factor
(RTF) on `P:` with `A:bench_api.py`, pointed at the current API (port 7860, `--fast-all`):
- cases: short and medium voice design, short and medium inline reference; 3 warm-ups, then the
  median of 10 runs (tasks.md "SC-007 method"; the first plan of 3 runs proved too noisy);
- saved to `research/baseline-<date>.md`.

The 002 baseline (RTX 4090: short design 75 ms / 0.379) is for reference only; it was taken on a
different branch.

**Rationale**: The comparison has to be made on the same machine and branch.

## R18. Cross-process determinism of inline reference codes

**Decision**: `breeze_infer/audio.py`'s `encode_prompt_waveform` wraps the codec's `encode` call
in `_deterministic_cudnn_encode()`, a small save-and-restore context manager -- not
`torch.backends.cudnn.flags()` (review finding 1, this round): that context manager resets
every argument you don't pass to *its own* default (`allow_tf32=True`, `benchmark_limit=10`,
`fp32_precision="none"`), not to the caller's actual value, so a caller running with e.g.
`allow_tf32=False` would silently get TF32 turned back on for the duration of the encode.
`_deterministic_cudnn_encode()` instead saves exactly `torch.backends.cudnn.benchmark` and
`torch.backends.cudnn.deterministic`, sets `benchmark=False, deterministic=True`, and restores
both saved values in a `finally` -- even when the encode raises -- once the `with` block exits;
every other cudnn flag (`enabled`, `allow_tf32`, `benchmark_limit`, ...) is never touched, so
whatever the caller had stays exactly as it was, inside the scope and out. The codec's decode
path outside this scope keeps cuDNN's autotuned algorithms. `infer.py` and
`breeze_infer/synthesis.py`'s `resolve_reference` are the only two places that encode reference
audio in the live serving path, and both already call `encode_prompt_waveform` rather than the
codec directly, so no other call site needed changing. (`models/breeze.py`'s
`_get_audio_token_from_batch` also calls a codec's `.encode(...)` directly, but it has no caller
anywhere in this repo -- dead training-script code ported from the base model, not part of the
inference server -- so it was left alone.)

This is a deviation the user approved on 2026-09-25, not a BC item: `--fast-all` sets
`torch.backends.cudnn.benchmark = True` for the whole process
(`models/stream_runtime/stream/runtime.py`'s `MultiRequestStreamRuntime.__init__`, which now has
a short comment pointing back here), so cuDNN autotunes whichever conv algorithm looks fastest
*in that process* on first use and reuses it. Autotuning races candidate kernels against a wall
clock, so two separate server processes can settle on different algorithms for the identical
convolution, and those algorithms round differently -- about 1% of the fine-codebook codes
(codebooks 6-15) then differ for the same input. The observable symptom: the same inline
reference wav, the same text and the same seed produced audio 5.20 s long in one process and
6.96 s long after a restart, at `bea6767`. Pinning the encoded codes (bypassing a fresh `encode`
call) gave byte-identical output at both `bea6767` and `HEAD`, isolating the divergence to the
encode step rather than generation or decode. A probe that forced `benchmark=False` around only
the encode call, in two otherwise-identical fresh processes, produced identical codes both times;
without it, the same two processes produced 22-23 differing entries out of 2,064 (129 frames x 16
codebooks) -- consistent with the ~1% figure above.

**Rationale**: Scoping the fix to the encode call, rather than turning `cudnn.benchmark` off for
the whole process, keeps the decode path's autotuned performance (the reason `--fast-all` sets
`benchmark = True` in the first place) while making the one thing that must be reproducible --
what a saved or inline reference encodes to -- independent of which process encoded it. The
guarantee this buys is narrower than "always identical," and is stated that way on purpose
(review finding 4, this round): codes are identical across processes **on the same GPU and
software stack** (GPU model, driver, CUDA, cuDNN and torch versions all held fixed). A different
stack, or cuDNN falling back to a different deterministic engine under GPU memory/workspace
pressure than it did last time, is free to produce different codes for the identical input --
`deterministic=True` only promises that *one* stack, run twice under the same conditions, picks
the same algorithm every time; it says nothing about agreement across stacks. PyTorch's cuDNN
plan cache is keyed on `deterministic` and `allow_tf32` but not on `benchmark`, so if anything
ever set `cudnn.deterministic = True` globally while `benchmark` stayed `True`, a plan autotuned
outside this scope could be looked up and reused inside it -- one more reason `deterministic`
must only ever be scoped to the one call that needs it, as `_deterministic_cudnn_encode()` does,
never set globally alongside `benchmark = True`. This has a direct
implication for stored voice codes (T06x, not yet built): a voice saved by encoding the same
reference wav on a different GPU/driver/CUDA/cuDNN stack can end up with different stored codes
for what a person would call "the same voice" than encoding it fresh on this one. The
`codec_fingerprint` that travels with saved codes (`breeze_infer/audio.py`) doesn't cover this --
it hashes the codec's own identity (its config's identity fields and its weight files' tensor
`{dtype, shape}`, not the trained values), which tells a loader whether *this checkpoint's codec*
can decode a saved voice's codes at all, not whether those codes match what today's stack would
have produced. That's fine for what saved voices actually need: `codec_fingerprint`'s own
docstring notes there is no re-encode path for a saved voice (the original recording isn't kept),
so the stored codes are authoritative for that voice once saved, by design -- but it's one reason
a stored voice is pinned to whatever codes it was saved with rather than silently re-derived, and
worth knowing if a saved-voice migration or backup ever moves those codes to different hardware.

Measured cost, **warm** (bench reference wav, median of 20 calls after 5 warm-up calls at that
same length, one process, fast codec's `cudnn.benchmark=True` already enabled): ~25.6 ms/call
with the scoped flags versus ~24.9 ms/call without them -- within each other's run-to-run noise,
so the fix has no measurable cost once cuDNN has already picked an algorithm for that exact input
shape during warm-up.

That warm number is not the whole story (review finding 9, this round): every request can bring
a differently-sized inline reference, and cuDNN's algorithm choice is keyed on input shape, so
the *first* encode at each new reference length is a fresh "cold" case regardless of how many
other lengths that process has already warmed up. A first attempt at measuring this ran one
(length, scope) pair per fresh process, so the timed call was also the first CUDA call that
process ever made -- it measured CUDA context creation, cuDNN handle setup and kernel loading,
not the per-length cost, which is why even the fastest row came out at several seconds. Redone
correctly: 3 fresh processes per scope setting; each process loads the codec, sets
`cudnn.benchmark = True` (matching `--fast-all`), does one *unrecorded* warm-up encode of a 2 s
slice to pay the CUDA/cuDNN start-up cost up front, then a **cold pass** -- the bench reference
wav (`$REFERENCE_VOICES_DIR/eric/eric.wav`, 44.1 kHz mono, 10.245 s) sliced to 1, 3, 5,
7 and 9.5 s, encoded once each in that order and recorded (first call at that exact shape, but
not first call in the process) -- then a **warm pass** over the same five lengths again, recorded.
Values below are the median over the 3 processes for each cell:

| reference length | with scope, cold | with scope, warm | without scope, cold | without scope, warm |
| --- | --- | --- | --- | --- |
| 1.0 s | 85.5 ms | 23.7 ms | 233.9 ms | 24.3 ms |
| 3.0 s | 35.3 ms | 25.9 ms | 207.7 ms | 24.7 ms |
| 5.0 s | 36.1 ms | 21.6 ms | 208.4 ms | 26.0 ms |
| 7.0 s | 73.5 ms | 34.9 ms | 528.2 ms | 29.5 ms |
| 9.5 s | 40.8 ms | 28.3 ms | 131.0 ms | 29.6 ms |

**Conclusion**: once CUDA/cuDNN start-up is paid once (by the warm-up call) and excluded, the
real per-length cold cost is modest -- tens to a few hundred ms, not the multi-second-to-tens-
of-seconds figures the first attempt reported; those were an artifact of conflating process
start-up with per-shape cost, not a property of the fix. With the scope, cold costs range
~35-86 ms; without it, ~131-528 ms -- **higher without the scope at every length measured**,
because cuDNN's `benchmark=True` runs a real timed search over candidate algorithms the first
time it sees a new shape, while forcing `benchmark=False` (a heuristic pick, no search) is
consistently cheaper. Both settle to ~20-35 ms on the warm pass either way, matching the
original steady-state figure above. So the fix's cold-path cost is real but small, and it is
never a net cost versus not having the fix -- at every length measured here it was a modest,
consistent saving instead.

The first-call-in-process cost measured by the first (flawed) attempt still exists either way --
it just isn't the encode scope's cost, it's CUDA/cuDNN's own start-up cost, paid once per
process regardless of `encode_prompt_waveform`. It lands on whatever is the first CUDA work that
process does. The server's own warm-up (`FastBreezeStreamingRuntime.warmup_from_profile`,
`models/fast_streaming.py`) does not encode a reference: its codec stage calls
`codec.decode_request_chunk(...)` on synthetic zero codes -- decode only -- and nothing in
warm-up calls `encode_prompt_waveform` or the codec's `.encode(...)`. So the first inline
reference request after start-up is the first time the *encode* path runs in that process at
all. By then, though, warm-up's own decode/backbone/depth-decoder graphs have already created
the CUDA context and loaded the relevant kernels, so that first request pays this table's
"cold" column, not a from-scratch CUDA cold start -- consistent with the story above, not a new
cost.

**Evidence**: `tests/test_audio.py`'s
`test_encode_prompt_waveform_scopes_cudnn_to_deterministic_no_benchmark` and
`test_encode_prompt_waveform_restores_cudnn_flags_when_encode_raises` (a fake codec records all
six cudnn flags -- not just the two this fix changes -- in effect during its own `encode`, on
the CPU suite, confirming `enabled`/`allow_tf32`/`benchmark_limit`/`fp32_precision` pass through
untouched while `benchmark`/`deterministic` are pinned, and that all six are restored even when
`encode` raises)
and `tests/gpu/test_reference_encode_determinism.py`:

- `test_same_reference_wav_encodes_identically_across_processes` runs
  `_encode_determinism_worker.py` in three fresh processes. Each worker loads only the codec --
  not a full `MultiRequestStreamRuntime`, which costs ~85 s per worker (a lazy `torch.compile` of
  every `SnakeBeta` module plus a CUDA graph warmup, both unrelated to what this test checks) just
  to reach the one line that sets `cudnn.benchmark = True` -- and sets that flag directly instead.
  It then records the cudnn flags in effect during the codec's own `encode` call and the test
  asserts `benchmark=False, deterministic=True` there in all three workers, in addition to
  asserting the three workers' codes are identical to each other. Asserting the flags directly
  (not just comparing codes) matters because the bug this guards against is probabilistic:
  without the fix, cuDNN's autotuner often lands on the same algorithm across processes anyway, so
  a bare "codes are equal" check can pass by luck with the regression still present.
- `test_multi_request_stream_runtime_sets_cudnn_benchmark_for_fast_codec` is a second, cheap GPU
  test/assert that builds a real `MultiRequestStreamRuntime` -- not a reimplementation of its
  `__init__`, a source-level check would not be acceptable here -- with its two slowest,
  unrelated init steps patched to no-ops (`_validate_and_get_samples_per_code`'s decoder forward
  pass, which is what actually triggers the lazy `SnakeBeta` compile, and
  `_maybe_warmup_fast_codec`'s CUDA graph capture), and asserts `cudnn.benchmark` really is `True`
  afterward. This is the direct proof that the condition the worker above reproduces by hand is
  real, without paying the ~85 s full-construction cost: measured directly (not inferred),
  `MultiRequestStreamRuntime(...)` with both patches applied takes ~7 s, versus ~85 s unpatched.
  That's the saving this patching buys; it is not the whole per-worker wall time. On this
  machine, a fresh process spends ~80 s just importing `qwen_tts`/`transformers`/
  `models.stream_runtime` (`transformers` eagerly scans its own `models/` package tree at import
  time, and this repo's checkpoints and dependencies sit on a slow, WSL-mounted drive) plus ~7 s
  loading the codec -- both fixed costs of any fresh-process worker, old test or new, unrelated to
  what this patch skips. That import/load cost is why the reworked GPU test file's total wall
  time (four fresh processes: three encode workers plus this one) is still several minutes on
  this machine, not "a few seconds" -- but the ~85 s to ~7 s improvement inside the construction
  itself is real and is what this test isolates.

**Open item: codec decode variation** (found while porting `tests/gpu/test_voice_tier1_equivalence.py`,
tasks.md T067). Only the reference *encode* is pinned deterministic
(`breeze_infer/audio.py`'s `_deterministic_cudnn_encode`: `benchmark=False, deterministic=True`,
scoped to that one call); codec *decode* keeps `--fast-all`'s autotuned, non-deterministic
`cudnn.benchmark=True`. This was known to make decode differ across a server restart, but a GPU
investigation for T067 found it also differs **within one warmed server process, request to
request, for the exact same generated codec tokens**: decoding one identical set of captured
token frames twice, under two different request ids, left 72,960 of 76,800 samples different,
max absolute difference about 0.072 of the codec's [-1, 1] float output -- about 2,365 to 2,654
int16 steps (roughly 8% of full scale), measured across a few such pairs on an RTX 4090. Pinning
decode deterministic the same way as encode was not evaluated (it would cost every request's own
decode, not a one-time reference encode, so the trade-off needs its own benchmark before deciding
either way). Until then, no test in this repo may assert PCM/audio-sample equality across two
separate requests, even with identical inputs and seed; `test_voice_tier1_equivalence.py` instead
asserts the generated *token frames* match exactly and checks the decoded PCM only for basic
plausibility (non-empty, not silent, matching length).
