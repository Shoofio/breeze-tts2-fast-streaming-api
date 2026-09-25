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
- **Form parsing:** endpoints call
  `request.form(max_files=1, max_fields=32, max_part_size=64 KiB)`.
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

**Decision**: `soundfile` (libsndfile 1.2.2) on a `BytesIO`, in three steps:
1. Enforce the size cap.
2. Call `sf.info` and check the header: the format is in {WAV, WAVEX, FLAC, OGG}, channels are
   1–8, the sample rate is 8,000–192,000 Hz, and the duration is 30 s or less.
3. Only then call `sf.read(dtype="float32", always_2d=True)`, downmix, and hand the result to the
   codec's own resampler.

A `RuntimeError` (the base of `LibsndfileError`) maps to `400 invalid_audio`.

**Rationale**: Every malformed case was rejected cleanly:
- a data length of `0xFFFFFFFF` was clamped to the real size;
- a truncated `fmt` chunk, zero channels, zero bits or a zero sample rate each raised an error;
- 18,000 random mutations produced no crash, and memory peaked at 41 MiB.

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
- **Still valid:** 11 split goldens and 7 drain goldens.
- **Change:** 5 split goldens (the first-budget pair, empty, whitespace-only, NUL) and 5 drain
  goldens (ellipsis, the two byte-vs-weight cases, quotes, fullwidth period).

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
4. **Prefix guard:** relax `build_reference_prefix` to `prefix_len >= max_seq_len - 1`. Today any
   voice prefix over 548 tokens is rejected (`P:fast_streaming.py:864-867`).
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
- **Codec fingerprint:** sha256 of `config.json`, the codebook count, the codebook size and the
  sample rate, with no path.

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
  - `try_acquire()` is for HTTP. It fails if the gate is held *or* a WebSocket piece is waiting;
    the caller then answers `409 busy`.
  - `await acquire()` is for WebSocket pieces; the caller sends `queued` first when it has to wait.
  - `release()` hands the gate directly to the next waiter.
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
  - `cancel` bumps the epoch, drops pieces and end markers, and signals the piece in flight. It
    enqueues exactly one `CancelMark`.
  - `end` enqueues exactly one `EndMark`.
  - `start` validates first, cancels only when work is pending, and then enqueues `StartMark`.
- **Slow client:** if the outgoing queue would overflow, the piece in flight is cancelled (the GPU
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
- cases: short and medium voice design, short inline reference, 3 runs each, median;
- saved to `research/baseline-<date>.md`.

The 002 baseline (RTX 4090: short design 75 ms / 0.379) is for reference only; it was taken on a
different branch.

**Rationale**: The comparison has to be made on the same machine and branch.
