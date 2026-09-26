---

description: "Task list for the C++-compatible API (fixed)"
---

# Tasks: C++-Compatible API (Fixed)

**Input**: Design documents from `specs/003-cpp-compatible-api/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: Required. SC-002 asks for one check per BC id that fails against the defective C++
behavior, SC-003 asks for a malformed corpus, and Constitution V asks for failing-first tests on
bug fixes. Write each phase's tests first and confirm they fail before implementing.

**Organization**: tasks are grouped by user story. The phases here map onto the plan's delivery
phases as follows:

| Plan phase | Tasks phases |
|---|---|
| 0 | Phase 1 (Setup) |
| 1 | Phase 2 (Foundational) and Phase 3 (US5) |
| 2 | Phases 4–6 (US1 speech, US2, US3) |
| 3 | Phase 7 (US1 voices) |
| 4 | Phase 8 (US4) |
| 5 | Phase 9 (Polish) |

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (a different file, and no dependency on an incomplete task).
- **[Story]**: US1–US5 from spec.md.
- Paths are relative to the repo root `<repo>`.
- `A:` means `git show api-alignment:<path>`, the source to port from. Read it with `git show`;
  **never check out that branch**.

## Standing rules (apply to every task)

1. **Branch and baseline**: work on `perf-and-fixes`. Read spec.md, plan.md, the relevant contract
   section and the research decision (Rnn) that a task cites before starting it.
2. **Delegation**, as in plan.md "Delivery process":
   - **Sonnet** for tasks without a model tag;
   - **Opus** for tasks tagged *(Opus)*;
   - the main session reviews every agent's output before committing.
3. **Verification**: every task ends with `.venv/bin/ruff check .` and `.venv/bin/pytest` passing.
   Add `BREEZE_MODEL=<model> .venv/bin/pytest -m gpu` when the runtime, synthesis or speech path
   changed. Show the real output.
4. **Commits**: one small, focused commit per task (or per tightly coupled pair). The message says
   what changed and why, and cites FR/BC ids.
5. **Review loop after each commit**:
   1. `review-agent` pass 1 on the commit;
   2. fix the valid findings in a follow-up commit, with a failing-first test for bugs;
   3. `review-agent` pass 2 on the fixes;
   4. address anything still outstanding, then move on. There is no third pass.

   Disputed findings are recorded in the phase's `research/live-<phase>.md`.
6. **Two-strike rule**: when a fix attempt fails, stop. Explain why it failed, list at least 3
   different approaches, and propose one before making further changes.
7. **SillyTavern safety**:
   - Only use the "Breeze validation" chat under Seraphina.
   - Never delete the `eric` or `vale` voices. Throwaway voices are named `st_live_tmp*`.
   - Stop the C++ `breeze-server` before starting this server (same ports and GPU).
8. **Version before a live gate**: bump `breeze_infer/__init__.py` and `CHANGELOG.md` *before*
   starting the server for a live gate.

---

## Phase 1: Setup (baseline, dependencies, live harness)

**Purpose**: capture the before-state and build the tools that every later gate uses.

- [X] T001 (after T003) Port the benchmark from `A:breeze_infer/bench_api.py` to `breeze_infer/bench_api.py`:
  - Add `--api old|new`. `old` drives the current API: port 7860, form fields `text`,
    `instruction`, `cfg_scale`, `seed`, `ref_audio`, `ref_text`. `new` drives
    contracts/http-api.md at 8080.
  - Cases: `short_design`, `medium_design`, `short_inline`, `medium_inline` (added after T002), plus
    `short_voice` for `new` only.
  - Report the median time to first audio (TTFA, ms) and real-time factor (RTF) over `--runs`.
  - Output JSON events to stdout.
  - Add `tests/test_bench_api.py`, covering argument parsing and median maths.
- [X] T002 Record the performance baseline (quickstart Scenario 0.1, SC-007):
  1. Start the current API, `.venv/bin/python -m breeze_infer.api <model> --port 7860 --fast-all`.
  2. Run `python -m breeze_infer.bench_api --api old --url http://127.0.0.1:7860` (defaults:
     `--warmup 3 --runs 10`; see "SC-007 method" below).
  3. Write the GPU name, commit SHA, command lines and medians to
     `specs/003-cpp-compatible-api/research/baseline-2026-09-24.md` (use the actual date).
- [X] T003 Update `requirements.txt` (R4, plan "Dependency changes"):
  - Add `websockets>=17.1,<18` (the tested major version; changed from `>=15` after review, 2026-09-24).
  - Pin exactly: `fastapi==0.141.1`, `starlette==1.6.0`, `uvicorn==0.52.4`,
    `python-multipart==0.0.32`.
  - Add `httpx==0.28.1` (test dependency).
  - Install with `uv pip install -r requirements.txt`, and verify with
    `.venv/bin/python -c "import websockets, httpx; print(websockets.__version__)"`.
  - Mirror the new dependency in the `scripts/start_breeze.ps1` bootstrap if it installs packages
    separately.
- [X] T004 [P] Create `tests/live/sillytavern/config.mjs`. It exports, each overridable by an env
  variable:
  - the SillyTavern URL (`http://127.0.0.1:8000/`);
  - Breeze `http_url` (`http://127.0.0.1:8080`) and `ws_url` (`ws://127.0.0.1:8081`);
  - the `playwright-core` path (`~/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core`);
  - the chromium path (`~/.cache/ms-playwright/chromium-1234`);
  - the headless flag, and the launch args `--autoplay-policy=no-user-gesture-required`;
  - the reference voice samples dir (`$REFERENCE_VOICES_DIR`).
- [X] T005 Create `tests/live/sillytavern/lib.mjs`, adapted from
  `<SillyTavern checkout>/specs/001-breeze-tts-provider/us*-browser-validation.mjs` and
  `make-validation-chat.mjs`. Helpers:
  - `openSillyTavern()`;
  - `selectBreezeProvider(httpUrl, wsUrl)`: wait for `#tts_provider option[value="Breeze"]`, set
    it, enable `#tts_enabled`, wait for "TTS Provider Loaded" in `#tts_status`, set
    `#breeze_http_url` and `#breeze_ws_url`, then click `#tts_refresh`;
  - `openValidationChat()`: wait for `SillyTavern.getContext().chat.length > 0`, select
    Seraphina, then `openCharacterChat('Breeze validation')`;
  - `captureBreezeEvents(page)`: from `page.on('console')` messages whose first argument is
    `'breeze'`;
  - `readToasts()`: from `#toast-container`;
  - `writeRecord(phase, results)`: writes JSON and markdown to
    `specs/003-cpp-compatible-api/research/live-<phase>.md`.
- [X] T006 [P] Create `tests/live/sillytavern/phases/health.mjs`: provider loads, the
  `breeze.health` event is present, no `breeze.check_failed` event, no toast, and no CORS console
  errors.
- [X] T007 [P] Create `tests/live/sillytavern/phases/voices.mjs`:
  - upload `st_live_tmp` through `#breeze_upload_file`, `#breeze_upload_transcript`,
    `#breeze_upload_name` and `#breeze_upload_save` (accept the confirm dialog);
  - check that it appears in `#breeze_voice_list`;
  - re-upload the same name and accept "Replace" (expect success once the extension's
    DELETE-then-POST change has landed; otherwise record the step as "blocked on extension");
  - delete it via `.breeze_voice_delete[data-id="st_live_tmp"]`;
  - check the `voice.uploaded` and `voice.deleted` events, with no toasts;
  - check that `eric` and `vale` are still listed.
- [X] T008 [P] Create `tests/live/sillytavern/phases/speech.mjs`: in the SillyTavern page,
  `page.evaluate(fetch(httpUrl + '/v1/audio/speech', {method:'POST', body: FormData{text}}))`.
  Assert:
  - `200`;
  - `X-Sample-Rate` is readable (expose-headers work);
  - the body is non-empty and has an even byte length;
  - a request with `cfg_scale=banana` returns `400` with a JSON `error` key readable from the page.
- [X] T009 [P] Create `tests/live/sillytavern/phases/full.mjs`:
  - narrate the last message with quoted text via `.mes_narrate`: expect `synth.request`, then
    `synth.event` `started`, then `synth.done`, with the time to first audio recorded;
  - narrate, then click `#tts_media_control` to stop: expect `synth.cancelled` and no toast;
  - voice preview for `eric`;
  - two back-to-back narrations: expect a `queued` event on the second;
  - repeat the narration with `cfg_scale` set to 1, 4 and 7.5.
- [X] T010 Create `tests/live/sillytavern/run.mjs <health|voices|speech|full>`:
  - `full` runs health, then voices, then full; `speech` runs health, then speech.
  - Exit code 0 only if every step passes.
  - Add the command to the README development section (Constitution X).
- [X] T011 Run `node tests/live/sillytavern/run.mjs full` against the **C++ server**
  (`<Breeze-TTS-2.cpp checkout>`, started with `--cors`), with `speech.mjs` skipped because the
  C++ server returns `200` for `cfg_scale=banana`, which that script treats as a failure. Record the result as the reference behavior in
  `research/live-phase0.md`, then stop the C++ server.

**Checkpoint**: the baseline is recorded and the harness is green against the C++ server.

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: the composition root, error envelope, limits, GPU gate and thread, segmenter,
`/health`, launch scripts. No user story can start before this phase is done.

- [X] T012 [P] Create `breeze_infer/limits.py` with every constant in data-model.md "Limits".
  Plain module-level constants only (no mutable state).
- [X] T013 [P] Port `A:breeze_infer/events.py` to `breeze_infer/events.py` and `A:tests/test_events.py`
  to `tests/test_events.py`. The clock and output stream are injected; the emitter is built at the
  composition root and passed down.
- [X] T014 [P] Create `breeze_infer/errors.py` (R7), porting `A:breeze_infer/api.py` lines
  ~539–599. It holds `ApiError(status, code, message)` and `install_error_handlers(app, events)`:
  - `StarletteHTTPException` maps to the envelope, passing `exc.headers` through (so `405` keeps
    `Allow`), with codes `not_found` (404) and `method_not_allowed` (405);
  - `ApiError` maps to the envelope;
  - `python_multipart.exceptions.FormParserError` maps to `400 invalid_field`;
  - `RequestValidationError` maps to `400 invalid_field`;
  - `Exception` maps to `500 internal_error`, emits `request.failed` with `request_id`, and never
    includes the exception text.

  Port `A:tests/test_api_errors.py` to `tests/test_api_errors.py`, add `code` assertions, and name
  the tests `test_bc_18_*`.
- [X] T015 [P] Create `breeze_infer/body_limit.py` (R6): a pure-ASGI middleware. A
  `Content-Length` over `MAX_BODY_BYTES` is rejected immediately. Otherwise it wraps `receive`,
  counts bytes, and raises `ApiError(413, "payload_too_large", "request body is too large")`.

  `tests/test_body_limit.py` covers `Content-Length` over the limit, a chunked body over the limit,
  and a body under the limit passing (`test_bc_06_*`).
- [X] T016 [P] Create `breeze_infer/settings.py`: a frozen `Settings` plus `build_parser()` and
  `settings_from_args(argv)` for every field in data-model.md "Settings":
  - `--port` defaults to 8080;
  - `--ws-port` takes a port or `disabled`, and defaults to `port + 1`;
  - `--cors [ORIGINS]`: a bare flag means `*`;
  - `--split-chars`, `--chunk-first`, `--chunk-max`, `--voices-dir`;
  - the existing flags `--fast-*`, `--attn-implementation` and `--compile-cache-dir`, from
    `breeze_infer/api.py` `main()`.

  Port `_parse_cors_origins` from `A:breeze_infer/api.py` (~1384–1431): trim entries, reject `*`
  mixed with others. Reject negative `--split-chars` (don't clamp) and ports outside 1–65535.
  Clamp `chunk_first` to `chunk_max`.

  Create `tests/test_settings.py`, and move the flag tests out of `tests/test_runtime_flags.py`
  where they target the old parser.
- [X] T017 *(Opus)* Create `breeze_infer/gpu.py` `GpuGate` (R14): asyncio only, used only on the
  event loop.
  - `try_acquire() -> GpuLease | None` is `None` when the gate is held or any waiter is pending.
  - `async acquire(on_wait)` calls `on_wait` synchronously just before it blocks, and only if it
    blocks, so the caller can enqueue `queued` before waiting (changed after review: a returned
    flag arrives only after the wait).
  - `lease.release()` hands the gate directly to the next non-cancelled waiter; only the current
    lease can release (changed after review). `GpuSession(lease, gpu, gen)` closes the generator on
    the GPU thread before releasing, and is what T039 and T077 use.

  `tests/test_gpu_gate.py` covers try/acquire, handoff order, a cancelled waiter being skipped, and
  HTTP `try_acquire` failing while a WebSocket waiter is queued.
- [X] T018 *(Opus)* Add `GpuThread` to `breeze_infer/gpu.py` (R14): a single-thread executor that
  calls `torch.cuda.set_device(device)` once when the thread starts. Methods:
  - `run(fn, *args)` (awaitable);
  - `step(gen)`, which returns the next item or a sentinel on `StopIteration`;
  - `close(gen)`;
  - `shutdown()`.

  `tests/test_gpu_thread.py` checks that all calls run on one thread, that `close` queues behind an
  in-flight `step`, and that exceptions propagate. Pass a stub `set_device` callable so the test
  needs no GPU.
- [X] T019 *(Opus)* Port `A:breeze_infer/text_split.py` to `breeze_infer/text_split.py` with the
  R11 fixes. It exposes `weigh(text)`, `segment(buffer, *, budget, first_budget=0, final)` and
  `split_text(text, *, budget, first_budget=0)`, and keeps `ANCHOR_CHARS` in `limits.py`.
  1. Drop the NUL-as-closer quirk.
  2. Use one stop set: `\n`; `.!?;` followed by space, tab, CR, LF or U+3000 (not NBSP) or the end
     of a final buffer; and `。！？；…．` (not `．` between digits).
  3. Absorb closing quotes and brackets on both interfaces (ASCII, curly and CJK closers; the full
     list is in contracts/ws-api.md).
  4. A sentence end at the end of a non-final buffer doesn't cut.
  5. Use weighted length everywhere.
  6. Over-budget sentences are cut into clauses at the first break (space-like characters, `，`,
     `、`; `,` and `:` only via the whitespace after them) once a clause reaches the budget; a
     clause passing 2 × budget closes at its last break; a run with no break is hard-cut past
     2 × budget into chunks within budget, between grapheme clusters (review changes).
  7. `first_budget` applies to the first returned piece only.
  8. `budget == 0` means no length splitting.
  9. Strip pieces and drop pieces with no letter or digit; keep inner `\t` and `\r`.
- [X] T020 Port `A:tests/cpp_golden/{README.md,harness.cpp,gen_goldens.py,golden.json}` into
  `tests/cpp_golden/`. Change `gen_goldens.py` so it no longer rewrites the test file.

  Port `A:tests/test_text_split.py` to `tests/test_text_split.py`. It compares the normalized C++
  goldens (stripped, empties dropped) and holds an explicit `INTENTIONAL_DIFFERENCES` table: each
  entry has a case name, the expected new output, and a BC id (BC-39 or BC-46), as listed in
  research R11.

  New cases:
  - `"3."` then `"14"`;
  - `"Hi."` non-final waits;
  - `"好。"` at the end waits;
  - the opening budget applies to the first piece only;
  - CJK with no punctuation stays bounded;
  - `budget=0`;
  - `\n` cuts;
  - `\t` and `\r` are preserved;
  - `segment(x, final=True)[0] == split_text(x)` for every golden input;
  - a property test that a non-final leftover's weight stays within `max(budget, first_budget)`
    or 2 × budget.

  Name the tests `test_bc_39_*` and `test_bc_46_*`.
- [X] T021 Port `A:tests/fakes.py` to `tests/fakes.py` and extend it with `FakeRuntime`:
  - configurable chunks per piece, a per-chunk `threading.Event` gate, and fail-after-N;
  - it records the seeds, the sampling overrides and the reference passed to each piece;
  - it reports `sample_rate=24000`.

  Also add `FakeCodec` (`encode(wav, sr)` gives deterministic codes; the frame count follows the
  same formula as the prediction in T036). Add a docstring citing the Principle V deviation in
  plan.md Complexity Tracking.
- [X] T022 *(Opus)* Rewrite `breeze_infer/api.py` as the composition root only (Constitution III;
  R3, R5, R14):
  - `main(argv)`: build `Settings`, then `Events`, `GpuGate`, `GpuThread` and a `Readiness` holder,
    then `create_app(components)`.
  - Load the model in the background: on the `GpuThread`, call `configure_compile_cache`,
    `load_runtime` and warmup; then mark it ready and emit `model.loaded`.
  - Pre-bind the HTTP socket on `settings.host:settings.port`, with `TCP_USER_TIMEOUT` set when
    `socket` has it. Serve with `uvicorn.Server(Config(app, lifespan="off")).serve(sockets=[sock])`
    and emit `server.started` with every bound address (`host`, `addresses`).
  - Install one SIGINT/SIGTERM handler with `loop.add_signal_handler`, setting `should_exit`.
  - Remove the old `/v1/audio/speech` handler, `_settings`, `_request_lock`, the `os.environ` read,
    `print`s, and `MAX_*` duplicates (use `limits.py`).
  - Delete `tests/test_api.py`.
  - Keep `python -m breeze_infer.api` working.
- [X] T023 Create `breeze_infer/routes_health.py` (contracts/http-api.md `GET /health`):
  - `GET`/`HEAD` return `200` with exactly `{"status","sample_rate","ws_port"}`. `ws_port` comes
    from an injected provider, which returns 0 until Phase 8.
  - While loading, the body is `503 {"status":"loading","error":"model is loading","code":"loading"}`.
  - Add a `require_ready` dependency used by every other route: `503` with the same body.

  `tests/test_health.py` covers the loading body, the exact key set when ready, `HEAD`, a `404`
  envelope for unknown paths, and a `405` with `Allow` for a known path, including `OPTIONS` with
  CORS off (`test_bc_18_*`, `test_bc_24_*` partly). It also checks FR-001's excluded routes:
  `POST /v1/audio/convert`, `GET /`, `GET /app.js` and `GET /style.css` all return the `404`
  envelope.
- [X] T024 [P] Update the launch scripts:
  - `scripts/start_breeze.sh`: `--port 8080`; pass extra arguments through, so `--cors` works.
  - `scripts/start_breeze.ps1`: `-Port 8080`, and new `-Cors` and `-WsPort` parameters.
  - `docker/run.sh`: publish `8080:8080` and `8081:8081` and pass `--port 8080`.
  - `docker/README.md` if it mentions 7860.
- [X] T025 [P] Set `breeze_infer/__init__.py` `__version__ = "2.0.0.dev1"`. Create
  `breeze_infer/version_header.py`, a pure-ASGI middleware that adds
  `X-Breeze-Version: <__version__>` to every HTTP response start message (FR-037a). Wire it in
  `api.py` as the outermost layer, so CORS's own preflight and `403` responses carry it too (changed
  after review, 2026-09-24; it was "just inside CORS").
  `tests/test_version_header.py` checks the header on `200`, `404`, `413`, `500` and a streamed
  response. Create `CHANGELOG.md`
  with an "Unreleased — 2.0.0" section that states the old Python API is removed and lists
  BC-18–BC-24. Add a README "Development" section with the quickstart command table.

**Checkpoint**: the server starts, `/health` goes from `503` to `200`, and the unit suite is green.
User stories can begin.

---

## Phase 3: User Story 5 — Browser clients and cross-origin safety (P3, done first so the live gates work in a browser)

**Goal**: SillyTavern's browser can call the API; disallowed origins can't write or spend GPU.

**Independent Test**: CORS headers and `403`s for allowed, disallowed and absent origins on every
route (quickstart Scenario 1.3–1.4); the SillyTavern `health` live gate passes.

### Tests (write first; they must fail)

- [X] T026 [P] [US5] `tests/test_cors.py`: port the parse tests from `A:tests/test_cors.py`
  (~39–100), then add the following (contracts/http-api.md "CORS"):
  - `test_bc_19_allowlist_entries_are_trimmed`;
  - `test_bc_20_vary_origin_on_every_response_in_allowlist_mode`, including no Origin and a
    disallowed origin;
  - `test_bc_21_preflight_is_route_aware`: `204` with the route's methods, `404` for an unknown
    path, `405` for a method the route lacks, requested headers echoed, `Max-Age 86400`;
  - `test_bc_22_star_mixed_with_origins_fails_at_startup`;
  - `test_bc_23_post_and_delete_from_disallowed_origin_get_403_before_the_endpoint_runs`, both
    with CORS off and with an allowlist; use an endpoint that counts calls;
  - no-Origin requests pass;
  - `X-Breeze-Version` is listed in `Access-Control-Expose-Headers`;
  - allowed origins get `Access-Control-Allow-Origin` and expose-headers on `200`, `404`, `413`,
    `500` and a streamed response;
  - a disallowed preflight gets `403`.

  Note (review): the `A:tests/test_cors.py` parse tests weren't ported here after all -- they're
  covered by `tests/test_settings.py`, since `settings.py` replaced `_parse_cors_origins` (and now
  `canonical_origin`, in `breeze_infer/origins.py`) as the place `--cors` parsing and validation
  actually lives.

### Implementation

- [X] T027 [US5] Create `breeze_infer/cors.py` (R8):
  - Pure functions: `origin_allowed(policy, origin)`, `preflight_headers(policy, route_methods,
    requested_headers)`, `response_headers(policy, origin)`.
  - A pure-ASGI `CorsMiddleware(app, policy, router)`. It finds a route's methods with
    `route.matches(scope)` and `route.methods`, and it rejects unsafe methods (`POST`/`DELETE`)
    with a disallowed `Origin` by returning the `403` envelope before calling the app.
- [X] T028 [US5] Wire it up in `breeze_infer/api.py`:
  `VersionHeaderMiddleware(CorsMiddleware(BodyLimitMiddleware(fastapi_app)))`, so CORS sits
  outside the body limit and the app, and even `413`s and `500`s carry its headers; the version
  header stays outermost (T025). Build the policy from `Settings`. With CORS off the policy allows no
  origin, and every `OPTIONS` falls through to `405`.
- [X] T029 [US5] **Live gate (health)**. Standing rule 8: the version is already `2.0.0.dev1`;
  update the CHANGELOG for BC-19–BC-23.
  1. Start `scripts/start_breeze.sh --cors http://127.0.0.1:8000`.
  2. Check with curl as in quickstart Scenario 1.1–1.4.
  3. Run `node tests/live/sillytavern/run.mjs health`.
  4. Record the result in `research/live-phase1.md`, along with the review-loop outcomes for every
     commit in Phases 2–3.

**Checkpoint**: the plan's Phase 1 is complete; SillyTavern loads the provider against the new
server.

---

## Phase 4: User Story 1 — speech over HTTP (P1) 🎯 MVP for HTTP clients

**Goal**: a C++-contract client gets streamed PCM from `POST /v1/audio/speech` for voice design
and inline reference. A busy server returns `409`.

**Independent Test**: quickstart Scenario 2.1; `tests/test_routes_speech.py` with `FakeRuntime`;
the GPU smoke test.

### Tests (write first)

- [X] T030 [P] [US1] Update `tests/test_fast_streaming.py`:
  - Replace the `_prefill_plan` bool assertions (lines ~234–240) with `A:`'s tuple test
    (`A:tests/test_fast_streaming.py` ~575–580).
  - Port `A:`'s override tests (backbone only, NaN and inf rejected, temperature floor),
    `max_new_tokens_room` tests (~463–610) and prefix-guard tests (~678–705).
  - The runtime raises `ValueError` on invalid overrides; it doesn't coerce.
- [X] T031 [P] [US1] Port the `A:` codes-path, prefix and suffix tests into
  `tests/test_templates.py` (`ref_audio_codes`, `_check_reference_source`,
  `split_reference_prefix`, `prepare_prefix_inputs` / `prepare_suffix_inputs`). Drop the
  empty-instruction pass-through test (BC-09 now maps a blank instruction to the default at the
  boundary).
- [X] T032 [P] [US1] Port the `A:` audio tests into `tests/test_audio.py`: `encode_prompt_waveform`,
  `pcm16` clipping, and `codec_fingerprint`, which must not change when the checkpoint is moved to
  another absolute path.
- [X] T033 [P] [US1] Port `A:tests/gpu/test_runtime_request_overrides.py` and
  `A:tests/gpu/conftest.py` into `tests/gpu/`. Add `tests/gpu/test_cfg_values.py`: `cfg_scale` 2.5,
  7.5 and 0 each produce finite audio on the warmed graphs, with no recapture (check the graph
  cache size before and after).

### Implementation

- [X] T034 [US1] *(Opus)* Update `models/fast_streaming.py` (R12, points 1–4), porting the hunks of
  `A:` commits `86647b8` and `220b3ca` onto 81a5ca7's code:
  - `iter_audio_chunks(..., temperature, top_k, top_p, repetition_penalty, max_new_tokens)`, where
    `None` means the default. Overrides apply to the backbone only, with a temperature floor of
    1e-5.
  - `_prefill_plan` returns `(use_graph, prefill_len)`, and both callers are updated.
  - `max_new_tokens_room(requested, inputs, prefix_len)`.
  - `build_reference_prefix` rejects only when `prefix_len >= max_seq_len - 1`.
  - The default length comes from `model.generation_config.max_new_tokens` (750);
    `config.max_new_tokens` (1500) is the ceiling.
  - Warmup exercises the top-k/top-p and penalty kernels.
  - Add the repetition-penalty semantics docstring in `models/cudagraph/sampling.py`.
- [X] T035 [P] [US1] Update `breeze_infer/templates.py` and `breeze_infer/audio.py`:
  - Port the `A:6fb3736` hunks: `ref_audio_codes` segments, `required_fields` without the path,
    `_check_reference_source`, the prefix/suffix split, `encode_prompt_waveform(tokenizer, wav,
    sr)`, and `codec_fingerprint` (sha256 of the codec config's required identity fields from
    both `encoder_config` and `decoder_config` in `audio_tokenizer/config.json`, plus the
    safetensors header -- tensor name/dtype/shape, not trained values -- of the one weight file
    the codec loader actually reads; no path hashed; detects an architecture/shape/dtype change,
    not a retrain of same-shaped weights).
  - Move `_pcm16` from the old `api.py` to `audio.pcm16`.
  - Remove `encode_prompt_audio(path)` and the temporary-file upload path.
- [X] T036 [P] [US1] Create `breeze_infer/reference_audio.py` (R9):
  - `decode(blob) -> DecodedAudio`: open with `sf.SoundFile` and check the header (format, 1–8
    channels, 8–192 kHz). A trustworthy, over-30 s frame count is rejected without decoding;
    otherwise decode in bounded blocks (capped at 30 s + 1 sample) and let the actual decoded
    length decide, downmixing to mono in float64 as each block arrives.
  - `predicted_frames(duration, sample_rate)`, using the 12 Hz codec's frame arithmetic. Document
    the formula; it is asserted against the real encode in T049.
  - Errors are `ApiError(400, invalid_audio | audio_too_long | audio_too_short)`.
  - `tests/test_reference_audio.py` generates valid inputs in-test: 8, 16 and 24-bit PCM, float,
    stereo 44.1 kHz, FLAC and OGG, all decoding to mono float32.
- [X] T037 [US1] Create `breeze_infer/http_fields.py`, happy path only (data-model.md
  `SpeechRequest`):
  - `async read_fields(request)`: `request.form(max_files=1, max_fields=32, max_part_size=64 KiB)`
    merged with the query string.
  - `parse_speech(fields, settings) -> SpeechRequest`, applying the defaults and treating an empty
    value as absent (BC-02). `0` means `None` for the sampling fields. A blank instruction becomes
    the default (BC-09).
  - Build the `ReferenceSpec` variants.
  - Missing or blank `text` gives `400 text_required` (BC-10).
  - `tests/test_http_fields.py` covers defaults and parsing of valid values.
- [X] T038 [US1] Create `breeze_infer/synthesis.py`, the single-reference part (data-model
  `Reference`, `Piece`):
  - the `Reference` variants;
  - `resolve_reference(spec, …)`: an inline reference is encoded once on the `GpuThread` into
    codes;
  - `piece_seed(s, i) = (s + i) & 0xFFFFFFFF`;
  - `prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)`;
  - `ramp_pcm(chunks, chunk_first, chunk_max)`, ported from `A:api.py` ~199–238;
  - `generate_piece(...)`, a sync generator of PCM bytes for the `GpuThread`.

  Nothing reads `app.state`; everything is passed in. `tests/test_synthesis.py` uses `FakeRuntime`:
  seeds per piece, inline reference encoded once for every piece and both CFG rows, and ramp
  growth.
- [X] T039 [US1] *(Opus)* Create `breeze_infer/streaming.py` (R2, R3). `SpeechResponse`, a
  `StreamingResponse` subclass:
  - `__init__` takes the primed first chunk and an async body.
  - `stream_response` wraps each `send` in `asyncio.timeout(HTTP_SEND_TIMEOUT_SECONDS)` and
    `aclose()`s the body in `finally`.
  - `__call__` runs, in a shielded `finally`: `GpuSession.aclose()` (from `gpu.py`: it closes the
    generator on the `GpuThread`, then releases the lease; don't re-implement that sequence), then
    emit `speech.completed`, `speech.aborted` or `speech.failed`.
  - An exception after headers propagates, so uvicorn closes without the terminator. **Never use
    `BaseHTTPMiddleware` anywhere in the app.**
- [X] T040 [US1] Create `tests/test_speech_abort.py`. It runs a **real uvicorn** on an ephemeral
  port with `FakeRuntime`:
  - `test_bc_17_failure_after_streaming_starts_aborts_the_response`: httpx raises
    `RemoteProtocolError`, and curl-style checks confirm there is no `0\r\n\r\n` terminator;
  - `test_bc_17_failure_before_first_chunk_returns_an_error_status`;
  - `test_client_disconnect_releases_the_gate_within_one_chunk`;
  - `test_gate_released_if_body_never_starts`;
  - `test_stalled_reader_hits_send_timeout` (patch the timeout down to 0.5 s via injection).
- [X] T041 [US1] Create `breeze_infer/routes_speech.py`, `POST /v1/audio/speech`:
  1. `require_ready`;
  2. `read_fields` and `parse_speech`;
  3. reference checks (the full order arrives with US2);
  4. decode the reference on a worker thread;
  5. `gate.try_acquire()`, else `409 busy`;
  6. prepare on the `GpuThread`;
  7. prime the first chunk;
  8. return `SpeechResponse` with the contract headers.

  Split with `split_text(text, budget=split_chars)`; anchoring arrives in US3. Emit
  `speech.accepted` and `speech.first_audio`. Register the router in `api.py`.
- [X] T042 [US1] Create `tests/test_routes_speech.py` (`FakeRuntime`, TestClient):
  - `200` headers `audio/pcm`, `X-Sample-Rate`, `X-Sample-Format` and `Cache-Control`, with a
    non-empty body;
  - voice design and inline reference;
  - `409 busy` while the gate is held;
  - `409` while a WebSocket waiter is queued (drive `GpuGate` directly);
  - piece seeds;
  - `503 loading` before ready.
- [X] T043 [US1] Create the GPU smoke test `tests/gpu/test_speech_http.py`: voice design and inline
  reference return a PCM stream of plausible length, with the whole app on the real runtime. Then
  run `bench_api --api new` and compare with the T002 baseline ("SC-007 method" below), recording
  the numbers in `research/bench-phase2.md`. If a gating case's time to first audio regresses more
  than 10%, stop and
  investigate before continuing (SC-007).

**Checkpoint**: HTTP speech works end to end on the GPU.

---

## Phase 5: User Story 2 — invalid input is rejected with a clear error (P1)

**Goal**: every malformed input gets a specific `4xx` with the envelope, in the FR-007 order, and
never crashes or silently changes behavior.

**Independent Test**: quickstart Scenario 2.2 (the malformed corpus), plus the `test_bc_01`–`16`
and `test_bc_46` tests.

### Tests (write first; each docstring states the C++ behavior it rejects)

- [X] T044 [P] [US2] Add table-driven tests to `tests/test_http_fields.py`:
  - `test_bc_01_unparseable_numbers_get_400`: `banana`, `1e`, `0x10`, `inf`, `nan`, `1.5` for an
    integer field;
  - `test_bc_02_empty_value_is_absent`;
  - `test_bc_03_out_of_range_sampling_values_get_400`: negative values, `top_p` over 1,
    `cfg_scale` over 100, NaN; `0` means the default;
  - `test_bc_04_max_new_tokens_over_ceiling_gets_400`;
  - `test_bc_05_text_or_instruction_too_long_gets_400`;
  - `test_bc_08_duplicate_field_in_body_query_or_both_gets_400`;
  - `test_bc_09_blank_instruction_uses_default`;
  - `test_bc_10_whitespace_text_gets_400`;
  - `test_bc_46_control_characters_get_400`: NUL, `\x07` and `\x1b` rejected; `\t`, `\r` and
    `\n` allowed;
  - seed range 0–4294967295; `split_chars` 0–10,000.
- [X] T045 [P] [US2] Add route tests to `tests/test_routes_reference.py` (TestClient,
  `FakeRuntime`, soundfile for real):
  - `test_bc_11_undecodable_or_empty_ref_audio_gets_400`;
  - `test_bc_12_ref_audio_without_ref_text_gets_400`;
  - `test_bc_13_ref_text_without_reference_gets_400`;
  - `test_bc_14_voice_id_with_ref_audio_gets_400`;
  - `test_bc_15_malformed_wavs_are_rejected_safely`: data chunk length `0xFFFFFFFF`, a truncated
    `fmt` chunk, zero channels, zero bits, a 0 Hz sample rate, a 1 Hz sample rate, 65,535
    channels, an unknown format tag; also check that 8, 24-bit and float WAVs now decode;
  - `test_bc_16_too_long_or_too_short_reference_gets_400`.
- [X] T046 [P] [US2] Create `tests/test_validation_order.py` (FR-007):
  - `test_bc_07_invalid_request_while_busy_gets_400_not_409`;
  - field errors come before an unknown `voice_id`, which comes before decoding, which comes
    before busy;
  - `test_bc_06_oversize_multipart_gets_413_envelope`.
- [X] T047 [US2] Create `tests/test_malformed_corpus.py` (SC-003). It sends **at least 50**
  malformed HTTP requests:
  - bad numbers, duplicates, control characters, reference conflicts;
  - corrupt, truncated and oversize audio;
  - a multipart body with no boundary, junk after the boundary, and a truncated body with no
    closing boundary (it must produce `400`s for the missing required fields);
  - bad `Content-Type`.

  Every item gets a `4xx` with the JSON `error` and `code` keys. `/health` stays `200` throughout,
  and a normal request afterwards still returns `200`.

### Implementation

- [X] T048 [US2] Complete `breeze_infer/http_fields.py`:
  - the strict number grammar from contracts/http-api.md, with ranges;
  - duplicate detection (`_check_no_duplicate_names`: every raw name seen on the wire, file
    parts included, counted across the query string and the body combined, so a name
    repeated within a single source and one split across both are both `400
    duplicate_field`); run once over the query string alone before the body is even
    sniffed, and again per body branch over the query's names combined with that branch's;
    also run over whatever names a multipart or urlencoded parsing limit left behind, so a
    name repeated enough times to trip that limit is still `duplicate_field`, not the
    parser's generic error;
  - length limits and the control-character rule;
  - `ReferenceSpec` checks in order: `reference_conflict`, then `ref_text_required`, then
    `reference_required`;
  - error messages exactly as in the contract's error table.
- [X] T049 [US2] Enforce the FR-007 order in `breeze_infer/routes_speech.py`:
  1. body limit;
  2. fields;
  3. reference rules;
  4. unknown voice (a stub lookup returns 404 until Phase 7);
  5. decode, including the too-short check using `predicted_frames`;
  6. `try_acquire`.

  After the encode on the GPU, assert that the predicted frame count equals the actual count.
  Emit `speech.frame_prediction_mismatch` on a mismatch, and add a GPU assertion to
  `tests/gpu/test_speech_http.py`.

**Checkpoint**: all of `test_bc_01`–`17` and `test_bc_46` pass, and so does the malformed corpus.

---

## Phase 6: User Story 3 — long text completes (P1)

**Goal**: long text is split, anchored and seeded as in C++, with explicit room handling.

**Independent Test**: quickstart Scenario 2.4 (3,000 characters × 5 on the GPU),
`tests/test_long_text.py`.

### Tests (write first)

- [X] T050 [P] [US3] Create `tests/test_long_text.py` (`FakeRuntime`), porting the matching `A:`
  anchoring cases from `A:tests/test_api_speech.py` (~462–494):
  - `test_first_piece_anchors_every_later_piece`;
  - the anchor skips all-pad frames, and there's no anchor when piece 0 produced 0 frames;
  - with a reference, every piece uses that reference;
  - the opening budget applies only with no reference;
  - `split_chars=0` gives one piece;
  - `test_bc_47_first_piece_without_room_gets_400_text_too_long`;
  - `test_bc_47_later_piece_without_room_aborts_the_stream`, with a real uvicorn as in T040;
  - `test_partial_room_clamps_and_emits_piece_clamped` (FR-036a).
- [X] T051 [P] [US3] Port `A:tests/gpu/test_speech_long_text.py` to
  `tests/gpu/test_speech_long_text.py`: a fixed 3,000-character passage, no reference,
  `--fast-all`, 5 runs. Every stream completes, and the audio duration exceeds a floor derived
  from the text length (SC-004). Save the first run's audio as
  `specs/003-cpp-compatible-api/research/long-text-sample.wav` for the listening check in T054
  (not committed; `*.wav` is ignored).

### Implementation

- [X] T052 [US3] Extend `breeze_infer/synthesis.py`:
  - `anchor_codes(frames, pad_id)`: port `A:api.py` ~1002–1010, dropping all-pad frames;
  - anchor after piece 0 succeeds with at least one frame;
  - `piece_room(...)` using `max_new_tokens_room`;
  - the piece cap is `min(max_new_tokens or default, room)`;
  - no room for piece 0 raises `ApiError(400, text_too_long)` before streaming; no room for a later
    piece raises, which aborts the stream;
  - a partial room clamps and emits `speech.piece_clamped`.
- [X] T053 [US3] Update `breeze_infer/routes_speech.py`:
  - use `split_text(text, budget=split_chars, first_budget=ANCHOR_CHARS if no reference else 0)`;
  - check the first piece's room on the CPU before `try_acquire` (predicted reference frames plus
    tokenized text via `_prefill_plan`);
  - prepare later pieces on the `GpuThread` inside the body.
- [X] T054 [US3] **Live gate (speech)**:
  1. Set the version to `2.0.0.dev2` (re-deploys: `2.0.0.dev2+1` after the review fixes, `+2` after the reference-encode fix; plan.md's live-gate step), and add BC-01–BC-17, BC-46 and BC-47 to the CHANGELOG
     (standing rule 8).
  2. Start the server with `--cors http://127.0.0.1:8000`.
  3. Run quickstart Scenario 2.1–2.5 by hand, and `node tests/live/sillytavern/run.mjs speech`.
     Listen to one of the 3,000-character outputs from T051 (saved as a WAV by the test) and
     record in the live record whether one speaker is heard throughout (SC-004 / US3's
     independent test; a manual check, since there is no automated speaker comparison).
  4. Re-run the benchmark (SC-007).
  5. Record the results in `research/live-phase2.md`, along with the review outcomes for Phases
     4–6.

**Checkpoint**: the plan's Phase 2 is complete; HTTP speech is compatible, strict and handles long
text.

---

## Phase 7: User Story 1 — voices (P1, continued)

**Goal**: register, list, delete and use voices by `voice_id`, with the new file format and name
rules.

**Independent Test**: quickstart Scenario 3; the SillyTavern `voices` live gate.

### Tests (write first)

- [X] T055 [P] [US1] Create `tests/test_voice_file.py` (pure): v1 round trip. Rejections:
  - an unknown `format` or `version`;
  - a stem that differs from `id`;
  - a bad name, or a `v_` name;
  - a codes length or sha256 mismatch;
  - a code outside `[0, codebook_size)`;
  - a fingerprint mismatch.
- [X] T056 [P] [US1] Create `tests/test_voice_store.py` (real `tmp_path`, with `sleep` injected):
  - atomic create that never overwrites;
  - case-duplicate creation refused;
  - delete renames to `.del-*` and retries 5× on `PermissionError`;
  - leftover `.del-*` and `.tmp-*` files are swept;
  - `test_bc_29_breeze_files_are_ignored_and_counted`;
  - `test_bc_25_invalid_files_are_skipped_with_an_event`;
  - a skipped file with a valid name reserves that name.
- [X] T057 [P] [US1] Create `tests/test_voice_registry.py` (pure):
  - `test_bc_26_names_are_unique_ignoring_case_and_v_prefix_is_reserved`;
  - `test_bc_48_cap_counts_only_unnamed_voices`;
  - oldest unnamed voice evicted first;
  - an identical unnamed registration returns the existing entry;
  - `test_bc_25_list_order_is_saved_sorted_then_unnamed_by_registration`.
- [X] T058 [P] [US1] Port `A:tests/test_voice_prefix_cache.py` to `tests/test_voice_prefix_cache.py`,
  re-keyed by `(voice_id, content_hash)` (`voice_file.prefix_key`: `ref_text` and codes). Add the
  delete-and-re-register case: the stale KV is never returned, including when the delete lands
  during the build.
- [X] T059 [P] [US1] Create `tests/test_routes_voices.py` (TestClient, `FakeRuntime`, `tmp_path`):
  - the POST check order from contracts/http-api.md;
  - `test_bc_27_existing_name_gets_409_voice_exists`, also for different case;
  - an unnamed dedupe returns `200` without the gate, even while busy;
  - `409 busy`;
  - `test_bc_28_delete_removes_the_file_and_it_stays_gone_after_restart`: rebuild the app on the
    same directory;
  - `file_kept` is always false;
  - `404 unknown_voice`;
  - deleting a skipped file's name releases it;
  - `GET` list shape and order;
  - the `200` response shape (`seconds` with 2 decimals, integer `encode_ms`).
- [X] T060 [P] [US1] Add speech-by-voice tests to `tests/test_routes_speech.py`:
  - `voice_id` uses the prefix path;
  - `voice_id` plus `ref_text` uses the codes path with the override;
  - an unknown or deleted voice gets `404`;
  - a KV prefix is reused across requests.

  Port the matching cases from `A:tests/test_api_speech.py` (~378–521).

### Implementation

- [X] T061 [P] [US1] Create `breeze_infer/voice_file.py`, the v1 codec and validation from
  data-model.md "Voice file v1". Codes are base64 of int16 little-endian, row-major.
- [X] T062 [US1] Create `breeze_infer/voice_store.py`: `scan() -> (voices, skipped, breeze_count)`,
  `create(record)` and `remove(id)`.
  - Port from `A:breeze_infer/voices.py`: `_rename_with_retry` (~545–554), the fsync helpers
    (~563–586), the leftover sweep (~503–510) and the write-lock pattern (~190–194).
  - `sleep`, the clock (for `created_at`) and the nonce source (for `.del-<nonce>` and
    `.tmp-<nonce>` names) are injected; tests pass fixed ones.
  - Emit `voices.loaded{loaded, skipped, breeze_ignored}` and `voice.skipped{file, reason}`.
- [X] T063 [US1] Create `breeze_infer/voice_registry.py`:
  - in-memory voices, a case-insensitive name index (including reserved skipped names), the
    unnamed cap and eviction, and ordering;
  - `unnamed_id(wav_bytes, text) = "v_" + blake2b(len ‖ wav ‖ text, digest_size=8).hexdigest()`;
  - one lock, never held across I/O or the GPU;
  - the clock used for `encode_ms` timing is injected.

  Borrow `api_record` and `voice_seconds` from `A:breeze_infer/voice_index.py` (~61–85).
- [X] T064 [P] [US1] Port `A:breeze_infer/voice_prefix.py` to `breeze_infer/voice_prefix.py`: an
  LRU of `ReferencePrefix` keyed by `(voice_id, content_hash)` (`voice_file.prefix_key`) and
  bounded by `VOICE_PREFIX_CACHE_BYTES`, built on the `GpuThread` while the gate is held. A
  build's out-of-memory error propagates with nothing cached; the codes-path fallback is T066's.
  Delete invalidates the entry.
- [X] T065 [US1] Create `breeze_infer/routes_voices.py`:
  - `POST /v1/voices`, in the contract order, with a commit-time name re-check under the store
    lock;
  - `GET /v1/voices`;
  - `DELETE /v1/voices/{id}`.

  Emit `voice.created` and `voice.deleted`. Build the store and registry in `api.py` from
  `settings.voices_dir`. The startup order is: model load, then the voice scan (on a worker
  thread, since checking files needs the loaded model's codec fingerprint and codebook size),
  then mark ready. Pass the clock and nonce source in from the composition root.
- [X] T066 [US1] Connect `VoiceRef` to `breeze_infer/synthesis.py` and `routes_speech.py`:
  - replace the stub lookup with the real one;
  - with no override, use the prefix path;
  - with an override, use the codes path;
  - on a prefix build that raises CUDA out-of-memory, fall back to the codes path for that
    request (emit an event) instead of a 500;
  - read `VoicePrefixCache.token()` in the same event-loop step that resolves the voice from the
    registry (before waiting for the gate), and pass it to `get_or_build` as `resolved_token`,
    with the request's `request_id`; DELETE passes its `request_id` to `invalidate`;
  - the registry's saved and unnamed records must carry codes and ref_text (or a way to load
    them) so a VoiceRef can be resolved; decide which during T066. Decided: the records hold
    the codes and `ref_text` in memory, loaded at scan or registration (`VoiceRegistry.lookup`).
- [X] T067 [P] [US1] Port `A:tests/gpu/{test_voice_equivalence,test_voice_prefill_buckets,
  test_voice_tier1_equivalence}.py` to `tests/gpu/`, adapted to the new store (register through the
  route). Include a prefix longer than 548 tokens that now builds (R12, point 4).
- [X] T068 [US1] Coordinate with `sillytavern-agent` via SendMessage: confirm whether the
  extension's DELETE-then-POST replace flow and the new delete wording have landed. Record the
  answer in `research/live-phase3.md`.
- [X] T069 [US1] **Live gate (voices)**:
  1. Set the version to `2.0.0.dev3`, and add BC-25–BC-29 and BC-48 to the CHANGELOG.
  2. Start the server and register `eric` and `vale` with curl (quickstart Scenario 3.1), using the
     samples and transcripts in `$REFERENCE_VOICES_DIR/{eric,vale}`.
  3. Run quickstart Scenario 3.2–3.4 and `node tests/live/sillytavern/run.mjs voices`.
  4. Record the results and the review outcomes for Phase 7 in `research/live-phase3.md`.

**Checkpoint**: the plan's Phase 3 is complete; SillyTavern can manage voices on the new server.

---

## Phase 8: User Story 4 — incremental text over a WebSocket session (P2)

**Goal**: SillyTavern (and other C++ WebSocket clients) synthesize over the WebSocket, with exact
`done`/`cancelled` semantics and isolation from slow clients.

**Independent Test**: quickstart Scenario 4; the SillyTavern `full` live gate; the extension's
`npm run test:live`.

- [ ] T070 [US4] *(Opus)* **Prototype first (R4 plan check)** in the session scratchpad, not the
  repo. Confirm with a `websockets` native server that:
  1. a peer that stops reading is aborted after `close_timeout`, and the socket is gone;
  2. `process_request` can return a JSON `403` and `503` with the envelope;
  3. a pre-bound socket works, and a bind failure can be caught;
  4. it runs on the same loop as uvicorn.

  Write the findings to `specs/003-cpp-compatible-api/research/ws-prototype.md`. **If any check
  fails, stop and re-plan with the user** (two-strike rule).

### Tests (write first)

- [ ] T071 [P] [US4] Create `tests/test_ws_messages.py` (pure):
  - `test_bc_32_unicode_escapes_are_decoded` (`"你好"`);
  - `invalid_json`;
  - `unknown_type` with the message `unknown type`;
  - wrong field types and out-of-range values give `invalid_field` (sharing the range rules with
    `http_fields`);
  - `start` accepts `top_p`, `repetition_penalty` and `max_new_tokens`;
  - `test_bc_38_split_chars_zero_means_no_splitting_negative_is_invalid`;
  - `test_bc_46_control_characters_rejected`;
  - `test_bc_13_start_ref_text_without_voice_id_is_an_error`.
- [ ] T072 [P] [US4] Create `tests/test_ws_session.py` (pure session plus a scripted fake worker):
  - `test_bc_34_end_with_nothing_left_still_sends_one_done`;
  - `test_bc_35_every_cancel_gets_exactly_one_cancelled_even_when_idle`;
  - `test_bc_35_cancel_between_pieces_never_drops_later_pieces`;
  - `test_bc_36_start_with_pending_work_cancels_then_starts`; a `start` on an idle session sends
    only `started`;
  - `test_bc_37_blank_instruction_resets_to_default`;
  - `test_bc_39_end_of_buffer_punctuation_waits`; the opening budget applies to the first piece
    only; the opening flag resets when piece 0 is cancelled;
  - `test_bc_40_buffer_over_limit_is_an_error_and_not_appended`;
  - a later `cancel` replaces a pending `done`;
  - `not_started` before `start`;
  - an unknown `voice_id` on `start` leaves the previous session untouched;
  - **SC-005**: 1,000 seeded random sequences of `text`, `flush`, `cancel`, `end`, `start` and
    `instruction`, asserting exactly one `done` per un-superseded `end`, exactly one `cancelled`
    per `cancel`, and no lost uncancelled piece.
- [ ] T073 [P] [US4] Create `tests/test_ws_server.py` (a real server on ephemeral ports,
  `FakeRuntime`, the `websockets` client, and raw sockets for protocol probes):
  - `ready` first, with `sample_rate` from the runtime (`test_bc_33_*`);
  - every handshake response, accepted or refused, carries `X-Breeze-Version` (FR-037a);
  - `test_bc_31_disallowed_origin_gets_403_json`, and no-Origin connections allowed;
  - `test_bc_30_binds_configured_host_only`;
  - `503 loading` and `503 too_many_connections` above 16;
  - handshake timeout;
  - `test_bc_43_close_codes`: an echoed client close gives 1000, an unmasked frame 1002, bad UTF-8
    1007, a message over 1 MiB 1009;
  - `test_bc_44_speaking_text_is_exact`: tabs and CR preserved;
  - `test_bc_45_binary_frame_gets_error`;
  - `test_bc_41_generation_failure_is_an_error_event_and_session_continues`;
  - `test_bc_42_slow_client_closed_1008_and_gpu_freed`;
  - `queued` sent only when actually waiting;
  - piece seeds continue from `start`.
- [ ] T074 [P] [US4] Create `tests/test_ws_isolation.py` (SC-006): WebSocket client A stops
  reading mid-piece. An HTTP speech request from client B starts streaming within the in-flight
  piece plus 1 s. Client A is closed with 1008.

### Implementation

- [ ] T075 [US4] Create `breeze_infer/ws_messages.py`: parse text frames with `json.loads` into
  typed messages (`Start`, `Text`, `Flush`, `End`, `Instruction`, `Cancel`) or a
  `WsError(code, message, request_type)`. Reuse the range and grammar helpers from `http_fields.py`
  (extract them into shared functions there if needed). Emit no I/O.
- [ ] T076 [US4] *(Opus)* Create `breeze_infer/ws_session.py`: the pure state machine from
  data-model.md "WebSocket Session" and R15. `apply(message) -> list[immediate events]`, plus the
  work-deque operations the worker consumes (`next_item`, `mark_piece_done(anchor)`,
  `is_stale(piece)`). No I/O, no GPU, no locks.
- [ ] T077 [US4] *(Opus)* Create `breeze_infer/ws_server.py`:
  - `serve(settings, components, sock)` using `websockets.asyncio.server.serve`, with
    `process_request` handling the Origin check via `cors.origin_allowed`, loading, and the
    connection cap, and a `process_response` hook that adds `X-Breeze-Version` to every handshake
    response; `open_timeout=10`, `max_size=1 MiB`, `ping_interval=20`, `ping_timeout=20`,
    `close_timeout=2`.
  - Per connection: send `ready`, then run a reader task (parse, `session.apply`, enqueue
    immediate events) and one worker coroutine:
    - `CancelMark` sends `cancelled`, `StartMark` sends `started`, and `EndMark` sends `done` if
      its epoch is current;
    - a `Piece` does `gate.acquire` (sending `queued` when it waits), sends `speaking`, and steps
      the generator on the `GpuThread`, checking the cancel event between chunks;
    - the piece anchors on success; on failure it sends `generation_failed`; `finally` closes the
      generator and releases the gate.
  - One ordered outbox bounded at `WS_OUTBOX_BYTES`, plus a sender task. On overflow: cancel the
    piece in flight, then close with 1008 `client too slow`.
  - On disconnect: bump the epoch, cancel, and join.
  - Emit the `ws.*` events with `session_id` and `piece_index`.
- [ ] T078 [US4] Wire the WebSocket into `breeze_infer/api.py`:
  - Unless `ws_port` is `disabled`, pre-bind the socket on `settings.host:ws_port`.
  - On `OSError`, emit `ws.bind_failed` and report 0 (`test_bc_24_ws_bind_failure_reports_zero`
    goes in `tests/test_health.py`).
  - `/health`'s `ws_port` provider returns the bound port.
  - Run the WebSocket server on the same loop as uvicorn, and shut both down on the one signal
    handler.
- [ ] T079 [US4] **Live gate (full)**:
  1. Set the version to `2.0.0.dev4`, and add BC-30–BC-45 to the CHANGELOG.
  2. Start the server with `--cors http://127.0.0.1:8000`.
  3. Run `node tests/live/sillytavern/run.mjs full`, then
     `cd <SillyTavern checkout>/extensions/SillyTavern-BreezeTTS && npm run test:live`.
  4. Ask `sillytavern-agent` about any extension-side failure.
  5. Record the results, including time to first audio versus Phase 0's C++ run and the review
     outcomes for Phase 8, in `research/live-phase4.md`.

**Checkpoint**: the plan's Phase 4 is complete; SillyTavern narrates, cancels, previews and queues
on the new server.

---

## Phase 9: Polish and sign-off

**Purpose**: prove the success criteria, write the documentation, release 2.0.0.

- [ ] T080 [P] Create `tests/live/cpp_examples.py` (SC-001): run every HTTP and WebSocket example
  from `<Breeze-TTS-2.cpp checkout>/docs/{server,voices,websocket}.md` against
  `--url`. Compare status, headers and body or event shape. Every difference must match an entry
  in an `EXPECTED_DIFFERENCES` table keyed by BC id. Exit non-zero on any unexplained difference.
- [ ] T081 [P] Create `tests/test_bc_coverage.py` (SC-002): parse the BC ids from
  `specs/003-cpp-compatible-api/spec.md`, collect the test names under `tests/`, and assert that
  every BC-01 to BC-48 has at least one `test_bc_NN_*` whose docstring states the C++ behavior it
  rejects. Fill any gaps it reveals.
- [ ] T082 Final benchmark (SC-007): `bench_api --api new` (`--warmup 3 --runs 10`), compared with the
  T002 baseline as described in "SC-007 method" below.
  Record it in `research/bench-final.md`. A regression over 10% blocks sign-off.
- [ ] T083 [P] Rewrite `README.md`'s API section (SC-008, FR-038):
  - every endpoint, field, range, default and error code, from the contracts;
  - every launch option;
  - the WebSocket messages and close codes;
  - the Breaking Changes table (BC-01 to BC-48) and Known Differences, linking to the C++ docs;
  - the development command table (Constitution X);
  - removal of old API examples and port 7860 references (`rg -n 7860` must come back clean
    outside `specs/` and the benchmark's `--api old`).
- [ ] T084 Set `breeze_infer/__init__.py` to `__version__ = "2.0.0"`, and finalize `CHANGELOG.md`
  2.0.0 with the date, the full BC list, Known Differences and "old Python API removed". This
  happens before the final live gate (standing rule 8).
- [ ] T085 Rehearse rollback as in quickstart "Rollback": time it (it must take under 5 minutes),
  and record it in `research/live-phase5.md`.
- [ ] T086 **Final live gate (full)**:
  - run `node tests/live/sillytavern/run.mjs full`, the extension's `npm run test:live`, and
    `python -m tests.live.cpp_examples`;
  - run the full `pytest` suite, the `pytest -m gpu` suite and `ruff`;
  - record everything in `research/live-phase5.md`, and tell `sillytavern-agent` that 2.0.0 is
    live;
  - propose merging `perf-and-fixes` to `main` (the Principle IX deviation's expiry in plan.md).
    Merge only after the user confirms.

---

## SC-007 method (decided 2026-09-24, after T002)

T002 showed that 3 runs with one warm-up leave cold runs in the median (medium_design: 223 ms over
3 runs against a steady 75 ms over 10). So every benchmark in this feature (T002, T043, T054, T082)
runs `--warmup 3 --runs 10`, and compares its 10-run medians with the T002 10-run medians.

- **Gating cases:** `short_design`, `medium_design`, `short_inline` and `medium_inline`. All four
  are one piece on both APIs (text within `split_chars` stays one piece), so they compare like for
  like. `medium_design` was first left out on the mistaken belief that it splits in two; the user
  moved it into the gate on 2026-09-24.
- **Reported only:** `short_voice`, which has no baseline.
- **Context, not a gate:** TTFA min and p25, which show outliers that move a median.
- **Conditions:** nothing else runs while a benchmark runs (no test suites or builds; the first
  T002 recording ran alongside agents' test suites, and its RTF was about 17% worse). Record the
  per-run values and the GPU clock/power state.

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)** comes first. T002 must run before any code change, and T011 before T022
  removes the old API.
- **Foundational (Phase 2)** depends on T003 (dependencies) and blocks every story.
- **US5 (Phase 3)** depends on Phase 2. It comes first among the stories because every live gate
  runs in a browser.
- **US1 speech (Phase 4)** depends on Phase 3. **US2 (Phase 5)** and **US3 (Phase 6)** depend on
  Phase 4; each touches `http_fields.py` or `synthesis.py` and `routes_speech.py`, so run them in
  sequence (US2, then US3).
- **US1 voices (Phase 7)** depends on Phase 6, because `routes_speech.py` gains the voice path.
- **US4 (Phase 8)** depends on Phase 7, since SillyTavern only synthesizes by `voice_id`. T070
  gates the rest of Phase 8.
- **Polish (Phase 9)** depends on everything.

### Within each phase

1. Tests first; confirm they fail.
2. Then pure modules, then I/O modules, then routes and wiring.
3. Each commit then gets the two-pass review loop.
4. The live gate is the last task in its phase.

## Parallel opportunities

- **Phase 1:** T004 and T006–T009 (separate files). T001 runs after T003, because the ported
  benchmark may import `httpx`.
- **Phase 2:** T012, T013, T014, T015, T016 and T024/T025 in parallel. T017 and T018 share
  `gpu.py`, so they run in sequence. T019 and T020 run in sequence.
- **Phase 4:** tests T030–T033 in parallel; T035 and T036 in parallel with T034.
- **Phase 5:** T044, T045 and T046 in parallel.
- **Phase 7:** T055–T060 in parallel; T061 and T064 in parallel.
- **Phase 8:** T071–T074 in parallel once T070 passes.
- **Phase 9:** T080, T081 and T083 in parallel.

Example: launching the Phase 2 pure modules together.

```text
Agent (Sonnet): T012 limits.py
Agent (Sonnet): T013 events.py and tests
Agent (Sonnet): T014 errors.py and tests
Agent (Sonnet): T015 body_limit.py and tests
Agent (Sonnet): T016 settings.py and tests
```

Each lands as its own commit, reviewed by the main session, then run through the `review-agent`
two-pass loop.

## Implementation Strategy

- **MVP for HTTP clients of the C++ contract**: Phases 1–4 (T001–T043). Streamed speech with a
  strict-enough envelope.
- **MVP for SillyTavern** (the primary real client): through Phase 8, since it synthesizes only
  over the WebSocket with `voice_id`.
- **Incremental delivery**:
  - each checkpoint is a working, live-tested server;
  - each phase's development version and CHANGELOG entry come before its live gate;
  - rollback at any point means restarting the previous commit, or the C++ server.
