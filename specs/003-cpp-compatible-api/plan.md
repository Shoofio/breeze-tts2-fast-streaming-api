# Implementation Plan: C++-Compatible API (Fixed)

**Branch**: `perf-and-fixes` | **Date**: 2026-09-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-cpp-compatible-api/spec.md`

## Summary

Replace the Python server's API with the C++ `breeze-server` contract: `/health`,
`/v1/audio/speech`, the `/v1/voices` routes, and a WebSocket session on `ws_port`. The known C++
defects are fixed rather than copied; the 48 breaking changes are listed in the spec as BC-01 to
BC-48.

The approach:
- Keep FastAPI on uvicorn for HTTP, with pure-ASGI middleware for CORS, body limits and the error
  envelope.
- Add a native `websockets` server on the same event loop for the session protocol.
- Run all CUDA work on one GPU thread, behind an asyncio gate shared by HTTP (try, then `409`) and
  WebSocket (`queued` sent just before waiting).
- Use one pure text segmenter for both interfaces, and a small versioned voice-file store.
- Port the earlier `api-alignment` work where it survives revalidation (research R10) and rewrite
  the rest.

Delivery happens in five phases. Every commit gets two `review-agent` passes. Every phase ends with
a live test through the SillyTavern TTS UI, the primary real client.

## Technical Context

**Language/Version**: Python 3.12.

**Primary Dependencies**:
- Existing: FastAPI 0.141.1, Starlette 1.6.0, uvicorn 0.52.4 (h11), python-multipart 0.0.32,
  soundfile 0.14 (libsndfile 1.2.2), torch 2.9.1, transformers 4.57.3, qwen-tts 0.1.1.
- **New: `websockets` >= 17.1, < 18** (17.1 tested; 17.x needs Python 3.11+). See [Dependency changes](#dependency-changes).
- The web-stack versions become exact pins, because the abort and disconnect behavior relies on
  their internals (R2, R3).

**Storage**: Files, one `<voices_dir>/<id>.voice.json` (format v1) per saved voice. Unnamed voices
and KV prefixes live in memory.

**Testing**:
- pytest, plus httpx against a real uvicorn server for stream and abort cases, plus the
  `websockets` client for session and protocol tests.
- `gpu`-marked tests with `BREEZE_MODEL`.
- ruff.
- SillyTavern live tests: Node, playwright-core, headless chromium.

**Target Platform**: Linux (WSL2) and native Windows (`scripts/start_breeze.ps1`), with a single
CUDA GPU.

**Project Type**: A single-process web service (HTTP plus a WebSocket listener) over a PyTorch
inference runtime.

**Performance Goals**:
- Time to first audio (TTFA) and real-time factor (RTF) within 10% of the pre-change baseline
  (SC-007).
- A stalled WebSocket client delays others by at most the in-flight piece plus 1 s (SC-006).
- SillyTavern's time limits: first audio within 10 s of `started`.

**Constraints**:
- One generation at a time.
- A static 2,048-token context (BC-47).
- No new processes or services.
- Limits as in [data-model.md](data-model.md#limits-constants).

**Scale/Scope**: Local and LAN clients; at most 16 WebSocket connections; about 15 new or changed
modules; 48 breaking-change tests (one or more per BC id).

## Constitution Check

*GATE: must pass before Phase 0 research and again after Phase 1 design.*

| # | Principle | Pre-research | Post-design | Notes |
|---|---|---|---|---|
| I | Do not distribute | Pass | Pass | Two listeners in **one** process on one event loop. No new service or queue. |
| II | Optimize for deletion | Pass | Pass | Each module has one job (Project Structure) and can be rewritten in a day. Limits are constants, not options. No interface with a single implementation (the `GpuGate` and `GpuThread` classes are concrete). `form_fields.py`, the 422 handler, `X-Breeze-Reference` and `bench_voices` are **not** ported. |
| III | Explicit dependencies | Pass, with a fix | Pass | Today's `api.py` breaks this: module globals `_settings` and `_request_lock`, an `os.environ` read mid-function, `print`. The rewrite builds everything at one composition root (`api.py`) and passes it down, with the clock, sleep, output stream and RNG seed injected. |
| IV | Contract at the boundary | **Deviation** | **Deviation (justified)** | Versioned contracts for HTTP and WebSocket (v2.0.0), with the version on the wire as `X-Breeze-Version` on every HTTP response and WebSocket handshake (FR-037a), and the voice file (`format`/`version`). The boundary modules (`http_fields`, `ws_messages`, `voice_file`) own their types and map to internal ones explicitly. **Deviation**: the old Python API is removed without a new version or a migration path, at the user's explicit request. See Complexity Tracking. |
| V | Test the transformation | **Deviation** | **Deviation (justified)** | Pure units: `http_fields`, `text_split`, `ws_messages`, `ws_session`, CORS policy, `voice_file`, `GpuGate`, the ramp. Boundaries run for real: uvicorn, `websockets`, soundfile, the filesystem. Every BC has a test that fails against the defective behavior (SC-002). **Deviation**: a fake runtime at the GPU edge for API tests without a GPU. See Complexity Tracking. |
| VI | Structured events | Pass | Pass | `events.py` (JSON lines) with `request_id`, `session_id` and `piece_index`. No `print`. Event catalog in data-model.md. |
| VII | Recovery over prevention | Pass | Pass | Rollback means restarting the previous tag or the C++ server. It takes under 5 minutes, has no data migration, and leaves `.breeze` files untouched. It is rehearsed in Phase 5 ([quickstart](quickstart.md#rollback-constitution-vii)). **No feature flag, with a written blast-radius argument** (VII "How To Apply"): the change replaces the API of one self-hosted, single-GPU server whose only known clients are the SillyTavern extension and Liveva, both already written against the C++ contract this change adopts. No shared data is migrated (voice files are new files; `.breeze` files are never touched), so a bad release affects only that server's current clients until it is stopped. The kill switch is operational, not a flag: stop the server and start the previous tag or the C++ `breeze-server` on the same ports. SillyTavern keeps working on either. A flag that kept both APIs in one server would double the surface for clients that don't exist (see the IV deviation). |
| VIII | Attention is finite | N/A | N/A | No alerts or dashboards are added. |
| IX | Value at the user | **Deviation** | **Deviation (justified)** | Each phase ends with a live SillyTavern run, and instrumentation (events) ships with each change. **Deviation**: the work accumulates on the `perf-and-fixes` branch, not `main`, as the user directed. See Complexity Tracking. |
| X | Discoverable commands | Pass | Pass | The README development section lists the run, test, GPU-test, lint, benchmark and live-test commands ([quickstart Commands](quickstart.md#commands)). |

Result: **PASS**, with the three deviations (IV, V, IX) recorded under Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/003-cpp-compatible-api/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/http-api.md, contracts/ws-api.md
├── checklists/requirements.md
├── research/baseline-<date>.md      # Phase 0 output
├── research/live-<phase>.md         # SillyTavern live-test record, one per phase
└── tasks.md                         # /speckit-tasks
```

### Source Code (repository root)

```text
breeze_infer/
├── __init__.py          # __version__
├── api.py               # composition root only: settings → components → HTTP + WS on one loop
├── settings.py          # frozen Settings from argparse (FR-003); CORS list parsing
├── limits.py            # constants (data-model Limits)
├── events.py            # structured JSON events; clock and stream injected   [port A:6fb3736]
├── errors.py            # ApiError(status, code, message); envelope handlers  [port A:45295d2+]
├── cors.py              # pure-ASGI CORS and origin guard; pure policy functions   [new, R8]
├── body_limit.py        # pure-ASGI 26 MiB limit → 413                              [new, R6]
├── http_fields.py       # strict form/query parsing → SpeechRequest/VoiceRequest    [new]
├── reference_audio.py   # bounds-safe decode and limits (soundfile)                  [new, R9]
├── audio.py             # encode_prompt_waveform, codec_fingerprint, pcm16    [port A:6fb3736, fixed]
├── templates.py         # prompts; codes path; prefix/suffix split            [extend, port A:6fb3736]
├── text_split.py        # the one segmenter (R11)                             [port A:72d9bdd+, fixed]
├── synthesis.py         # Reference, pieces, room, seeds, anchor, ramp        [port ideas A:59a3528]
├── gpu.py               # GpuGate (asyncio) + GpuThread (single CUDA thread)  [new, R14]
├── streaming.py         # speech response: prime, abort, send timeout, cleanup   [new, R2/R3]
├── routes_health.py     # GET/HEAD /health
├── routes_speech.py     # POST /v1/audio/speech
├── routes_voices.py     # /v1/voices routes
├── voice_file.py        # v1 record encode/decode and validation (pure)       [new, R13]
├── voice_store.py       # directory I/O: scan, create, remove-with-retry      [port parts of A:voices.py]
├── voice_registry.py    # in-memory voices, name rules, cap, order            [new]
├── voice_prefix.py      # byte-bounded LRU of KV prefixes, key (id, content)  [port A:6fb3736, re-keyed]
├── ws_messages.py       # strict client-message schema (pure)                 [new]
├── ws_session.py        # pure session state machine (R15)                    [new]
├── ws_server.py         # websockets server: handshake, reader, worker, outbox   [new, R4]
├── bench_api.py         # TTFA/RTF benchmark                                  [port A]
├── runtime.py           # unchanged, except device selection on the GPU thread
└── compile_cache.py     # unchanged
models/
├── fast_streaming.py    # per-request overrides, _prefill_plan tuple, max_new_tokens_room,
│                        # relaxed prefix guard, sampling warmup               [port A:86647b8/220b3ca]
└── cudagraph/sampling.py  # docstring: repetition-penalty semantics
tests/
├── fakes.py             # FakeRuntime at the GPU edge (Complexity Tracking)   [port A, extended]
├── test_*.py            # units and real-server integration, one file per module; bc_ ids in names
├── test_malformed_corpus.py, test_speech_abort.py, test_ws_isolation.py
├── cpp_golden/          # C++ segmenter oracle (the differences table is in test_text_split.py)   [port A]
├── gpu/                 # long text, cfg values, overrides, voice equivalence, prefix buckets
└── live/
    ├── cpp_examples.py  # SC-001 runner over the C++ docs examples
    └── sillytavern/     # Playwright harness: config.mjs, run.mjs, phases/{health,voices,speech,full}.mjs
scripts/start_breeze.{sh,ps1}   # port 8080, --cors, ws port
docker/run.sh                   # publish 8080 and 8081
CHANGELOG.md, README.md
```

**Removed**: the old `/v1/audio/speech` form handler and globals in `api.py`, and
`tests/test_api.py` (the old contract).

**Structure Decision**: Keep the existing single-package layout (`breeze_infer/`, `models/`,
`tests/`). `api.py` keeps its module name, so the launch scripts and Docker keep working, but it
becomes the composition root only.

## Dependency changes

| Change | Why | Alternatives considered |
|---|---|---|
| **Add `websockets` >= 17.1, < 18** | The venv has no WebSocket library, and without one uvicorn answers upgrades with `404`. Its native server gives four things we need: RFC-conformant close codes, a handshake timeout, a `process_request` hook (Origin `403`, `503` while loading or at the connection cap), and aborting a stalled peer when `close_timeout` expires (R4). | `wsproto` drops the connection on protocol errors without a close frame. uvicorn with `websockets-sansio` has no way to evict a stalled peer without `TCP_USER_TIMEOUT`, plus signal and `sys.exit` workarounds. A hand-rolled implementation is where C++'s defects came from. |
| Pin `fastapi`, `starlette`, `uvicorn`, `python-multipart` exactly | The abort-on-failure and disconnect behavior depends on library internals (R2, R3). A pin makes an upgrade a deliberate change, re-verified by `test_speech_abort.py`. | Floating `>=` pins (a silent regression risk). |
| Add `httpx` as a test dependency | Real-server tests assert `RemoteProtocolError` on an incomplete chunked transfer. It is installed today only as a transitive dependency. | TestClient (can't observe an incomplete transfer). |

## Delivery process (applies to every phase)

### Per-commit loop

1. **Implement** the task in a small, focused commit. Delegation follows the global guidance:
   - **Sonnet** for mechanical work: porting, test tables, scripts, docs.
   - **Opus** for the hard parts: `gpu.py`, `streaming.py`, `ws_session.py`, `ws_server.py`, and
     the runtime changes in `fast_streaming.py`.
   - The main session reviews the agent's output before it is committed.
2. **Verify locally**: `.venv/bin/ruff check .` and `.venv/bin/pytest`, plus
   `BREEZE_MODEL=… .venv/bin/pytest -m gpu` whenever the runtime, synthesis or speech path changed.
   Show the real output.
3. **Commit.**
4. **Review pass 1**: send the commit SHA and its spec and contract references to the peer session
   `review-agent` (check `ListAgents` first). Triage its findings, reject invalid ones with a
   stated reason, and fix the valid ones in a follow-up commit, with a failing-first test for any
   bug (Constitution V).
5. **Review pass 2**: send the fix commit to `review-agent`. Address anything still outstanding,
   then move on. **There is no third pass.** Anything still disputed is recorded in the phase's
   live record for the user.
6. **Debugging rule**: if a fix attempt fails, don't try a similar variation. Stop, explain why it
   failed, list at least 3 fundamentally different approaches, and propose the best one before
   changing anything.

### Per-phase live gate

1. **Bump the version first.** Before the live run, set the development version
   (`2.0.0.devN`, where N is the phase number) and append the phase's BC entries to the
   `CHANGELOG.md` "Unreleased" section. A re-deploy within the same phase adds a local label
   (`2.0.0.devN+1`, `+2`, ...), so `X-Breeze-Version` still names the phase and each deploy
   is unique. New entries always go under whichever named subsection they belong to --
   BC items under "Breaking changes", fixes under "Fixed" -- never appended wherever the file
   currently ends, since reordering the file doesn't make that unambiguous.
2. **Start the server** (`scripts/start_breeze.sh --cors http://127.0.0.1:8000`), after stopping
   the C++ server.
3. **Run the SillyTavern live test**: `node tests/live/sillytavern/run.mjs <phase>` against
   `http://127.0.0.1:8000/`. It uses only the "Breeze validation" chat under Seraphina, never
   deletes `eric` or `vale`, and uses throwaway voice names. Capture the `console.debug('breeze',
   …)` events and any toasts.
4. **Ask `sillytavern-agent`** when an extension behavior is unclear or a failure could be on the
   extension's side. Coordinate any extension change through it; it needs its own user's approval.
5. **Record the results** in `research/live-<phase>.md`: server version, commands, events,
   timings, and pass or fail per scenario. A phase is done only when its gate passes.

## Implementation Phases

Each phase is independently testable and ends with the live gate. The user stories and BC ids
covered are listed per phase.

### Phase 0: Baseline and live harness

- Record the performance baseline on the **current** API (SC-007), following
  [quickstart Scenario 0](quickstart.md#scenario-0-baseline-before-any-code-change).
- Build `tests/live/sillytavern/`: config, a runner, and the `health`, `voices`, `speech` and
  `full` phases, adapted from the extension's `us*-browser-validation.mjs`.
- Run `full` against the **C++ server** to capture reference behavior.
- Port `bench_api.py` so it can drive both the old and new API.
- **Live gate**: the harness runs green against the C++ server. Recorded in `live-phase0.md`.

### Phase 1: Server foundation (US1-1, US5; BC-18 to BC-24, FR-001–003, FR-023/024, FR-034–036)

- Add `settings.py`, `limits.py`, `events.py`, `errors.py`, `cors.py` and `body_limit.py`.
- Add `gpu.py` (`GpuGate`, and `GpuThread` with background model load and warmup).
- Rewrite `api.py` as the composition root, with a pre-bound HTTP socket and `TCP_USER_TIMEOUT`
  where available.
- `/health` serves `503` while loading; unknown routes get `404`/`405`.
- Remove the old speech route and `tests/test_api.py`.
- Update the launch scripts and Docker ports (8080/8081), and add `--cors`.
- Set the version to `2.0.0.dev1` and start `CHANGELOG.md`.
- **Live gate (`health`)**: SillyTavern Refresh shows "TTS Provider Loaded", with no CORS errors,
  and the extension's `/health` deep-equality check holds.

### Phase 2: Speech over HTTP (US1-2, US1-5, US2, US3; BC-01 to BC-17, BC-46, BC-47)

**Runtime** (`models/fast_streaming.py`, R12):
- per-request overrides;
- the `_prefill_plan` tuple;
- `max_new_tokens_room`;
- the relaxed prefix guard;
- sampling warmup;
- GPU tests for cfg 2.5, 7.5 and 0, and for the overrides.

**Request handling:**
- `templates.py` codes path and `audio.encode_prompt_waveform`, so each reference is encoded once
  per request.
- `reference_audio.py`, `http_fields.py`, `text_split.py` (with the golden oracle and the table of
  differences), `synthesis.py`, `streaming.py` and `routes_speech.py`.
- The malformed corpus (SC-003), the abort test (BC-17), and the long-text GPU test (SC-004).

**Benchmark** against the Phase 0 baseline; SC-007 is checked early here.

**Live gate (`speech`)**: a browser `fetch` of `/v1/audio/speech` from the SillyTavern origin.
The expose-headers are readable and the PCM decodes. The `health` gate is re-run.

### Phase 3: Voices (US1-3, US1-4; BC-25 to BC-29, BC-48, FR-017–022)

- Add `voice_file.py`, `voice_store.py`, `voice_registry.py`, `voice_prefix.py` and
  `routes_voices.py`, and connect `voice_id` to speech.
- Add the prefix-cache GPU tests and the voice-equivalence GPU tests.
- Register `eric` and `vale` from `$REFERENCE_VOICES_DIR`.
- **Coordination**: confirm with `sillytavern-agent` that the extension's DELETE-then-POST replace
  flow and the new delete wording have landed. If they haven't, the replace step is recorded as
  "blocked on extension" rather than failed.
- **Live gate (`voices`)**: upload, list, replace and delete of a throwaway voice through the
  SillyTavern UI. Voice preview is deferred to Phase 4, since previews synthesize over the
  WebSocket.

### Phase 4: WebSocket sessions (US4; BC-30 to BC-45, FR-025–033)

**Prototype first**: check R4's assumptions, namely that `close_timeout` aborts a stalled peer and
that `process_request` can return a JSON `403`/`503`. If either fails, stop and re-plan (two-strike
rule).

**Build**:
- `ws_messages.py`, `ws_session.py` (pure, with SC-005's 1,000 randomized sequences) and
  `ws_server.py` (outbox, 1008 eviction, connection cap, Origin, loading);
- the WebSocket wired into `api.py`, with `/health` reporting the real `ws_port` or 0 when the
  bind fails;
- the isolation test (SC-006).

**Live gate (`full`)**:
- narrate, stop/cancel, voice preview, queued back-to-back narrations, and `cfg_scale` 1, 4 and
  7.5;
- the extension's `npm run test:live`.

### Phase 5: Sign-off (SC-001, SC-002, SC-007, SC-008; FR-037/038)

- `tests/live/cpp_examples.py` (SC-001) and the BC coverage audit: one test per BC id that fails
  against the defect (SC-002).
- The final benchmark (SC-007).
- The README with the full API reference and the breaking-changes list (SC-008).
- Set the version to `2.0.0` and finalize `CHANGELOG.md`.
- Rehearse rollback.
- **Live gate (`full`)**: a full regression on 2.0.0, plus the extension's live suite.

## Risks (ranked)

1. **Context budget (BC-47).** Long single pieces, 30 s references with long transcripts, or dense
   CJK can hit the 2,048-token context where C++ doesn't. Mitigations: FR-036a, a
   `speech.piece_clamped` event, and the default `split_chars` of 600. Raising `max_seq_len` is
   deferred until the events show a real need.
2. **New WebSocket stack.** The Phase 4 prototype checks R4 before building on it.
3. **Ordering in the WebSocket session (SC-005).** `ws_session` is kept pure and tested with
   randomized sequences.
4. **FR-007 order versus checks that need the codec.** "Too short", and the first piece's room,
   use a *predicted* frame count before the gate is taken. The prediction is asserted equal to the
   actual count, with an event on a mismatch and a GPU test covering it.
5. **Library internals.** Abort and disconnect behavior relies on uvicorn and Starlette internals.
   Mitigated by exact pins and a real-server test.
6. **Performance (SC-007).** One executor hop per chunk, plus parsing overhead, could cost TTFA.
   Measured in Phase 2, not assumed.
7. **Windows.** `TCP_USER_TIMEOUT` is Linux-only; on Windows, stalled HTTP streams rely on the
   30 s send timeout. The NTFS rename-retry and case-insensitivity behavior needs one real run on
   `/mnt/m`.
8. **SillyTavern extension coordination.** The replace flow depends on the extension change
   (spec Clarifications). Until it lands, only that step is recorded as blocked.
9. **Repetition-penalty semantics** differ from C++ (spec Known Differences). They affect the
   audio, not the contract.

## Complexity Tracking

| Deviation | Why needed | Simpler or compliant alternative rejected because | Expiry |
|---|---|---|---|
| **Principle IV**: the old Python API is removed with no new version or migration path | The user explicitly requires the C++ contract as the new default with no backwards compatibility (spec Input). The known clients (SillyTavern, Liveva) already target the C++ contract. | Serving both APIs side by side would double the surface and tests for clients that don't exist. | Once 2.0.0 ships. Any later breaking change needs a new contract version. |
| **Principle V**: API, streaming and WebSocket tests use `tests/fakes.py` `FakeRuntime` in place of the real model (re-recorded from 001's deviation, `api-alignment:specs/001-saved-voice-references/plan.md:167-176`) | The checkpoint needs a CUDA GPU and about 8 GiB. The fake stands in at the GPU edge only: every module this repo owns runs for real (uvicorn, `websockets`, soundfile, the filesystem, the session machine, the segmenter). The real path is covered by the `gpu` suite and the live SillyTavern gates, which must pass before each phase closes. | Running everything on the GPU makes the no-GPU suite impossible and puts SC-005's 1,000 sequences out of reach. | When a CPU-loadable miniature Breeze checkpoint fixture exists under `tests/fixtures/`. The fakes are then deleted and the tests switched to the real runtime. |
| **Principle IX**: the feature is built on the long-lived branch `perf-and-fixes` instead of integrating to `main` behind flags | The user directed that this work build on `perf-and-fixes`, which also carries unmerged performance work (reference-prefix KV reuse, the compile cache, the eager fallback) that this feature depends on. Each phase is still deployed and live-tested on the real server, so value reaches the one real client per phase. | Integrating each phase to `main` behind a flag would mean merging the unreviewed-on-`main` performance commits first, and a flag that switches between two APIs is rejected under VII/IV above. | When 2.0.0 ships (T086): `perf-and-fixes` is merged to `main`, and later work integrates to `main` directly. |
