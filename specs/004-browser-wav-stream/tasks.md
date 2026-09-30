---

description: "Task list for browser-playable WAV streaming"
---

# Tasks: Browser-Playable WAV Streaming

**Input**: Design documents from `specs/004-browser-wav-stream/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/http-wav-stream.md](contracts/http-wav-stream.md),
[quickstart.md](quickstart.md)

**Tests**: A small set only, as the plan says. The user asked to keep this tight and not over-index
on unit or corner-case tests:
- about 6 automated tests, listed in the plan's Project Structure;
- browser behaviour and long texts are checked at the live gate.

Don't add tests beyond those named in a task.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (a different file, and no dependency on an incomplete task).
- **[Story]**: US1–US5 from spec.md.
- Paths are relative to the repo root `<repo>`.

## Standing rules (apply to every task)

1. **Branch**: work on `004-browser-wav-stream`. Before starting a task, read the research decision
   (Rn) it cites.
2. **Delegation**:
   - **Sonnet** for tasks without a model tag; **Opus** for tasks tagged *(Opus)*.
   - The main session reviews every agent's output before committing.
   - Agent briefs forbid `git stash`, `git checkout` and `git reset`, because agents share one
     working tree. Stage by path.
3. **Verification**: every task ends with `.venv/bin/ruff check .` and `.venv/bin/pytest` passing.
   Add `BREEZE_MODEL=<model> .venv/bin/pytest -m gpu` where a task says so. Without
   `BREEZE_MODEL`, every GPU test skips and still exits 0, which proves nothing. Show the real output.
4. **Commits**: one small commit per task. The message cites FR ids.
5. **Review loop after each commit**: two `review-agent` passes, then fix the valid findings. Run
   only one agent at a time, and no review while a subagent is still working.
6. **Two-strike rule**: when a fix attempt fails, stop. Explain why, list at least 3 different
   approaches, and propose one.
7. **The POST route stays unchanged** (FR-017). Every existing test must keep passing unmodified. If
   one needs changing, stop and ask.

---

## Phase 1: Setup

- [ ] T001 Bump `__version__` to `2.1.0.dev1` in `breeze_infer/__init__.py`. Add an
  `## Unreleased` section with an `### Added` subsection in `CHANGELOG.md`, containing one line for
  `GET /v1/audio/speech.wav`. This task and T002 go in one commit.

---

## Phase 2: Foundational (blocks every story)

- [ ] T002 In `breeze_infer/streaming.py`, give `SpeechResponse.__init__` two new keyword
  parameters (research R4). Neither changes anything when left at its default.
  - **`media_type: str = "audio/pcm"`**: used for the contract `content-type` instead of the class
    attribute.
  - **`event_fields: Mapping[str, str] | None = None`**: merged into the fields of the one
    outcome event (`speech.completed`/`aborted`/`failed`), by passing them through `_Outcome`.

  No new tests; the existing ones cover the defaults.
- [ ] T003 [P] Add three constants to `breeze_infer/limits.py`, each with a one-line comment saying
  why (spec FR-004, FR-006, FR-016):
  - `WAV_GPU_WAIT_SECONDS = 60`
  - `WAV_SEND_TIMEOUT_SECONDS = 600`
  - `MAX_REQUEST_LINE_BYTES = 128 * 1024`

**Checkpoint**: `pytest` is green, with no behaviour change.

---

## Phase 3: User Story 1 - Gapless streamed playback (Priority: P1) 🎯 MVP

**Goal**: `GET /v1/audio/speech.wav` streams a progressive WAV of the same synthesis as the POST
route.

**Independent test**: GET the route with the fake runtime. Check that the first 44 bytes are the
contract's header, the headers are right, and the PCM after the header equals the POST route's
body.

- [ ] T004 [US1] Add a pure function `wav_header(sample_rate: int) -> bytes` to
  `breeze_infer/routes_speech.py` (research R5).
  - It returns the 44-byte layout in [contracts/http-wav-stream.md](contracts/http-wav-stream.md),
    built with `struct.pack("<4sI4s4sIHHIIHH4sI", ...)`.
  - Test it byte for byte against a hand-written expected value at 24,000 Hz, in the new file
    `tests/test_speech_wav.py`.
- [ ] T005 *(Opus)* [US1] Serve the route by reusing `_serve_speech` in
  `breeze_infer/routes_speech.py` (research R1). Pieces:
  - **New parameter**: `_serve_speech` gains the keyword `wav: bool = False`. When true, it returns
    `SpeechResponse` with:
    - `first_chunk=wav_header(sample_rate) + first_chunk`
    - `media_type="audio/wav"`
    - `headers={"Accept-Ranges": "none"}`
    - `event_fields={"format": "wav"}`
  - **Route**: `install_speech` also registers `@app.get("/v1/audio/speech.wav")`, with the same
    `require_ready` dependency. It calls `_serve_speech(..., wav=True)`.
  - **At this point** the GET still uses `try_acquire` and `409`. US3 changes that.
  - **Checks**:
    - `Range` needs no code; the test below proves it's ignored.
    - Confirm `cors.route_methods_for_path` reports `GET` for the new path (preflight). Change
      nothing there unless it doesn't.
  - **Tests**: add to `tests/test_speech_wav.py`, using the `TestClient` and fake-runtime setup from
    `tests/test_routes_speech.py:73-124`. Reuse its helpers by importing them, not copying them:
    1. A GET with `text` and `seed`, sending `Range: bytes=0-`, returns:
       - `200`, `audio/wav`, `Accept-Ranges: none`, `Cache-Control: no-store` and `X-Sample-Rate`;
       - a body that is `wav_header(24000)` followed by exactly the bytes a POST with the same
         fields returns.

       The fake runtime is deterministic.
- [ ] T006 [P] [US1] Add `tests/gpu/test_speech_wav.py` with one `@pytest.mark.gpu` test. On the
  session's warmed runtime (`tests/gpu/conftest.py`), GET a short text with `seed=7` and POST the
  same fields. Check:
  - the header is valid;
  - the PCM lengths are within the frame tolerance used for piece 0 in
    `tests/gpu/test_speech_long_text.py` (GPU output is not bit-reproducible; see the comment at
    line 281).

  Run `BREEZE_MODEL=<model> .venv/bin/pytest -m gpu tests/gpu/test_speech_wav.py`.

**Checkpoint**: the MVP route works end to end with `curl` (quickstart §1).

---

## Phase 4: User Story 2 - Slow or paused readers never hold the GPU (Priority: P1)

**Goal**: generation runs at full speed into a buffer, and the GPU is released at the end of
generation. Delivery follows the client's pace, with a 600 s send timeout and no minimum rate.

**Independent test**: with real uvicorn, a client that reads nothing still sees the gate freed and
`speech.generated` emitted. It then reads the whole body.

- [ ] T007 *(Opus)* [US2] Add the tests first, in the new file `tests/test_speech_wav_stream.py`,
  using the real uvicorn and httpx harness pattern of `tests/test_speech_abort.py`. Serve the real
  app built as in `tests/test_routes_speech.py`, with a fake runtime whose generation yields enough
  chunks to outrun the socket buffers. Two tests:
  1. **Buffered delivery**: open a GET and read only the headers.
     - Wait until `speech.generated` has been emitted. Then check the gate is free: run
       `gate.try_acquire()` and release it inside one coroutine on the server's loop, via
       `LiveServer`'s `run_coroutine_threadsafe` helper (`tests/test_speech_abort.py:206`).
       `GpuGate` is loop-bound (gpu.py:80), so never touch it from the test thread while the
       server is running.
     - Then read the rest of the body. Its length equals header + all generated PCM, and
       `speech.completed format=wav` is emitted.
  2. **Disconnect**: close the connection while generation is running (the fake generation is gated
     on an event). Check that the gate frees within one chunk and `speech.aborted
     reason=client_disconnect` is emitted.

  Confirm test 1 fails before T008.
- [ ] T008 *(Opus)* [US2] Implement buffered delivery in `breeze_infer/routes_speech.py`
  (research R4).
  - **The generator**: an async generator `_buffered(session, events, request_id)`. On its first
    iteration it starts a producer task that loops `session.step()` into an unbounded
    `asyncio.Queue`.
    - At `DONE`, it runs `await session.aclose()`, suppressing exceptions: the response's final
      close reports them. Then it emits `speech.generated` with `audio_seconds`, and puts an
      end-of-stream marker on the queue.
    - On an exception from `step()`, it puts the exception on the queue.
    - The generator yields chunks and re-raises a queued exception. Its `finally` cancels the
      producer and awaits it.
  - **Wiring**: when `wav=True`, `_serve_speech` passes `body=_buffered(...)`,
    `send_timeout=WAV_SEND_TIMEOUT_SECONDS` and `min_rate_grace=math.inf`.
  - **Comments**: explain *why* the GPU is released before the response ends, and why a second
    `aclose()` from `SpeechResponse` is safe.

**Checkpoint**: T007's tests pass, and so does the whole suite, unmodified.

---

## Phase 5: User Story 3 - Wait for a busy GPU instead of failing (Priority: P2)

**Goal**: the GET waits for the GPU in FIFO order for up to 60 s, then answers `503 busy_timeout`,
and it leaves the queue on disconnect.

**Independent test**: hold the gate and GET with a short wait. With the gate held throughout, the
result is `503 busy_timeout`. With the gate released during the wait, the result is `200`.

- [ ] T009 *(Opus)* [US3] Implement `_wait_for_gpu(gate, http_request, timeout)` in
  `breeze_infer/routes_speech.py` (research R3):
  - **The race**: race `gate.acquire()` inside `asyncio.timeout(timeout)` against a task that awaits
    `http_request.receive()` until `http.disconnect`. Cancel the loser, and await it.
  - **Timeout**: emit `speech.queued_timeout` with `waited_s`, then raise
    `ApiError(503, "busy_timeout", "the GPU stayed busy")`.
  - **Disconnect**: emit `speech.aborted reason=client_disconnect queued=true`, then end the request
    without a response body. Pick the ending that doesn't produce a `request.failed` event or a
    traceback, and justify it in a comment.
  - **Wiring**:
    - `_serve_speech` gets `gpu_wait: float | None = None`. `None` keeps `try_acquire`/`409`
      exactly as today.
    - `install_speech` gains `wav_gpu_wait: float = WAV_GPU_WAIT_SECONDS`, which the GET route
      passes in.
    - A poisoned gate still raises `GpuUnavailable` (`503 gpu_unavailable`).
- [ ] T010 [US3] Add two tests:
  - **(a)** In `tests/test_speech_wav.py`: install the app with `wav_gpu_wait=0.2`, hold the gate
    with `try_acquire()` from the test thread, and check that a GET returns `503` with code
    `busy_timeout`. This is safe across threads because the waiter times out and leaves the queue
    before the test releases the gate.
  - **(b)** In `tests/test_speech_wav_stream.py`, the real-uvicorn harness: acquire the gate on the
    server's loop (`LiveServer`'s `run_coroutine_threadsafe`), start a GET, then release the gate on
    that same loop. The GET returns `200` with a WAV body. `GpuGate` is loop-bound (gpu.py:80), so
    releasing from another thread would resolve its waiter's future from the wrong thread.

  Don't write a disconnect-while-queued test; the live gate covers it (quickstart §3).

---

## Phase 6: User Story 4 - Long and non-ASCII text works in a URL (Priority: P2)

**Goal**: request lines of up to 128 KiB are accepted.

**Independent test**: over real uvicorn, a GET whose URL carries about 90 KB of percent-encoded
Chinese reaches the route.

- [ ] T011 [US4] In `breeze_infer/api.py`'s `uvicorn.Config(...)`, add
  `h11_max_incomplete_event_size=MAX_REQUEST_LINE_BYTES`, with a comment giving the reason: 10,000
  CJK characters come to about 90 KB percent-encoded, and h11's default is 16 KiB.
  - In `tests/test_speech_wav_stream.py`, pass the same constant in the test server's own
    `uvicorn.Config`.
  - Add one test: a GET with `text` of 10,000 CJK characters with a fake runtime gets `200`.
    Assert only the status and the WAV header.

---

## Phase 7: User Story 5 - Detection and input errors (Priority: P3)

**Goal**: clients detect the feature by version, and see the same errors as on the POST route.

**Independent test**: the same invalid fields get the same status and body on both routes.

- [ ] T012 [US5] Add one parametrized test to `tests/test_speech_wav.py`. For four inputs, the GET
  and the POST (same fields as a form) return an identical status and JSON body. The four inputs are:
  - unknown `voice_id`
  - empty `text`
  - `seed=-1`
  - `ref_audio` in the query

  Version detection needs no task: `X-Breeze-Version` already carries `__version__` (T001).

---

## Phase 8: Polish and release

- [ ] T013 [P] Update `README.md`:
  - **API section**: a `GET /v1/audio/speech.wav` subsection covering fields, headers, the WAV
    header, queueing (60 s, then `503 busy_timeout`), buffered delivery and the 600 s send timeout.
    Link the contract addendum.
  - **Security**: a note next to "no authentication": any web page can trigger synthesis through
    this route (spec FR-020).
  - **Kept behaviours**: the list says "no WAV"; amend it to note this route is the exception.
- [ ] T014 [P] Finish the `CHANGELOG.md` Unreleased entries:
  - the route
  - the 128 KiB request-line limit (it affects all routes)
  - the new error code `busy_timeout`
  - the new events `speech.generated` and `speech.queued_timeout`
- [ ] T015 Run the GPU suite (`BREEZE_MODEL=... .venv/bin/pytest -m gpu`), and record the result.
- [ ] T016 Live gate:
  1. Bump to `2.1.0.devN` first.
  2. Start the server as in the quickstart.
  3. Run [quickstart.md](quickstart.md) §1–5 in Chrome and Firefox. SillyTavern safety rules apply:
     only the "Breeze validation" chat, and never delete `eric` or `vale`.
  4. Record every check, the bench numbers and anything a browser did unexpectedly (re-requests,
     `0xFFFFFFFF` handling) in `specs/004-browser-wav-stream/research/live-004.md`.

  The user drives the browser steps that automation can't.
- [ ] T017 Release, with explicit user confirmation before touching `main` or the tag:
  1. Set the version to `2.1.0`, and date the CHANGELOG section.
  2. Merge to `main` and tag `v2.1.0`. Nothing is pushed unless the user asks.
  3. Restart the live server on 2.1.0, and tell `st-agent` that 2.1.0 is live.

---

## Dependencies and execution order

```text
T001 ─► T002, T003 ─► T004 ─► T005 ─┬─► T006 (GPU, any time after T005)
                                    ├─► T007 ─► T008            (US2)
                                    ├─► T009 ─► T010            (US3)
                                    ├─► T011                    (US4)
                                    └─► T012                    (US5)
T008, T010, T011, T012 ─► T013, T014 ─► T015 ─► T016 ─► T017
```

- **Stories**: US1 blocks the rest, because every later story extends the route T005 creates. After
  that, US2–US5 touch the same `routes_speech.py`. They run **one after another** (US2 → US3 → US4
  → US5), not in parallel, because agents share one working tree.
- **Parallel opportunities**:
  - T003 alongside T002;
  - T006 (GPU test file) alongside US2–US5;
  - T013 and T014 (docs) together.

## Implementation strategy

- **MVP = Phases 1–3 (US1)**. The route streams WAV using the POST route's 409 rule. That is enough
  to try in a browser with `curl` and a short text.
- **Before any live use: US2.** Without it, long messages can be cut at slow playback speeds.
- **Then US3, US4, US5, polish, the live gate and the release**, each in its own commit with the
  review loop.
