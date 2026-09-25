# Quickstart & Validation: C++-Compatible API (Fixed)

**Feature**: [spec.md](spec.md) | **Contracts**: [http-api.md](contracts/http-api.md),
[ws-api.md](contracts/ws-api.md)

This guide proves the feature end to end. Each scenario names the requirement or breaking change it
covers. Exact fields and codes live in the contracts; they are not repeated here.

## Prerequisites

- **Environment:** repo at `<repo>` on `enhanced-api`, with the venv at `.venv`
  and dependencies installed by `uv pip install -r requirements.txt` (`uv sync` doesn't read
  `requirements.txt`: `pyproject.toml` has no `[project]` table).
- **GPU and model:**
  - CUDA GPU with the Breeze checkpoint.
  - `scripts/start_breeze.sh` resolves the model path.
  - `BREEZE_MODEL=<path>` enables the `gpu` tests.
- **Ports:** 8080 and 8081 must be free. **Stop the C++ `breeze-server` first**, since it uses the
  same ports and GPU.
- **SillyTavern:**
  - Running at `http://127.0.0.1:8000/` (Docker container `sillytavern`) with the Breeze TTS
    extension.
  - Playwright: `playwright-core` from the npx cache and chromium from `~/.cache/ms-playwright`.
    Paths are set in `tests/live/sillytavern/config.mjs`.
- **Reference voices:** the samples in `$REFERENCE_VOICES_DIR/{eric,vale}` (WAV plus
  transcript).

## Commands

These are also listed in the README development section (Constitution X).

| Action | Command |
|---|---|
| Run the server (browser clients) | `scripts/start_breeze.sh --cors http://127.0.0.1:8000` |
| Unit and integration tests (no GPU) | `.venv/bin/pytest` |
| GPU tests | `BREEZE_MODEL=<path> .venv/bin/pytest -m gpu` |
| Lint | `.venv/bin/ruff check .` |
| Benchmark | `.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080` (`--warmup 3 --runs 10` by default) |
| SillyTavern live test | `node tests/live/sillytavern/run.mjs <phase>` (`health`, `voices`, `speech`, `full`) |
| C++ docs example check | `.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080` |

## Scenario 0: Baseline (before any code change)

1. **Record performance on the current API:** check out `perf-and-fixes` HEAD, start the current
   API (`python -m breeze_infer.api <model> --port 7860 --fast-all`), and run `bench_api` in
   old-API mode. Save the medians to `research/baseline-<date>.md` (SC-007).
2. **Record SillyTavern against the C++ server:** start the C++ `breeze-server --cors` and run
   `node tests/live/sillytavern/run.mjs full`. The events and timings go into the baseline file.
   This is the reference behavior that the new server must match or improve on.

## Scenario 1: Health, CORS, errors (spec US1-1, US5; BC-18 to BC-24)

1. **Loading:** start the server and immediately `curl -i :8080/health`. Expect
   `503 {"status":"loading",...}` while loading, then exactly `{"status":"ok","sample_rate":24000,
   "ws_port":8081}`.
2. **Unknown route:** `curl -i :8080/nope` returns `404` with a JSON envelope.
   `curl -i -X PUT :8080/v1/voices` returns `405` with an `Allow` header.
3. **CORS preflight:** with `--cors http://127.0.0.1:8000`, send a preflight
   `OPTIONS /v1/voices/x` with `Origin: http://127.0.0.1:8000` and
   `Access-Control-Request-Method: DELETE`. Expect `204` and allow-methods `DELETE, OPTIONS`.
4. **Disallowed origin:** a `POST /v1/voices` with `Origin: http://evil.test` returns `403`, and
   no file is written.
5. **WebSocket port in use:** occupy 8081 with `nc -l 8081`, then start the server. `/health`
   reports `ws_port: 0`, and a `ws.bind_failed` event is logged.
6. **Live (SillyTavern `health`):** Refresh in the SillyTavern TTS panel shows "TTS Provider
   Loaded". The console shows a `breeze.health` event, with no CORS errors in the page.

## Scenario 2: Speech over HTTP (US1-2, US2, US3; BC-01 to BC-17, BC-46, BC-47)

1. **Voice design:** `curl -sS -D- -F text='Hello there.' :8080/v1/audio/speech -o out.pcm`.
   Expect `200`, the four headers, and a non-empty file.
   `ffplay -f s16le -ar 24000 -ac 1 out.pcm` plays speech.
2. **Malformed corpus:** `.venv/bin/pytest tests/test_malformed_corpus.py` sends at least 50
   malformed requests. Every one returns a `4xx` envelope, and `/health` stays `200` (SC-003).
3. **Mid-stream abort:** `.venv/bin/pytest tests/test_speech_abort.py` uses a real uvicorn server
   and a fake runtime that fails after 3 chunks. httpx raises `RemoteProtocolError` (BC-17).
4. **Long text:** `BREEZE_MODEL=... pytest -m gpu tests/gpu/test_speech_long_text.py` sends a
   3,000-character passage with no reference, 5 runs. Every run completes and covers the whole
   text (SC-004).
5. **CFG values:** `cfg_scale` values 2.5, 7.5 and 0 (GPU test) produce finite audio, with no
   graph recapture.
6. **Live (SillyTavern `speech`):** from the SillyTavern page, via Playwright `page.evaluate`,
   `fetch` `POST /v1/audio/speech` with the page's origin. The CORS expose-headers are readable,
   and the bytes decode as PCM.

## Scenario 3: Voices (US1-3/4; BC-25 to BC-29, BC-48)

1. **Register the test voices once:**
   `curl -F ref_audio=@eric.wav -F ref_text="$(cat eric.txt)" -F name=eric :8080/v1/voices`, and
   the same for `vale`. Expect `200` with `saved: true`.
2. **Duplicate name:** posting `name=ERIC` returns `409 voice_exists`.
3. **Delete and restart:** register `st_live_tmp`, delete it (`file_kept: false`), and restart the
   server. The voice is not listed (BC-28).
4. **Old voice files:** place a `.breeze` file in `voices/` and restart. A `voices.loaded` event
   reports `breeze_ignored: 1` (BC-29).
5. **Live (SillyTavern `voices`):** in the voice panel, upload `st_live_tmp` (transcript
   required), confirm it appears, and use "Replace" (after the extension's DELETE-then-POST change
   lands). Then delete it. `eric` and `vale` are never touched. Expect `voice.uploaded` and
   `voice.deleted` events, and no toasts.

## Scenario 4: WebSocket sessions (US4; BC-30 to BC-45)

1. **Session state machine:** `.venv/bin/pytest tests/test_ws_session.py` runs 1,000 seeded
   random sequences of `text`, `flush`, `cancel`, `end` and `start`. It checks exactly one
   `done`/`cancelled` per message and no lost pieces (SC-005).
2. **Protocol conformance:** `.venv/bin/pytest tests/test_ws_server.py` checks the `\uXXXX`
   round-trip, the close-code matrix, the Origin `403`, the connection cap, and eviction of a
   slow client (1008).
3. **Stalled client:** `.venv/bin/pytest tests/test_ws_isolation.py` stops one client from reading
   while an HTTP speech request starts. The HTTP request starts streaming within the in-flight
   piece plus 1 s (SC-006).
4. **Live (SillyTavern `full`):** in the "Breeze validation" chat under Seraphina:
   - narrate the last message with quoted text; expect `synth.request`, then `started`, the audio
     frames, `done`, and `synth.done`;
   - narrate again and press stop: expect `synth.cancelled`, with no error toast;
   - use voice preview for `eric`;
   - run two narrations back to back and expect `queued` on the second;
   - check `cfg_scale` 1, 4 and 7.5 in the settings.

## Scenario 5: Compatibility and performance sign-off (SC-001, SC-002, SC-007, SC-008)

1. **C++ docs examples:** `python -m tests.live.cpp_examples` runs every example from the C++
   `docs/server.md`, `docs/voices.md` and `docs/websocket.md`. Every difference from the documented
   result must be a listed BC id (SC-001).
2. **Breaking-change coverage:** `.venv/bin/pytest -k bc_` runs one test per BC id. Each fails
   against the C++ behavior, as recorded in the test docstrings (SC-002).
3. **Performance:** the 10-run benchmark medians for time to first audio and real-time factor on
   the gating cases (`short_design`, `short_inline`, `medium_inline`) are within 10% of the
   Scenario 0 10-run baseline (SC-007; the method is in tasks.md "SC-007 method").
4. **Documentation:** the README lists every endpoint, field, error code, launch option, WebSocket
   message and BC id (SC-008).
5. **SillyTavern's own live suite:** `cd <SillyTavern checkout>/extensions/SillyTavern-BreezeTTS && npm
   run test:live` passes against the new server.

## Rollback (Constitution VII)

Rollback must take under 5 minutes with no code change, and is rehearsed once before sign-off:
1. Stop the server.
2. Either `git checkout <previous tag>` and start `scripts/start_breeze.sh`, or start the C++
   `breeze-server --cors` on 8080/8081. SillyTavern keeps working on the C++ server.
3. Voice files in `voices/` are left untouched by a rollback. The `eric` and `vale` `.breeze` files
   of the C++ server are never modified by this feature.
