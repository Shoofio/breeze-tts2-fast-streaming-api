# Live record: Phase 5 sign-off (T085, T086), 2.0.0

## T086: final live gate (2026-09-29)

- **Code**: `enhanced-api` at `c5c7527` (version `2.0.0`, T084), plus the working-tree change that
  clears the `verify_live` flags once this gate confirmed them (committed with this record).
- **Servers**, one at a time (one model load at a time on this host):
  1. WSL, `scripts/start_breeze.sh` (no `--cors`): the T082 benchmark (`bench-final.md`), then
     the C++ examples with CORS off. Stopped with SIGTERM, exit 0.
  2. WSL, `scripts/start_breeze.sh --cors http://127.0.0.1:8000`: the C++ examples with the
     allowlist, SillyTavern `run.mjs full`, and the extension's `npm run test:live`. Stopped with
     SIGTERM, exit 0.
  3. Windows, `scripts/start_breeze.ps1 -BindHost 127.0.0.1` (`.venv-win`, no `--cors`): the
     smoke check and the C++ examples with CORS off. Stopped with `Stop-Process`.

  Every `/health` answered `{"status":"ok","sample_rate":24000,"ws_port":8081}` with
  `X-Breeze-Version: 2.0.0`. None of the three server logs has an error- or warning-level event.
  The voice list was `eric`, `vale` after every run.
- **Result: the gate passes.**

### Test suites

| Suite | Result |
|---|---|
| CPU `pytest` | 2,081 passed, 28 skipped (GPU), 3 min 14 s |
| `pytest -m gpu` (`BREEZE_MODEL` = snapshot `c1c8ca18`) | 28 passed, 14 min 41 s |
| `tests/test_cpp_examples.py`, `tests/test_bc_coverage.py` after the flag change | 38 passed |
| `ruff check .` | the 29 errors in `models/` only (see below) |

**ruff and `models/`**: `ruff check .` reports 29 errors, all in `models/` (`RUF012` mutable class
attributes, `F841` unused locals, `TRY004`, `B023`, `SIM102`, `SIM118`, one `I001`). This
feature never changed `models/`; the repo has no ruff config, and ruff 0.16.5 reports these by
default. With the user's agreement they are recorded here and not fixed (no refactoring of
working model code). Everything else is clean.

### C++ doc examples (`python -m tests.live.cpp_examples`, SC-001)

All 25 examples from `Breeze-TTS-2.cpp/docs/{server,voices,websocket}.md`.

| Server | Result |
|---|---|
| WSL, CORS off | 18 pass, 6 explained, 0 fail, 1 skipped (cpp13 needs an allowlist); RESULT OK |
| WSL, `--cors http://127.0.0.1:8000` | 19 pass, 5 explained, 1 fail (cpp25, below), 0 skipped; cpp25 rerun alone: PASS |
| Windows, CORS off | 18 pass, 6 explained, 0 fail, 1 skipped; RESULT OK |

The explained differences, each seen live:

- **ADD-2**: `code` next to `error` (cpp08 `text_required`, cpp09 `unknown_voice`, cpp10
  `busy`, cpp24 `unknown_voice` on the WebSocket).
- **ADD-3**: `request_type: "start"` on the WebSocket error (cpp24).
- **BC-28**: `file_kept: false` on DELETE (cpp18).
- **BC-18**: with CORS off, `OPTIONS` is `405`, where C++ gave `404` (cpp12).

Every entry was confirmed, so the `verify_live` flags are cleared. The flag stays in the harness
for future entries.

**The cpp25 failure was another client, not the server.** cpp25's voice registration got
`409 {"error":"busy","code":"busy"}` for its whole retry window (30 × 0.5 s). The server log shows
a WebSocket session the harness didn't open (voice `eric`, three pieces, closed with 1005) holding
the GPU from about 160 s to 185 s into that minute. It was the user narrating in SillyTavern by
hand, which the allowlist launch let in (confirmed by the user; `sillytavern-agent` was idle).
`409 busy` is the documented C++ behavior, so the server was correct, and the rerun passed.

### SillyTavern

- **`run.mjs full`**: 48 of 48 steps passed (record: `research/live-phase5-full.md`). This covered
  health, and voice upload, replace and delete of `st_live_tmp`; T079's replace-dialog flake did
  not recur. It also covered narration, stop mid-narration (`synth.cancelled`, 3,840 bytes, no
  `done`), the `eric` preview (130,560 bytes), `queued` while a Node client held the GPU, and
  `cfg_scale` 4, 7.5 and 1 reaching the request.
- **Extension `npm run test:live`** (`SillyTavern-BreezeTTS` 0.1.3, `d628cee`): 15 of 15 passed.
  These cover the WebSocket full session, an unknown voice, cancel after the first frame, two
  concurrent sessions (one `queued`), streaming chunks, the trailing-comma drain, and instruction
  plus `cfg_scale`. The HTTP checks cover health, the voice list, unnamed uploads, the upload
  errors and deleting an unknown voice.
- `sillytavern-agent` was asked to stay off the server during the gate. It is told that 2.0.0 is
  live once the rollback rehearsal (T085) is done and 2.0.0 is running again.

### Windows smoke check

Run natively on Windows (`.venv-win`, the production launcher), with requests from Windows
`curl.exe`:

- `GET /health`: `200`, `X-Breeze-Version: 2.0.0`.
- Voice design (`text`, `seed=7`): `200`, `audio/pcm`, `X-Sample-Rate: 24000`,
  `X-Sample-Format: s16le`, 230,400 bytes (4.8 s).
- `voice_id=vale`: `200`, 168,960 bytes.
- `GET /v1/voices`: `eric` and `vale`, both `saved: true`.

WSL reaches the Windows server on `127.0.0.1`, so the C++ examples were run against it too (the
table above). The Windows CPU `pytest` suite was not run in this gate.

## T085: rollback rehearsal (2026-09-29)

Rehearsed as in quickstart "Rollback", taking the C++ route (the one SillyTavern keeps working
on). The repo has no tags, so the "previous tag" route has no target yet; tagging `v2.0.0` at
release gives future rollbacks one.

- **Start state**: 2.0.0 serving on WSL (`scripts/start_breeze.sh --cors http://127.0.0.1:8000`,
  warm, `/health` ok).
- **Steps, all from one timed script**:
  1. SIGTERM to the 2.0.0 server, then wait for ports 8080/8081 to be free.
  2. Start the C++ server with `<Breeze-TTS-2.cpp checkout>/start_breeze.sh` (`breeze-server`
     at `edb927c`, `breeze-tts-2-q4_k.gguf`, `--cors --host 0.0.0.0 --port 8080`, its own
     `voices/` directory).
  3. Poll `/health` until `ok`. The answer carried no `X-Breeze-Version`, so it came from the C++
     server.
  4. `node tests/live/sillytavern/run.mjs health --record rollback-cpp`: 7 of 7 passed
     (`research/live-rollback-cpp.md`). The provider loaded, `breeze.health` and
     `voices.refreshed` were logged, and there were no error toasts or CORS errors.
- **Timeline (EDT)**: SIGTERM at 11:25:00; C++ launched at 11:25:03; SillyTavern checking against
  it at 11:25:18.1; all checks passed at 11:25:18.75.
- **Result: rollback takes about 19 s**, well under the 5-minute limit, with no code change. The
  seconds are taken from the logs, because the script's own interval arithmetic failed
  (`bc: command not found`).
- **Voices**: neither server's voice files were touched. This server's `voices/` (`eric`,
  `vale`) and the C++ server's `.breeze` files are separate directories.
- **Not covered**: narration on the C++ server; Phase 0 (T011, `live-phase0.md`) already ran it.
  The C++ server was stopped with SIGTERM afterwards.

## Review loop outcomes (Phase 9)

- **T080/T081 (`9af5679`, `fd89358`), review pass 1 (Opus)** found the following, fixed in
  `16296e3` (Sonnet, with failing-first unit tests):
  - voice cleanup that could fail silently;
  - cpp16's delete not in a `finally`;
  - cpp10 carrying on after an unexpected first status;
  - WebSocket examples erroring instead of skipping when `ws_port` is 0;
  - ADD-2/ADD-3 patterns missing prefixed event paths;
  - no check of `OPTIONS` without CORS (now cpp12 under BC-18);
  - a coverage check weaker than T081 asked (it now requires the exact BC-01..BC-48 set and a
    docstring naming the C++ behavior);
  - docstrings that misstated the C++ behavior. BC-18 had wrongly called C++'s 400/409 errors
    non-JSON, and BC-15 and BC-27 never named the C++ behavior.
- **Declined in pass 1**: correcting spec.md's "`type` on WebSocket error events" to
  `request_type` (the contract's and server's name) was left for the user to decide.
- **Pass 2** on `16296e3` and T084's `7a88a17` was stopped by the user before it reported, so
  these commits had one review pass.
