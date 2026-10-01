# Live gate: 004 browser-playable WAV streaming (T016)

**Date**: 2026-10-01 · **Build**: `2.1.0.dev2` (`a6d2714` + docs) · **Server**: WSL,
`scripts/start_breeze.sh --cors http://127.0.0.1:8000` · **Browser**: Chromium via Playwright, on
the SillyTavern page `http://127.0.0.1:8000` · **Voice**: `Eric01`

**The gate was ended early at the user's decision**, after the 1× and 0.9× runs in Chromium. The
section "Not run" lists what was left out.

## Before the gate

| Check | Result |
|---|---|
| `ruff check breeze_infer tests` | clean (the 29 errors in `models/` are the known 003 baseline) |
| CPU suite | 2093 passed, 29 skipped (GPU tests without `BREEZE_MODEL`) |
| GPU suite (T015), with `BREEZE_MODEL` | **29 passed, 0 skipped**, 14 min 12 s |

## Results

| # | Check | Result |
|---|---|---|
| §1 | `200`, `audio/wav`, chunked, `Accept-Ranges: none`, `Cache-Control: no-store`, `X-Sample-Rate: 24000`, `X-Breeze-Version: 2.1.0.dev2` | pass |
| §1 | WAV header bytes match the contract; `ff ff ff ff` at offsets 4 and 40. Python's `wave` reads 1 ch, 24000 Hz, 16 bit | pass |
| §1 | `Range: bytes=0-` gives `200` | pass |
| §1 | `voice_id=nope` gives the same `404 unknown_voice` body on GET and POST | pass |
| SC-001 | Time to first audio, 3 runs each: GET 49.2/51.0/50.9 ms, POST 49.6/51.1/56.4 ms. Medians 51.0 vs 51.1 ms | pass (within 10%) |
| SC-002 | Chromium, 1×, 3,653-character English text in 7 pieces: 227.2 s of audio played to `ended` in 229.9 s wall. One `waiting` of 2.25 s at startup, none during playback | pass |
| SC-002 | Chromium, 0.9×, same text, different seed: 222.7 s of audio, ended `speech.completed` | pass (server side; the page's playback record was lost when the gate was stopped) |
| FR-014 | 1×: `rtf` 0.37, so generation finished about 84 s in, about 143 s before playback ended. The GPU was free for the rest | pass |
| FR-018 | Exactly one `speech.generated` and one `speech.completed format=wav` per playback | pass |
| — | After the gate, a new GET got first audio in 52 ms (GPU free) | pass |

Chromium read each whole stream long before playback ended (the `speech.completed` time matches
generation time). So Chromium never stalled the server's send. The Firefox read-ahead behaviour
that motivated buffered delivery (spec Clarifications) was not observed, because Firefox was not
tested.

## Not run (ended early)

- **Firefox**: all of SC-002, and the 0.5× run in both browsers.
- **SC-003**: changing `src` while synthesis is still running. Covered on CPU by
  `tests/test_speech_wav_stream.py`'s disconnect test.
- **SC-004**: a WebSocket preview during a drain. **SC-005**: a GET queued behind a WebSocket
  session. **FR-007**: a disconnect while queued. All three are covered on CPU by real-uvicorn
  tests.
- **SC-006**: 10,000-character Chinese and English texts on the real model. The request-head limit
  is covered on CPU, including on the production `api.serve`.
- **SC-008**: the `bench_api` regression benchmark. Spot check: POST time to first audio of 50–56 ms
  matches 2.0.0.
- The extension's `node tests/live/sillytavern/run.mjs full`.

## Process notes

- `review-agent` reviewed Phases 1–6, two passes each. Phase 7 (T012, one test) went unreviewed
  because `review-agent` was down. Phase 8 had no review, at the user's decision.
- The quickstart used voice `eric` and `xxd`. The real voices are `Eric01` and `Vale01`, and WSL has
  no `xxd`. Both are fixed in `quickstart.md`.
