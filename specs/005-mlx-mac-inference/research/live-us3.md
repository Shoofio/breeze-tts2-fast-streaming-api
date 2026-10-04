# Live gate, User Story 3 (T028): 2026-10-04

Quickstart step 4 on the reference Mac (Apple M5, 16 GB), 8-bit, at commit `dc3e5d6`.

## C++ server examples (`tests.live.cpp_examples`)

The examples suite ran unmodified. The reference clip was the server-made clip from the US2
gate, with its transcript beside it as `reference.txt`.

| Server started with | Result | Pass | Explained | Fail | Skipped |
|---|---|---|---|---|---|
| `scripts/start_breeze_mac.sh` (CORS `*`) | `RESULT: OK`, exit 0 | 19 | 5 | 0 | 1 (cpp13 needs a `--cors` allowlist) |
| `scripts/start_breeze_mac.sh --cors http://127.0.0.1:8000` | `RESULT: OK`, exit 0 | 20 | 5 | 0 | 0 |
| `python -m breeze_infer.api <8-bit>` (no CORS) | `RESULT: OK`, exit 0 | 18 | 6 | 0 | 1 (cpp13) |

- **Every difference matched a recorded expected difference**, with the same Breaking-Change ids
  as on CUDA. BC-18 (`OPTIONS` without CORS is `405` with `Allow`) only shows when CORS is off,
  hence the third run.
- **WebSocket examples cpp19–cpp25 all passed**: connect, a session, feeding text as it arrives,
  cancel, a queued second session, the error shape, and the client sketch.
- **The allowlist run also confirms that a later `--cors` overrides the launcher's `*`.**

## SillyTavern (`node tests/live/sillytavern/run.mjs full`)

**Not run yet.** The suite drives a running SillyTavern (`ST_URL`, default
`http://127.0.0.1:8000`) through Playwright's Chromium, using the reference voices in
`REFERENCE_VOICES_DIR`. None of these are installed on this Mac. Its defaults are Linux paths
(`chrome-linux64`).

## Automated coverage

`tests/mlx/test_mlx_server.py::test_websocket_session_streams_audio_then_done` (T027) runs a real
WebSocket session against the MLX server at both precisions. It observed `ready, started,
speaking, <binary frames>, done`: 3.76 s of audio at 8-bit and 3.28 s at bf16.
