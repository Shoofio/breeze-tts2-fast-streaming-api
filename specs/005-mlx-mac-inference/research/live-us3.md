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

## SillyTavern (`node tests/live/sillytavern/run.mjs full --record mac-full`)

**Run 1, 2026-10-04 19:40 UTC: 30/34 steps passed.** The run was made by the user from the Linux
machine's SillyTavern, with `BREEZE_HTTP_URL=http://<mac>:8080` and
`BREEZE_WS_URL=ws://<mac>:8081`, against the Mac launcher at 8-bit. The Mac had the user's saved
voices `Eric01` and `Vale01`.

**What passed:**
- **Health:** the provider loaded, `/health` reported `version 2.2.0` and `wavStream: true`, the
  voice list refreshed, and there were no CORS errors.
- **Narration:** streamed in the "Breeze validation" chat, 42 frames delivered.
- **Stop:** cancelled an active session (`synth.cancelled`).
- **Guidance:** `cfg_scale` 4, 7.5 and 1 each reached the request and completed.
- **Settings:** restored and verified.

**The four failures all come from the test suite or its setup, not from the MLX backend.** The
`breezetts-linux` session confirmed each one against the CUDA history:

| Step | Cause |
|---|---|
| voices phase aborted (`page.click … dialog[open] .popup-button-ok … element was detached from the DOM`) | A timing race in `acceptPopupIfPresent` (`tests/live/sillytavern/lib.mjs:229-235`). The same message, word for word, appeared against CUDA in `live-full.md` (2026-09-27, 35/36), and the voices phase alone passed 20/20 a minute later. The aborted phase left an `st_live_tmp` voice behind, and the ST suite has no cleanup for it. |
| voice preview for `eric` timed out | The suite hard-codes `eric` (`phases/full.mjs:194`). The voices on both machines were renamed to `Eric01`/`Vale01` on 2026-09-29, after the last full CUDA pass (`live-phase5-full.md`, 48/48). A CUDA run would fail this step today too. |
| Node WS client: `unknown voice_id` | The same cause, with `vale` (`phases/full.mjs:205`). |
| queued narration after the GPU frees | Skipped, because the previous step failed. |

**Next:** delete `st_live_tmp` on the Mac, add the voices under the names the suite expects
(`eric`, `vale`), and run it again. The suite itself is unchanged, since T028 requires running
the checks without edits.

## Automated coverage

`tests/mlx/test_mlx_server.py::test_websocket_session_streams_audio_then_done` (T027) runs a real
WebSocket session against the MLX server at both precisions. It observed `ready, started,
speaking, <binary frames>, done`: 3.76 s of audio at 8-bit and 3.28 s at bf16.
