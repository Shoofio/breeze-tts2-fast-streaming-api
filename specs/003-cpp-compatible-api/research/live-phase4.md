# Live record: Phase 4 gate (T079), WebSocket sessions

## T079: full live gate (2026-09-26)

- **Server**: `enhanced-api` at `e81702d`, version `2.0.0.dev4`, started with
  `scripts/start_breeze.sh --cors http://127.0.0.1:8000` (`--fast-all`). `/health` answered
  `{"status":"ok","sample_rate":24000,"ws_port":8081}` with `X-Breeze-Version: 2.0.0.dev4`.
  Before the gate, the CPU suite passed at `ffc2e0c` (2,044) and the GPU suite at `b4bf010`
  (28 passed, 13 min 28 s). The C++ server was not running.
- **Result: the gate passes.** SillyTavern narrates, streams, cancels, previews and queues on the
  new server. The server log has no error-level events; SIGTERM stopped it in 5 s.

### SillyTavern harness

- **`run.mjs full`**: 35 of 36 steps passed (record: `research/live-full.md`). Narration
  (3,663,360 bytes, 76.3 s of audio, in 28 s), stop mid-narration (`synth.cancelled`, no
  `done`), the `eric` preview, `queued` while a Node client held the GPU, and `cfg_scale` 4, 7.5
  and 1 all passed.
- **The one failure was in the harness UI, not the server**: in the `voices` phase, Playwright's
  click on the replace dialog's OK found the element detached from the DOM (`page.click: Timeout
  30000ms exceeded`). The upload before it had succeeded, so `st_live_tmp` was left behind; it was
  deleted with `DELETE /v1/voices/st_live_tmp`, and `run.mjs voices` then passed 20 of 20
  (`research/live-voices.md`). That upload was the first reference encode after start-up
  (`encode_ms: 4991`, the known slow first encode in `live-phase3.md`), which is the likely reason
  the dialog's timing differed.
- **Extension `npm run test:live`** (v0.1.3 working tree): 15 of 15 passed, including the
  WebSocket full session, unknown voice, cancel after the first frame, two concurrent sessions
  (one `queued`), streaming chunks, and instruction plus `cfg_scale`.

### `sillytavern-agent`'s check

Headless Playwright on the "Breeze validation" chat under Seraphina, voice `vale`, no throwaway
voices, settings and messages restored afterwards. Every server-side check passed:

- Raw protocol probe: `ready {sample_rate: 24000, format: "s16le"}`, then `started` with
  `voice_id: "vale"`, `speaking`, and exactly one `done`; a cancel after the first frame gives
  exactly one `cancelled` and no `done`; an unknown voice gives
  `{"type":"error","code":"unknown_voice","message":"unknown voice_id","request_type":"start"}`.
- Through the UI: buffer narration (6 sentences, 4,277,760 bytes in 32.4 s); streaming narration
  (first chunk after 900 ms, 43 chunks, the same bytes as buffer mode); stop mid-stream and
  mid-buffer (one `cancelled`, nothing after it); cancel right after `started`; voice preview;
  queueing behind a raw client holding the GPU (`ready`, `started`, `queued`, `speaking`,
  `done`). No console errors and no `*_failed` events.
- **Close code 1005**: the clients close without a status code after `done`, and the server
  echoes that close, so they see 1005 ("no status"). This is RFC 6455's echo (BC-43); a client
  that closes with 1000 gets 1000 back (`test_bc_43_close_codes`).
- **Extension-side finding (not the server)**: a stop pressed about 30 ms after narrate reaches
  the provider before the job does, so the SillyTavern framework dispatches the narration seconds
  later, and it cancelled a preview started in between. The server queued and cancelled correctly.
  `sillytavern-agent` is raising it with its user.

### Time to first audio

`scratchpad/ws_ttfa.py`: `start` with a voice, then `end` with one sentence (78 characters),
timed to the first binary frame, sequential runs on the warm server.

| Voice | Runs | TTFA median (first run excluded) | First run | Audio / wall time |
|---|---|---|---|---|
| `vale` | 8 | 47 ms | 241 ms (builds the voice's prefix) | 6.16 s in 2.24 s |
| `eric` | 6 | 47 ms | 47 ms | 4.64 s in 1.69 s |

**No C++ time to first audio exists to compare with.** Phase 0 (`live-phase0.md`) recorded only
whole-narration times, in buffer mode. On those: C++ Q4 produced 33.4 s of audio in 24 s (wall
time ÷ audio 0.72, `cfg_scale` 6); this server produced 76.3 s in 28 s (0.37, `cfg_scale` 1).
The two models speak the same message at different lengths, so only the ratio is comparable.

## Review loop outcomes (Phase 8)

Every Phase 8 commit had two `review-agent` passes (reviews 44–49), with no third; the final
fixes were verified here with the CPU and GPU suites. Findings that changed the design or
behaviour:

- **T070 prototype**: `websockets`' `close_timeout` never starts for a peer that stopped reading
  mid-send (`ws.close()` awaits `drain()` before enforcing its deadline). The user chose to keep
  the library and bound every close ourselves (2 s, then `SO_LINGER(1, 0)` and abort), with
  `TCP_USER_TIMEOUT` as a Linux-only backstop.
- **Reviews 44–45 (design)**: a stall watchdog on the write buffer (30 s, user decision) also
  evicts a client stalled under 2 MiB or idle behind its own pong replies; every close path is
  bounded, including handler exits; one registry shared by every listener, holding the cap, the
  connection set and the shutdown flag, releasing a slot when the transport closes;
  concurrent shutdown; synchronous handshake hooks; `503 shutting_down`.
- **Reviews 46–47 (parser and session)**: deeply nested JSON, `NaN` and lone surrogates no longer
  crash the reader or reach the encoder; `start` follows HTTP's field rules (BC-02, caps, blank
  `ref_text`, `voice_id` syntax), through one shared text-field helper; a `done` owed by an
  earlier session is no longer lost, since every `start` begins an epoch and a `cancel` removes
  only its own session's `done`; `Session.close()` on disconnect; the SC-005 checker was
  tightened twice, each time after planted bugs got through it.
- **Reviews 48–49 (server)**: two gate leaks (a double release after a close timeout, and a slow
  client at `speaking`) and a shutdown hang with a stuck GPU, all with regression tests; a
  cancel reaches a piece waiting for its anchor sizing or the gate; the out-of-memory prefix
  fallback lasts for the session and shares HTTP's code; GPU health checked just before the 101;
  malformed upgrades refused before readiness and the cap.
- **Declined**: moving WebSocket anchor sizing onto HTTP's anchor-sizing worker (review 48 #7).
  That worker runs only the lease holder's sizing, whose timeout changes HTTP's behaviour; on
  the pre-gate worker a WebSocket check can only add latency.

## Decisions made with the user during Phase 8

- Keep `websockets` and bound every server-initiated close ourselves, with `TCP_USER_TIMEOUT` as
  a backstop.
- A send stall of 30 s evicts a client.
- The anchor rule over the WebSocket is checked per piece: a later piece that the anchor would
  shorten is spoken without it.
