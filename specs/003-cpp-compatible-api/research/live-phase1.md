# Live record: Phase 1 gate (T029), health and CORS on the new server

## Summary

- **Server**: `enhanced-api` at `4cb9df9`, version `2.0.0.dev1`, started with
  `scripts/start_breeze.sh --cors http://127.0.0.1:8000` (host 0.0.0.0, `--fast-all`). The model
  loaded in about 80 s.
- **Result: the plan's Phase 1 gate passes.** The provider loads ("TTS Provider Loaded"), the
  `breeze.health` event deep-equals `{status, sample_rate, ws_port}`, and there are no CORS errors in
  the page. The SillyTavern settings were restored and checked against the server afterwards.
- **Expected failures (2 of 7 steps):** the extension's `checkReady()` also calls `GET /v1/voices`,
  which doesn't exist until T065 (Phase 7). That gives `breeze.check_failed {stage: voices,
  message: "not found"}` and a "using the cached voice list" toast. The harness has checked this
  stage strictly since its review fixes, so it can't pass before Phase 7. This is a phase-ordering
  gap in the plan, not a server defect, and the T069 voices gate re-checks it.

## Quickstart Scenario 1 by curl

| Check | Result |
|---|---|
| 1.1 `/health` while loading | `503 {"status":"loading","error":"model is loading","code":"loading"}` with `x-breeze-version: 2.0.0.dev1` |
| 1.1 `/health` when ready | `200 {"status":"ok","sample_rate":24000,"ws_port":0}` (`ws_port` is 0 until Phase 8) |
| 1.2 unknown route | `404 {"error":"not found","code":"not_found"}` |
| 1.2 `PUT /v1/voices` | `404` for now: the voice routes arrive in T065 (a `405` with `Allow` is covered by unit tests on `/health`) |
| 1.3 preflight `DELETE /v1/voices/x` | `404` envelope with CORS headers, since the route doesn't exist yet; route-aware `204` preflights are covered by `tests/test_cors.py` |
| 1.4 `POST /v1/voices` from `http://evil.test` | `403 {"error":"origin not allowed","code":"origin_not_allowed"}`, before any work (BC-23) |
| Allowed-origin `GET /health` | `access-control-allow-origin: http://127.0.0.1:8000`, `vary: Origin`, expose-headers including `X-Breeze-Version` |
| 1.5 WebSocket port in use | Not applicable until Phase 8 |

## Known issue found in review, being fixed now

A preflight for `POST /v1/voices` would get `405`, because the CORS middleware used only the first
route matching a path, and FastAPI registers GET and POST separately. It is fixed in the CORS
follow-up before T065 adds those routes.

## Review loop outcomes (Phases 2–3)

Every Phase 2–3 commit had two `review-agent` passes. All findings were accepted, apart from
recorded declines: event ts order is kept over serializing outside the lock; the web-stack versions
are duplicated in the smoke check; httpx stays in requirements.txt. Decisions made with the user
along the way:
- the SC-007 method (10 runs after 3 warm-ups; medium_design gated);
- the websockets pin and Python 3.12 as the tested version;
- the codec fingerprint (config fields plus the weights header digest);
- an 80 ms minimum reference clip;
- a privacy rewrite of the branch history.

Notable fixes from review:
- 413 had turned into 400 on Form routes;
- `Connection: close` made clients lose the 413 (added, then removed);
- a GPU close could be skipped at shutdown;
- a second signal couldn't force an exit, and Windows signals didn't work;
- the segmenter was quadratic on combining runs, and unpunctuated spaced text was never cut
  (a C++ quirk);
- CORS origins are now canonicalized the same way at startup and per request.

## Raw record (written by run.mjs)

Generated 2026-09-25T10:24:50.004Z. 5/7 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-25T10:24:49.186Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":0} |
| FAIL | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"breeze.check_failed","at":"2026-09-25T10:24:49.189Z","httpUrl":"http://127.0.0.1:8080","stage":"voices","message":"not found"} |
| FAIL | health: no error toast after loading the provider | Breeze TTSnot found. Using the voice list from 9/24/2026, 10:25:44 PM.Error: not foundBreeze TTSnot found. Using the voi |
| ok | health: no CORS errors in the page console |  |
| ok | health: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-phase1-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

```json
{
  "settings_version": 1,
  "http_url": "http://127.0.0.1:8080",
  "ws_url": "ws://127.0.0.1:8081",
  "delivery_mode": "buffer",
  "streaming_chunk_seconds": 1,
  "guidance_baseline": 1,
  "guidance_direction": 6,
  "guidance_vocal_event": 6,
  "seed": 73,
  "direction_enabled": true,
  "direction_for_user_messages": false,
  "direction_prompt": "",
  "direction_max_words": 20,
  "inline_tags_enabled": true,
  "inline_tag_keyword": "[redacted]",
  "vocal_events_enabled": true
}
```

```json
[
  {
    "step": "health: SillyTavern loaded with the Breeze provider registered",
    "ok": true,
    "detail": ""
  },
  {
    "step": "health: provider reports \"TTS Provider Loaded\"",
    "ok": true,
    "detail": ""
  },
  {
    "step": "health: breeze.health (not breeze.check_failed) logged for this run's httpUrl",
    "ok": true,
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-25T10:24:49.186Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":0}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": false,
    "detail": "{\"event\":\"breeze.check_failed\",\"at\":\"2026-09-25T10:24:49.189Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"stage\":\"voices\",\"message\":\"not found\"}"
  },
  {
    "step": "health: no error toast after loading the provider",
    "ok": false,
    "detail": "Breeze TTSnot found. Using the voice list from 9/24/2026, 10:25:44 PM.Error: not foundBreeze TTSnot found. Using the voi"
  },
  {
    "step": "health: no CORS errors in the page console",
    "ok": true,
    "detail": ""
  },
  {
    "step": "health: settings restored and verified on disk",
    "ok": true,
    "detail": ""
  }
]
```
