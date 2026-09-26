# Live SillyTavern run: speech

Generated 2026-09-26T02:01:18.567Z. 9/11 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-26T02:01:16.949Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":0} |
| FAIL | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"breeze.check_failed","at":"2026-09-26T02:01:16.966Z","httpUrl":"http://127.0.0.1:8080","stage":"voices","message":"not found"} |
| FAIL | health: no error toast after loading the provider | Breeze TTSnot found. Using the voice list from 9/24/2026, 10:25:44 PM.Error: not foundBreeze TTSnot found. Using the voi |
| ok | health: no CORS errors in the page console |  |
| ok | speech: POST /v1/audio/speech returns 200 | status=200 |
| ok | speech: X-Sample-Rate header is readable (expose-headers work) | X-Sample-Rate=24000 |
| ok | speech: body is non-empty with an even byte length | bytes=72960 |
| ok | speech: cfg_scale=banana returns 400 with a JSON error key | {"status":400,"json":{"error":"cfg_scale must be a number","code":"invalid_field"}} |
| ok | speech: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-speech-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

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
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-26T02:01:16.949Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":0}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": false,
    "detail": "{\"event\":\"breeze.check_failed\",\"at\":\"2026-09-26T02:01:16.966Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"stage\":\"voices\",\"message\":\"not found\"}"
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
    "step": "speech: POST /v1/audio/speech returns 200",
    "ok": true,
    "detail": "status=200"
  },
  {
    "step": "speech: X-Sample-Rate header is readable (expose-headers work)",
    "ok": true,
    "detail": "X-Sample-Rate=24000"
  },
  {
    "step": "speech: body is non-empty with an even byte length",
    "ok": true,
    "detail": "bytes=72960"
  },
  {
    "step": "speech: cfg_scale=banana returns 400 with a JSON error key",
    "ok": true,
    "detail": "{\"status\":400,\"json\":{\"error\":\"cfg_scale must be a number\",\"code\":\"invalid_field\"}}"
  },
  {
    "step": "speech: settings restored and verified on disk",
    "ok": true,
    "detail": ""
  }
]
```
