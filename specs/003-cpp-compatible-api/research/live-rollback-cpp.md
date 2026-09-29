# Live SillyTavern run: rollback-cpp

Generated 2026-09-29T15:25:18.752Z. 7/7 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-29T15:25:18.061Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":8081} |
| ok | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"voices.refreshed","at":"2026-09-29T15:25:18.109Z","count":4} |
| ok | health: no error toast after loading the provider |  |
| ok | health: no CORS errors in the page console |  |
| ok | health: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-rollback-cpp-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

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
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-29T15:25:18.061Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":8081}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-29T15:25:18.109Z\",\"count\":4}"
  },
  {
    "step": "health: no error toast after loading the provider",
    "ok": true,
    "detail": ""
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
