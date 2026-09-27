# Live SillyTavern run: voices

Generated 2026-09-27T01:32:52.147Z. 20/20 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-27T01:32:47.823Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":8081} |
| ok | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"voices.refreshed","at":"2026-09-27T01:32:47.827Z","count":2} |
| ok | health: no error toast after loading the provider |  |
| ok | health: no CORS errors in the page console |  |
| ok | voices: st_live_tmp uploaded (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-27T01:32:48.254Z","voiceId":"st_live_tmp","seconds":10.32,"saved":true} |
| ok | voices: voice list refreshed after upload | {"event":"voices.refreshed","at":"2026-09-27T01:32:48.258Z","count":3} |
| ok | voices: st_live_tmp appears in the voice list |  |
| ok | voices: st_live_tmp replaced (voice.replaced event) | {"event":"voice.replaced","at":"2026-09-27T01:32:51.185Z","voiceId":"st_live_tmp"} |
| ok | voices: st_live_tmp re-uploaded after replace (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-27T01:32:51.350Z","voiceId":"st_live_tmp","seconds":10.32,"saved":true} |
| ok | voices: replace did not fail (no voice.upload_failed) | null |
| ok | voices: voice list refreshed after replace | {"event":"voices.refreshed","at":"2026-09-27T01:32:51.354Z","count":3} |
| ok | voices: st_live_tmp deleted (voice.deleted event) | {"event":"voice.deleted","at":"2026-09-27T01:32:51.667Z","voiceId":"st_live_tmp","orphaned":[]} |
| ok | voices: voice list refreshed after delete | {"event":"voices.refreshed","at":"2026-09-27T01:32:51.682Z","count":2} |
| ok | voices: st_live_tmp no longer in the voice list |  |
| ok | voices: no error toast across the upload/replace/delete run | Breeze TTSVoice "st_live_tmp" removed.Breeze TTSVoice "st_live_tmp" saved on the Breeze server.Breeze TTSVoice "st_live_tmp" saved on the Breeze server. |
| ok | voices: eric is still listed | eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/26/2026, 9:32:51 PM |
| ok | voices: vale is still listed | eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/26/2026, 9:32:51 PM |
| ok | voices: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-voices-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

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
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-27T01:32:47.823Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":8081}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-27T01:32:47.827Z\",\"count\":2}"
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
    "step": "voices: st_live_tmp uploaded (voice.uploaded event)",
    "ok": true,
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-27T01:32:48.254Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.32,\"saved\":true}"
  },
  {
    "step": "voices: voice list refreshed after upload",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-27T01:32:48.258Z\",\"count\":3}"
  },
  {
    "step": "voices: st_live_tmp appears in the voice list",
    "ok": true,
    "detail": ""
  },
  {
    "step": "voices: st_live_tmp replaced (voice.replaced event)",
    "ok": true,
    "detail": "{\"event\":\"voice.replaced\",\"at\":\"2026-09-27T01:32:51.185Z\",\"voiceId\":\"st_live_tmp\"}"
  },
  {
    "step": "voices: st_live_tmp re-uploaded after replace (voice.uploaded event)",
    "ok": true,
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-27T01:32:51.350Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.32,\"saved\":true}"
  },
  {
    "step": "voices: replace did not fail (no voice.upload_failed)",
    "ok": true,
    "detail": "null"
  },
  {
    "step": "voices: voice list refreshed after replace",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-27T01:32:51.354Z\",\"count\":3}"
  },
  {
    "step": "voices: st_live_tmp deleted (voice.deleted event)",
    "ok": true,
    "detail": "{\"event\":\"voice.deleted\",\"at\":\"2026-09-27T01:32:51.667Z\",\"voiceId\":\"st_live_tmp\",\"orphaned\":[]}"
  },
  {
    "step": "voices: voice list refreshed after delete",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-27T01:32:51.682Z\",\"count\":2}"
  },
  {
    "step": "voices: st_live_tmp no longer in the voice list",
    "ok": true,
    "detail": ""
  },
  {
    "step": "voices: no error toast across the upload/replace/delete run",
    "ok": true,
    "detail": "Breeze TTSVoice \"st_live_tmp\" removed.Breeze TTSVoice \"st_live_tmp\" saved on the Breeze server.Breeze TTSVoice \"st_live_tmp\" saved on the Breeze server."
  },
  {
    "step": "voices: eric is still listed",
    "ok": true,
    "detail": "eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/26/2026, 9:32:51 PM"
  },
  {
    "step": "voices: vale is still listed",
    "ok": true,
    "detail": "eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/26/2026, 9:32:51 PM"
  },
  {
    "step": "voices: settings restored and verified on disk",
    "ok": true,
    "detail": ""
  }
]
```
