# Live SillyTavern run: phase5-full

Generated 2026-09-29T14:50:19.270Z. 48/48 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-29T14:47:29.657Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":8081} |
| ok | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"voices.refreshed","at":"2026-09-29T14:47:29.660Z","count":2} |
| ok | health: no error toast after loading the provider |  |
| ok | health: no CORS errors in the page console |  |
| ok | voices: st_live_tmp uploaded (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-29T14:47:30.220Z","voiceId":"st_live_tmp","seconds":10.32,"saved":true} |
| ok | voices: voice list refreshed after upload | {"event":"voices.refreshed","at":"2026-09-29T14:47:30.223Z","count":3} |
| ok | voices: st_live_tmp appears in the voice list |  |
| ok | voices: st_live_tmp replaced (voice.replaced event) | {"event":"voice.replaced","at":"2026-09-29T14:47:33.021Z","voiceId":"st_live_tmp"} |
| ok | voices: st_live_tmp re-uploaded after replace (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-29T14:47:33.089Z","voiceId":"st_live_tmp","seconds":10.32,"saved":true} |
| ok | voices: replace did not fail (no voice.upload_failed) | null |
| ok | voices: voice list refreshed after replace | {"event":"voices.refreshed","at":"2026-09-29T14:47:33.093Z","count":3} |
| ok | voices: st_live_tmp deleted (voice.deleted event) | {"event":"voice.deleted","at":"2026-09-29T14:47:33.285Z","voiceId":"st_live_tmp","orphaned":[]} |
| ok | voices: voice list refreshed after delete | {"event":"voices.refreshed","at":"2026-09-29T14:47:33.290Z","count":2} |
| ok | voices: st_live_tmp no longer in the voice list |  |
| ok | voices: no error toast across the upload/replace/delete run | Breeze TTSVoice "st_live_tmp" removed.Breeze TTSVoice "st_live_tmp" saved on the Breeze server.Breeze TTSVoice "st_live_tmp" saved on the Breeze server. |
| ok | voices: eric is still listed | eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/29/2026, 10:47:33 AM |
| ok | voices: vale is still listed | eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/29/2026, 10:47:33 AM |
| ok | full: "Breeze validation" chat open under Seraphina |  |
| ok | full: synth.request logged for the narration | {"event":"synth.request","at":"2026-09-29T14:47:34.697Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":1,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: synth.event "started" logged | {"event":"synth.event","at":"2026-09-29T14:47:34.701Z","voiceId":"vale","type":"started"} |
| ok | full: synth.done logged, audio delivered | {"event":"synth.done","at":"2026-09-29T14:48:03.165Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: audio bytes were delivered | bytes=3663360 frames=50 |
| ok | full: cancel test: throwaway line added |  |
| ok | full: narration (for the cancel test) reaches "started" | {"event":"synth.event","at":"2026-09-29T14:48:05.699Z","voiceId":"vale","type":"started"} |
| ok | full: stop click was sent (narration was still active) |  |
| ok | full: stop cancelled the active session (not synth.done) | {"event":"synth.cancelled","at":"2026-09-29T14:48:05.850Z","voiceId":"vale","bytes":3840} |
| ok | full: no error toast after the cancel attempt |  |
| ok | full: cancel test: throwaway line removed |  |
| ok | full: voice preview for eric synthesizes | {"event":"synth.done","at":"2026-09-29T14:48:07.308Z","voiceId":"eric","bytes":130560,"frames":7} |
| ok | full: preview delivered audio bytes | bytes=130560 |
| ok | full: Node WS client is generating (holds the GPU) |  |
| ok | full: narration is queued while the Node client holds the GPU | {"event":"synth.event","at":"2026-09-29T14:48:07.700Z","voiceId":"vale","type":"queued"} |
| ok | full: queued narration completes once the GPU frees up | {"event":"synth.done","at":"2026-09-29T14:48:49.206Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: cfg_scale 4: synth.request logged | {"event":"synth.request","at":"2026-09-29T14:48:49.693Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":4,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 4: synth.done logged | {"event":"synth.done","at":"2026-09-29T14:49:22.588Z","voiceId":"vale","bytes":3479040,"frames":48} |
| ok | full: cfg_scale 4: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 4 reached the request | cfgScale=4 |
| ok | full: cfg_scale 7.5: synth.request logged | {"event":"synth.request","at":"2026-09-29T14:49:23.690Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":7.5,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 7.5: synth.done logged | {"event":"synth.done","at":"2026-09-29T14:49:50.006Z","voiceId":"vale","bytes":2883840,"frames":42} |
| ok | full: cfg_scale 7.5: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 7.5 reached the request | cfgScale=7.5 |
| ok | full: cfg_scale 1: synth.request logged | {"event":"synth.request","at":"2026-09-29T14:49:50.689Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":1,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 1: synth.done logged | {"event":"synth.done","at":"2026-09-29T14:50:18.853Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: cfg_scale 1: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 1 reached the request | cfgScale=1 |
| ok | full: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-phase5-full-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

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
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-29T14:47:29.657Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":8081}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-29T14:47:29.660Z\",\"count\":2}"
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
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-29T14:47:30.220Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.32,\"saved\":true}"
  },
  {
    "step": "voices: voice list refreshed after upload",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-29T14:47:30.223Z\",\"count\":3}"
  },
  {
    "step": "voices: st_live_tmp appears in the voice list",
    "ok": true,
    "detail": ""
  },
  {
    "step": "voices: st_live_tmp replaced (voice.replaced event)",
    "ok": true,
    "detail": "{\"event\":\"voice.replaced\",\"at\":\"2026-09-29T14:47:33.021Z\",\"voiceId\":\"st_live_tmp\"}"
  },
  {
    "step": "voices: st_live_tmp re-uploaded after replace (voice.uploaded event)",
    "ok": true,
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-29T14:47:33.089Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.32,\"saved\":true}"
  },
  {
    "step": "voices: replace did not fail (no voice.upload_failed)",
    "ok": true,
    "detail": "null"
  },
  {
    "step": "voices: voice list refreshed after replace",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-29T14:47:33.093Z\",\"count\":3}"
  },
  {
    "step": "voices: st_live_tmp deleted (voice.deleted event)",
    "ok": true,
    "detail": "{\"event\":\"voice.deleted\",\"at\":\"2026-09-29T14:47:33.285Z\",\"voiceId\":\"st_live_tmp\",\"orphaned\":[]}"
  },
  {
    "step": "voices: voice list refreshed after delete",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-29T14:47:33.290Z\",\"count\":2}"
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
    "detail": "eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/29/2026, 10:47:33 AM"
  },
  {
    "step": "voices: vale is still listed",
    "ok": true,
    "detail": "eric — saved, 10.3 svale — saved, 8.4 sList refreshed 9/29/2026, 10:47:33 AM"
  },
  {
    "step": "full: \"Breeze validation\" chat open under Seraphina",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: synth.request logged for the narration",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-29T14:47:34.697Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":1,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: synth.event \"started\" logged",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-29T14:47:34.701Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: synth.done logged, audio delivered",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:48:03.165Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
  },
  {
    "step": "full: audio bytes were delivered",
    "ok": true,
    "detail": "bytes=3663360 frames=50"
  },
  {
    "step": "full: cancel test: throwaway line added",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: narration (for the cancel test) reaches \"started\"",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-29T14:48:05.699Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: stop click was sent (narration was still active)",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: stop cancelled the active session (not synth.done)",
    "ok": true,
    "detail": "{\"event\":\"synth.cancelled\",\"at\":\"2026-09-29T14:48:05.850Z\",\"voiceId\":\"vale\",\"bytes\":3840}"
  },
  {
    "step": "full: no error toast after the cancel attempt",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: cancel test: throwaway line removed",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: voice preview for eric synthesizes",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:48:07.308Z\",\"voiceId\":\"eric\",\"bytes\":130560,\"frames\":7}"
  },
  {
    "step": "full: preview delivered audio bytes",
    "ok": true,
    "detail": "bytes=130560"
  },
  {
    "step": "full: Node WS client is generating (holds the GPU)",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: narration is queued while the Node client holds the GPU",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-29T14:48:07.700Z\",\"voiceId\":\"vale\",\"type\":\"queued\"}"
  },
  {
    "step": "full: queued narration completes once the GPU frees up",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:48:49.206Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
  },
  {
    "step": "full: cfg_scale 4: synth.request logged",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-29T14:48:49.693Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":4,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 4: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:49:22.588Z\",\"voiceId\":\"vale\",\"bytes\":3479040,\"frames\":48}"
  },
  {
    "step": "full: cfg_scale 4: no direction/tag active (hasInstruction false)",
    "ok": true,
    "detail": "hasInstruction=false"
  },
  {
    "step": "full: cfg_scale 4 reached the request",
    "ok": true,
    "detail": "cfgScale=4"
  },
  {
    "step": "full: cfg_scale 7.5: synth.request logged",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-29T14:49:23.690Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":7.5,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 7.5: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:49:50.006Z\",\"voiceId\":\"vale\",\"bytes\":2883840,\"frames\":42}"
  },
  {
    "step": "full: cfg_scale 7.5: no direction/tag active (hasInstruction false)",
    "ok": true,
    "detail": "hasInstruction=false"
  },
  {
    "step": "full: cfg_scale 7.5 reached the request",
    "ok": true,
    "detail": "cfgScale=7.5"
  },
  {
    "step": "full: cfg_scale 1: synth.request logged",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-29T14:49:50.689Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":1,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 1: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-29T14:50:18.853Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
  },
  {
    "step": "full: cfg_scale 1: no direction/tag active (hasInstruction false)",
    "ok": true,
    "detail": "hasInstruction=false"
  },
  {
    "step": "full: cfg_scale 1 reached the request",
    "ok": true,
    "detail": "cfgScale=1"
  },
  {
    "step": "full: settings restored and verified on disk",
    "ok": true,
    "detail": ""
  }
]
```
