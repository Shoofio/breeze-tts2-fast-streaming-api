# Live SillyTavern run: full

Generated 2026-09-27T01:32:24.499Z. 35/36 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| ok | health: provider reports "TTS Provider Loaded" |  |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-27T01:29:24.466Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":8081} |
| ok | health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed) | {"event":"voices.refreshed","at":"2026-09-27T01:29:24.475Z","count":2} |
| ok | health: no error toast after loading the provider |  |
| ok | health: no CORS errors in the page console |  |
| FAIL | voices: voices phase aborted | page.click: Timeout 30000ms exceeded.
Call log:
[2m  - waiting for locator('dialog[open] .popup-button-ok')[22m
[2m    - locator resolved to <div autofocus="" tabindex="0" data-i18n="OK" data-result="1" class="popup-button-ok menu_button result-control menu_button_default interactable">OK</div>[22m
[2m  - attempting click action[22m
[2m    - waiting for element to be visible, enabled and stable[22m
[2m  - element was detached from the DOM, retrying[22m
 |
| ok | full: "Breeze validation" chat open under Seraphina |  |
| ok | full: synth.request logged for the narration | {"event":"synth.request","at":"2026-09-27T01:29:55.508Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":1,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: synth.event "started" logged | {"event":"synth.event","at":"2026-09-27T01:29:55.513Z","voiceId":"vale","type":"started"} |
| ok | full: synth.done logged, audio delivered | {"event":"synth.done","at":"2026-09-27T01:30:23.443Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: audio bytes were delivered | bytes=3663360 frames=50 |
| ok | full: cancel test: throwaway line added |  |
| ok | full: narration (for the cancel test) reaches "started" | {"event":"synth.event","at":"2026-09-27T01:30:25.510Z","voiceId":"vale","type":"started"} |
| ok | full: stop click was sent (narration was still active) |  |
| ok | full: stop cancelled the active session (not synth.done) | {"event":"synth.cancelled","at":"2026-09-27T01:30:25.774Z","voiceId":"vale","bytes":11520} |
| ok | full: no error toast after the cancel attempt |  |
| ok | full: cancel test: throwaway line removed |  |
| ok | full: voice preview for eric synthesizes | {"event":"synth.done","at":"2026-09-27T01:30:27.148Z","voiceId":"eric","bytes":130560,"frames":7} |
| ok | full: preview delivered audio bytes | bytes=130560 |
| ok | full: Node WS client is generating (holds the GPU) |  |
| ok | full: narration is queued while the Node client holds the GPU | {"event":"synth.event","at":"2026-09-27T01:30:28.513Z","voiceId":"vale","type":"queued"} |
| ok | full: queued narration completes once the GPU frees up | {"event":"synth.done","at":"2026-09-27T01:30:56.512Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: cfg_scale 4: synth.request logged | {"event":"synth.request","at":"2026-09-27T01:30:57.505Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":4,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 4: synth.done logged | {"event":"synth.done","at":"2026-09-27T01:31:28.532Z","voiceId":"vale","bytes":3479040,"frames":48} |
| ok | full: cfg_scale 4: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 4 reached the request | cfgScale=4 |
| ok | full: cfg_scale 7.5: synth.request logged | {"event":"synth.request","at":"2026-09-27T01:31:29.503Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":7.5,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 7.5: synth.done logged | {"event":"synth.done","at":"2026-09-27T01:31:55.119Z","voiceId":"vale","bytes":2883840,"frames":42} |
| ok | full: cfg_scale 7.5: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 7.5 reached the request | cfgScale=7.5 |
| ok | full: cfg_scale 1: synth.request logged | {"event":"synth.request","at":"2026-09-27T01:31:56.501Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":1,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 1: synth.done logged | {"event":"synth.done","at":"2026-09-27T01:32:24.162Z","voiceId":"vale","bytes":3663360,"frames":50} |
| ok | full: cfg_scale 1: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 1 reached the request | cfgScale=1 |
| ok | full: settings restored and verified on disk |  |

## Breeze settings in effect (scalars only; secrets redacted)

The complete snapshot — every TTS provider's settings, not just Breeze's — was written to `/tmp/breeze-live-full-settings-snapshot.json` (untracked; not this record). Restore from there by hand if the automatic restore failed.

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
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-27T01:29:24.466Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":8081}"
  },
  {
    "step": "health: checkReady() completes end to end (voices.refreshed, not breeze.check_failed)",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-27T01:29:24.475Z\",\"count\":2}"
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
    "step": "voices: voices phase aborted",
    "ok": false,
    "detail": "page.click: Timeout 30000ms exceeded.\nCall log:\n\u001b[2m  - waiting for locator('dialog[open] .popup-button-ok')\u001b[22m\n\u001b[2m    - locator resolved to <div autofocus=\"\" tabindex=\"0\" data-i18n=\"OK\" data-result=\"1\" class=\"popup-button-ok menu_button result-control menu_button_default interactable\">OK</div>\u001b[22m\n\u001b[2m  - attempting click action\u001b[22m\n\u001b[2m    - waiting for element to be visible, enabled and stable\u001b[22m\n\u001b[2m  - element was detached from the DOM, retrying\u001b[22m\n"
  },
  {
    "step": "full: \"Breeze validation\" chat open under Seraphina",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: synth.request logged for the narration",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-27T01:29:55.508Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":1,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: synth.event \"started\" logged",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-27T01:29:55.513Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: synth.done logged, audio delivered",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:30:23.443Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
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
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-27T01:30:25.510Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: stop click was sent (narration was still active)",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: stop cancelled the active session (not synth.done)",
    "ok": true,
    "detail": "{\"event\":\"synth.cancelled\",\"at\":\"2026-09-27T01:30:25.774Z\",\"voiceId\":\"vale\",\"bytes\":11520}"
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
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:30:27.148Z\",\"voiceId\":\"eric\",\"bytes\":130560,\"frames\":7}"
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
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-27T01:30:28.513Z\",\"voiceId\":\"vale\",\"type\":\"queued\"}"
  },
  {
    "step": "full: queued narration completes once the GPU frees up",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:30:56.512Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
  },
  {
    "step": "full: cfg_scale 4: synth.request logged",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-27T01:30:57.505Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":4,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 4: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:31:28.532Z\",\"voiceId\":\"vale\",\"bytes\":3479040,\"frames\":48}"
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
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-27T01:31:29.503Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":7.5,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 7.5: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:31:55.119Z\",\"voiceId\":\"vale\",\"bytes\":2883840,\"frames\":42}"
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
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-27T01:31:56.501Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":1,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 1: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-27T01:32:24.162Z\",\"voiceId\":\"vale\",\"bytes\":3663360,\"frames\":50}"
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
