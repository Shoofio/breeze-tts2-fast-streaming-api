# Live record: Phase 0 (T011), SillyTavern against the C++ server

## Summary

- **Server**: C++ `breeze-server` from `<Breeze-TTS-2.cpp checkout>` at `edb927c` (branch
  `server-cors`), plus the uncommitted `ws_api.cpp` drain fix. Started with its own
  `start_breeze.sh`: `breeze-tts-2-q4_k.gguf --voices-dir voices --cors --host 0.0.0.0 --port 8080`.
- **Harness**: `node tests/live/sillytavern/run.mjs full --record phase0` at `0dbfad6`.
  `speech` is not part of `full` (tasks.md T011 asked to skip it; that was already the case).
- **Result**: 43/44 steps pass, which makes this the reference behavior. The one failure is a
  harness read race, not server behavior: the health phase re-read `#tts_status` after SillyTavern
  had already replaced "TTS Provider Loaded" with "Successfully applied settings". The
  `breeze.health` check in the same run passed. It is fixed in `health.mjs` (the step records
  `selectBreezeProvider`'s own wait).
- **Reference timings**: narration of the validation chat's last message (`vale`, cfg 6, buffer
  mode) had 1,605,120 bytes (33.4 s of audio) in 24 s. The cfg 4, 7.5 and 1 narrations took
  37–40 s each. Voice preview for `eric` took about 2 s. With the Node client holding the GPU, the page's
  narration reported `queued` and finished once the GPU was free.
- **Earlier runs**:
  - Run 1 (harness at `9a21a21`) got 39/40. `synth.done` at cfg 4 took more than the old 60 s
    timeout, since the C++ Q4 model runs near real time on long narrations; the timeout is now
    180 s.
  - Run 2 got 38/40, failing health on an event-timing bug (review pass 2 #2, fixed).
  - Both runs left `direction_enabled` and `vocal_events_enabled` off in the user's SillyTavern
    settings, because the debounced save was lost on browser close (review pass 2 #1). They were
    restored through the page (`saveSettings()`) and checked against the 2026-09-24 19:54 UTC
    backup. The harness now saves its restore explicitly, and run 3 left the settings intact.

## Review loop outcomes (Phase 0 / Phase 1 commits)

### T003 (720719d → fixes 76cb493, 156b444)
- P1#1 websockets>=15 unbounded / py3.10 → SPEC ISSUE raised to user
- P1#2 transitive deps float, httptools → comment; http="h11" deferred to T022
- P1#3 gradio vs exact starlette pin → accepted risk
- P1#4 quickstart `uv sync` doesn't install requirements → SPEC ISSUE raised to user
- P1#5 httpx test dep in prod requirements → accepted (existing pytest/ruff pattern)
- P1#6 smoke check → fixed 76cb493
- P2#1 comment named non-existent test → fixed 156b444
- P2#2 versions duplicated in smoke_check → declined (deliberate, simple)
- P2#3/#4 imports in smoke check → fixed 156b444
- P2#5 constraints file/uv.lock → accepted risk (same as P1#2)
- P2#6 h11 → T022
- P2#7 comment dup → fixed
### T001 bench (f0eca3e → 70f7b98)
- P1: 10 findings, all accepted and fixed (quickstart --api default, lazy ref, MEDIUM text, truncated label, 409 retry, per-run values, bad headers, voice cleanup, input validation, tests). Main session also: warm-up uses retry; docstring BC-02 miscitation fixed.
### Harness T004-T010 (5724c1b)
- Main-session fix before commit: voices composition runs health first.
- P1: 10 findings, all accepted → sent back to Sonnet (provider wait, queued via 2nd client, idle stop, per-step catch, voices.refreshed, replace assertion, restore settings, health event filter, pinned cfg settings, event ordering).
- tasks.md inconsistency: T011 "skip speech" vs T010 full composition (no speech) — no-op, noted.
### T001 pass 2 — 10 findings all accepted → agent fixing (voice ownership via GET, empty cases, sr<=0, register retry, cleanup scope, truncated on any mid-stream error, aligned lists, simplify, test literal, ref_audio Path)
### T002/T012 pass 1 (44cc760, ab8fbe8)
- Baseline #1-5,#8 accepted: re-record with --warmup 3, 10 runs, versions recorded.
- #6/#7 → USER DECISION 2026-09-24: SC-007 = 10 runs, 3 warm-ups, 10-run vs 10-run medians; add medium_inline like-for-like case; gate on short_design, short_inline, medium_inline; medium_design reported only. Update tasks.md T043/T054/T082 + quickstart.
- #9 fixed 8d75f92; #10 message overstatement, api.py MAX_NEW_TOKENS removed in T022.

### Harness pass 2 (0dbfad6)
- 10 findings, all accepted and fixed: settings saved to disk, health timing, setup inside the try,
  real cancel test, GPU-hold timeout, pinned instruction sources, URL normalization, handle
  disposal, 60 s replace waits. Main-session addition: the throwaway line is deleted only while it
  is still the last message.
### T013 events (9477a03 → 0ce1790; pass 2 fixes in progress)
- Pass 1: 10 findings accepted (reserved keys, ASCII, isolation of telemetry failures, lock,
  positional-only name, level field, tests, docstring). #5 (synchronous sink) documented as a
  known risk.
- Pass 2: 10 findings accepted (fallback keeps the schema, ValueError on a closed stream, numpy-only
  hook with catch-all, clock inside the lock, RLock, dead reserved key, tests).
### T014/T015 errors and body limit (f2ffc77; fixes in progress)
- 9 findings accepted, including 2 reproduced bugs: a mid-stream 413 was rewrapped as a 400 on
  Form routes, and Starlette multipart limits gave a non-catalog code. The "http_error" code was
  replaced by a fixed status table.
### T016 settings (e06d4c6; fixes in progress)
- 10 findings accepted (bare-origin validation, messages, model_path check, no dataclass defaults,
  strict ports, tests). The empty --cors rule is now recorded in data-model.md.
### T024 launch scripts (7cbe64b; fixes in progress)
- 8 findings accepted. #1: the scripts pass flags the old parser lacks, so they only work once
  T022 lands; the README marks those rows as pending.
### T025 version header (3465ce3; fixes in progress)
- 9 findings accepted. The middleware wasn't wired in, so T025 is unmarked until T022. Spec
  change: the version header is outermost, so CORS preflights and 403s carry it (tasks.md T025
  and T028).
### Spec changes decided with the user (2026-09-24)
- SC-007 method: 10 runs, 3 warm-ups, 10-run medians, and the medium_inline gating case.
- websockets>=17.1,<18 and README Python 3.11; the quickstart installs with `uv pip install -r`.
- The stray `fg` line is removed from start_breeze.ps1.
- R11's golden-change counts are corrected to what was measured (1 split, 3 drain).
- The working branch is enhanced-api (the spec said perf-and-fixes).

## Raw record (run 3, written by run.mjs)

Generated 2026-09-25T02:28:43.425Z. 43/44 steps passed.

| Result | Step | Detail |
| --- | --- | --- |
| ok | health: SillyTavern loaded with the Breeze provider registered |  |
| FAIL | health: provider reports "TTS Provider Loaded" | Successfully applied settings |
| ok | health: breeze.health (not breeze.check_failed) logged for this run's httpUrl | {"event":"breeze.health","at":"2026-09-25T02:25:45.268Z","httpUrl":"http://127.0.0.1:8080","status":"ok","sample_rate":24000,"ws_port":8081} |
| ok | health: no error toast after loading the provider |  |
| ok | health: no CORS errors in the page console |  |
| ok | voices: st_live_tmp uploaded (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-25T02:25:46.251Z","voiceId":"st_live_tmp","seconds":10.24,"saved":true} |
| ok | voices: voice list refreshed after upload | {"event":"voices.refreshed","at":"2026-09-25T02:25:46.297Z","count":4} |
| ok | voices: st_live_tmp appears in the voice list |  |
| ok | voices: st_live_tmp replaced (voice.replaced event) | {"event":"voice.replaced","at":"2026-09-25T02:25:48.690Z","voiceId":"st_live_tmp"} |
| ok | voices: st_live_tmp re-uploaded after replace (voice.uploaded event) | {"event":"voice.uploaded","at":"2026-09-25T02:25:49.163Z","voiceId":"st_live_tmp","seconds":10.24,"saved":true} |
| ok | voices: replace did not fail (no voice.upload_failed) | null |
| ok | voices: voice list refreshed after replace | {"event":"voices.refreshed","at":"2026-09-25T02:25:49.210Z","count":4} |
| ok | voices: st_live_tmp deleted (voice.deleted event) | {"event":"voice.deleted","at":"2026-09-25T02:25:49.372Z","voiceId":"st_live_tmp","orphaned":[]} |
| ok | voices: voice list refreshed after delete | {"event":"voices.refreshed","at":"2026-09-25T02:25:49.374Z","count":4} |
| ok | voices: st_live_tmp no longer in the voice list |  |
| ok | voices: no error toast across the upload/replace/delete run | Breeze TTSVoice "st_live_tmp" removed.Breeze TTSVoice "st_live_tmp" saved on the Breeze server.Breeze TTSVoice "st_live_tmp" saved on the Breeze server. |
| ok | voices: eric is still listed | breeze_probe — saved, 10.2 seric — saved, 10.2 svale — saved, 8.3 sList refreshed 9/24/2026, 10:25:49 PM |
| ok | voices: vale is still listed | breeze_probe — saved, 10.2 seric — saved, 10.2 svale — saved, 8.3 sList refreshed 9/24/2026, 10:25:49 PM |
| ok | full: "Breeze validation" chat open under Seraphina |  |
| ok | full: synth.request logged for the narration | {"event":"synth.request","at":"2026-09-25T02:25:50.374Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":6,"hasInstruction":true,"hasTag":false,"instruction":"Gentle, airy, and slow; deliver with intimate warmth and a soothing, compassionate intensity","text":"\"Ah, you're awake at last. I was so worried, I found you bloodied and unconsciou","mode":"buffer"} |
| ok | full: synth.event "started" logged | {"event":"synth.event","at":"2026-09-25T02:25:50.378Z","voiceId":"vale","type":"started"} |
| ok | full: synth.done logged, audio delivered | {"event":"synth.done","at":"2026-09-25T02:26:14.406Z","voiceId":"vale","bytes":1605120,"frames":20} |
| ok | full: audio bytes were delivered | bytes=1605120 frames=20 |
| ok | full: narration (for the cancel test) reaches "started" | {"event":"synth.event","at":"2026-09-25T02:26:17.639Z","voiceId":"vale","type":"started"} |
| ok | full: stop click was sent (narration was still active) |  |
| ok | full: stop cancelled the active session (not synth.done) | {"event":"synth.cancelled","at":"2026-09-25T02:26:18.093Z","voiceId":"vale","bytes":0} |
| ok | full: no error toast after the cancel attempt |  |
| ok | full: voice preview for eric synthesizes | {"event":"synth.done","at":"2026-09-25T02:26:19.954Z","voiceId":"eric","bytes":111360,"frames":4} |
| ok | full: preview delivered audio bytes | bytes=111360 |
| ok | full: Node WS client is generating (holds the GPU) |  |
| ok | full: narration is queued while the Node client holds the GPU | {"event":"synth.event","at":"2026-09-25T02:26:20.375Z","voiceId":"vale","type":"queued"} |
| ok | full: queued narration completes once the GPU frees up | {"event":"synth.done","at":"2026-09-25T02:26:45.587Z","voiceId":"vale","bytes":1605120,"frames":20} |
| ok | full: cfg_scale 4: synth.request logged | {"event":"synth.request","at":"2026-09-25T02:26:46.368Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":4,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 4: synth.done logged | {"event":"synth.done","at":"2026-09-25T02:27:24.641Z","voiceId":"vale","bytes":2680320,"frames":35} |
| ok | full: cfg_scale 4: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 4 reached the request | cfgScale=4 |
| ok | full: cfg_scale 7.5: synth.request logged | {"event":"synth.request","at":"2026-09-25T02:27:25.367Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":7.5,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 7.5: synth.done logged | {"event":"synth.done","at":"2026-09-25T02:28:02.117Z","voiceId":"vale","bytes":2542080,"frames":34} |
| ok | full: cfg_scale 7.5: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 7.5 reached the request | cfgScale=7.5 |
| ok | full: cfg_scale 1: synth.request logged | {"event":"synth.request","at":"2026-09-25T02:28:03.364Z","voiceMapKey":"Seraphina","voiceId":"vale","cfgScale":1,"hasInstruction":false,"hasTag":false,"instruction":"","text":"*You wake with a start, recalling the events that led you deep into the forest a","mode":"buffer"} |
| ok | full: cfg_scale 1: synth.done logged | {"event":"synth.done","at":"2026-09-25T02:28:43.321Z","voiceId":"vale","bytes":2860800,"frames":36} |
| ok | full: cfg_scale 1: no direction/tag active (hasInstruction false) | hasInstruction=false |
| ok | full: cfg_scale 1 reached the request | cfgScale=1 |

## Breeze settings in effect for this run

Scalar settings only: voice lists, voice maps and cached transcripts are left out as personal data.

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
  "inline_tag_keyword": "dir",
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
    "ok": false,
    "detail": "Successfully applied settings"
  },
  {
    "step": "health: breeze.health (not breeze.check_failed) logged for this run's httpUrl",
    "ok": true,
    "detail": "{\"event\":\"breeze.health\",\"at\":\"2026-09-25T02:25:45.268Z\",\"httpUrl\":\"http://127.0.0.1:8080\",\"status\":\"ok\",\"sample_rate\":24000,\"ws_port\":8081}"
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
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-25T02:25:46.251Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.24,\"saved\":true}"
  },
  {
    "step": "voices: voice list refreshed after upload",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-25T02:25:46.297Z\",\"count\":4}"
  },
  {
    "step": "voices: st_live_tmp appears in the voice list",
    "ok": true,
    "detail": ""
  },
  {
    "step": "voices: st_live_tmp replaced (voice.replaced event)",
    "ok": true,
    "detail": "{\"event\":\"voice.replaced\",\"at\":\"2026-09-25T02:25:48.690Z\",\"voiceId\":\"st_live_tmp\"}"
  },
  {
    "step": "voices: st_live_tmp re-uploaded after replace (voice.uploaded event)",
    "ok": true,
    "detail": "{\"event\":\"voice.uploaded\",\"at\":\"2026-09-25T02:25:49.163Z\",\"voiceId\":\"st_live_tmp\",\"seconds\":10.24,\"saved\":true}"
  },
  {
    "step": "voices: replace did not fail (no voice.upload_failed)",
    "ok": true,
    "detail": "null"
  },
  {
    "step": "voices: voice list refreshed after replace",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-25T02:25:49.210Z\",\"count\":4}"
  },
  {
    "step": "voices: st_live_tmp deleted (voice.deleted event)",
    "ok": true,
    "detail": "{\"event\":\"voice.deleted\",\"at\":\"2026-09-25T02:25:49.372Z\",\"voiceId\":\"st_live_tmp\",\"orphaned\":[]}"
  },
  {
    "step": "voices: voice list refreshed after delete",
    "ok": true,
    "detail": "{\"event\":\"voices.refreshed\",\"at\":\"2026-09-25T02:25:49.374Z\",\"count\":4}"
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
    "detail": "breeze_probe — saved, 10.2 seric — saved, 10.2 svale — saved, 8.3 sList refreshed 9/24/2026, 10:25:49 PM"
  },
  {
    "step": "voices: vale is still listed",
    "ok": true,
    "detail": "breeze_probe — saved, 10.2 seric — saved, 10.2 svale — saved, 8.3 sList refreshed 9/24/2026, 10:25:49 PM"
  },
  {
    "step": "full: \"Breeze validation\" chat open under Seraphina",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: synth.request logged for the narration",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-25T02:25:50.374Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":6,\"hasInstruction\":true,\"hasTag\":false,\"instruction\":\"Gentle, airy, and slow; deliver with intimate warmth and a soothing, compassionate intensity\",\"text\":\"\\\"Ah, you're awake at last. I was so worried, I found you bloodied and unconsciou\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: synth.event \"started\" logged",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-25T02:25:50.378Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: synth.done logged, audio delivered",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:26:14.406Z\",\"voiceId\":\"vale\",\"bytes\":1605120,\"frames\":20}"
  },
  {
    "step": "full: audio bytes were delivered",
    "ok": true,
    "detail": "bytes=1605120 frames=20"
  },
  {
    "step": "full: narration (for the cancel test) reaches \"started\"",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-25T02:26:17.639Z\",\"voiceId\":\"vale\",\"type\":\"started\"}"
  },
  {
    "step": "full: stop click was sent (narration was still active)",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: stop cancelled the active session (not synth.done)",
    "ok": true,
    "detail": "{\"event\":\"synth.cancelled\",\"at\":\"2026-09-25T02:26:18.093Z\",\"voiceId\":\"vale\",\"bytes\":0}"
  },
  {
    "step": "full: no error toast after the cancel attempt",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: voice preview for eric synthesizes",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:26:19.954Z\",\"voiceId\":\"eric\",\"bytes\":111360,\"frames\":4}"
  },
  {
    "step": "full: preview delivered audio bytes",
    "ok": true,
    "detail": "bytes=111360"
  },
  {
    "step": "full: Node WS client is generating (holds the GPU)",
    "ok": true,
    "detail": ""
  },
  {
    "step": "full: narration is queued while the Node client holds the GPU",
    "ok": true,
    "detail": "{\"event\":\"synth.event\",\"at\":\"2026-09-25T02:26:20.375Z\",\"voiceId\":\"vale\",\"type\":\"queued\"}"
  },
  {
    "step": "full: queued narration completes once the GPU frees up",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:26:45.587Z\",\"voiceId\":\"vale\",\"bytes\":1605120,\"frames\":20}"
  },
  {
    "step": "full: cfg_scale 4: synth.request logged",
    "ok": true,
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-25T02:26:46.368Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":4,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 4: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:27:24.641Z\",\"voiceId\":\"vale\",\"bytes\":2680320,\"frames\":35}"
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
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-25T02:27:25.367Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":7.5,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 7.5: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:28:02.117Z\",\"voiceId\":\"vale\",\"bytes\":2542080,\"frames\":34}"
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
    "detail": "{\"event\":\"synth.request\",\"at\":\"2026-09-25T02:28:03.364Z\",\"voiceMapKey\":\"Seraphina\",\"voiceId\":\"vale\",\"cfgScale\":1,\"hasInstruction\":false,\"hasTag\":false,\"instruction\":\"\",\"text\":\"*You wake with a start, recalling the events that led you deep into the forest a\",\"mode\":\"buffer\"}"
  },
  {
    "step": "full: cfg_scale 1: synth.done logged",
    "ok": true,
    "detail": "{\"event\":\"synth.done\",\"at\":\"2026-09-25T02:28:43.321Z\",\"voiceId\":\"vale\",\"bytes\":2860800,\"frames\":36}"
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
  }
]
```
