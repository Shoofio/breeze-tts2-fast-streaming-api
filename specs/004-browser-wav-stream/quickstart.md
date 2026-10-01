# Quickstart: Validating Browser-Playable WAV Streaming

**Feature**: [spec.md](spec.md) | **Contract**: [contracts/http-wav-stream.md](contracts/http-wav-stream.md)

The live gate for 004 replaces the skipped browser spike. Results go in
`research/live-004.md`, one line per check, each marked pass or fail with any notes.

## Prerequisites

- The branch is built and the server is running on 8080:
  `scripts/start_breeze.sh --cors http://127.0.0.1:8000`.
  - Stop 2.0.0 first; 8080 must be free.
  - Keep its stdout, which is the JSON event log.
- SillyTavern is on `http://127.0.0.1:8000`, with Chrome and Firefox available.
- A saved voice, for example `Eric01`. Never delete `Eric01` or `Vale01` (the voices 003 calls `eric` and `vale`).
- `BASE=http://127.0.0.1:8080`

## 1. Wire format (curl)

```bash
curl -sN -D - "$BASE/v1/audio/speech.wav?voice_id=Eric01&seed=7&text=Hello%20there." -o /tmp/h.wav
od -A d -t x1 -N 44 /tmp/h.wav   # xxd is not installed in WSL
```

Expected:
- `200` with `audio/wav`, `Accept-Ranges: none` and `X-Breeze-Version: 2.1.0`.
- Header bytes as in the contract table, with `ff ff ff ff` at offsets 4 and 40.
- `curl -H 'Range: bytes=0-'` still gives `200`.
- `?voice_id=nope` gives the same `404 unknown_voice` body as the POST route.

**Time to first audio (SC-001):** run the same text 3 times through each route and compare the
medians. They must be within 10% of each other.

```bash
for i in 1 2 3; do curl -s -o /dev/null -w '%{time_starttransfer}\n' "$BASE/v1/audio/speech.wav?voice_id=Eric01&seed=7&text=Hello%20there."; done
for i in 1 2 3; do curl -s -o /dev/null -w '%{time_starttransfer}\n' -X POST "$BASE/v1/audio/speech" -d voice_id=Eric01 -d seed=7 --data-urlencode 'text=Hello there.'; done
```

## 2. Browser playback (SC-002, SC-003)

In **Chrome**, then **Firefox**, open the SillyTavern page and run this in the devtools console:

```js
a = new Audio(`http://127.0.0.1:8080/v1/audio/speech.wav?voice_id=Eric01&seed=7&text=${encodeURIComponent(LONG)}`);
a.playbackRate = 1; a.play();
```

`LONG` is text that produces more than 3 minutes of audio. Repeat at `playbackRate` 0.5 and 0.9.

Expected:
- Audio starts before generation ends.
- No gaps.
- Plays to the end.
- The event log shows one `speech.generated` well before playback ends.
- Exactly one `speech.completed format=wav` per `play()`.

Next, set `a.src = ''` during the first seconds of generation. Expected: `speech.aborted
reason=client_disconnect` within one chunk, and a new `play()` starts at once.

## 3. GPU sharing (SC-004, SC-005)

- **Preview during a drain:** play a long message at 0.5×. After `speech.generated`, use a
  SillyTavern voice preview (WebSocket). It must start at once, with no `queued`.
- **Busy wait:** start a long WebSocket narration, then run a GET with `curl`. The GET waits and
  then streams, and never gets `409`. With the GPU held for more than 60 s, it gets `503
  busy_timeout`.
- **Disconnect while queued (FR-007):** while the GPU is held, start a GET with `curl` and press
  Ctrl-C before it gets audio. The event log shows `speech.aborted reason=client_disconnect
  queued=true`. When the GPU frees, the next request starts at once.

## 4. Long text (SC-006)

- A GET with 10,000 Chinese characters (about 90 KB URL) streams to completion.
- A GET with 10,000 English characters streams to completion.

## 5. No regression (SC-008)

```bash
.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080
```

Time to first audio and real-time factor must be within 10% of the 2.0.0 numbers in
`specs/003-cpp-compatible-api/research/`.

Then run the extension's phase `node tests/live/sillytavern/run.mjs full`, which must still pass
unchanged.

## Automated suites

```bash
.venv/bin/ruff check . && .venv/bin/pytest
BREEZE_MODEL=<checkpoint dir> .venv/bin/pytest -m gpu
```

Without `BREEZE_MODEL`, every GPU test skips and still exits 0.
