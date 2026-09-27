<div align="center">
  <a href="https://breezeblue.ai/"><img src="assets/breezeblue-logo.png" alt="BreezeBlue" width="35%"></a>
  <br><br>
  <a href="https://huggingface.co/BreezeBlue/breeze-tts-2"><img src="https://img.shields.io/badge/Hugging%20Face-breeze--tts--2-FFD21E" alt="Hugging Face"></a>
  <a href="https://breezeblue.ai/breeze-tts-2"><img src="https://img.shields.io/badge/Blog-Breeze%20TTS%202-2563EB" alt="Blog"></a>
  <a href="https://breezeblue.ai/"><img src="https://img.shields.io/badge/Website-BreezeBlue-0EA5E9" alt="Website"></a>
  <a href="https://discord.com/invite/6H7AgPe9pA"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/BreezeBlueX"><img src="https://img.shields.io/badge/X-Follow%20BreezeBlue-000000?logo=x&logoColor=white" alt="X"></a>
</div>

> [!IMPORTANT]
> Source code is licensed under Apache 2.0. Breeze TTS 2 model weights, derivative models, and self-hosted outputs are for research and non-commercial use only. See [License](#license-and-responsible-use).

## 📰 News

- **[2026.08.25]** 🎉 We open-source [Breeze TTS 2](https://huggingface.co/BreezeBlue/breeze-tts-2) model weights and the [PyTorch inference code](https://github.com/breezeblue-ai/breeze-tts).
- **[2026.08.07]** 🔥 We release the TTS benchmark suite for [voice design](https://github.com/breezeblue-ai/tts-voice-design-benchmark), [voice direction](https://github.com/breezeblue-ai/TTS-Voice-Direction-Benchmark), and [latency evaluation](https://github.com/breezeblue-ai/TTS-Latency-Benchmark).

## 📖 Introduction

Breeze TTS 2 is an open-weight text-to-speech model built for real-time interaction. It ranks #1 among open-weight models on the Artificial Analysis TTS leaderboard, while outperforming frontier proprietary systems. Its open-ended natural-language instruction-following capability supports reference-free voice design and reference-guided voice direction, while ultra-low-latency streaming enables responsive, expressive interaction.

<div align="center">
  <img src="assets/tts-elo-leaderboard.svg" alt="Text-to-speech models ranked by Artificial Analysis Elo score" width="100%">
</div>

## ✨ Highlights

- 🎙️ **Voice Clone** — Uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style.
- 🎨 **Voice Design** — Creates a distinctive voice from a natural-language description, without reference audio.
- 🎛️ **Voice Direction** — Clones a voice from reference audio while steering tone, emotion, pace, and delivery.
- 🎭 **Vocal Events** — Adds expressive inline events directly in the text: use parentheses in English, such as `(laugh)`, `(cough)`, `(clears throat)`, and `(sigh)`; use square brackets in Chinese, such as `[笑]`, `[咳嗽]`, `[清嗓子]`, and `[叹气]`.
- ⚡ **Ultra-Low Latency** — Achieves under 40 ms time to first audio (TTFA) with the warmed-up fast path on an NVIDIA H100.
- 🌊 **Real-Time Streaming** — Reaches a 0.32 real-time factor (RTF), generating audio at approximately 3.1× real time with the warmed-up fast path on an NVIDIA H100.
- 💾 **GPU-Efficient** — Eager inference uses approximately 7.7 GiB of GPU memory; a 12 GB GPU is the minimum recommended configuration.
- 🌏 **Bilingual Support** — Generates natural English and Chinese speech with a single model.

## 🚀 Quick Start

### Requirements

- Linux and Python 3.12 (tested; 3.11 is the minimum the `websockets` pin allows)
- A CUDA-capable NVIDIA GPU
- GPU memory: approximately 7.7 GiB for eager inference or 14.4 GiB with `--fast-all`; use a 12 GB GPU for eager or a 24 GB GPU for the fast path
- The Breeze TTS 2 checkpoint

### Installation

Download the inference code:

```bash
git clone https://github.com/breezeblue-ai/breeze-tts.git
cd breeze-tts
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

All required model components are included in the Breeze TTS 2 checkpoint.

For the tested CUDA environment, build the included Docker image:

```bash
bash docker/build.sh
```

The default image targets H100/Hopper (sm90). For A100:

```bash
FLASH_ATTN_CUDA_ARCHS=80 bash docker/build.sh
```

### 🎙️ Voice Clone

Clone a speaker from clean reference audio and its exact transcript.

#### English

```bash
python infer.py ../breeze-tts-2 \
  --ref-audio reference_en.wav \
  --ref-text "This is the exact transcript of the English reference audio." \
  --text "(sigh) It is good to hear your voice again after all this time." \
  --output outputs/voice_clone_en.wav
```

#### Chinese

```bash
python infer.py ../breeze-tts-2 \
  --ref-audio reference_zh.wav \
  --ref-text "这是中文参考音频的准确文字稿。" \
  --text "[叹气] 没想到过了这么久，你还记得我的声音。" \
  --output outputs/voice_clone_zh.wav
```

Reference audio should contain clean speech with minimal background noise.

### 🎨 Voice Design

Create a voice from a natural-language description without reference audio. Match the instruction language to the target text. Use `--cfg-scale 4` to strengthen instruction-following.

#### English

```bash
python infer.py ../breeze-tts-2 \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." \
  --cfg-scale 4 \
  --output outputs/voice_design_en.wav
```

#### Chinese

```bash
python infer.py ../breeze-tts-2 \
  --text "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。" \
  --cfg-scale 4 \
  --output outputs/voice_design_zh.wav
```

### 🎛️ Voice Direction

Keep the identity of a reference speaker while directing tone, emotion, pace, and delivery. Use `--cfg-scale 4` to strengthen instruction-following.

```bash
python infer.py ../breeze-tts-2 \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(clears throat) We need to discuss what happened last night." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --cfg-scale 4 \
  --output outputs/voice_direction.wav
```

## API Server

The server exposes an HTTP API and a WebSocket session endpoint compatible with the
[HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) reference server's
contract, except where listed under [Breaking Changes](#breaking-changes-from-the-c-server) below.
Contract version `2.0.0`. This section is the user-facing reference; the machine-readable source
is `specs/003-cpp-compatible-api/contracts/http-api.md` and `contracts/ws-api.md`.

Start it the same way as the CLI, pointing at the checkpoint directory:

```bash
python -m breeze_infer.api ../breeze-tts-2 --host 0.0.0.0 --port 8080 --fast-all
```

The HTTP API listens on `--port` (default `8080`). A WebSocket session endpoint listens on
`--ws-port` on the same host (default: the HTTP port + 1, so `8081`). Both bind only to `--host`;
the WebSocket never widens to all interfaces regardless of what `--host` resolves to.
`GET /health` reports which WebSocket port is actually listening, or `0` when it is disabled or
failed to bind.

### Launch options

| Option | Default | Meaning |
| --- | --- | --- |
| `model_path` (positional) | required | Path to the model checkpoint directory |
| `--host HOST` | `127.0.0.1` | Bind address for HTTP and WebSocket |
| `--port N` | `8080` | HTTP port, `1`–`65535` |
| `--ws-port N\|disabled` | HTTP port + 1 | WebSocket port, `1`–`65535`, must differ from `--port`; `disabled` turns the WebSocket server off |
| `--cors [ORIGINS]` | disabled | Enable CORS. Bare flag (or `--cors` with no value) allows any origin (`*`); `--cors=http://a,http://b` sets a comma-separated allowlist. Entries are trimmed, canonicalized (lowercase scheme/host, IDN hosts in punycode, IP literals canonical, default port dropped) and deduplicated; `*` cannot be mixed with other entries |
| `--split-chars N` | `600` | Default text-splitting budget for `POST /v1/audio/speech` and WebSocket `start`; `0` disables length splitting |
| `--chunk-first N` | `1` | Codec frames (1,920 samples each) in the first streamed chunk |
| `--chunk-max N` | `25` | Codec frames the stream ramps up to |
| `--voices-dir PATH` | `./voices` | Directory holding saved voice files |
| `--fast-all` / `--no-fast-all` | unset | Enable or disable every fast-path stage at once |
| `--fast-text-encoder` / `--no-fast-text-encoder` | off | Fast path for the text encoder stage |
| `--fast-backbone-prefill` / `--no-fast-backbone-prefill` | off | Fast path for backbone prefill |
| `--fast-backbone-decode` / `--no-fast-backbone-decode` | off | Fast path for backbone decode |
| `--fast-depth-decoder` / `--no-fast-depth-decoder` | off | Fast path for the depth decoder |
| `--fast-codec` / `--no-fast-codec` | off | Fast path for the codec |
| `--attn-implementation {eager,sdpa}` | `eager` | Attention kernel for the backbone and text encoder (the fast text-encoder stage always runs `sdpa` regardless of this setting) |
| `--compile-cache-dir PATH` | `$TORCHINDUCTOR_CACHE_DIR` if set, else `./.cache/torchinductor` | Where `torch.compile` artifacts persist between starts |

`--port`, `--ws-port`, `--split-chars`, `--chunk-first` and `--chunk-max` must be plain unsigned
integers (no leading `+`/`-`, no decimal point, no exponent). See
[Fast Inference Options](#-fast-inference-options) above for what each fast-path stage changes.

### HTTP API

**Transport and body**: HTTP/1.1. `POST /v1/audio/speech` and `POST /v1/voices` accept
`multipart/form-data` or `application/x-www-form-urlencoded` bodies; JSON bodies are not accepted.
Fields may also come from the query string. A field present more than once, anywhere, gets
`400 duplicate_field`. An empty field value means absent, so its default applies. A body over
26 MiB gets `413 payload_too_large`, whether or not `Content-Length` is set.

**Version header**: every response — including errors, preflights and streamed speech — carries
`X-Breeze-Version: 2.0.0` (the running server's version; a development build reports
`2.0.0.devN`, plus a `+M` local label on a within-phase re-deploy), so clients can pin the
contract version.

**Errors**: every error has body `{"error": "<message>", "code": "<code>"}` and
`Content-Type: application/json`. This includes `404`, `405` (with an `Allow` header), `413` and
`500`. Every `500` closes the connection.

**Validation order**: (1) body size (`413`); (2) field syntax and ranges (`400`); (3) reference
consistency (`400`); (4) unknown `voice_id` (`404`); (5) audio decode and limits (`400`); (6) busy
(`409`).

**Busy and loading**: only one generation or voice encode runs at a time. HTTP never waits: a
request made while busy gets `409 {"error":"busy","code":"busy"}` at once — this includes while a
WebSocket piece is running or queued. Until the model is ready, every route returns
`503 {"status":"loading","error":"model is loading","code":"loading"}`. Once the GPU has stopped
responding (a generation's cleanup ran past 30 s), every route returns
`503 {"status":"error","error":"gpu is not responding","code":"gpu_unavailable"}` until the server
is restarted.

#### CORS

Off by default: no CORS headers are sent, and every `OPTIONS` request gets
`405 method_not_allowed`. Enable with `--cors` (any origin) or `--cors=<allowlist>`.

| Request | Response |
| --- | --- |
| Any response to an allowed `Origin` | `Access-Control-Allow-Origin: <origin or *>` and `Access-Control-Expose-Headers: X-Sample-Rate, X-Sample-Format, X-Breeze-Version`. Applies to errors, `500`s and streams |
| Any response in allowlist mode | `Vary: Origin` |
| `OPTIONS` preflight on an existing path | `204`, `Access-Control-Allow-Methods: <that route's methods>, OPTIONS`, `Access-Control-Allow-Headers: <echo of the request>`, `Access-Control-Max-Age: 86400` |
| `OPTIONS` preflight on an unknown path | `404 not_found` |
| `OPTIONS` preflight asking for a method the route lacks | `405 method_not_allowed` |
| `OPTIONS` preflight from a disallowed origin | `403 origin_not_allowed` |

Independent of the CORS setting: a `POST` or `DELETE` whose `Origin` header is present and not
allowed gets `403 origin_not_allowed` before any work is done. With CORS off, no origin is allowed.
Requests without `Origin` are unaffected.

#### `GET /health`

`200` with exactly `{"status":"ok","sample_rate":24000,"ws_port":8081}` and no other fields.
`sample_rate` is the loaded model's rate; `ws_port` is the port actually listening, or `0` when the
WebSocket is disabled or failed to bind.

`503` with the loading body while the model loads, and with the `gpu_unavailable` body once the
GPU has stopped responding (it stays so until restart, so a supervisor can restart the server).
`HEAD` is supported.

#### `POST /v1/audio/speech`

| Field | Type | Default | Valid |
| --- | --- | --- | --- |
| `text` | string | required | Non-blank; ≤ 10,000 characters; no control characters except TAB, CR and LF |
| `instruction` | string | `Speak clearly and naturally.` | ≤ 2,000 characters; blank → default; control-character rule |
| `voice_id` | string | — | A registered id |
| `ref_audio` | file part | — | WAV, WAVEX, FLAC or OGG; 1–8 channels; 8–192 kHz; ≤ 25 MiB; ≤ 30 s; at least 80 ms, one full codec frame |
| `ref_text` | string | — | ≤ 2,000 characters; control-character rule |
| `cfg_scale` | decimal | `1.0` | Finite, 0–100. `1` = no CFG; `0` = unconditional only |
| `seed` | integer | `42` | 0–4294967295 |
| `temperature` | decimal | `0` = model default | 0 or (0, 10] |
| `top_k` | integer | `0` = model default | 0–10,000 |
| `top_p` | decimal | `0` = model default | 0 or (0, 1] |
| `repetition_penalty` | decimal | `0` = model default | 0 or [0.0001, 10] (below 1e-4 the logits overflow) |
| `max_new_tokens` | integer | `0` = model default (750) | 0–1,500 |
| `split_chars` | integer | server `--split-chars` (600) | 0–10,000. `0` = no length splitting |

Integers match `^[+-]?\d+$`; decimals match `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`. Anything
else is `400 invalid_field`.

**Reference rules**:

| Fields sent | Result |
| --- | --- |
| `voice_id` and `ref_audio` | `400 reference_conflict` |
| `ref_audio` without `ref_text` | `400 ref_text_required` |
| `ref_text` without `ref_audio` or `voice_id` | `400 reference_required` |
| `voice_id` and `ref_text` | The given text overrides the voice's stored transcript for this request |

**Splitting**: long text is split with the shared segmenter, using `budget = split_chars`. With no
reference, the first piece gets a soft opening budget of 200 (or `split_chars`, if that is
smaller); its generated audio and text then become the reference for every later piece. Anchoring
is skipped, and the later pieces stay voice design, when the first piece was truncated, or when
the anchor would leave any later piece a smaller effective frame limit than it would have without
the anchor. Piece `i` is generated with seed `(seed + i) mod 2^32`.

**`200` response**: headers `Content-Type: audio/pcm`, `X-Sample-Rate: 24000`,
`X-Sample-Format: s16le`, `Cache-Control: no-store`, `Transfer-Encoding: chunked`. Body: headerless
mono signed 16-bit little-endian PCM. Chunks grow from `--chunk-first` to `--chunk-max` codec
frames (1,920 samples per frame). The response is sent only after validation, reference
preparation and the first audio chunk have all succeeded.

**Failure after streaming starts**: the connection is closed without the chunked terminator, so
clients see an incomplete transfer. This covers a later piece with no room left and generation
errors. Normal endings: reaching `max_new_tokens`, or a piece clamped to the room left (recorded as
an event).

**Client disconnect**: generation stops within one chunk, and the GPU is released.

**Errors**:

| Status | Code | Message |
| --- | --- | --- |
| 400 | `invalid_field` | `<field> must be <rule>`, e.g. numbers "an integer"/"a number"; control characters "free of control characters"; `cfg_scale` "finite and between 0 and 100"; `seed` "an integer between 0 and 4294967295"; `temperature` "0, or greater than 0 and at most 10"; `top_k` "0, or an integer between 1 and 10,000"; `top_p` "0, or greater than 0 and at most 1"; `repetition_penalty` "0, or between 0.0001 and 10"; `max_new_tokens` "0, or an integer between 1 and 1,500"; `split_chars` "an integer between 0 and 10,000"; `instruction`/`ref_text` "at most 2,000 characters"; `voice_id` "a voice name or v_ id" |
| 400 | `invalid_field` | `content type must be multipart/form-data or application/x-www-form-urlencoded` |
| 400 | `duplicate_field` | `<field> was given more than once` |
| 400 | `text_required` | `text is required` |
| 400 | `text_too_long` | `text is too long` (over the limit, or the first piece doesn't fit the context; for a `voice_id` without `ref_text`, the first piece must fit with the voice's cached prefix; with `ref_text`, it must fit with the voice's codes) |
| 400 | `reference_conflict` | `voice_id and ref_audio cannot be used together` |
| 400 | `ref_text_required` | `ref_text is required with ref_audio` |
| 400 | `reference_required` | `ref_text needs ref_audio or voice_id` |
| 400 | `invalid_audio` | `could not read ref_audio` |
| 400 | `audio_too_long` | `ref_audio is longer than 30 seconds` |
| 400 | `audio_too_short` | `ref_audio is too short` |
| 404 | `unknown_voice` | `unknown voice_id` |
| 409 | `busy` | `busy` |
| 413 | `payload_too_large` | `request body is too large` |
| 503 | `gpu_out_of_memory` | `not enough GPU memory for this voice right now` (a `voice_id` without `ref_text` whose prefix ran out of GPU memory while building, and the fallback doesn't fit either; retrying later may succeed) |

Example:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -F "cfg_scale=4" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=This is the exact transcript of the reference audio." \
  -F "text=(clears throat) We need to discuss what happened last night." \
  -F "instruction=Speak slowly with a restrained, serious tone." \
  -F "seed=42" \
  --output voice_direction.pcm
```

The response is streaming mono 24 kHz signed 16-bit little-endian PCM.

#### `POST /v1/voices`

| Field | Type | Valid |
| --- | --- | --- |
| `ref_audio` | file part, required | Same rules as speech |
| `ref_text` | string, required | Non-blank; ≤ 2,000 characters; control-character rule |
| `name` | string, optional | `[A-Za-z0-9_-]{1,64}`; must not start with `v_` |

**Order of checks**: body size; `ref_text` syntax → `400 invalid_field`; missing `ref_audio` or
`ref_text` → `400 voice_fields_required`; bad name → `400 invalid_name`; named voice, name already
taken (ignoring case) → `409 voice_exists`; unnamed voice whose computed id already exists → `200`
with the existing entry, without checking busy; decode and limits (`400`), then the reference
prefix the voice would build → `400 voice_too_long` if it leaves fewer than the minimum suffix room
(an unnamed voice registered meanwhile by an identical request returns `200` with that entry); busy
(`409`); encode (`500 internal_error` if the codes no voice can hold, `400 voice_too_long` if a
re-sized frame count no longer fits); named voice: write the file (`409 voice_exists` on a name
clash found at commit, `500 voice_write_failed` on a write failure).

Unnamed ids are `v_` followed by 16 lowercase hex characters, derived deterministically from the
audio bytes and the transcript. Unnamed voices live in memory only, at most 64 of them, and the
oldest unnamed voice is evicted first; saved voices don't count toward the cap.

**`200` response**: `{"id":"alice","frames":375,"seconds":30.00,"encode_ms":812,"saved":true,"ref_text":"…"}`.
`seconds` has 2 decimals; `encode_ms` is an integer. `200`, not `201`, is kept from C++.

**Errors**:

| Status | Code | Message |
| --- | --- | --- |
| 400 | `invalid_field` | `ref_text` over 2,000 characters or with control characters |
| 400 | `voice_fields_required` | `ref_audio and ref_text are required` |
| 400 | `invalid_name` | `name can only use letters, digits, dash and underscore` (also used for the `v_` prefix) |
| 400 | `invalid_audio`, `audio_too_long`, `audio_too_short` | as speech |
| 400 | `voice_too_long` | `the reference is too long for the model's context` |
| 409 | `voice_exists` | `voice already exists` |
| 409 | `busy` | `busy` |
| 500 | `internal_error` | `internal error` (the encode produced codes no voice can hold; nothing is written or registered) |
| 500 | `voice_write_failed` | `could not write the voice file` |

Every `500` here (and `voice_delete_failed` on `DELETE`) closes the connection and emits a
`request.failed` event with the request id.

#### `GET /v1/voices`

`200` with an array of voice objects: saved voices sorted by id, then unnamed voices in
registration order. Files that failed validation at startup are not listed.

#### `DELETE /v1/voices/{id}`

`200` with `{"deleted":"<id>","file_kept":false}`. A saved voice's file is removed, so it does not
come back after a restart; an unnamed voice is removed from memory; an id reserved by a skipped
file has that file removed and the name released; the cached reference prefix is dropped. `DELETE`
never checks busy.

| Status | Code | Message |
| --- | --- | --- |
| 404 | `unknown_voice` | `unknown voice_id` |
| 500 | `voice_delete_failed` | `could not delete the voice file`. The voice stays registered |

#### Other routes

| Case | Response |
| --- | --- |
| Unknown path | `404 not_found` |
| Known path, wrong method | `405 method_not_allowed`, with an `Allow` header |
| `/v1/audio/convert` | `404` (voice conversion is out of scope) |
| Web UI routes (`/`, `/app.js`, `/style.css`) | `404` |

#### HTTP error codes

`invalid_field`, `duplicate_field`, `text_required`, `text_too_long`, `reference_conflict`,
`ref_text_required`, `reference_required`, `invalid_audio`, `audio_too_long`, `audio_too_short`,
`voice_fields_required`, `invalid_name`, `voice_too_long`, `unknown_voice`, `voice_exists`, `busy`,
`loading`, `gpu_unavailable`, `gpu_out_of_memory`, `not_found`, `method_not_allowed`,
`payload_too_large`, `origin_not_allowed`, `voice_write_failed`, `voice_delete_failed`,
`internal_error`. Codes are stable identifiers; messages may change, except where they match the
C++ server's strings.

### WebSocket API

Behavior matches the C++ server's session protocol except where marked as a breaking change below.

**Where to connect**: `ws://<host>:<ws_port>/`, by default the HTTP port + 1, and only on `--host`.
Any path is accepted, as in C++. The port can be discovered with `GET /health`.

**Handshake**: RFC 6455 only. Every handshake response, accepted or refused, carries
`X-Breeze-Version: <server version>`. Refusals get a JSON body `{"error","code"}`:

- `403 origin_not_allowed` if an `Origin` header is present and not allowed by the CORS setting.
  With CORS off, no browser origin is allowed.
- `503 loading` while the model loads, and `503 gpu_unavailable` once the GPU has stopped
  responding (checked again just before the `101`, so a handshake under way when the GPU stops
  still gets the `503`).
- `503 too_many_connections` above 16 connections.
- `503 shutting_down` once the server has started shutting down.
- `426 upgrade_required` for a request that isn't a valid WebSocket upgrade (no
  `Upgrade: websocket` or no `Connection: upgrade`), or `400 bad_handshake` (no
  `Sec-WebSocket-Key`, an unsupported `Sec-WebSocket-Version`, or a malformed key), and an
  unexpected handshake failure `500 internal_error`.
- Checks run in this order, and the first that fails answers: `shutting_down`, then the upgrade
  headers, then `origin_not_allowed`, then `gpu_unavailable`, then `loading`, then
  `too_many_connections`.
- The handshake must complete within 10 s.

**Frames**: text frames carry JSON messages; binary frames from the server carry PCM. Inbound
messages are limited to 1 MiB (close `1009` above that). The server pings every 20 s and closes
with `1011` if there is no pong within 20 s.

**Close codes**:

| Code | When |
| --- | --- |
| 1000 | echoed on a client close |
| 1001 | server shutdown. A client that doesn't read the close frame within 2 s is dropped without one (it sees 1006) |
| 1002 | protocol error (for example an unmasked frame) |
| 1007 | invalid UTF-8 |
| 1008 | `client too slow`: more than 2 MiB of undelivered output, or no output delivered for 30 s while some is waiting; the piece in flight is cancelled first. A client that doesn't read the close frame within 2 s is dropped without one (it sees 1006) |
| 1009 | message too big |
| 1011 | ping timeout or internal error; also `gpu is not responding` when the GPU stopped responding between the `101` and the session's start |

**Server → client**: on connect, before any client message, the server sends
`{"type":"ready","sample_rate":24000,"format":"s16le"}`.

| type | Fields | Meaning |
| --- | --- | --- |
| `ready` | `sample_rate`, `format` | Sent once on connect |
| `started` | `voice_id` (string, `""` when none) | A `start` was accepted |
| `queued` | — | The next piece is waiting for the GPU |
| `speaking` | `text` | A piece begins; `text` is the exact piece text. Its binary frames follow |
| *(binary)* | — | Mono s16le PCM at `sample_rate` for the current piece |
| `instruction_set` | — | An `instruction` message was applied |
| `cancelled` | — | Exactly one per `cancel`, and per `start` that interrupts pending work, after the last audio frame of the cancelled work |
| `done` | — | Exactly one per `end`, after every piece queued before it, even when nothing was left. A later `cancel` in the same session (before any further `start`) replaces it with `cancelled` |
| `error` | `message`, `code`, `request_type` | See error codes below; `request_type` is the client message type that caused the error, or `null` |

All events and audio frames go through one ordered queue: within a session, `speaking`, then that
piece's frames, then the next `speaking`, and so on; `cancelled`/`done` are emitted strictly in the
order of the client messages that caused them.

**Client → server**: messages are JSON objects with a string `type`, parsed with a real JSON
parser (`\uXXXX` escapes work). Unknown fields are ignored, and a field whose value is JSON `null`
counts as absent; on `start`, an empty string also counts as absent. A message that is valid JSON
but not an object gets `invalid_json`; a missing or non-string `type` gets `invalid_field`. Wrong
types or out-of-range values produce `error{code: invalid_field}` with no other effect. A binary
frame produces `error{code: unsupported_binary}`. The connection stays open after any of these
errors.

| type | Fields | Effect |
| --- | --- | --- |
| `start` | `voice_id`, `instruction`, `ref_text`, `cfg_scale`, `seed`, `temperature`, `top_k`, `split_chars`; additive: `top_p`, `repetition_penalty`, `max_new_tokens` | Validates with the same types, ranges and defaults as HTTP. If work is pending, cancels it (one `cancelled`), resets the session, and sends `started` |
| `text` | `text` (string) | Appends text and speaks each complete sentence |
| `flush` | `text` (optional) | Appends, then speaks everything buffered |
| `end` | `text` (optional) | Like `flush`, then one `done` once everything before it has been spoken. The session stays usable afterwards |
| `instruction` | `instruction` | Applies to pieces that start after this message. Blank means the default instruction. Replies `instruction_set` |
| `cancel` | — | Discards buffered text and queued pieces, stops the piece in flight, replaces a pending `done` of the same session, and replies with one `cancelled`, even when idle |

**`start` details**: `split_chars` is 0–10,000; `0` means no length limit (each drain's ready text
becomes one piece, and unpunctuated text waits for a sentence end, `flush` or `end`; the
10,000-character buffer limit still applies); absent means the server default; a negative value
gets `invalid_field`. `start` is rejected with an `error`, leaving the previous session unchanged,
when `voice_id` is unknown (`unknown_voice`) or `ref_text` comes without `voice_id`
(`reference_required`).

**`text` details**: a sentence end at the very end of the buffer waits for more text, `flush` or
`end`. Text is cut at sentence ends and packed into pieces up to the budget (soft: a single unit
heavier than it stays whole). A sentence over the budget is cut into clauses at the first break
after the clause reaches the budget; a clause that would pass 2× the budget is closed at its last
break instead, and a run with no break at all is hard-cut into budget-sized chunks once it is over
2× the budget, never inside a grapheme cluster. A piece with no letter or digit (emoji-only or
punctuation-only) is dropped. If the buffer plus the new text would exceed 10,000 characters, the
server sends `error{code: text_too_long}` and does not append the text. Control characters are
rejected with `invalid_field` in `text`, `instruction` and `ref_text`.

**Other rules**: any message other than `start` before a successful `start` gets
`error{code: not_started, message: "send start first"}`. The opening budget of 200 applies only to
the first piece of a session that has no reference. Piece `i`, counted from `start`, uses seed
`(seed + i) mod 2^32`.

**Generation**: pieces wait for the GPU in order; `queued` is sent only when a piece actually has
to wait. The GPU is never held while waiting on the socket. If generating a piece fails, the server
sends `error{code: generation_failed}` and moves on to the next item; the session and server keep
running. If the GPU stops responding during a session, each later piece gets the same
`generation_failed` error and the session stays open; new connections are refused with
`503 gpu_unavailable`. A piece that doesn't fit the model's context with its reference gets
`error{code: text_too_long, request_type: null}` and is skipped. Without a voice, the first piece
that succeeds becomes the anchor for the later pieces, unless it would make a later piece shorter
than it would be without it, in which case that piece is spoken without the anchor.

**WebSocket error codes**: `invalid_json`, `unknown_type`, `invalid_field`, `not_started`,
`unknown_voice`, `reference_required`, `text_too_long`, `unsupported_binary`, `generation_failed`,
`internal_error`.

**Compatibility notes for existing clients**: clients written for the C++ server keep working if
they wait for `ready`, send `start → started → text/end`, and handle `queued`, `done`, `cancelled`
and `error` with its `message` key. The known client is the SillyTavern extension, which sends a
single `end`; it gets `\uXXXX` escapes decoded correctly, a `cancelled` for every `cancel`
(including when idle), and a clean `1000` close instead of the connection dropping (`1006`).

### Breaking Changes from the C++ server

The C++ server's documented contract lives in
[`docs/server.md`](https://github.com/HoppouAI/Breeze-TTS-2.cpp/blob/main/docs/server.md),
[`docs/voices.md`](https://github.com/HoppouAI/Breeze-TTS-2.cpp/blob/main/docs/voices.md) and
[`docs/websocket.md`](https://github.com/HoppouAI/Breeze-TTS-2.cpp/blob/main/docs/websocket.md) in
[HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp). "Compatible" means a
client written against that contract works against this server by changing only the host and
port, except where listed here. Every change below is a deliberate break, made because the C++
behavior is a **defect**: it crashes or stalls the server, silently discards or reinterprets what
the client asked for, corrupts or loses data, creates a security exposure, or contradicts the C++
server's own documentation. IDs are stable and referenced by tests and `CHANGELOG.md`; they are
grouped by area, not listed in numeric order (BC-46 to BC-48 were added later).

**HTTP: parsing and validation**

| ID | C++ behavior | New behavior | Affected |
| --- | --- | --- | --- |
| BC-01 | Unparseable numbers become `0` (`cfg_scale=banana` → CFG 0, `seed=x` → 0) | `400` naming the field | Clients sending malformed numbers |
| BC-02 | Empty field value (`seed=`) is used as `""`/`0` | Treated as absent; default applies | Clients sending empty fields |
| BC-03 | Negative, NaN or out-of-range sampling values silently mean "default" or pass through unchecked (`top_p` > 1, negative `cfg_scale`) | `400`; only `0` means default | Clients sending such values |
| BC-04 | `max_new_tokens` unbounded (can exhaust memory and kill the process) | `400` above the server maximum | Clients asking for more than the maximum |
| BC-05 | `text` and `instruction` unbounded (url-encoded bodies over 8 KB instead get an empty `413`) | `400` above the maximum text or instruction length; url-encoded bodies accepted up to the request limit | Clients sending very long text |
| BC-06 | Multipart bodies unbounded | `413` with the error envelope above the upload limit | Clients uploading very large files |
| BC-07 | Busy check runs before validation | Validation first, so an invalid request gets `400`/`404` even while busy | Clients sending invalid requests during generation |
| BC-08 | A field repeated or present in both query and body resolves silently | `400` | Clients sending duplicate fields |
| BC-09 | Empty `instruction` on HTTP is used literally | The default instruction is used | Clients sending `instruction=` |
| BC-10 | Whitespace-only `text` accepted | `400 text is required` | Clients sending blank text |

**HTTP: reference audio**

| ID | C++ behavior | New behavior | Affected |
| --- | --- | --- | --- |
| BC-11 | Undecodable or empty `ref_audio` ignored; request silently becomes voice design | `400` | Clients uploading bad audio |
| BC-12 | `ref_audio` without `ref_text` ignored silently | `400` | Clients omitting the transcript |
| BC-13 | `ref_text` without `ref_audio`/`voice_id` ignored silently (HTTP and WebSocket `start`) | `400` on HTTP; `error` event on WebSocket | Clients sending a stray transcript |
| BC-14 | `voice_id` plus `ref_audio`: the upload is ignored | `400` (mutually exclusive) | Clients sending both |
| BC-15 | Malformed WAV can over-read memory or divide by zero; 8/24-bit PCM decodes as silence | Safe decode of all common PCM/float formats; undecodable input gets `400` | Clients with such files (now work or get a clear error) |
| BC-16 | Reference clip of unlimited length; a clip too short for any frame is silently ignored | `400` over 30 s or under one full codec frame (80 ms) | Clients sending such clips |

**HTTP: responses, errors and CORS**

| ID | C++ behavior | New behavior | Affected |
| --- | --- | --- | --- |
| BC-46 | Control characters accepted in text; NUL counts as sentence-closing punctuation | `400` (`error` event on WebSocket) for control characters other than tab, CR and LF | Clients sending control characters |
| BC-47 | No fixed context limit: a single piece of any length can generate (the cache is sized per piece) | A piece that cannot fit the 2,048-token context gets `400 text is too long` (first piece) or an aborted stream (later piece) | Clients sending `split_chars=0` or very large `split_chars` with long text, or long references with long transcripts |
| BC-17 | Failure mid-stream ends the stream like success; failure before audio gives `200` with an empty body | Failures before streaming get a proper error status; failures after streaming starts abort the response | Clients that treated a truncated stream as complete |
| BC-18 | Unknown route `404`, wrong method `400`/`404`, oversize `413`: empty non-JSON bodies; `OPTIONS` without CORS gets `404` | JSON error envelope; wrong method (including `OPTIONS` without CORS) is `405` with `Allow` | Clients parsing these responses |
| BC-19 | CORS allowlist entries not trimmed (`"a, b"` never matches `b`) | Entries trimmed | Operators with spaced lists (now works) |
| BC-20 | `Vary: Origin` only on matching responses in allowlist mode | On every response in allowlist mode | Caches (fixes mixing) |
| BC-21 | Preflight returns `204` for any path, advertising every method | Existing routes only, with that route's methods; unknown paths get `404` | Clients preflighting non-existent paths |
| BC-22 | `*` mixed into an allowlist is a dead entry | Startup error | Operators with such configs |
| BC-23 | Cross-origin browser `POST`/`DELETE` runs (and can write voice files) even when CORS is off | `403` when `Origin` is present and not allowed | Browser pages on disallowed origins |
| BC-24 | `/health` reports `ws_port` even when the WebSocket failed to bind | Reports `0` in that case | Clients discovering the WebSocket |

**Voices**

| ID | C++ behavior | New behavior | Affected |
| --- | --- | --- | --- |
| BC-25 | Voice files with unsafe names load and break `GET /v1/voices` JSON; list order is unspecified | Invalid files skipped; deterministic order | Operators with hand-placed files |
| BC-26 | Names may start with `v_`, colliding with generated ids; names differing only by case collide on case-insensitive filesystems | `v_` prefix reserved (`400`); case-insensitive duplicates rejected | Clients using such names |
| BC-27 | Named `POST` for an existing name silently overwrites it | `409` with code `voice_exists`; delete then register to replace | Clients that re-POST a name to update it |
| BC-28 | `DELETE` keeps the file (`file_kept: true`), so the voice comes back on restart | The file is removed; `file_kept` is always `false` | Clients relying on deleted voices returning |
| BC-48 | The 64-voice cap counts saved voices too, so many saved voices leave little room for unnamed ones | The cap counts only unnamed voices | Clients registering many unnamed voices alongside saved ones |
| BC-29 | Voices stored as `.breeze` files | Own versioned format; `.breeze` files are ignored, so C++ voices must be registered again | Operators switching from the C++ server |

**WebSocket**

| ID | C++ behavior | New behavior | Affected |
| --- | --- | --- | --- |
| BC-30 | WebSocket binds `0.0.0.0` when the host is not an IPv4 literal | Binds only the configured host | Deployments using `localhost`/IPv6 |
| BC-31 | No `Origin` check on the handshake (cross-site hijacking) | `403` for disallowed browser origins | Browser pages on disallowed origins |
| BC-32 | Hand-rolled JSON reader: `\uXXXX` escapes deleted, keys matched inside values, `null` misread, invalid JSON accepted, wrong types become defaults | Real JSON with schema validation; an `error` event on invalid input | All clients escaping non-ASCII (now works); clients sending bad messages |
| BC-33 | `ready.sample_rate` hard-coded to 24000 | Real model rate (24000 for current models) | None in practice |
| BC-34 | `done` missing if `end` arrives with nothing left to speak | Exactly one `done` per `end` | Clients that worked around the hang |
| BC-35 | `cancel` sometimes unacknowledged, sometimes spurious, and can swallow the next piece | Exactly one `cancelled` per `cancel`; later pieces never dropped | Clients counting or ignoring `cancelled` |
| BC-36 | `start` mid-speech mutates the running session (a data race) | Cancels in-flight work (`cancelled`), then `started` | Clients restarting mid-speech |
| BC-37 | Empty `instruction` message stores `""` | Resets to the default instruction | Clients sending an empty instruction |
| BC-38 | `split_chars ≤ 0` at `start` means 600; on HTTP `0` means no splitting | `0` means no length splitting on both; negative gets an `error` | Clients sending `split_chars: 0` |
| BC-39 | Period at the end of the buffer cuts immediately (splits `Dr.`, `3.` mid-stream); CJK without spaces never drains; the 200-char opening budget applies to every piece of that drain; different stop set from HTTP | Shared segmenter; end-of-buffer punctuation waits for the next character; CJK fallback; opening budget on the first piece only | All clients: streaming piece boundaries differ, and so do HTTP piece boundaries for long text |
| BC-40 | Buffered text unbounded | `error` event above the maximum text length | Clients buffering very long text without punctuation |
| BC-41 | Generation error in a session terminates the whole server process | `error` event; session and server continue | All |
| BC-42 | A client that stops reading holds the GPU and blocks every other client; unbounded connection threads; no handshake timeout | Bounded outgoing buffer; slow client disconnected; connection and handshake limits | Very slow clients; connection floods |
| BC-43 | Client close not answered with a close frame (browser sees 1006); unmasked frames, bad UTF-8 and malformed control frames accepted | Standard-conformant WebSocket | Non-conformant clients |
| BC-44 | `speaking.text` drops tabs and carriage returns | Exact piece text | Clients comparing piece text |
| BC-45 | Binary frames from the client silently ignored | `error` event | Clients sending binary frames |

**Additive changes (not breaking)**: an `X-Breeze-Version` header on every HTTP response and
WebSocket handshake; an error `code` field alongside `error` on HTTP and WebSocket errors; `type`
on WebSocket error events; `top_p`, `repetition_penalty`, `max_new_tokens` on WebSocket `start`;
any sample rate and channel count, 8/24-bit PCM and other common audio containers accepted as
`ref_audio`.

#### Known Differences Outside the API Contract

- **Repetition penalty**: this server applies it once per distinct generated token (the reference
  model implementation's semantics). The C++ server compounds it once per occurrence. The field,
  its range and its default are the same; only the resulting audio differs.
- **First chunk size**: streams start at 1 codec frame (80 ms) by default and ramp to 25, where
  C++ starts at 4. This lowers time to first audio. It changes chunk boundaries, not the audio
  format, and can be set back with `--chunk-first`/`--chunk-max`.
- **Default length**: with `max_new_tokens` absent or `0`, a piece is capped at the model default
  of 750 frames (60 s), as in C++. The previous Python API used 1,500.

#### Intentionally Kept C++ Behaviors

These C++ choices are unconventional but are kept for compatibility, not fixed:

- `409` with `{"error":"busy"}` when the GPU is busy on HTTP (`503` with `Retry-After` would be
  more conventional).
- `200` rather than `201` on voice creation.
- `{"error": "<string>"}` as the error body key (a `code` is added alongside).
- `Content-Type: audio/pcm` with rate and format in headers; no WAV or `response_format` option.
- Fields accepted from the query string.
- `0` meaning "model default" for sampling fields and `max_new_tokens` (greedy decoding cannot be
  requested).
- Per-piece seed `seed + i`, `ref_text` overriding a voice's stored transcript, the unnamed voice
  id format, the 64-entry unnamed voice cap, WebSocket on a separate port, and no authentication.

### ⚡ Fast Inference Options

Both the CLI and API use eager streaming by default and skip graph warmup. Pass `--fast-all` to enable the best configuration for every inference stage when the additional cold-start time is acceptable. Each stage can also be controlled independently:

| Stage | Fast parameter | Disabled | Enabled |
| --- | --- | --- | --- |
| Text encoder | `--[no-]fast-text-encoder` | Native eager forward | Static CUDA Graph selected by CFG shape and text-length bucket |
| Backbone prefill | `--[no-]fast-backbone-prefill` | Native eager prefill | CUDA Graph selected by CFG shape and prompt-length bucket |
| Backbone decode | `--[no-]fast-backbone-decode` | Native eager token step | StaticCache-backed graph selected by CFG shape |
| Depth decoder | `--[no-]fast-depth-decoder` | Native eager depth loop | Full-graph compilation with CFG-shape CUDA Graphs |
| Codec | `--[no-]fast-codec` | Eager streaming decode | Single-request streaming CUDA Graph with one-frame chunks |

Individual stage flags are intended for profiling and debugging.

`--attn-implementation {eager,sdpa}` (default `eager`) selects the attention kernel for the
backbone and text encoder. The fast text-encoder stage always runs `sdpa` for graph capture
regardless of this setting. On an RTX 4090 with `--fast-all`, `sdpa` was no faster than `eager`
(about 9% slower on a ~2000-character request) but used about 2.7 GB less peak VRAM.
FlashAttention 2 is not offered: Hugging Face's FA2 path rejects the backbone's 4D attention
masks in both eager and fast modes.

#### Startup time and the compile cache

The fast path captures every CUDA graph again on each start. A captured graph is bound to live device memory (its static buffers, the KV cache, the shared graph pool), and neither PyTorch nor CUDA can serialize one, so capture is unavoidable. Most of the warmup time is not capture, though: it is `torch.compile` of the depth decoder and the codec's SnakeBeta activations, which runs lazily during the eager warmup passes before capture. That work is cacheable, and torch caches it on disk by default, but in the system temp directory, which Ubuntu and WSL clear on boot.

The server pins that cache to a persistent location, chosen in this order: `--compile-cache-dir`, then an existing `TORCHINDUCTOR_CACHE_DIR`, then `./.cache/torchinductor`. It also stores torch's own source-tree hash there (`torch_key.json`), which Inductor otherwise recomputes on every start by reading every Python file in the torch package; the stored value is reused while the torch version, install path, wheel `RECORD`, and the sizes and mtimes of that source tree are all unchanged. Checking that costs about 1.6 s on native Windows against 12 s for the hash after a reboot; on WSL2 with the environment on a Windows drive the directory listing itself is slow enough that it is roughly break-even. After every fast warmup the server writes `warmup_manifest.json` next to the cache with per-stage timings, torch's per-phase compile timers, and cache hit/miss counters, and prints a one-line summary such as `fast warmup: 33668 ms (fx graph cache hits 65 / misses 0)`. A warm start shows hits and no misses; a cold one shows the reverse.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--compile-cache-dir PATH` | env or `./.cache/torchinductor` | Where compiled kernels persist between starts |

Measured on an RTX 4090 with `--fast-all` (warmup only, after the checkpoint is loaded):

| Start | Windows native | WSL2 (repo on a Windows drive) |
| --- | --- | --- |
| Empty cache | 102 s | 157 s |
| Warm cache, torch hash recomputed | 50 s | 79 s |
| Warm cache, torch hash reused | 34 s | not measured |

What remains on a warm start is capture and its eager warmup passes (about 10 s for the 63 text-encoder and prefill graphs), the codec (about 3 s), and Dynamo re-tracing the compiled depth decoder (about 13 s), which torch 2.9 cannot cache for module compiles: its experimental precompile cache was tried and made startup slower.

On Linux, Triton builds a small helper with `gcc` the first time it populates a cache directory. Run the server from a directory other than the repo root when doing that: gcc treats a `./specs` directory in its working directory as a spec file and aborts.


## Development

| Action | Command |
| --- | --- |
| Run the server (browser clients) | `scripts/start_breeze.sh --cors http://127.0.0.1:8000` |
| Unit and integration tests (no GPU) | `.venv/bin/pytest` |
| GPU tests | `BREEZE_MODEL=<path> .venv/bin/pytest -m gpu` |
| Lint | `.venv/bin/ruff check .` |
| Benchmark | `.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080` (defaults `--warmup 3 --runs 10`) |
| SillyTavern live test | `node tests/live/sillytavern/run.mjs <health\|voices\|speech\|full>` |
| C++ docs example check | `.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080` |

See `specs/003-cpp-compatible-api/quickstart.md` for the full walkthrough these commands are
drawn from.


## License and Responsible Use

The source code is licensed under the [Apache License, Version 2.0](https://github.com/breezeblue-ai/breeze-tts/blob/main/LICENSE). The audio tokenizer is based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen Team and is licensed under the Apache License, Version 2.0. Model weights, checkpoints, adapters, derivative models, and self-hosted outputs are governed separately by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). The Apache License does not grant rights to use the model commercially.

If you have an active paid subscription, outputs you generate through BreezeBlue's hosted platform or API at [breezeblue.ai](https://breezeblue.ai/) can be used commercially, subject to our [Terms of Service](https://breezeblue.ai/legal/terms). A paid subscription does not grant commercial rights to the open-weight model or self-hosted outputs.

You are responsible for complying with applicable laws and obtaining all necessary rights and consents for inputs, reference audio, voices, and outputs. Unauthorized voice cloning, impersonation, fraud, and other unlawful or harmful uses are prohibited.

The code and Model Materials are provided "AS IS," without warranties or liability to the maximum extent permitted by law. Third-party components remain subject to their respective licenses.
