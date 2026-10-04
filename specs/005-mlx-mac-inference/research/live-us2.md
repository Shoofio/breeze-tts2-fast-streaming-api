# Live gate, User Story 2 (T026): 2026-10-04

Quickstart step 3 on the reference Mac (Apple M5, 16 GB). The server was started with
`scripts/start_breeze_mac.sh [--precision bf16] --host 127.0.0.1 --voices-dir <tmp>`. The
requests were sent with `curl`. The script is in the session scratchpad
(`live-us2/gate.sh`).

The reference clip was made by the server itself: a POST of "This is the exact transcript of
the reference audio." with seed 7, rewrapped as a 24 kHz mono 16-bit WAV. That text is the
`ref_text`.

## Results

| Check | 8-bit | bf16 |
|---|---|---|
| Clone (`ref_audio` + `ref_text`) | 200, first byte 0.41 s, 3.36 s audio | 200, first byte 0.66 s, 2.48 s |
| Design (instruction + `cfg_scale=4`, no reference) | 200, 0.35 s, 2.80 s | 200, 0.54 s, 2.88 s |
| Direction (`docs/api.md` example: reference + instruction + `cfg_scale=4`, "(clears throat) We need to discuss what happened last night.") | 200, 0.50 s, 4.88 s | 200, 0.60 s, 5.52 s |
| `POST /v1/voices name=alice` | 200, `saved: true` | 200, `saved: true` |
| GET `.wav` with `voice_id=alice` | 200 | 200 |
| Restart on the same `--voices-dir`: `alice` listed and used | yes, 200 | yes, 200 |
| A copy with an edited `codec_fingerprint` | `voices.loaded loaded=1 skipped=1`, then `voice.skipped file=foreign.voice.json reason="codec_fingerprint does not match the running codec"` | same |
| Request for the skipped id | `404 {"code":"unknown_voice"}` | same |

The automated real-server tests (T024, T025) cover the same cases at both precisions.

Whether each output sounds like the intended voice is judged in the listening test (T033). The
WAVs from this run are in the scratchpad (`live-us2/{8bit,bf16}/*.wav`).

## Finding: the docs' curl example is broken by curl's `-F` syntax

The first run used `-F "text=(clears throat) …"` and produced 34–60 s of rambling audio. This
was not an MLX or model fault. In curl's `-F` syntax, a value that starts with `(` opens a nested
`multipart/mixed` part, so:
- the words are never sent;
- the server receives MIME headers and a random boundary string as `text`;
- every later `-F` field (`instruction`, `seed`) disappears inside that part, so their defaults
  apply.

The captured request body and an instrumented server run showed this. Sending the same text with
`--form-string` gives a normal-length result, 53 frames at 8-bit.

The broken examples are `docs/api.md:192` and `README.md:173`. The problem is the same on CUDA
and predates this feature.

The same investigation found the MLX path deterministic. A fixed seed gives bit-identical frames
across fresh processes, with and without a reference.
