# Quickstart: validating MLX inference on a Mac

This guide proves the feature end to end. The reference machine is an Apple M5 with 16 GB, macOS,
with `uv` installed. Results go in `research/live-*.md`, as in features 003 and 004.

Paths: `$MLX_BF16` and `$MLX_8BIT` are the pinned snapshot directories under
`$HF_HOME/hub/models--mlx-community--Breeze-TTS-2-mlx{,-8bit}/snapshots/<sha>` (shas in
research R3).

## 0. Phase-0 gate (before any build work): research R7

```bash
uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx --revision 3c8829fb7fd335818f085cd2ef49b4100c0e46c8
uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit --revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49
```

- Use a scratch venv with mlx-audio at the pin, then run stock `generate()` four ways: bf16 and
  8-bit, each with no CFG and with `cfg_scale` 4.
- Each run uses a one-sentence input and a passage of about 60 s.
- Record RTF, time to first audio and peak memory (`/usr/bin/time -l`, maximum resident set size)
  in `research/live-phase0.md`.

**Expected:** 8-bit with CFG reaches RTF ≤ 1.3. If it doesn't, **stop and report**; don't
continue.

## 1. Install and start (User Story 1)

```bash
scripts/start_breeze_mac.sh                  # 8-bit (default)
scripts/start_breeze_mac.sh --precision bf16 # bf16
```

**Expected:**
- `/health` goes from `503 loading` to `200 {"status":"ok","sample_rate":24000,...}`.
- The `model.loaded` event shows `backend=mlx`, `device=mlx:gpu` and `weights=8bit` (or `bf16`).
- SC-001: under 10 minutes from a fresh clone, with the weights already downloaded.

## 2. First requests (User Story 1)

Run the README's "First request" `curl` commands unchanged (POST `.pcm` and GET `.wav`). Then open
the `.wav` URL in Chrome and in Firefox.

**Expected:**
- Playable 24 kHz mono audio.
- In `curl -w '%{time_starttransfer}'`, the first byte arrives before generation ends.
- A second POST while the first is running gets `409 busy`.
- Interrupting `curl` mid-stream frees the server for the next request within a second.

## 3. Voice features (User Story 2)

Run the voice clone, voice design and voice direction requests from `docs/api.md` against the Mac
server. Also upload a voice through `/v1/voices`, restart the server, and use it again.

**Expected:**
- Each request returns speech in the expected voice.
- The voice survives the restart.
- A voice file whose fingerprint doesn't match is skipped and reported at startup (User Story 2,
  scenario 4). To test this, edit a copy's `codec_fingerprint`.

## 4. WebSocket (User Story 3) and compatibility (SC-003)

```bash
node tests/live/sillytavern/run.mjs full
.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080
```

Both checks exercise the WebSocket API as well as HTTP.

**Expected:** both pass with no edits to the checks.

## 5. Speed and memory (SC-002, SC-002a, SC-006)

```bash
.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080
```

Run it at bf16 and at 8-bit, with a browser and an editor open.

**Expected:**
- 8-bit: first audio under 2 s, and RTF ≤ 1.0 over the passage.
- bf16: the same, or the docs recommend 8-bit for 16 GB Macs and publish the measured bf16
  numbers.
- No swap use during the run (`sysctl vm.swapusage` unchanged).

## 6. Listening test (SC-005)

Synthesize the 10 fixed prompts (3 clone, 3 design, 3 direction, 1 plain) on CUDA and on the Mac at
both precisions.

**Expected:** every Mac output is intelligible and artifact-free, and matches the CUDA output's
speaker or description. Record the results in `research/live-listening.md`.

## 7. Refusals (SC-007)

Each of these must print the usage line and the documented error message, and exit with status 2
(contracts/launch-and-events.md):
- `--fast-all` on the Mac;
- the PyTorch checkpoint with `--backend mlx`;
- the 4-bit MLX checkpoint;
- `--backend cuda` on the Mac.

## 8. Tests

```bash
.venv/bin/pytest                                   # model-free suite, on the Mac and on Linux
BREEZE_MLX_MODEL=$MLX_BF16 .venv/bin/pytest -m mlx # Mac, real weights
BREEZE_MLX_MODEL=$MLX_8BIT .venv/bin/pytest -m mlx
```

## 9. CUDA unchanged (User Story 4, SC-004)

On the CUDA machine:

```bash
BREEZE_MODEL=<path> .venv/bin/pytest -m gpu
.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080
```

**Expected:**
- All GPU tests pass.
- Time to first audio and throughput are within 5% of the 2.1.0 baseline
  (`specs/003-cpp-compatible-api/research/bench-final.md`).
- `scripts/start_breeze.sh` and `scripts/start_breeze.ps1` are byte-identical to 2.1.0.
