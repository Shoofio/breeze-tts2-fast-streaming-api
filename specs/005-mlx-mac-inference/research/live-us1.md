# Live gate, User Story 1 (T023): 2026-10-04

Quickstart steps 1, 2 and 7, plus the suspend check from the spec's Edge Cases, on the reference
Mac (Apple M5, 16 GB). The server was started with `scripts/start_breeze_mac.sh` at commit
`87a17ef`. The checks added `--host 127.0.0.1`, which also confirms that a later `--host`
overrides the launcher's default.

## Step 1: install and start

| Precision | Command | `/health` 200 after | `model.loaded` |
|---|---|---|---|
| 8-bit | `scripts/start_breeze_mac.sh` | 8 s | `backend=mlx`, `weights=8bit`, `device=mlx:gpu`, `warmup_ms` 1680 |
| bf16 | `scripts/start_breeze_mac.sh --precision bf16` | 10 s | `backend=mlx`, `weights=bf16`, `device=mlx:gpu` |

`/health` returned `{"status":"ok","sample_rate":24000,"ws_port":8081}`.

The launcher's error paths all exit 1:
- `--precision fp16` → `--precision must be 8bit or bf16 (got fp16)`.
- `--precision` with no value → `--precision needs a value: 8bit or bf16`.
- `HF_HOME=/nonexistent` → `Model snapshot not found: …` followed by the exact
  `uvx --from huggingface_hub hf download <repo> --revision <sha>` command, for each precision.

## Step 2: first requests

The README `curl` commands were run unchanged against the running server.

| Precision | Request | Status | First byte | Total | Bytes |
|---|---|---|---|---|---|
| 8-bit | POST `.pcm` | 200 | 0.34 s | 0.72 s | 30720 |
| 8-bit | GET `.wav` | 200 | 0.29 s | 0.68 s | 30764 |
| bf16 | POST `.pcm` | 200 | 0.51 s | 1.43 s | 38400 |
| bf16 | GET `.wav` | 200 | 0.47 s | 1.40 s | 38444 |

- **WAV header:** `RIFF ffffffff WAVE fmt … data ffffffff`, the streaming header with unknown
  sizes. `afinfo` reads both files as 1 channel, 24000 Hz, Int16.
- **Busy:** a second POST while a long one was running got `409 {"error":"busy","code":"busy"}`.
- **Interrupt, measured 3 times at 8-bit:** killing `curl` 1.5 s into a long request logged
  `speech.aborted` with reason `client_disconnect`.
  - A probe polled every 50 ms. It completed its first full request 1.10–1.18 s after the kill.
  - That time includes generating and reading the probe's own reply, about 0.7 s for a short
    text. So the gate was free within about 0.5 s.
- **Browsers:** the user played the `.wav` URL in Chrome and in Firefox on 2026-10-04: it plays
  in both.

## Step 7: refusals (SC-007)

Each one printed the usage line and the contract's message, and exited with status 2:

| Case | Message |
|---|---|
| `--fast-all` on the Mac | `--fast-all only apply to --backend cuda; remove them` |
| PyTorch checkpoint with `--backend mlx` | `<dir> is the PyTorch checkpoint; --backend mlx needs the MLX weights: uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit --revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49` |
| 4-bit MLX checkpoint | `<dir> is 4-bit mxfp4; the MLX backend supports bf16 and 8-bit (mxfp8)` |
| `--backend cuda` on the Mac | `--backend cuda is not available on macOS; use --backend mlx (the default here)` |

## Suspend mid-request (spec Edge Cases)

- **Setup:** a long POST (11.4 s of audio) at 8-bit. 1.5 s in, the server's Python process got
  `kill -STOP`. `ps` showed state `T`. After 10 s it got `kill -CONT`.
- **Result:**
  - The request completed with status 200 and all 549120 bytes (11.44 s of audio).
  - It took 19.6 s in total: the usual 9.5 s plus the 10 s pause.
  - `speech.completed` was logged.
  - The next request returned 200, with its first byte at 0.30 s.
- **Method note:** the launcher runs the server as a child of `uv run`. The first attempt
  stopped the `uv` parent, which left the server running, and was discarded. The process that
  must be stopped is the `.venv/bin/python3 -m breeze_infer.api` child.
