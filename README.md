# Breeze TTS 2 Streaming API

> [!IMPORTANT]
> This is an independent derivative of [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts). It is not affiliated with or endorsed by BreezeBlue. The code is Apache 2.0. The Breeze TTS 2 model weights, derivative models and self-hosted outputs are under the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE): research and non-commercial use only. See [License and responsible use](#license-and-responsible-use).

## What this is

Breeze TTS 2 is an open-weight text-to-speech model by BreezeBlue. This repository takes the upstream PyTorch inference code and adds a streaming HTTP and WebSocket server around it.

From upstream: the Breeze TTS 2 PyTorch inference code (model by BreezeBlue, [Hugging Face](https://huggingface.co/BreezeBlue/Breeze-TTS-2)).

- **Voice Clone**: uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style.
- **Voice Design**: creates a distinctive voice from a natural-language description, without reference audio.
- **Voice Direction**: clones a voice from reference audio while steering tone, emotion, pace, and delivery.

Added in this fork:

- An HTTP and WebSocket streaming server compatible with the [HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) server's API contract, with [documented differences](docs/api.md#breaking-changes-from-the-c-server).
- Saved voices, managed through `/v1/voices`.
- `GET /v1/audio/speech.wav`, a progressive WAV stream that a browser `<audio>` element can play.
- Single-GPU request handling: one generation at a time. `POST /v1/audio/speech` gets `409 busy` when the GPU is in use; the `.wav` route queues for up to 60 s.
- Optional CORS (`--cors`).
- Fast-path inference options, enabled together with `--fast-all`.
- A native Windows launcher, `scripts/start_breeze.ps1`.
- An MLX backend for Apple Silicon Macs, started with `scripts/start_breeze_mac.sh`. See [Quick start (macOS, Apple Silicon)](#quick-start-macos-apple-silicon).
- CPU and GPU test suites.

## Requirements

- An NVIDIA GPU with CUDA, or an Apple Silicon Mac (M1 or later) with at least 16 GB of memory.
- GPU memory: about 7.7 GiB for eager inference, about 14.4 GiB with `--fast-all`. A 12 GB GPU is recommended for eager and a 24 GB GPU for the fast path.
- Python 3.12 (uv installs it).
- [uv](https://docs.astral.sh/uv/).
- Linux or WSL2, native Windows 10/11, or macOS 14 (Sonoma) or later on Apple Silicon.
- On Windows, `--fast-all` needs the Visual Studio C++ build tools (MSVC). The launcher installs `triton-windows` for `torch.compile`, and it compiles through MSVC, which it finds from the Visual Studio install.

## Quick start (Linux / WSL)

```bash
git clone https://github.com/northcraftfoundries/breeze-tts2-fast-streaming-api.git
cd breeze-tts2-fast-streaming-api
uv venv --python 3.12
uv pip install -r requirements.txt
uvx --from huggingface_hub hf download BreezeBlue/Breeze-TTS-2
scripts/start_breeze.sh
```

The download goes to the HuggingFace cache (`$HF_HOME`, default `~/.cache/huggingface`), which is where the launcher looks. Set `HF_HOME` if your cache is elsewhere. Keep the repository name exactly as `BreezeBlue/Breeze-TTS-2`: the launcher looks for the cache directory `models--BreezeBlue--Breeze-TTS-2`.

The launcher binds `0.0.0.0:8080` (WebSocket on `8081`), turns on `--fast-all`, and enables CORS for every origin (`*`). Extra arguments are passed to the server after those defaults, and a later `--host` or `--cors` overrides the launcher's value:

```bash
scripts/start_breeze.sh --host 127.0.0.1 --cors=http://127.0.0.1:8000
```

> [!WARNING]
> With the launcher's defaults, any device on your network and any web page can use the server, including uploading and deleting voices. To keep it local, pass `--host 127.0.0.1`.

To run the server without the launcher, give it the checkpoint directory:

```bash
uv run python -m breeze_infer.api <checkpoint dir> --fast-all
```

Run this way, it listens on `127.0.0.1:8080` and CORS is off. See [docs/api.md](docs/api.md#launch-options) for every option.

## Quick start (Windows)

In PowerShell:

```powershell
git clone https://github.com/northcraftfoundries/breeze-tts2-fast-streaming-api.git
cd breeze-tts2-fast-streaming-api
uvx --from huggingface_hub hf download BreezeBlue/Breeze-TTS-2
.\scripts\start_breeze.ps1
```

If PowerShell refuses with "running scripts is disabled on this system", Windows' default execution policy is blocking local scripts. Allow them for your account once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run this launch as `powershell -ExecutionPolicy Bypass -File scripts\start_breeze.ps1`.

The first run builds a separate `.venv-win` (CUDA torch, the packages in `requirements.txt`, and `triton-windows`). Later runs skip that setup unless `requirements.txt` or the Triton pin changes. The script reads the model from `$env:HF_HOME`, or `~\.cache\huggingface` if that is unset, and uses the same defaults as the Linux launcher: `0.0.0.0:8080`, `--fast-all`, CORS `*`.

Common parameters:

| Parameter | Meaning |
| --- | --- |
| `-BindHost` | Interface to bind (default `0.0.0.0`). Use `127.0.0.1` to keep the server local. |
| `-Port` | HTTP port (default `8080`). |
| `-WsPort` | WebSocket port (default HTTP port + 1); `disabled` turns it off. |
| `-Cors` | CORS origins (default `*`); pass a list to narrow it. |
| `-ModelPath` | Use this checkpoint directory instead of the HuggingFace cache. |
| `-NoFastAll` | Run the eager path (about 7.7 GiB VRAM) instead of the fast path. |
| `-AttnImplementation` | `eager` (default) or `sdpa`. |
| `-SkipSetup` | Skip the environment check and launch. |
| `-Reinstall` | Delete `.venv-win` and rebuild it. |

Because it binds `0.0.0.0` by default, Windows Firewall will prompt on the first launch. `-BindHost 127.0.0.1` keeps the server local.

## Quick start (macOS, Apple Silicon)

Requirements: an Apple Silicon Mac (M1 or later), at least 16 GB of memory, and macOS 14 (Sonoma) or later, the oldest release MLX ships for. The server refuses to start on an Intel Mac or with less memory. Python 3.12 and [uv](https://docs.astral.sh/uv/) are needed, as on Linux.

The Mac backend runs community MLX conversions of the model, in 8-bit or bf16. Download one, pinned to a revision so every install gets the same weights:

```bash
# 8-bit (default, recommended for 16 GB Macs)
uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit --revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49

# bf16
uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx --revision 3c8829fb7fd335818f085cd2ef49b4100c0e46c8
```

Then, from a clone of this repository:

```bash
scripts/start_breeze_mac.sh [--precision 8bit|bf16|mixed] [server options...]
```

The precision defaults to `8bit`. If the Python environment lacks `mlx`, the launcher installs the dependencies with `requirements-mac-overrides.txt`. It looks for the pinned snapshot in the HuggingFace cache (`$HF_HOME`, default `~/.cache/huggingface`), and if it is missing it prints the download command above and exits. It has the same defaults as `start_breeze.sh`: `0.0.0.0:8080` and CORS `*`. It doesn't pass `--fast-all`, which is CUDA-only. Extra arguments go to the server, and a later `--host` or `--cors` overrides the launcher's value. The warning under [Quick start (Linux / WSL)](#quick-start-linux--wsl) about the open defaults applies here too: pass `--host 127.0.0.1` to keep the server local.

Measured on an Apple M5 with 16 GB with a browser and an editor open. The speed columns give the range of the per-case medians (10 runs each) over the five benchmark cases; the slowest single first audio was 0.42 s at 8-bit and 0.68 s at bf16. Peak memory is the server's physical footprint after medium-length requests.

| Precision | First audio | Real-time factor (RTF) | Peak memory |
| --- | --- | --- | --- |
| 8-bit | 0.32–0.41 s | 0.84–0.88 | 7.0 GB |
| bf16 | 0.49–0.55 s | 1.47–1.50 | 9.7 GB |

An RTF below 1 means audio is produced faster than it plays. bf16 is slower than real time on this machine, so streamed audio can't keep up with playback, and its memory use pushed the Mac into swap during the run. Use 8-bit on a 16 GB Mac. The details are in `specs/005-mlx-mac-inference/research/live-perf.md`.

### Real time on Apple Silicon: `--precision mixed`

```bash
uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx --revision 3c8829fb7fd335818f085cd2ef49b4100c0e46c8
scripts/start_breeze_mac.sh --precision mixed [server options...]
```

`mixed` loads the bf16 weights and turns on four settings, each also available on its own (`MlxSpeedOptions` in `models/mlx_streaming.py`):

| Setting | What it does |
| --- | --- |
| `BREEZE_MLX_QUANT=depth:8` | Quantizes only the depth decoder to int8 at load. The frame loop is memory-bound: each frame reads the backbone once and the depth decoder 15 times, so the depth decoder is about three quarters of the bytes. The backbone, which sets prosody and is where a voice direction lands, stays exact. |
| `BREEZE_MLX_COMPILE=1` | Runs each frame's sampling and its 15 depth-decoder steps as one `mx.compile` graph. |
| `BREEZE_MLX_FAST_FIRST=6` | Sends the first six frames one at a time, each decoded before the next frame is queued, with `--chunk-first 1 --chunk-max 1`. The first audio no longer waits for three frames. |
| `BREEZE_MLX_CACHE_GB=1` | Caps MLX's buffer cache, which otherwise kept ~6 GB of freed buffers. |

It also fixes a streaming bug in the pinned mlx-audio that made the decoded audio depend on the chunk size (the transposed convolutions' bias was added twice at every chunk boundary, worst in an utterance's first frames; fixed upstream in [Blaizzy/mlx-audio#1003](https://github.com/Blaizzy/mlx-audio/pull/1003)). This applies to every precision.

Measured on an Apple M4 Pro (14-core CPU, 20-core GPU, 48 GB) with `scripts/bench_mac.py`: 24 lines of one to four Harvard sentences, two in three with a voice direction at CFG 2, a 24 s LibriSpeech reference (speaker 1272, dev-clean `1272-128104-0000` to `-0002`, CC BY 4.0) registered once. "Stall-free start" is the first audio plus any buffer playback needs so it never runs dry: `8bit` and `mixed` needed none on any line; `bf16`, at RTF 0.86, needed ~70 ms on every line (its default 25-frame chunks starve playback outright, hence `--chunk-max 4`).

| Precision | Stall-free start, plain / directed (median) | RTF, directed | Memory |
| --- | --- | --- | --- |
| `8bit` (default) | 212 / 297 ms | 0.58 | 13 GB |
| `bf16` (`--chunk-max 4`) | 380 / 456 ms | 0.86 | 15 GB |
| `mixed` | 139 / 206 ms | 0.62 | 8.9 GB |

Quality against bf16, measured with `scripts/mlx_fidelity.py` (the same tokens through both models; KL of the samplers' logits, lower is closer):

| Precision | Backbone KL, directed | Depth decoder KL, directed |
| --- | --- | --- |
| `8bit` (mxfp8) | 0.0030 (plain lines: 0.0011) | 0.0067 |
| `mixed` | 0 (exact) | 0.0020 |

Voice directions amplify quantization error, because CFG multiplies the gap between its two rows; `8bit` quantizes the backbone too, `mixed` doesn't. In a blind listening test (one listener, ten lines, the same four builds), `8bit` was judged worst on seven lines and best on none, while `bf16`, `mixed` and an all-int8 build tied.

The HTTP and WebSocket API is the same on both backends. [docs/api.md](docs/api.md#differences-on-the-mlx-backend) lists the differences.

> [!NOTE]
> `infer.py` is CUDA-only and doesn't run on the MLX backend. On a Mac, synthesis goes through the server.

> [!IMPORTANT]
> The MLX weights are an unofficial community conversion, published by [mlx-community](https://huggingface.co/mlx-community) and not affiliated with BreezeBlue. They are derivative models of Breeze TTS 2, so the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) still applies: research and non-commercial use only. See [License and responsible use](#license-and-responsible-use).

## First request

With the server running, `POST /v1/audio/speech` takes form fields (only `text` is required) and streams raw mono 24 kHz signed 16-bit little-endian PCM:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -F "text=Hello there." \
  --output hello.pcm
```

For a file a player can open, or to listen in a browser, use `GET /v1/audio/speech.wav`. Open this URL in a browser:

```text
http://127.0.0.1:8080/v1/audio/speech.wav?text=Hello%20there.
```

or save it:

```bash
curl -G http://127.0.0.1:8080/v1/audio/speech.wav \
  --data-urlencode "text=Hello there." \
  --output hello.wav
```

Full reference: [docs/api.md](docs/api.md).

## Command-line synthesis

`infer.py` synthesizes a file directly, without the server. The first argument is the checkpoint directory: the downloaded snapshot directory (under `$HF_HOME/hub/models--BreezeBlue--Breeze-TTS-2/snapshots/`).

Voice clone, from clean reference audio and its exact transcript:

```bash
uv run python infer.py <checkpoint dir> \
  --ref-audio reference_en.wav \
  --ref-text "This is the exact transcript of the English reference audio." \
  --text "(sigh) It is good to hear your voice again after all this time." \
  --output outputs/voice_clone_en.wav
```

Reference audio should contain clean, non-looping speech with minimal background
noise. `--ref-text` should match the complete spoken content of the reference
audio; if speech is repeated in the audio, include those repetitions in the
transcript.

Voice design, from a description with no reference audio. Match the instruction language to the target text; `--cfg-scale 4` strengthens instruction-following:

```bash
uv run python infer.py <checkpoint dir> \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." \
  --cfg-scale 4 \
  --output outputs/voice_design_en.wav
```

Voice direction, which keeps the reference speaker's identity while steering delivery:

```bash
uv run python infer.py <checkpoint dir> \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(clears throat) We need to discuss what happened last night." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --cfg-scale 4 \
  --output outputs/voice_direction.wav
```

Chinese examples are in upstream's [README](https://github.com/breezeblue-ai/breeze-tts#readme).

## Docker

See [docker/README.md](docker/README.md). Build the image with:

```bash
bash docker/build.sh
```

## Development

| Action | Command |
| --- | --- |
| Run the server (CORS open to every origin by default; narrow it with `--cors=http://127.0.0.1:8000`) | `scripts/start_breeze.sh` |
| Run the server on an Apple Silicon Mac (MLX backend) | `scripts/start_breeze_mac.sh` |
| Unit and integration tests (no GPU) | `.venv/bin/pytest` |
| GPU tests | `BREEZE_MODEL=<path> .venv/bin/pytest -m gpu` |
| MLX tests (Apple Silicon Mac) | `BREEZE_MLX_MODEL=<path> .venv/bin/pytest -m mlx` |
| Lint | `.venv/bin/ruff check .` |
| Benchmark | `.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080` (defaults `--warmup 3 --runs 10`) |
| SillyTavern live test | `node tests/live/sillytavern/run.mjs <health\|voices\|speech\|full>` |
| C++ docs example check | `.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080` |

See `specs/003-cpp-compatible-api/quickstart.md` for the full walkthrough these commands are
drawn from.

Environment variables:

- `HF_HOME`: the HuggingFace cache the launchers read the model from.
- `BREEZE_MODEL`: checkpoint path; enables the GPU tests (without it they skip).
- `BREEZE_MLX_MODEL`: path to an MLX snapshot directory; enables the MLX tests (without it they skip).
- `REFERENCE_VOICES_DIR`: used by the reference-voice GPU tests, `bench_api` and the live scripts. It expects `<dir>/eric/eric.wav` and `eric.txt`.
- `BREEZE_CPP_ROOT`: path to a checkout of the C++ server, used by the C++ golden and docs tools.

## Relationship to upstream

This repository is derived from [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts). It branched at upstream commit `43e2ea1` (2026-09-04), and later upstream commits were merged selectively. The files in `breeze_infer/` and the launch scripts are substantially modified or new. See [CHANGELOG.md](CHANGELOG.md) for what changed and how the API differs.

Upstream's model card, blog and community links: [Hugging Face](https://huggingface.co/BreezeBlue/Breeze-TTS-2), [blog](https://breezeblue.ai/breeze-tts-2). Report issues with this fork in this repository, not upstream.

## License and responsible use

The source code is licensed under the [Apache License, Version 2.0](LICENSE). Upstream's code and this fork's modifications are both under Apache 2.0. The audio tokenizer is based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen Team and is licensed under the Apache License, Version 2.0.

Model weights, checkpoints, adapters, derivative models, and self-hosted outputs are governed separately by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE): research and non-commercial use only. Neither upstream's nor this fork's Apache license grants any right to use the model or its outputs commercially, and running this server does not change that.

Per BreezeBlue's [Terms of Service](https://breezeblue.ai/legal/terms), outputs generated through BreezeBlue's hosted platform or API at [breezeblue.ai](https://breezeblue.ai/) can be used commercially with an active paid subscription. A paid subscription does not grant commercial rights to the open-weight model or self-hosted outputs.

You are responsible for complying with applicable laws and obtaining all necessary rights and consents for inputs, reference audio, voices, and outputs. Unauthorized voice cloning, impersonation, fraud, and other unlawful or harmful uses are prohibited.

The code and Model Materials are provided "AS IS," without warranties from the authors of this fork or of upstream, and without liability to the maximum extent permitted by law. Third-party components remain subject to their respective licenses.
