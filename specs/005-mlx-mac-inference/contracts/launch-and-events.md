# Contract: launch options, refusals and the `model.loaded` event

The HTTP and WebSocket API do not change. They are specified in `docs/api.md` and
`specs/003-cpp-compatible-api/contracts/`, and both backends serve them. This file covers only
the surfaces that change.

Version: the server moves to **2.2.0** (additive). `X-Breeze-Version` carries it, as before.

## Server launch option (new)

```
python -m breeze_infer.api <checkpoint dir> [--backend {cuda,mlx}] [existing options]
```

| Platform | Default `--backend` |
|---|---|
| macOS (an Intel Mac is then refused: it needs Apple Silicon) | `mlx` |
| everything else | `cuda` (today's behaviour) |

`<checkpoint dir>` keeps its meaning: the snapshot directory to load. The MLX backend reads the
precision from the checkpoint itself (research R3). There is no precision option on the server.

### Refusals

Each refusal happens before any weights load, inside `settings_from_args`. It goes through
`parser.error()`, as every existing launch-option rejection does: argparse prints the usage line
and then `error: <message>` to stderr, with no traceback, and exits with status 2. The messages
below are the `<message>` part.

| Condition | Message (exact text, with `{…}` filled in) |
|---|---|
| `mlx` on non-macOS or non-arm64 | `--backend mlx needs an Apple Silicon Mac (this machine: {sys.platform} {machine})` |
| `cuda` on macOS | `--backend cuda is not available on macOS; use --backend mlx (the default here)` |
| `mlx` with < 16 GB memory | `--backend mlx needs at least 16 GB of memory (this machine: {n} GB)` |
| `mlx` with CUDA-only options | `{option[, option…]} only apply to --backend cuda; remove them` |
| checkpoint is CUDA-format, backend is `mlx` | `{dir} is the PyTorch checkpoint; --backend mlx needs the MLX weights: uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit --revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49` |
| backend is `mlx`, and `{dir}/config.json` is missing or not a JSON object | `{dir} has no readable config.json; --backend mlx needs the MLX weights: uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit --revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49` |
| backend is `mlx`, and `model_type` is neither `breeze` nor `breeze_tts` | `{dir} has model_type {value!r}; expected 'breeze' (PyTorch) or 'breeze_tts' (MLX)` |
| checkpoint is MLX-format, backend is `cuda` | `{dir} holds MLX weights; --backend cuda needs: uvx --from huggingface_hub hf download BreezeBlue/Breeze-TTS-2` |
| MLX checkpoint with unsupported quantization | `{dir} is {bits}-bit {mode}; the MLX backend supports bf16 and 8-bit (mxfp8)` |

CUDA-only options: `--fast-all`, `--no-fast-all`, `--fast-text-encoder`,
`--fast-backbone-prefill`, `--fast-backbone-decode`, `--fast-depth-decoder`, `--fast-codec`
(and their `--no-` forms), `--attn-implementation`, `--compile-cache-dir`.

The CUDA backend's options, defaults and refusals are unchanged. With `--backend cuda`, a
missing or unreadable `config.json` is ignored, as today; only one that says `breeze_tts` is
refused.

## macOS launcher (new)

```
scripts/start_breeze_mac.sh [--precision 8bit|bf16] [server options…]   # default 8bit
```

- Refuses on anything other than macOS arm64.
- Installs dependencies with the Mac overrides when they are missing.
- Resolves the pinned snapshot of the chosen repo in `$HF_HOME`, and prints the `hf download`
  command if it is missing.
- Runs the server with `--host 0.0.0.0 --port 8080 --cors '*'` followed by the user's arguments,
  as `start_breeze.sh` does, but without `--fast-all`. A later `--host` or `--cors` overrides.

## `model.loaded` event (additive fields)

Existing fields keep their names and meanings. New fields:

| Field | Type | CUDA value | MLX value |
|---|---|---|---|
| `backend` | `"cuda" \| "mlx"` | `"cuda"` | `"mlx"` |
| `weights` | string | `"bf16"` | `"bf16"` or `"8bit"` |

Existing fields on the MLX backend:
- `device` is `"mlx:gpu"`.
- `compile_cache_dir`, `torch_key`, `fx_graph_cache_hits` and `fx_graph_cache_misses` are `null`.
- `warmup_ms` is the time of the warmup generation.

`model.load_failed` is unchanged. The refusals above happen before the event loop starts, so
they don't emit it.

## `/health`

Unchanged in shape and status codes. The spec's FR-014 allowed adding fields there, and this plan
doesn't use that: the event carries the facts, and no client asked for them in `/health`.

## Saved voice files

Unchanged: format `breeze-tts-voice` version 1. The Mac backend writes the same format, stamped
with its codec fingerprint. That fingerprint equals the CUDA one, because the codec files are
identical (research R3).
