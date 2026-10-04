# Data Model: MLX Inference on Apple Silicon Macs

Nothing here is persisted anew. Saved voice files are unchanged. The entities are values that
exist for the life of the process or of one request.

## Backend (new launch setting)

| Field | Type | Rule |
|---|---|---|
| `backend` | `"cuda" \| "mlx"` | From `--backend`. The default comes from the platform (contracts/launch-and-events.md). Fixed for the process. |

Lives on the frozen `Settings` (`breeze_infer/settings.py`). Validation runs in
`settings_from_args` against a `Platform` value passed in from `api.main`.

## Platform (new, read once at the entry point)

| Field | Type | Source |
|---|---|---|
| `system` | `str` | `sys.platform` |
| `machine` | `str` | `platform.machine()` |
| `memory_bytes` | `int` | `sysctl hw.memsize` on macOS; unused elsewhere |

It is a frozen dataclass, built once in `api.main` and passed down (Constitution III). Tests build
it directly, with no patching.

## Checkpoint kind (new, derived from `<checkpoint dir>/config.json`)

| Field | Type | Rule |
|---|---|---|
| `format` | `"pytorch" \| "mlx"` | `model_type == "breeze"` → pytorch; `"breeze_tts"` → mlx; anything else is refused |
| `weights` | `"bf16" \| "8bit"` | MLX only. No `quantization` → bf16; `{bits: 8, mode: "mxfp8"}` → 8bit; anything else is refused. PyTorch → `"bf16"` |

It is read by a small pure function, `checkpoint_kind(config: dict) -> CheckpointKind`, which
raises a `ValueError` carrying the refusal message. The file read stays at the entry point.

State rule: `format` must match `backend` (`pytorch`↔`cuda`, `mlx`↔`mlx`), or startup is refused.

## MLX runtime (new, process lifetime)

`MlxBreezeStreamingRuntime` (`models/mlx_streaming.py`) holds:

| Field | Type | Notes |
|---|---|---|
| `model` | mlx-audio `breeze_tts.Model`, plus a config view | Weights in bf16 or mxfp8 |
| `codec` | mlx-audio Qwen3-TTS speech tokenizer | fp32, the same file as upstream |
| `tokenizer` | HF fast tokenizer | Identical files to upstream |
| `audio_tokenizer` | adapter (`encode`, `get_decode_upsample_rate`) | See contracts/runtime-seam.md |
| `config` | `FastStreamingConfig` | `max_seq_len = 2048`, `max_new_tokens` ceiling; `fast_*` all False |
| `codec_chunk_frames` | `int` = 2 | Frames per yielded chunk |

## Reference prefix (MLX variant of an existing concept)

| Field | Type | Notes |
|---|---|---|
| `prefix_len` | `int` | The only field the server reads |
| `kv` | per-layer MLX KV arrays (opaque) | Copied into each request's cache; never mutated |

It lives in the existing in-memory `VoicePrefixCache` (LRU, 1 GiB budget), whose byte estimate is
`prefix_bytes_per_token × prefix_len`. The formula is the same as for CUDA, because the KV cache is
bf16 on both backends.

## Request generation state (MLX, per request, freed in `finally`)

| Field | Notes |
|---|---|
| backbone KV cache (batch 1, or 2 with CFG) | Seeded from the reference prefix when there is one |
| depth-decoder KV cache (batch 1 or 2) | Reset every frame |
| codec streaming state | Convolution buffers and codec KV cache for `streaming_step` |
| random key | `mx.random.key(seed)`, split each step |
| token history | For the repetition penalty |

Lifecycle: created on the first `next()` → advanced once per frame → released on `is_final`, on
`close()` (abort), or on an exception.

## Startup report (existing, additive)

The `model.loaded` fields gain `backend` and `weights`. For MLX, `device = "mlx:gpu"` and the
compile-cache fields are `null`. See contracts/launch-and-events.md.

## Saved voice (unchanged)

The file format and validation are unchanged. Voices load only when their fingerprint matches the
running codec. On the Mac that fingerprint equals the CUDA one, because the codec files are
identical (research R3). That makes voices portable in practice, though the spec doesn't promise
it.
