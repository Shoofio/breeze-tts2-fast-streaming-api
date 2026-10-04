# Contract: the streaming runtime seam

This is an **in-process** contract, not a wire format. It lists everything the server
(`breeze_infer/`) uses on its runtime object. `FastBreezeStreamingRuntime` (CUDA),
`MlxBreezeStreamingRuntime` (new) and `tests/fakes.FakeRuntime` all implement it. A change here
means changing all three in the same diff.

The source is the 2026-10-03 seam map (research R4). The file references are where the server
calls each member.

## Attributes

| Member | Type | Used by | MLX runtime provides |
|---|---|---|---|
| `sample_rate` | `int` (24000) | `/health`, WAV header, WebSocket, voices | from the codec config |
| `dtype.itemsize` | `int` | `api.prefix_bytes_per_token` (KV cache budget) | `torch.bfloat16`, because the KV cache is bf16 at both precisions |
| `model.config` | object with `num_codebooks`, `codebook_pad_token_id`, `codec_config.codebook_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `head_dim` | `api.py`, `routes_speech.py`, `ws_server.py`, `synthesis.py` | a read-only view built from the MLX `config.json`, with the same field names |
| `model.device` | anything `torch.Tensor.to()` accepts | `templates.prepare_inputs` | `"cpu"` (the tensors are converted to MLX per request) |
| `tokenizer` | HF tokenizer, deep-copyable | `templates.py`, `routes_speech.CpuTokenizer` | `AutoTokenizer` (transformers 4.57.3) from the MLX checkpoint's tokenizer files, which are identical to the official ones |
| `audio_tokenizer.encode(wav: np.ndarray, sr: int)` | returns `{"audio_codes": [LongTensor[frames, 16]]}` | `audio.encode_prompt_waveform` | an adapter over mlx-audio's codec encoder. Its frame count MUST equal `reference_audio.predicted_frames` |
| `audio_tokenizer.get_decode_upsample_rate()` | `int` (1920) | `synthesis.codec_samples_per_frame` | from the codec config |
| `config.max_seq_len` | `int` (2048) | `synthesis.py` | 2048, so validation matches CUDA |
| `fast_enabled` | `bool` | `model_loading.load_model` (warmup branch) | `False`. MLX warmup goes through its own branch (research R6) |

## Methods

| Method | Contract |
|---|---|
| `frame_cap(requested: int \| None) -> int` | `None` → model default (750); clamp to `limits.MAX_NEW_TOKENS_CEILING` |
| `max_new_tokens_room(requested, inputs, *, prefix_len=0) -> int` | `room_for_length(requested, prompt_length(inputs), prefix_len=…)` |
| `room_for_length(requested, PromptLength, *, prefix_len=0) -> int` | Validates `requested` (`ValueError`), then `min(frame_cap(requested), context room)`. Context room is `max_seq_len - prefill_len - 1`, ≤ 0 meaning none. CUDA's `prefill_len` may include 32-token bucket padding under `--fast-backbone-prefill`; MLX's is always `prefix_len + seq_len` (research R4) |
| `build_reference_prefix(prefix_inputs: dict) -> obj` | Returns an object with `.prefix_len: int`; everything else is opaque to the server. Raises `ValueError` when `prefix_len > max_reference_prefix_len(max_seq_len)` |
| `iter_audio_chunks(inputs, *, request_id, seed, token_observer, prefix, temperature, top_k, top_p, repetition_penalty, max_new_tokens) -> Iterator[FastStreamingChunk]` | See below |

### `iter_audio_chunks`

- **Inputs.** `inputs` is the dict from `templates._collate_inputs`: torch tensors, with CFG
  carried in `cfg_scale` and the `cfg_negative_*` keys. Dual-CFG keys are rejected, as on CUDA.
- **Lazy.** No work happens until the first `next()`. Validation errors (`ValueError`,
  `NoRoomError`) are raised there.
- **Sampling.** `temperature`, `top_k`, `top_p` and `repetition_penalty` apply to the
  **backbone** only. `None` means the model default (repetition penalty 1.1). The depth decoder
  uses the model's `generation_config` defaults.
- **Seed.** `seed` fully determines the random stream for this request. No global random state
  is read or written.
- **`token_observer`.** If given, it is called once per generated frame, pad frames included,
  with a 1-D integer **torch** tensor of `1 + num_codebooks` codes.
- **Output.** Each yielded chunk is a `FastStreamingChunk`. `.audio` is 1-D float32 numpy at
  `sample_rate`, and `is_final` is `True` on the last one. Chunks come every
  `codec_chunk_frames` (2) frames.
- **Abort.** `close()` on the generator ends it at the next yield, and a `finally` block frees
  all per-request state. The server does this on the GPU thread.
- **Errors mid-stream** are any `Exception`. They follow the server's existing generic failure
  path.

## Module helpers the server imports (unchanged location)

`models.fast_streaming.NoRoomError`, `prompt_length`, `max_reference_prefix_len`,
`FastStreamingChunk`, `FastStreamingConfig`. The MLX runtime imports them from there; there are
no duplicates.

## Threading

Every call is made from the single `GpuThread`. Implementations may rely on that, for example
MLX's default stream or CUDA's current device.
