# Remote MLX options for Breeze TTS 2 (research, 2026-10-03)

## TL;DR
- **Runtime: Blaizzy/mlx-audio**, model `mlx_audio/tts/models/breeze_tts/` (MIT, very active). Use git main at `e1b19b9` or a release newer than 0.5.7. The bf16 speed fix is **not in PyPI 0.5.7**.
- **Weights: `mlx-community/Breeze-TTS-2-mlx`** (bf16, sha `3c8829fb`) and **`mlx-community/Breeze-TTS-2-mlx-8bit`** (sha `c6e4a2ff`, *mxfp8* group 32, not affine int8). Both are ungated and converted with mlx-audio's own generic converter.
- **Write our own generation loop** on top of mlx-audio's modules. Its `generate()` is missing: separate depth-decoder sampling, abort checks per frame, per-request RNG, pre-encoded reference codes, repetition penalty 1.1 by default, and a depth-decoder KV cache. Reuse its model classes, loader, prompt embedding, and the codec's `streaming_step`.
- **Blocking packaging conflict:** mlx-audio requires `transformers>=5.14.0`, and the repo pins `transformers==4.57.3` and `torch==2.9.1`. We need either a separate Mac dependency set or a vendored copy of the needed files (MIT). vanch007 already vendored them and runs on `transformers>=4.48`.

## Candidates

### 1. Blaizzy/mlx-audio, `breeze_tts` (recommended runtime)
- Repo: https://github.com/Blaizzy/mlx-audio. MIT, about 7,981 stars, pushed 2026-10-03. HEAD `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
- Breeze landed in PR #911 (2026-08-26, author "Luna", who also owns LunaFox) and PR #941 (text segmenting, 2026-09-04). PR #987 (bf16 dtype fix, commit `df761b3`) merged 2026-10-03, after PyPI `mlx-audio==0.5.7` (2026-09-28). Breeze support first shipped in v0.5.1.
- Files: `breeze_tts.py` (1243 lines), `config.py`. It reuses `qwen3_tts/speech_tokenizer.py` for the codec and the vendored `mlx_audio/lm` for Qwen3/Llama blocks, KVCache and the sampler. Tests: 32 unit tests on tiny configs (`tests/test_breeze_*.py`). None of them check numerical parity against PyTorch.
- Runtime dependencies: mlx>=0.31.1, transformers>=5.14 (only `PreTrainedTokenizerFast`), huggingface_hub>=1.0, numpy, scipy, miniaudio, sounddevice, tqdm. **No torch.**
- Components: everything runs in MLX. That covers the T5Gemma2 encoder (bidirectional, sliding and full masks), the Qwen3 backbone, the depth decoder (Llama blocks), and the Qwen3-TTS codec encoder and decoder. The codec is loaded from `audio_tokenizer/`, and its key set is validated strictly.
- Features (from the code):
  - **Streaming:** yes. `audio_tokenizer.decoder.streaming_step()` is truly incremental, with conv buffers and a transformer KV cache. Chunk size is `streaming_interval` in seconds, default 2.0.
  - **Clone, design and direction:** yes.
  - **CFG:** single CFG, and only when `instruct` is set. The negative branch is the same prompt without the instruction, which matches our `tts_instruction` / `ref_edit_tata` negatives.
  - **Seed:** yes, but through the global `mx.random.seed`.
  - **Sampling:** one shared `temperature`, `top_p` and `top_k` for **both** decoders. There is no separate depth setting.
  - **repetition_penalty:** present, default 1.0. The official fast_streaming default is 1.1.
  - **Abort:** only by closing the generator, which takes effect at yield points (chunk boundaries).
  - **Long text:** splits on `\n`, then at most 600 characters per chunk. The reference audio is re-encoded for every segment.
  - **Depth decoder:** recomputes the whole prefix for each codebook with no KV cache, and runs cond and uncond as separate passes.
- Faithfulness, checked against our `breeze_infer/templates.py` and `models/breeze.py`:
  - Matches: the `[S0]` speaker prefix, `{spk}{ref_text}`, then reference codes plus one EOS frame (`codebook_eos_token_id` on all 16 codebooks), then `{spk}<ins_bos>{ins}<ins_eos>{text}`. Each text segment is encoded separately by the T5 encoder and goes through `text_encoder_proj`. Reserved codec ids 2048–2050 are masked. EOS is index `vocab_size`. The default of 750 frames and the 0.9/1.0/50 sampling defaults also match.
  - Differences: mlx-lm `make_sampler` applies top-k/top-p in a different order (it only matters when top_p<1). Depth sampling parameters cannot be set separately. There is no dual CFG (our fork does not use it). The default repetition penalty differs.
- Performance: commit `df761b3` reports, on an M5 Max with bf16, backbone step 29.1→7.7 ms, depth step 9.0→2.3 ms, and **RTF 2.1→0.6** (elapsed/audio). CFG is presumably off in that measurement. There are no published numbers for time-to-first-audio, memory, the base M5 or 16 GB machines.
- Conversion: the generic `python -m mlx_audio.convert` (`--q-mode affine|mxfp4|nvfp4|mxfp8`) copies subdirectories and rewrites `model_type` to `breeze_tts`.

### 2. mlx-community weights (recommended)
- bf16 `mlx-community/Breeze-TTS-2-mlx` @ `3c8829fb7fd335818f085cd2ef49b4100c0e46c8`: 2 shards, 6.91 GB.
- 8-bit `mlx-community/Breeze-TTS-2-mlx-8bit` @ `c6e4a2ff6ab9afba68b7853de802273ffe23fb49`: 3.89 GB, `{"bits":8,"group_size":32,"mode":"mxfp8"}`.
- 4-bit `mlx-community/Breeze-TTS-2-mlx-4bit` @ `3a06d26b…` (mxfp4, out of scope).
- Layout: same key names as upstream. `codec_model.*` is dropped (an unused Mimi codec, 350 keys). The tied depth/backbone audio embedding is materialised. `audio_tokenizer/model.safetensors` is **byte-identical** to upstream (LFS oid `836b7b35`, fp32, 682 MB).
- What is quantized: in 8-bit, every Linear and Embedding in the text encoder, backbone, depth decoder, lm_head and text_encoder_proj. Norms, `codebooks_head` and the codec stay unquantized.
- Upstream weights have not changed since `a3bd0a6` (2026-08-25). Later commits only touch the README and licence (v1.1 on 2026-09-01). Upstream HEAD is `3e28c515`.
- `LunaFox/Breeze-TTS-2-mlx-4bit` @ `27be05f0`: its README says "moved" to mlx-community. `npario/Breeze-TTS-2-mlx-8bit` @ `351df91b` is a duplicate re-upload of the 8-bit conversion.

### 3. rishikksh20/breeze-tts-mlx + `rishikksh20/Breeze-TTS-2-mlx` (INT8)
- Code: https://github.com/rishikksh20/breeze-tts-mlx. Apache-2.0, 4 commits, last on 2026-08-30 (`4e045c13`), 4 stars. Not on PyPI, requires Python 3.12.
- **Hard dependencies on torch==2.9.1, torchaudio, transformers==4.57.3, qwen-tts==0.1.1 and the SoX binary.** The codec runs in **PyTorch** (MPS or CPU) through `qwen_tts`, and so do prompt building and the streaming codec runtime.
- MLX parts: text encoder, backbone, and a depth decoder **with a KV cache**. It has **separate backbone and depth sampling configs**, a NumPy RNG seed, single CFG (dual CFG raises an error), and `iter_audio_chunks` streaming. It is essentially a port of our `models/fast_streaming.py`.
- Weights @ `92266ef6`: split files per component, affine int8 g64, KV cache in fp16, codec fp32, source revision `c1c8ca18`. This layout does not load in mlx-audio. Converter: `breeze_tts_mlx/convert.py`.
- No published benchmark figures. "0.53x realtime" in the README is only an example of the output format.

### 4. vanch007/mlx-breeze-tts2 + `vanch007/Sirocco-MLX-{BF16,8bit,4bit}`
- Code: https://github.com/vanch007/mlx-breeze-tts2. MIT, `51c182dd` (2026-09-02). 51 commits in 3 days, then nothing; 2 stars. Install with pip from git (v0.1.0).
- It is a standalone **vendored derivative of mlx-audio's breeze_tts** (THIRD_PARTY_NOTICES says so). Dependencies: mlx>=0.31.1, `transformers>=4.48`, no torch (torch only in the optional evaluation extra).
- Additions: "fast depth" (a compiled incremental depth KV cache plus batched CFG), cancellation tested by closing the generator, a FastAPI server, a strict-audit converter (affine; the "sensitive-bf16" policy keeps the text encoder and depth decoder in bf16), and an extensive evidence bundle (CER, ECAPA, PyTorch parity).
- Still missing: separate depth sampling, and per-request RNG (global seed).
- **It still has the fp32-promotion bug** fixed upstream in `df761b3` (`model.py:227`). Its reported numbers (M3 Max 128 GB) are therefore pessimistic:
  - BF16: RTF 3.78, TTFA 3.6 s, peak memory 12.0 GB.
  - 8-bit with fast depth: RTF 1.15, 1.53 with CFG=4, TTFA 1.30 s, peak memory 8.7–10.8 GB.
- Weights are **gated** (manual licence click-through): BF16 `4c7eec64`, 8bit `45c58f3a`, 4bit `0c4f0950`.
- Main value for us: a reference for the depth-decoder KV-cache optimisation and for vendoring.

### 5. HoppouAI/Breeze-TTS-2.cpp (ggml)
- https://github.com/HoppouAI/Breeze-TTS-2.cpp. Apache-2.0, `a0e177f9` (2026-09-29), 24 stars. GGUF weights in `HoppouAI/Breeze-TTS-2.cpp` @ `81b22bad` (f16, q8_0, q6_k, q4_k).
- CMake only offers `BREEZE_VULKAN` (on by default) and `BREEZE_CUDA`. **There is no Metal option, and the docs never mention Metal or Apple.** At runtime it picks any GPU backend ggml was built with, and the ggml submodule enables Metal by default on Apple, so a Metal build *might* work. That is **unverified**.
- The depth decoder is dispatch-bound: it builds a new cgraph every step. CUDA came out at 0.58x versus Vulkan at 1.53x on an RTX 3060, which suggests Metal would suffer the same way.
- It does have streaming HTTP/WS, cancellation, CFG and seed. Our spec 003 already mirrors its API.

Other repos seen and out of scope: `cstr/breeze-tts-2-GGUF` (CrispASR) and `EvoAwaken-Workshop/Breeze-TTS-2-gguf` (Rust, "No streaming API"). Neither claims Metal.

## Feature matrix (from the code)

| | mlx-audio | rishikksh20 | vanch007 | .cpp |
|---|---|---|---|---|
| All components in MLX/no torch | yes | **no (torch codec)** | yes | n/a (ggml) |
| Incremental streaming | yes (streaming_step) | yes | yes | yes |
| Clone / design / direction | yes / yes / yes | yes / yes / yes | yes / yes / yes | yes |
| CFG | single (instruct only) | single | single + batched | yes |
| Separate depth sampling | **no** | yes | **no** | ? |
| Seed | global mx | NumPy RNG | global mx | yes |
| Abort mid-gen | generator close at chunk | generator close | generator close (tested) | yes (WS) |
| Depth KV cache | **no** | yes | yes (fast) | n/a |
| bf16 / 8-bit weights | mlx-community bf16 / mxfp8 | int8 only | bf16 / affine8 (gated) | f16 / q8_0 |
| Licence (code) | MIT | Apache-2.0 | MIT | Apache-2.0 |
| Activity | high | stale since Aug 30 | stale since Sep 2 | active |
| Conversion script | `mlx_audio.convert` | `convert.py` | `mlx-breeze-tts2 convert` | `apps/convert` |

## What to build versus reuse
**Reuse from mlx-audio:** `Model` / `ModelConfig`, `sanitize` and `post_load_hook` (tokenizer plus strict codec load), `_prompt_embeddings`, `_mask_reserved_codec_logits`, the backbone KVCache, and the codec `streaming_step` / `reset_streaming_state`.

**Build:**
1. Our own frame loop. It needs separate backbone and depth sampling, an abort `Event` checked every frame, a per-request RNG (`mx.random.key` splits instead of the global seed), repetition penalty 1.1, and 1–2-frame codec chunks to match `fast_streaming`.
2. Feed our stored reference codes straight into `backbone_model.embed_tokens`. mlx-audio only accepts audio.
3. A depth-decoder KV cache and batched cond/uncond depth. vanch007 got about 2.4x from this.
4. Disable mlx-audio's own text splitting (`split_pattern=None`) and keep `breeze_infer/text_split.py`.
5. Run all MLX work on one dedicated thread. The existing GPU executor fits.

## Risks
- **Licence:** the weights stay under BreezeBlue's Research and Non-Commercial licence. MLX conversion adds no new restriction. The code licences (MIT, Apache) are compatible with Apache-2.0. Only the vendored files need the MIT notice.
- **Dependency conflict:** transformers 5.14 versus 4.57.3, plus `breeze_infer/templates.py` imports torch.
- **PyPI 0.5.7** runs bf16 about 3.4x slower. Pin `e1b19b9`.
- **16 GB M5:** bf16 is about 7.6 GB of weights, and vanch007 measured a 12 GB peak. A base M5 has much less bandwidth than the M5 Max behind the RTF 0.6 figure, so real-time with CFG is **unproven**. Benchmark 8-bit first.
- **mxfp8 quality** has not been measured by anyone. The only quantization evidence is vanch007's affine runs.
- mlx-audio has no published numerical parity against PyTorch.

## Sources
- https://github.com/Blaizzy/mlx-audio @ e1b19b9054bf163f5d812221a54fcc346f1890e9 (breeze commits 903903f, 752b3d6, ecd0a90, 151a0bc, df761b3; PRs #911, #941, #987); https://pypi.org/project/mlx-audio/ 0.5.7
- https://huggingface.co/mlx-community/Breeze-TTS-2-mlx @ 3c8829fb7fd335818f085cd2ef49b4100c0e46c8
- https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-8bit @ c6e4a2ff6ab9afba68b7853de802273ffe23fb49
- https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-4bit @ 3a06d26b172ea4ae1da2f42d708383e9c79d5526
- https://huggingface.co/LunaFox/Breeze-TTS-2-mlx-4bit @ 27be05f01bd8aad9628022c2bac6ded0119eef8a
- https://huggingface.co/npario/Breeze-TTS-2-mlx-8bit @ 351df91b53b3b5df8dd29d5c71b11765b74873b4
- https://huggingface.co/rishikksh20/Breeze-TTS-2-mlx @ 92266ef6e15dd4be290be416dd6b0bf9a2bab81b; https://github.com/rishikksh20/breeze-tts-mlx @ 4e045c13c4627b3f1a49ea1172f24efe06d2c16b
- https://huggingface.co/vanch007/Sirocco-MLX-BF16 @ 4c7eec64281272a32376234075e955dce50c9252 (8bit 45c58f3a, 4bit 0c4f0950); https://github.com/vanch007/mlx-breeze-tts2 @ 51c182dd0d6453a37d6a2e1233ffa42bdb71016d
- https://github.com/HoppouAI/Breeze-TTS-2.cpp @ a0e177f91242ebbd2ea8d61548d742c89e4b9e06; https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp @ 81b22bad9f05b99970e30c5ee5e4bbc52fedf2f8
- https://huggingface.co/BreezeBlue/Breeze-TTS-2 @ 3e28c5151381a722f1d8661b4118c298caa77aa4 (weights from a3bd0a6)
