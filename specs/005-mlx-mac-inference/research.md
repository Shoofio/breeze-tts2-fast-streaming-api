# Research: MLX Inference on Apple Silicon Macs

Every decision below is backed by evidence that was read or run on 2026-10-03, not inferred. Two
sources were used:
- [research/remote-candidates-2026-10-03.md](research/remote-candidates-2026-10-03.md): the full
  survey of existing MLX ports, with URLs and commit hashes.
- A read-only map of the server's runtime seam, summarised in R4 and in
  [contracts/runtime-seam.md](contracts/runtime-seam.md).

The user asked that the plan reuse remote code and repositories wherever it makes sense. R1–R3
are where that happens.

## R1. Which MLX code runs the model

**Decision:** Use [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) as a pinned git
dependency, at commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Take from it:
- the Breeze model classes (`mlx_audio/tts/models/breeze_tts/`): T5Gemma2 text encoder, Qwen3
  backbone, depth decoder;
- its weight loader, including its strict codec-weights check;
- its prompt embedding and reserved-token logit masking;
- the Qwen3-TTS codec (`mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py`): its encoder, and
  its incremental decoder (`streaming_step`).

We write only the per-frame generation loop and the adapter that fits the server (R4).

**Rationale:**
- **Faithful port.** mlx-audio's Breeze port reproduces our prompt format: the `[S0]` speaker
  prefix, the reference text, the reference codes plus one EOS frame, then
  `<ins_bos>instruction<ins_eos>text`. It also uses the same reserved-codec-id masking and the
  same default sampling. This was checked against `breeze_infer/templates.py` and
  `models/breeze.py`.
- **Fully MLX.** Everything runs in MLX, including the codec, so no torch work is on the hot path.
- **Licence and activity.** MIT, about 8k stars, actively maintained. Breeze support has been in
  it since PR #911 (2026-08-26).
- **Why a git commit and not PyPI.** The bf16 performance fix (commit `df761b3`, PR #987) merged
  on 2026-10-03, after the latest PyPI release (0.5.7). Without it, bf16 prompt embeddings get
  promoted to float32, and the measured cost is about 3.4× slower generation. Verified:
  `git tag --contains df761b3` is empty.

**Why our own frame loop, and not mlx-audio's `generate()`.** `generate()` falls short of the
spec in six ways, all confirmed in `breeze_tts.py`:

| Gap in `generate()` | Spec need |
|---|---|
| The request's temperature, top_p and top_k also apply to the depth decoder (`breeze_tts.py:1137`) | The server applies them to the backbone only, and the depth decoder keeps the model defaults (FR-009) |
| It seeds the global `mx.random.seed` (`:1016`) | Per-request seed with no global state (Constitution III) |
| Repetition penalty defaults to 1.0 | The CUDA runtime's default is 1.1 (`FastStreamingConfig.repetition_penalty`) |
| It only accepts reference audio | The server passes stored codes (saved voices, the first piece's anchor codes) |
| It splits text on `\n` itself | The server's `text_split.py` already splits text, and must stay the only splitter |
| The depth decoder has no KV cache, and runs the conditional and unconditional passes separately | Speed (R6) |

**Alternatives considered:**
- **rishikksh20/Breeze-TTS-2-mlx.** Rejected. Its codec runs in torch through `qwen-tts` and
  needs SoX. Its weights are int8 only, in split files. It has 4 commits and no benchmark.
- **vanch007/mlx-breeze-tts2 (Sirocco).** Rejected as a dependency. It is a vendored fork of
  mlx-audio that still has the float32 bug (`model.py:227`), its weights are gated, and it has
  been stale since 2026-09-02. Kept as a **reference**: its batched conditional/unconditional
  depth decoder with a KV cache measured about 2.4× faster, and we copy that idea (R6).
- **HoppouAI/Breeze-TTS-2.cpp with ggml Metal.** Rejected. It has no Metal option or docs. Its
  depth decoder is slowed by many small GPU dispatches (0.58× real time on CUDA). It would also
  need a separate process (Constitution I).
- **PyTorch MPS.** Rejected by measurement. The user ran the existing code on MPS: it takes more
  than 5 seconds per second of audio (spec, Clarifications).
- **Writing our own MLX port.** Rejected. It duplicates mlx-audio's roughly 2,900 lines of model
  and codec code, which is exactly the remote code the user asked us to reuse.

## R2. Packaging: one dependency set, with two overrides on the Mac

**Decision:**
- `requirements.txt` gains one line, the only change to that file:
  `mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@e1b19b9…; sys_platform == "darwin" and platform_machine == "arm64"`.
- `qwen-tts` stays as it is, installed on every platform. Only the CUDA runtime uses it (a lazy
  import in `runtime.load_runtime`), so the Mac backend never imports it. Leaving it installed on
  the Mac keeps `requirements.txt` unchanged for Linux and Windows. It also keeps `librosa` (which
  arrives only through `qwen-tts`) available to `tests/test_reference_audio.py:705`, which FR-018
  needs passing on macOS.
- A new file, `requirements-mac-overrides.txt`, holds `transformers==4.57.3` and
  `huggingface-hub==0.36.2`. The Mac install passes it with `uv pip install --overrides`.
- Every other pin, torch included, is the same on every platform.

**Rationale:**
- mlx-audio declares `transformers>=5.14` and `huggingface_hub>=1.0`, but its Breeze path never
  imports transformers. In `sample_utils.py` the name appears only in a comment. It uses only
  `snapshot_download` from huggingface_hub, which 0.36 also has.
- **Measured.** A clean Python 3.12 venv on the reference Mac was installed from the full
  `requirements.txt` (qwen-tts included), plus mlx-audio at the pin, with the two overrides.
  - It imports `mlx_audio…breeze_tts.Model`, the qwen3_tts codec, `qwen_tts`, `librosa`,
    `models.fast_streaming`, `models.stream_runtime.core.compat`, `breeze_infer.routes_speech`
    and `breeze_infer.templates`.
  - Versions: transformers 4.57.3, huggingface_hub 0.36.2, mlx 0.32.3, torch 2.9.1,
    accelerate 1.12.0. The MLX default device is `gpu`.
  - qwen-tts's own `transformers==4.57.3` pin agrees with the override.
  - The model-free test suite gives the same result in this venv as in the existing `.venv`:
    2055 passed, 17 failed, 55 skipped. The 17 failures are already there on macOS (R10).
- The result is one transformers version everywhere, so the server's tokenizer-facing code
  (`templates.py`) behaves identically on both backends.
- Torch stays installed on the Mac, on the CPU only. The server's template code builds torch
  tensors, and changing that would touch working CUDA code paths for no gain (R4).

**Alternatives considered:**
- **A separate Mac environment on transformers 5.x.** Rejected. The server's import chain would
  run under a different transformers major version on each platform. On the Mac that chain
  includes `models/cudagraph/*`, which imports `transformers.StaticCache` and `masking_utils`.
  We would need a token-parity guard between the two versions.
- **Vendoring the files we need into the repo with the MIT notice.** Rejected for now. It means
  about 5,700 lines: the model, the codec, mimi modules and lm helpers. That conflicts with
  Constitution II, and we would lose upstream fixes like the bf16 one. **Fallback** if a future
  pin bump breaks the overrides.
- **`--no-deps` install.** Rejected. It leaves mlx-audio's real dependencies (mlx, scipy,
  miniaudio, sounddevice) undeclared (Constitution III).
- **Marking `qwen-tts` non-macOS.** Rejected. It saves only install size on the Mac. It changes a
  working line of `requirements.txt`. And it would remove `librosa` from the Mac, breaking
  `tests/test_reference_audio.py` there unless `librosa` became a direct dependency.

**Risk:** we call mlx-audio's private classes (`_Backbone`, `_DepthDecoder`,
`_prompt_embeddings`). The exact-commit pin turns that into a deliberate upgrade step. Bumping
the pin means re-running the `mlx`-marked tests and the live gate.

## R3. Weights: which checkpoints, which revision, and how to tell them apart

**Decision:**

| Precision | Repo | Revision (pinned) | Size |
|---|---|---|---|
| 8-bit (default) | `mlx-community/Breeze-TTS-2-mlx-8bit` | `c6e4a2ff6ab9afba68b7853de802273ffe23fb49` | 3.9 GB |
| bf16 | `mlx-community/Breeze-TTS-2-mlx` | `3c8829fb7fd335818f085cd2ef49b4100c0e46c8` | 6.9 GB |

8-bit is the default because bf16 can't stream in real time on the 16 GB reference Mac: RTF 1.45
with every change applied (research/live-phase0.md, re-gate).

- The server identifies an MLX checkpoint by `config.json` `model_type == "breeze_tts"`; the
  official one says `"breeze"`.
- It reads the precision from `config.json` `quantization`:
  - absent means bf16;
  - `{"bits": 8, "mode": "mxfp8"}` means 8-bit;
  - anything else is refused (FR-012). The 4-bit repo has `{"bits": 4, "mode": "mxfp4"}`.
- So the server needs no precision flag: the checkpoint the user points it at decides. The
  launcher picks the repo (R8).

**Evidence (Hugging Face tree API, 2026-10-03):** all three repos share the same file hashes for
these files:
- `tokenizer.json` (`d3ec9ac3…`), `tokenizer_config.json` and `special_tokens_map.json`. So the
  official checkpoint and both MLX checkpoints tokenize text identically.
- `audio_tokenizer/model.safetensors` (`836b7b35…`, fp32, 682 MB) and `audio_tokenizer/config.json`.
  The codec is byte-identical to upstream.

**Consequence for FR-010:** `audio.codec_fingerprint` hashes `audio_tokenizer/config.json`. So a
voice saved by the CUDA server has the same fingerprint as one saved by the Mac server, and the
Mac server will load it. The spec allows this ("MUST load voices whose fingerprint matches"),
and it costs nothing. The codes are codec tokens from the same codec weights. Portability is
still **not tested or promised**, which matches the clarification.

**Alternatives considered:** affine int8 (rishikksh20, and vanch007's gated repo). Rejected: its
files are in a layout mlx-audio won't load, and the repo is gated. Note that the mlx-community
8-bit is **mxfp8**, a microscaling float format, not affine int8. No one has measured its quality,
so SC-005 checks it.

## R4. The seam: where the MLX runtime plugs in

**Decision:** add `models/mlx_streaming.py` with `MlxBreezeStreamingRuntime`. It implements the
same duck-typed surface the server already uses on `FastBreezeStreamingRuntime`. The full list
is in [contracts/runtime-seam.md](contracts/runtime-seam.md). The server's routes, templates,
synthesis, voice prefix cache and WebSocket code do not change.

**What the seam map found (with file references):**
- **What the server uses.** It touches the runtime only through these attributes and methods:
  - attributes: `sample_rate`, `dtype.itemsize`, `model.config.*`, `model.device`, `tokenizer`,
    `audio_tokenizer.encode` and `.get_decode_upsample_rate()`, `config.max_seq_len`;
  - methods: `frame_cap`, `max_new_tokens_room`, `room_for_length`, `build_reference_prefix`
    (an opaque object with `.prefix_len`), and `iter_audio_chunks`;
  - module helpers from `models.fast_streaming`: `NoRoomError`, `prompt_length`,
    `max_reference_prefix_len` and `FastStreamingChunk`.
- **Abort.** There is no abort method: the server calls `.close()` on the `iter_audio_chunks`
  generator, on the GPU thread (`gpu.py:314`, `synthesis.py:620-638`). The MLX generator stops
  at its next `yield`, which comes every 1–2 frames (R6), and frees per-request state in a
  `finally` block.
- **Inputs.** `iter_audio_chunks` receives a dict of **torch** tensors built by
  `templates._collate_inputs`, with CFG riding inside it (`cfg_scale`, `cfg_negative_*`). The MLX
  runtime converts the dict to MLX arrays once per request (torch CPU, to numpy, to `mx.array`).
  Keeping `templates.py` shared guarantees the same prompt on both backends.
- **`token_observer`.** It must receive a 1-D integer torch tensor per frame, because
  `synthesis.anchor_codes` calls `torch.stack` on them. That is one small conversion per frame.
- **Thread.** All runtime calls already run on the single `GpuThread`. MLX's default stream is
  therefore used from one thread only.
- **`model.device`.** The server uses it only as the `.to()` target for template tensors. The
  MLX runtime reports `"cpu"`, because the tensors are converted to MLX anyway.

**Room arithmetic: MLX matches CUDA's exact-length rule. No refactor.**
- `room_for_length` and `max_new_tokens_room` decide when a request is refused for being too
  long (`400 text_too_long`) or clamped.
- **The CUDA rule is not one fixed number.**
  - `FastBreezeStreamingRuntime._prefill_plan` (`fast_streaming.py:1006`) returns the exact
    length (`prefix_len + seq_len`) when `--fast-backbone-prefill` is off.
  - With it on (`--fast-all`), it pads `seq_len` up to a 32-token bucket. That leaves up to 31
    fewer frames of room, and only while the bucket still leaves `MIN_SUFFIX_FRAMES`.
  - So CUDA's limit already depends on its launch flags. Room is `max_seq_len - prefill_len - 1`
    either way, capped by `frame_cap`.
- **The MLX backend has no prefill buckets**, so it uses the exact-length rule:
  `min(frame_cap(requested), max_seq_len - (prefix_len + seq_len) - 1)`. That is what CUDA does
  without `--fast-backbone-prefill`. The constants `MIN_SUFFIX_FRAMES` and `MIN_SUFFIX_ROOM` are
  server-side limits the routes already apply, and they don't depend on the backend.
- This meets FR-005: the MLX backend validates exactly as the CUDA backend does in its eager
  configuration, and its limit is never tighter than CUDA's fast path.
- `frame_cap` and the override validation (`_require_valid_overrides`) are reused by import from
  `models.fast_streaming`, as `tests/fakes.py` already does.
- **No change to `FastBreezeStreamingRuntime`.** An earlier draft of this plan proposed extracting
  the arithmetic into a shared function. That was wrong: it assumed the CUDA rule was fixed, and
  it would have touched working code for no gain.

**Alternatives considered:**
- **A `Protocol` or base class for runtimes.** Not added. Constitution II would allow it now that
  there are two implementations, but the duck-typed surface plus `tests/fakes.py` already pins
  the contract. A Protocol would add a third place to update.
- **Rewriting `templates.py` to emit numpy.** Rejected. It touches every CUDA call site for no
  user-visible gain.

## R5. Backend selection and launch-time refusals

**Decision:**
- **The flag.** A new launch option `--backend {cuda,mlx}`, read once in `settings.py`. If it
  isn't given, the default is `mlx` on macOS and `cuda` everywhere else. An Intel Mac therefore
  gets the accurate "needs an Apple Silicon Mac" refusal rather than a CUDA one (decided
  2026-10-04). The platform
  facts (`sys.platform`, `platform.machine()`, physical memory) are read once in `api.main` and
  passed into `settings_from_args` (Constitution III).
- **Refusals before any weights load.** Each one goes through `parser.error` (usage line, then
  the message; exit status 2), like every existing option rejection. The message names the
  cause:
  - `mlx` on anything other than macOS arm64 (this includes Intel Macs);
  - `cuda` on macOS;
  - `mlx` with less than 16 GB of physical memory;
  - `mlx` with any of `--fast-all`/`--no-fast-all`, `--fast-*`, `--attn-implementation` or
    `--compile-cache-dir`;
  - a checkpoint whose `model_type` doesn't match the backend. The message includes the exact
    `hf download` command (FR-013);
  - an MLX checkpoint whose quantization isn't bf16 or mxfp8 8-bit.
- **Detecting explicit flags.** Each CUDA-only option uses a small argparse action that records
  the option string as typed. That keeps argv order for the refusal message and catches
  `--no-fast-*`, which a `None` default can't. Defaults and `--help` are unchanged (decided
  2026-10-04; this replaces the planned `None` sentinel defaults).

**Rationale:** fail-loud refusals are what Constitution X requires (no silent fallbacks). The
16 GB minimum is the reference machine, and the smallest Apple Silicon Mac with room for bf16
weights (7.6 GB) plus the codec and KV cache. If the Phase-0 gate (R7) shows 8-bit fits
comfortably in 8 GB, the minimum can drop for 8-bit only, as a spec amendment.

**Alternative considered:** auto-detecting the backend from the checkpoint format. Rejected as
the only mechanism, because it hides the choice. It is kept as a consistency check instead: a
checkpoint/backend mismatch is refused.

## R6. The frame loop: speed choices

**Decision:** `MlxBreezeStreamingRuntime.iter_audio_chunks` works frame by frame:
- **Prefill.** Run the text encoder, then prefill the backbone (or start from a cached voice
  prefix, R4).
- **Each frame:**
  - Backbone step: CFG batches the conditional and unconditional branches together (batch 2),
    combined as `uncond + g·(cond − uncond)`. Then repetition penalty (default 1.1), the
    reserved-id mask, and sampling with the request's parameters.
  - Depth decoder: runs with **its own KV cache**, conditional and unconditional batched
    together (the vanch007 technique). It samples with the model's own `generation_config`
    defaults (0.9 / 1.0 / 50), and guidance is applied as on CUDA.
  - Codec: one incremental `streaming_step` every `codec_chunk_frames` frames, which is 2, as
    on the CUDA path without `--fast-codec`. Then yield a `FastStreamingChunk` (float32 numpy
    audio).
- **Randomness.** One `mx.random.key(seed)` per request, split each step. No global seed.
- **No per-token host syncs.** Sampling stays on the GPU: no `.item()`, `float()` or Python-side
  check on an `mx.array` inside the frame. Stock mlx-audio does 16 such syncs per frame. Checks
  that need a value on the host, such as EOS, read it at most once per frame, after that frame's
  `mx.eval`.
- **Evaluation.** One `mx.eval` per frame, so the work for each frame is a single graph.
- **Warmup.** At load, one short synthetic generation (no CFG, then CFG) compiles the Metal
  kernels. That way the first real request meets SC-002. The time is reported as `warmup_ms`.

**Rationale:** these mirror what the CUDA fast path does, and what vanch007 measured. Its 8-bit
model with the batched depth KV cache reached RTF 1.15 (1.53 with CFG=4) on an M3 Max, **with
the float32 bug still in**. mlx-audio after the fix reports RTF 0.6 at bf16 on an M5 Max. Using
`FastStreamingChunk` means `synthesis.ramp_pcm` and the server's flush ramp (`--chunk-first`,
`--chunk-max`) work unchanged.

**Measured (prototype, 2026-10-04; research/proto-2026-10-04.md):** at 8-bit on the M5:

| Change | ms/frame, no CFG | ms/frame, CFG |
|---|---|---|
| stock | 114.0 | 207.4 |
| + depth KV cache | 71.4 | 124.9 |
| + CFG as batch 2 | 71.0 | 72.8 |
| + no per-token syncs | **66.3** (RTF 0.83) | **68.0** (RTF 0.85) |
| + `mx.compile`, fixed KV buffer | 65.6 | 67.5 |
| + codec on a second GPU stream | 64.5 | 66.3 |

**What is not done:** no `mx.compile` and no second GPU stream for the codec. They saved about
1 ms per frame each, near noise, while adding fixed-capacity KV buffers and stream management
(Constitution II). The first three changes already meet SC-002 at 8-bit. The prototype for all
of them is kept for reference in `research/proto/`.

**Numerical parity:** exact frame-for-frame equality with stock is not achievable once the depth
decoder has a KV cache. MLX's quantized matmul uses different kernels for 1 token than for 2–16
tokens, so near-tied logits can flip. The prototype's teacher-forced check
(`research/proto/diag_teacher.py`) showed:
- every argmax mismatch is a tie, meaning the reference's top-2 margin is at most twice the
  observed logit error;
- EOS is predicted at the same step.
Tests compare with that rule, not with exact frames (T018).

## R7. Phase-0 gate: measure before building

No published numbers exist for a base M5 with 16 GB, for time to first audio, or for CFG on mxfp8
(remote research, Risks). So the **first implementation step is a measurement, not code**:
- Run stock mlx-audio at the pin on the reference Mac (M5, 16 GB).
- Measure bf16 and 8-bit, each with and without CFG (`cfg_scale` 4 with an instruction).
- Use a one-sentence input and a roughly 60 s passage.
- Record RTF, time to first audio and peak memory in `research/live-phase0.md`.

**Go/no-go:**
- **Go** if 8-bit with CFG reaches RTF ≤ 1.3 with stock `generate()`. R6's batched depth KV cache
  has measured about 2.4× faster than stock elsewhere, so that leaves room to reach the RTF ≤ 1.0
  that SC-002 needs.
- **Stop and report to the user** if it does not. Options then are `mx.compile`, a smaller
  default chunk, or relaxing SC-002 for 16 GB Macs.
- **Memory.** The same run sets the SC-006 peak-memory number, and confirms or revises the 16 GB
  minimum in R5.

## R8. Launcher

**Decision:** a new script, `scripts/start_breeze_mac.sh`. It:
1. refuses to run on anything other than macOS arm64;
2. installs dependencies when missing (`uv pip install -r requirements.txt --overrides requirements-mac-overrides.txt`);
3. resolves `$HF_HOME/hub/models--mlx-community--Breeze-TTS-2-mlx[-8bit]/snapshots/<pinned sha>`.
   If that directory is missing, it prints the exact `hf download <repo> --revision <sha>`
   command and exits non-zero;
4. execs the server with the same binding and CORS defaults as `start_breeze.sh`, but without
   `--fast-all`.

`--precision 8bit|bf16` is a launcher option, defaulting to `8bit`: it picks the repo and is
stripped before the server sees the arguments. All other arguments pass through.

**Rationale:**
- A separate script leaves `start_breeze.sh` and `start_breeze.ps1` byte-identical (User
  Story 4).
- It is deletable on its own (Constitution II).
- It pins the exact snapshot revision rather than `refs/main` (FR-011).

**Alternative considered:** a Darwin branch inside `start_breeze.sh`. Rejected: it edits the
working Linux launcher.

## R9. Tests

**Decision:**
- **Model-free, every platform** (`.venv/bin/pytest`):
  - settings refusals;
  - backend default selection;
  - checkpoint and precision detection, from small `config.json` fixtures;
  - the MLX room rule: it equals the exact-length formula, and it matches `FakeRuntime`'s
    `_context_room` (the exact-length model the HTTP tests already use) for the same inputs.
    This needs no model; the test builds the runtime's room methods from a config view.
    `models/mlx_streaming.py` must import without mlx installed (lazy mlx imports), so this
    test runs on Linux too.
- **New `mlx` marker.** It runs on macOS arm64 only when `BREEZE_MLX_MODEL` is set, and skips
  elsewhere. These tests use the real checkpoint:
  - token-id parity between `runtime.tokenizer` and the official tokenizer, on the existing
    `tests/` text corpus;
  - the codec encoder's frame count equals `reference_audio.predicted_frames` for several
    lengths;
  - streaming produces audio before generation ends;
  - closing the generator mid-stream frees the runtime;
  - the same seed produces the same audio twice; a different seed produces different audio;
  - CFG and no-CFG both run;
  - a saved voice round-trips;
  - `/health`, `/v1/audio/speech` and the WebSocket end to end, on real uvicorn.
- **No mocks of our own code** (Constitution V). Third-party mlx-audio is exercised for real on
  the Mac, not faked.
- **Live gate (quickstart):**
  - the SillyTavern `full` run and the C++ docs example check (both also cover the WebSocket)
    against the Mac server;
  - `bench_api` on the Mac (SC-002);
  - the 10-prompt listening test at bf16 and 8-bit (SC-005);
  - `bench_api` and `pytest -m gpu` on the CUDA machine (SC-004).

## R10. The existing test suite already fails on macOS

**Finding:** on the reference Mac, the model-free suite gives 2055 passed, **17 failed**, 55
skipped, in both the existing `.venv` and the R2 probe venv. So the failures predate this feature.
FR-018 requires the model-free suite to pass on macOS, so fixing them is in scope. There are three
causes, and all of them are assumptions in the tests about Linux:

| Tests | Cause |
|---|---|
| 14 real-server tests in `test_speech_abort.py`, `test_speech_wav_stream.py`, `test_long_text.py` | The tests' server harness calls `socket.TCP_USER_TIMEOUT` unguarded (`test_speech_abort.py:172`), and that constant is Linux-only. Production code already guards it with `getattr` (`api.py:166`). |
| `test_api_main.py::test_bound_socket_has_reuseaddr_and_is_listening` | It asserts `getsockopt(SO_REUSEADDR) == 1`. macOS returns the flag's bit value (4) for "on". The option is set correctly; the assertion should test for non-zero. |
| 2 case-duplicate tests in `test_voice_store.py` | APFS is case-insensitive by default, so `ALICE.voice.json` "exists" when `alice.voice.json` does. The tests assume a case-sensitive filesystem. |

**Decision:** fix the tests, not the production code.
- Guard the harness's `TCP_USER_TIMEOUT`. On the Mac, skip only the tests that observe kernel
  eviction, with a reason; the rest run.
- Assert that SO_REUSEADDR is non-zero.
- Make the case-duplicate assertions check the store's refusal and listing, not filesystem
  existence. Each fix keeps the same tests passing on Linux.

**A Mac behaviour gap found along the way (no change in this release):**
- `TCP_USER_TIMEOUT` is how the server lets the kernel evict a client that stopped reading
  (600 s; `streaming.py:42`). macOS has no such option, so on the Mac a stalled reader's
  connection can stay open longer. The GPU isn't held: the `.wav` route already releases it when
  generation ends, and the POST route has its own send timeout. Only an idle socket lingers.
- This is recorded in the docs' list of Mac differences. macOS's `TCP_RXT_CONNDROPTIME` is a
  possible later equivalent, but it isn't added, because nothing has shown it is needed
  (Constitution II).

## Resolved unknowns

| Unknown | Resolution |
|---|---|
| Is there an MLX port, and is it faithful? | R1: mlx-audio, prompt format checked |
| Can it install beside our pins? | R2: yes, with two overrides, measured |
| Which weights and revisions? | R3 |
| Is the tokenizer or codec different from upstream? | R3: byte-identical |
| How does it plug into the server? | R4 |
| Speed on a 16 GB M5? | **Unmeasured**. The R7 gate measures it before any build work |
| Memory minimum and peak? | 16 GB provisional (R5). R7 measures the peak (SC-006) |
| Do qwen-tts and mlx-audio install together? | R2: yes, measured |
| Does today's test suite pass on macOS? | R10: no, 17 failures in the tests' Linux assumptions; fixed in scope |
