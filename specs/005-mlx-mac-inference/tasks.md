---

description: "Task list for MLX inference on Apple Silicon Macs"
---

# Tasks: MLX Inference on Apple Silicon Macs

**Input**: Design documents from `specs/005-mlx-mac-inference/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/runtime-seam.md](contracts/runtime-seam.md),
[contracts/launch-and-events.md](contracts/launch-and-events.md), [quickstart.md](quickstart.md)

**Tests**: FR-018 asks for them. They are the ones named in the plan's Project Structure and in
research R9 and R10. Don't add tests beyond those named in a task.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (a different file, and no dependency on an incomplete task).
- **[Story]**: US1–US4 from spec.md. US4 is "Linux and Windows users are unaffected".
- Paths are relative to the repo root.

## Standing rules (apply to every task)

1. **Branch**: work on `005-mlx-mac-inference`. Before starting a task, read the research decision
   (Rn) and the contract it cites.
2. **Delegation**:
   - **Sonnet** for tasks without a tag; **Opus** for tasks tagged *(Opus)*.
   - Tasks tagged *(main)* stay in the main session. Tasks tagged *(user)* need the user: the CUDA
     machine, or a human listener.
   - The main session reviews every agent's output before committing.
   - Agent briefs forbid `git stash`, `git checkout` and `git reset`, because agents share one
     working tree. Stage by path.
3. **Verification**: every task ends with `.venv/bin/ruff check breeze_infer tests models/mlx_streaming.py`
   and `.venv/bin/pytest` passing, with the real output shown.
   - `ruff check .` is not used: `main` already has 29 findings in upstream model code
     (`models/generation_breeze.py`, `models/t5gemma2_compat.py`, `models/warmup_profile.py` and
     others), which this feature doesn't touch.
   - From T009 on, the model-free suite must show **0 failures** on this Mac. Before T009 it has
     the 17 known failures listed in research R10.
   - Tasks that say so also run `BREEZE_MLX_MODEL=<8-bit snapshot> .venv/bin/pytest -m mlx`.
     8-bit is the default precision.
     Without `BREEZE_MLX_MODEL` every `mlx` test skips and still exits 0, which proves nothing.
   - The snapshot paths are under `$HF_HOME/hub/` (default `~/.cache/huggingface/hub/`):
     - bf16: `models--mlx-community--Breeze-TTS-2-mlx/snapshots/3c8829fb7fd335818f085cd2ef49b4100c0e46c8`
     - 8-bit: `models--mlx-community--Breeze-TTS-2-mlx-8bit/snapshots/c6e4a2ff6ab9afba68b7853de802273ffe23fb49`
     - 4-bit (config only): `models--mlx-community--Breeze-TTS-2-mlx-4bit/snapshots/3a06d26b172ea4ae1da2f42d708383e9c79d5526`
     - official PyTorch: `models--BreezeBlue--Breeze-TTS-2/snapshots/3e28c5151381a722f1d8661b4118c298caa77aa4`
4. **Commits**: one small commit per task. The message cites FR ids.
5. **Review loop at the end of each phase**, starting at the end of Phase 2 (Phases 1 and 2 are
   reviewed together):
   - A `feature-dev:code-reviewer` agent reviews the phase's commits, and the valid findings are
     fixed.
   - A second pass reviews the fixes, and anything left is fixed. There is no third pass.
   - Run only one agent at a time, and no review while a subagent is still working.
6. **Two-strike rule**: when a fix attempt fails, stop. Explain why, list at least 3 different
   approaches, and propose one.
7. **The CUDA path stays unchanged** (FR-017, User Story 4):
   - Don't edit `models/fast_streaming.py`, `models/cudagraph/`, `models/stream_runtime/`,
     `breeze_infer/runtime.py`, `infer.py`, `scripts/start_breeze.sh`, `scripts/start_breeze.ps1`
     or `docker/`.
   - Existing tests change only in T006–T008 (R10) and where T010 says. If any other existing
     test needs changing, stop and ask.
8. **mlx-audio is a dependency, not our code.** Never edit it in `.venv`. Use only the pinned
   commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Read its source from `.venv` (or the
   research clone) when a task needs its internals.
9. **Mac-only imports are lazy.** `models/mlx_streaming.py` must import on Linux without mlx
   installed: import `mlx`/`mlx_audio` inside functions, never at module top.

10. **Reference prototype.** `specs/005-mlx-mac-inference/research/proto/` holds the measured
    prototype loop (`proto.py`) and the teacher-forced check (`diag_teacher.py`). T014–T018 follow
    its structure. Don't import from it, and don't copy `mx.compile` or the second codec stream
    (research R6).

---

## Phase 1: Setup

- [X] T001 Bump `__version__` to `2.2.0.dev1` in `breeze_infer/__init__.py`. Add an
  `## Unreleased` → `### Added` line to `CHANGELOG.md`: "MLX backend for Apple Silicon Macs
  (`--backend mlx`, `scripts/start_breeze_mac.sh`)". One commit.
- [X] T002 *(main)* **Phase-0 speed gate (research R7).** Do it before any other code task.
  **Done 2026-10-04:** no-go on stock mlx-audio (8-bit CFG RTF 2.61). The user then asked for a
  prototype re-gate: **go** at 8-bit (RTF 0.83 with CFG, re-run by the main session). See
  `research/live-phase0.md`.
  - Use the scratch venv from the R2 probe, or a new one: `requirements.txt` plus
    `mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@e1b19b9054bf163f5d812221a54fcc346f1890e9`,
    installed with `--overrides` pinning `transformers==4.57.3` and `huggingface-hub==0.36.2`.
  - Write a throwaway script in the session scratchpad, not the repo. It loads each snapshot with
    mlx-audio's loader and calls stock `Model.generate()` with `stream=True`.
  - Inputs: one sentence ("Hello there, this is a short test of the streaming voice.") and a
    passage of about 60 s (the first paragraphs of `README.md`'s "What this is").
  - Run four configurations: bf16 and 8-bit, each with no instruction (no CFG) and with
    instruction "A calm, warm voice." plus `cfg_scale=4`.
  - Record, for each configuration and input: time to first audio, RTF (generation time divided
    by audio seconds), and peak memory (`/usr/bin/time -l`, maximum resident set size).
  - Write the results, the machine (M5, 16 GB, macOS version), the commands and the go/no-go
    verdict to `specs/005-mlx-mac-inference/research/live-phase0.md`.
  - **Go** if 8-bit with CFG has RTF ≤ 1.3 on the passage. Otherwise **stop and report to the
    user** with the numbers. Don't start T003.
- [X] T003 Add this line at the end of `requirements.txt`, under a comment
  `# Apple Silicon only: the MLX backend (specs/005-mlx-mac-inference research R1, R2).`:
  `mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@e1b19b9054bf163f5d812221a54fcc346f1890e9; sys_platform == "darwin" and platform_machine == "arm64"`.
  - Create `requirements-mac-overrides.txt` with exactly `transformers==4.57.3` and
    `huggingface-hub==0.36.2`. Add a top comment explaining why: mlx-audio declares
    `transformers>=5.14` and `huggingface_hub>=1.0`, but its Breeze path uses neither (R2).
  - Install into `.venv` with
    `uv pip install -r requirements.txt --overrides requirements-mac-overrides.txt`.
  - Confirm with `.venv/bin/python -c "import mlx.core as mx, transformers; from mlx_audio.tts.models.breeze_tts.breeze_tts import Model; print(transformers.__version__, mx.default_device())"`,
    which should print `4.57.3 Device(gpu, 0)`.
  - Run the model-free suite and confirm the same 17 failures as R10, no more.
- [X] T004 Register the `mlx` marker in `pyproject.toml` (`[tool.pytest.ini_options]` `markers`):
  `"mlx: loads the MLX checkpoint on Apple Silicon; skipped unless BREEZE_MLX_MODEL is set"`.
  - In `tests/conftest.py`, add `MLX_ENV = "BREEZE_MLX_MODEL"`. Extend
    `pytest_collection_modifyitems` so that `mlx`-marked items skip unless that variable is set
    **and** `sys.platform == "darwin"` and `platform.machine() == "arm64"`. Keep the existing
    `gpu` rule byte-for-byte.
  - Add a session fixture `mlx_model() -> Path` mirroring `breeze_model`. Add a session fixture
    `official_model() -> Path | None` that reads `BREEZE_MODEL` and returns `None` when unset.
- [X] T005 Create `tests/mlx/__init__.py` (empty) and `tests/mlx/conftest.py`. In the conftest,
  add a session-scoped fixture `mlx_runtime` that, given `mlx_model`, will load the runtime (T014
  provides `load_mlx_runtime`). Until T014 lands, the fixture body calls `pytest.skip("T014")`.

---

## Phase 2: Foundational (blocks every story)

### macOS green baseline (research R10)

- [X] T006 [P] In `tests/test_speech_abort.py`, make the `LiveServer` harness set
  `TCP_USER_TIMEOUT` only when `hasattr(socket, "TCP_USER_TIMEOUT")` (line 172).
  - Tests that **observe kernel eviction** get
    `@pytest.mark.skipif(not hasattr(socket, "TCP_USER_TIMEOUT"), reason="kernel eviction needs Linux TCP_USER_TIMEOUT")`.
    Decide per test by reading it: does it wait for the connection to be dropped by the kernel?
  - Every other failing test in this file, `tests/test_speech_wav_stream.py` and
    `tests/test_long_text.py` must then pass on this Mac with no other change.
  - List which tests were skipped, and why, in the commit message.
- [X] T007 [P] In `tests/test_api_main.py::test_bound_socket_has_reuseaddr_and_is_listening`,
  assert `!= 0` instead of `== 1`, with a comment explaining that BSD/macOS returns the option's
  bit value (4) for "on".
- [X] T008 [P] In `tests/test_voice_store.py`, fix `test_create_refuses_a_case_duplicate` and
  `test_scan_skips_a_case_duplicate_of_an_earlier_file` for case-insensitive filesystems (APFS).
  - Assert on the store's behaviour: the refusal raised, and the voices listed/skipped, with the
    skip event. Don't use `Path.exists()` on the differently-cased name.
  - Keep what each test proves on Linux.
  - Don't change `breeze_infer/voice_store.py`. If the behaviour itself is wrong on APFS, stop and
    report.
- [X] T009 *(main)* Run the full model-free suite on this Mac and require **0 failures**. Record
  the before (17 failed) and after counts in the Phase 2 review notes.
  **Done 2026-10-04:** before 17 failed / 2055 passed / 55 skipped; after 0 failed / 2069 passed /
  58 skipped (2 eviction tests need Linux `TCP_USER_TIMEOUT`; 1 scan test needs a case-sensitive
  filesystem).

### Backend selection and refusals (research R5, contracts/launch-and-events.md)

- [x] T010 Write the tests first, in `tests/test_settings.py` (new cases only) and the new
  `tests/test_checkpoint_kind.py`. They must fail before T011.
  - **Default and platform:**
    - `settings_from_args([...])` with no `platform` argument gives `backend == "cuda"` for an
      existing CUDA argv. This proves existing calls keep today's behaviour.
    - `platform=Platform("darwin", "arm64", 16 GiB)` with no `--backend` gives `mlx`.
    - `Platform("linux", "x86_64", …)` gives `cuda`.
  - **Each refusal** in contracts/launch-and-events.md: assert `SystemExit` code 2 and the exact
    `<message>` text in stderr (`capsys`):
    - `mlx` on `("linux","x86_64")`;
    - `mlx` on `("darwin","x86_64")`;
    - `mlx` with `memory_bytes = 8 GiB`;
    - `cuda` on `("darwin","arm64")`;
    - `mlx` with `--fast-all`;
    - `mlx` with `--no-fast-all`;
    - `mlx` with `--fast-codec`;
    - `mlx` with `--attn-implementation sdpa`;
    - `mlx` with `--compile-cache-dir X`. The option list in the message is in argv order.
  - **`--attn-implementation`:** omitted on CUDA still resolves to `eager`.
  - **`test_checkpoint_kind.py`:** `checkpoint_kind(dict)` with:
    - `{"model_type": "breeze"}` → (`pytorch`, `bf16`);
    - `{"model_type": "breeze_tts"}` → (`mlx`, `bf16`);
    - `{"model_type": "breeze_tts", "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}` → (`mlx`, `8bit`);
    - the 4-bit dict (`bits: 4, mode: "mxfp4"`) → `ValueError` with the contract's message;
    - unknown `model_type` → `ValueError`.
  - **Checkpoint/backend mismatch:** through `settings_from_args`, using a `tmp_path` checkpoint
    dir with a `config.json`, both directions give the contract's messages.
- [x] T011 Implement in `breeze_infer/settings.py` until T010 passes.
  - **`Platform`:** a frozen dataclass with `system: str`, `machine: str`, `memory_bytes: int`,
    and a `Platform.detect()` classmethod using `sys.platform`, `platform.machine()` and
    `os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")` (macOS only; `0` elsewhere,
    because Windows has no `os.sysconf`).
  - **`CheckpointKind`:** a frozen dataclass with `format: Literal["pytorch","mlx"]` and
    `weights: Literal["bf16","8bit"]`, plus `checkpoint_kind(config: dict) -> CheckpointKind`.
    Rules quoted from data-model.md:
    - "`model_type == "breeze"` → pytorch; `"breeze_tts"` → mlx; anything else is refused";
    - "No `quantization` → bf16; `{bits: 8, mode: "mxfp8"}` → 8bit; anything else is refused.
      PyTorch → `"bf16"`".
  - **`Settings` fields:** add `backend: Literal["cuda","mlx"]` and `weights: str`.
  - **Parser:** add `--backend` with `choices=("cuda","mlx")` and `default=None`.
  - **`settings_from_args`:** gains `platform: Platform | None = None`.
    - `None` means "not an Apple Silicon Mac".
    - Detect explicit CUDA-only options with an argparse action that records each option
      string as typed (decided 2026-10-04, replacing `None` sentinel defaults). Defaults and
      CUDA `Settings` values stay identical.
    - Read `<model_path>/config.json` after the existing directory check.
    - Every refusal goes through `parser.error` with the contract's exact message.
  - Don't change any existing message or default as seen by a CUDA user.
- [x] T012 In `breeze_infer/api.py` `main`, call `Platform.detect()` once and pass it to
  `settings_from_args(argv, platform=...)`.
  - When `settings.backend == "mlx"`, use `device = "mlx:gpu"` and
    `set_device = _select_no_device`.
  - Otherwise keep today's `_cuda_device(environ)` and `torch.cuda.set_device` lines unchanged.
  - Extend `tests/test_api_main.py` only by adding one test that `main`'s device choice for an mlx
    `Settings` is `"mlx:gpu"`. If that needs a seam, factor out a tiny pure
    `choose_device(settings, environ) -> tuple[str, Callable]` and test that.

### The MLX runtime (research R1, R4, R6; contracts/runtime-seam.md)

- [x] T013 Write `tests/test_mlx_room.py`. It runs everywhere, with no mlx and no model.
  - Build `MlxBreezeStreamingRuntime`'s room logic from a small config view (`max_seq_len=2048`,
    the default `max_new_tokens` 750, ceiling `limits.MAX_NEW_TOKENS_CEILING`), without loading
    weights. Expose this as a classmethod or a constructor path that takes a config view, decided
    in T014.
  - **Cases:**
    - `frame_cap(None) == 750`;
    - `frame_cap(5000) == ceiling`;
    - `room_for_length(None, PromptLength(1, 100), prefix_len=0) == min(750, 2048 - 100 - 1)`;
    - prefix_len 1500 with seq_len 500 gives `2048 - 2000 - 1 = 47`;
    - room ≤ 0 when `prefix_len + seq_len ≥ 2047`;
    - `room_for_length(0, …)` raises `ValueError`, as the CUDA validator does;
    - equality with `tests.fakes.FakeRuntime`'s `room_for_length` for the same inputs, with
      `fast_backbone_prefill=False`.
  - The test must fail until T014.
- [x] T014 *(Opus)* Create `models/mlx_streaming.py`, part 1: loading and attributes.
  - **`load_mlx_runtime(path: Path) -> MlxBreezeStreamingRuntime`:** load weights with
    mlx-audio's loader for `breeze_tts`, and the codec from `path / "audio_tokenizer"`.
  - **The tokenizer:** use `transformers.AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)`.
    If `model_type: breeze_tts` makes `AutoTokenizer` fail, load it via the class named in
    `tokenizer_config.json` (`tokenizer_class`) from the same files. Document which path was
    needed in a comment.
  - **Attributes**, per contracts/runtime-seam.md:
    - `sample_rate`;
    - `dtype = torch.bfloat16`;
    - `model`: a small object with `.config` (a read-only view mapping the MLX `config.json` onto
      the field names in the contract table) and `.device = "cpu"`;
    - `tokenizer`;
    - `config = FastStreamingConfig(max_new_tokens=limits.MAX_NEW_TOKENS_CEILING, max_seq_len=2048)`,
      with all `fast_*` False;
    - `fast_enabled = False`;
    - `codec_chunk_frames = 2`.
  - **Room:** `frame_cap`, `room_for_length` and `max_new_tokens_room` use the exact-length rule
    of research R4. `frame_cap` uses the checkpoint's `generation_config.json` `max_new_tokens`
    (750). Import `NoRoomError`, `prompt_length`, `max_reference_prefix_len`,
    `FastStreamingChunk`, `FastStreamingConfig` and `_require_valid_overrides` from
    `models.fast_streaming`, lazily if that import pulls in torch-heavy modules on Linux CI. It
    is fine as is today.
  - **Lazy imports:** all `mlx`/`mlx_audio` imports inside functions (rule 9).
  - Unblock the T005 fixture.
  - **mlx tests**, in `tests/mlx/test_mlx_runtime.py`:
    - the runtime loads, and `sample_rate == 24000`;
    - the `model.config` fields exist with the official model's values;
    - **tokenizer parity:** for every text in a corpus of at least 50 lines, drawn from existing
      test inputs (for example the strings in `tests/test_text_split.py` and
      `tests/test_templates.py`), the token ids equal those of
      `AutoTokenizer.from_pretrained(official_model, fix_mistral_regex=False)`. Skip this check
      when `official_model` is `None`.
  - Verify: T013 passes on the Mac. `pytest -m mlx` passes at 8-bit and at bf16.
- [x] T015 *(Opus)* `models/mlx_streaming.py`, part 2: the codec-encode adapter.
  - `audio_tokenizer` gets `.encode(wav: np.ndarray, sr: int)`, returning
    `{"audio_codes": [torch.LongTensor[frames, 16]]}`, using mlx-audio's Qwen3-TTS encoder. It
    also gets `.get_decode_upsample_rate() -> 1920` from the codec config.
  - Resample exactly as `qwen_tts` does. `breeze_infer/reference_audio.py`'s `predicted_frames`
    documents that arithmetic: reuse it, don't reinvent.
  - **mlx test:** for synthetic mono inputs of 0.5 s, 1 s, 3.7 s and 10 s, at 24 kHz and at
    44.1 kHz, the frame count equals `reference_audio.predicted_frames`, and every code is in
    `[0, codebook_size)`.
- [x] T016 *(Opus)* `models/mlx_streaming.py`, part 3: `iter_audio_chunks` with no CFG and no
  prefix. Follow research R6 and the `iter_audio_chunks` section of
  contracts/runtime-seam.md exactly.
  - **Before the first `next()`:** nothing runs. Then validate the overrides with
    `_require_valid_overrides`, and raise `NoRoomError` when the room is ≤ 0.
  - **Inputs:** convert the template dict (torch CPU → numpy → `mx.array`) in one helper; this is
    the only conversion point.
  - **Prefill:** run the text encoder and the backbone prefill through mlx-audio's model classes
    and `_prompt_embeddings`, and take `input_values` (reference codes) straight from the dict.
  - **Per frame:**
    - Backbone sampling uses the request's `temperature`/`top_k`/`top_p`, with a repetition
      penalty that defaults to `FastStreamingConfig.repetition_penalty` (1.1). Reserved codec ids
      are masked.
    - Depth decoder runs with its own KV cache, sampling with the checkpoint
      `generation_config.json` depth defaults (0.9 / 1.0 / 50). Read them from the file; don't
      hard-code them.
    - Randomness: `key = mx.random.key(seed if seed is not None else 0)`, split per sampling
      call. Never call `mx.random.seed`.
    - `token_observer(torch.tensor(frame_codes, dtype=torch.long))`, with pad frames included.
    - No per-token host syncs: sampling stays on the GPU, with no `.item()` or `float()` on
      an `mx.array` inside the frame. EOS is read at most once per frame, after that frame's
      `mx.eval` (research R6).
    - One `mx.eval` per frame.
    - Don't use `mx.compile`, and don't run the codec on a second stream (research R6).
  - **Codec:** run mlx-audio's codec `streaming_step` every `codec_chunk_frames` frames, and on
    the last frame. Yield `FastStreamingChunk(audio=<float32 numpy>, sample_rate, codec_frames, is_final, timing={})`.
  - **Stop** at EOS or at the room/frame limit, whichever comes first.
  - **Cleanup:** wrap everything after the first `next()` in `try/finally` that drops all
    per-request MLX state, so `close()` aborts cleanly.
  - **mlx tests:**
    - **streaming:** the first chunk arrives before `is_final`, and there is more than one chunk
      for a 3-sentence input;
    - **seed:** the same seed gives identical audio twice, and seed+1 gives different audio;
    - **abort:** calling `.close()` after 3 chunks returns within 1 s, and a following request
      works;
    - `max_new_tokens=24` stops at 24 frames or fewer;
    - `token_observer` is called once per frame, with `num_codebooks` (16) codes: codebook 0
      plus the 15 depth codes, as CUDA's `torch.cat([token, depth_tokens[0]])` sends.
  - Use `templates.prepare_inputs` to build inputs, exactly as `synthesis.prepare_piece` does.
- [x] T017 *(Opus)* `models/mlx_streaming.py`, part 4: CFG.
  - When the inputs carry `cfg_scale` and `cfg_negative_*`, run the conditional and
    unconditional backbone branches **batched** (batch 2) and combine them as
    `uncond + g·(cond − uncond)`, as `fast_streaming.py:1374-1383` does.
  - Pass guidance to the depth decoder as `fast_streaming.py:1408-1420` does, with the depth
    decoder's two branches batched in its KV cache.
  - Reject dual-CFG keys exactly as `select_fast_cfg` does: reuse it by import.
  - `cfg_scale == 0` means the negative prompt alone, as on CUDA.
  - **mlx tests:** an instruction with `cfg_scale=4` produces audio. `cfg_scale=1.0` produces
    audio, and the inputs from `templates.prepare_inputs` carry no `cfg_negative_*` keys. Assert
    on the template output, the observable fact. Don't add a test-only attribute to the
    runtime.
- [X] T018 *(Opus)* `models/mlx_streaming.py`, part 5: reference prefixes.
  - `build_reference_prefix(prefix_inputs) -> MlxReferencePrefix(prefix_len: int, kv)`. It runs
    the batch-1 backbone prefill over the prefix and keeps a per-layer KV snapshot. It raises
    `ValueError` when `prefix_len > max_reference_prefix_len(2048)`, with the same message as
    `fast_streaming.build_reference_prefix`.
  - `iter_audio_chunks(prefix=…)` seeds a **copy** of the snapshot into the request's cache and
    never mutates the cached arrays. Room uses `prefix_len`.
  - **mlx tests:**
    - **Teacher-forced parity, prefix vs inline.** Generate frames for one request with the
      reference inline (seed 42). Then feed those exact frames through both paths, inline and
      with the built prefix, and compare the logits at every sampling point (backbone and each
      depth codebook), following `research/proto/diag_teacher.py`. Assert:
      - (1) every argmax mismatch is a tie, meaning the inline path's top-2 margin is at most
        twice the observed |inline − prefix| logit error at that point;
      - (2) EOS is predicted at the same step.
      Print the maximum logit error and the mismatch count. Exact frame equality is NOT the
      criterion: prefix and inline run different matmul shapes (research R6, "Numerical
      parity"). A mismatch that is not a tie is a bug in prefix seeding: apply the two-strike
      rule.
    - Two requests sharing one prefix both succeed, and the prefix's arrays are unchanged.
- [X] T019 `models/mlx_streaming.py`, part 6: `warmup() -> float`. It runs one short synthetic
  generation without CFG and one with CFG (`cfg_scale=4`), each about 12 frames long, drains them,
  and returns the elapsed ms. Use `templates.prepare_inputs` with the runtime's tokenizer and the
  text "Warm up." **mlx test:** after `warmup()`, the time to first chunk of a one-sentence
  request is under 2 s on the reference Mac. Print the measured value.

---

## Phase 3: User Story 1 - Streamed speech from a Mac (P1) 🎯 MVP

**Goal**: The server starts on the Mac with one command and streams speech on the HTTP routes.

**Independent Test**: quickstart steps 1 and 2.

- [X] T020 [US1] Add the MLX branch to `breeze_infer/model_loading.py` `load_model`.
  - When `settings.backend == "mlx"`: skip `configure_compile_cache`, call
    `models.mlx_streaming.load_mlx_runtime(settings.model_path)`, then `warmup()`. Build the
    report as `{"backend": "mlx", "weights": settings.weights, "device": device, "compile_cache_dir": None, "torch_key": None, "warmup_ms": <ms>, "fx_graph_cache_hits": None, "fx_graph_cache_misses": None}`.
    Return `LoadedModel.from_runtime(runtime, report)`.
  - The CUDA branch adds only `"backend": "cuda", "weights": "bf16"` to its existing report.
  - Extend the existing `model.loaded` test, if there is one in `tests/test_api_main.py`, to
    assert the two new CUDA fields. Otherwise add one assertion where the report is built.
- [X] T021 [US1] Write `tests/mlx/test_mlx_server.py`, which starts the **real** server on real
  uvicorn with the MLX runtime. Reuse `tests.test_speech_abort.LiveServer`, or start
  `python -m breeze_infer.api` as a subprocess on a free port with `--backend mlx --ws-port disabled`.
  - `/health` goes from `503 loading` to `200`, with `sample_rate` 24000.
  - The `model.loaded` event line on stdout has `backend=mlx`, `device=mlx:gpu` and `weights`
    matching the snapshot (`8bit` or `bf16`).
  - `POST /v1/audio/speech` with `text` streams s16le bytes, and the first bytes arrive before
    the response completes.
  - `GET /v1/audio/speech.wav` starts with a 44-byte RIFF header with sizes `0xFFFFFFFF`.
  - A second POST during the first gets `409` with code `busy`.
  - Closing the client mid-stream frees the gate: the next POST succeeds within 5 s.
  - A text long enough to be split into at least 3 pieces (longer than 3× the default
    `--split-chars`) streams every piece in order with no error. This exercises the anchor-codes
    path: piece 0's frames go through `token_observer` and come back as `input_values` for the
    later pieces.
  - Run with `pytest -m mlx`, at 8-bit, and once at bf16 (`BREEZE_MLX_MODEL=<bf16>`).
- [X] T022 [US1] Create `scripts/start_breeze_mac.sh` (POSIX `sh`, executable), modelled on
  `scripts/start_breeze.sh`, per research R8 and contracts/launch-and-events.md.
  - Refuse unless `uname -s` is `Darwin` and `uname -m` is `arm64`.
  - Parse and strip a leading `--precision 8bit|bf16` (default `8bit`, spec FR-012); anything
    else is an error.
  - If `.venv` is missing or `mlx` isn't importable, run
    `uv venv --python 3.12` (when there's no `.venv`), then
    `uv pip install -r requirements.txt --overrides requirements-mac-overrides.txt`.
  - Resolve `$HF_HOME/hub/models--mlx-community--Breeze-TTS-2-mlx[-8bit]/snapshots/<pinned sha>`
    (shas in research R3; `HF_HOME` defaults to `~/.cache/huggingface`). If it's missing, print
    the exact `uvx --from huggingface_hub hf download <repo> --revision <sha>` command and exit 1.
  - `exec uv run python -m breeze_infer.api "$MODEL" --host 0.0.0.0 --port 8080 --cors '*' "$@"`,
    with no `--fast-all`.
  - Verify by hand: no option starts the 8-bit server, `--precision bf16` starts bf16, and a
    wrong `--precision` errors.
  - Confirm `git diff --stat main -- scripts/start_breeze.sh scripts/start_breeze.ps1` is empty.
- [X] T023 [US1] *(main)* Live gate, quickstart steps 1, 2 and 7, on this Mac:
  - the launcher at both precisions;
  - the README `curl` examples;
  - the `.wav` URL in Chrome and in Firefox;
  - the four refusals;
  - suspend the server mid-request (`kill -STOP <pid>`, wait 10 s, `kill -CONT <pid>`). The
    request must either complete or end with the existing error, and the next request must
    succeed (spec Edge Cases, "The Mac sleeps or the process is suspended").
  Record the results in `specs/005-mlx-mac-inference/research/live-us1.md`.

**Checkpoint**: MVP. A Mac user can stream speech over HTTP.

---

## Phase 4: User Story 2 - Voice features on a Mac (P2)

**Goal**: Voice clone, design and direction, and saved voices, on the Mac server.

**Independent Test**: quickstart step 3.

- [x] T024 [US2] Extend `tests/mlx/test_mlx_server.py` with real-server voice cases:
  - a clone request with reference audio plus `ref_text` (use a short WAV synthesized by the
    server itself in an earlier step, with its text, so the test needs no external file);
  - a design request (instruction, `cfg_scale=4`);
  - a direction request (reference plus instruction).
  Each returns non-empty audio of plausible length (more than 0.5 s for one sentence).
- [x] T025 [US2] Extend `tests/mlx/test_mlx_server.py`, saved voices:
  - Upload through `/v1/voices`, synthesize with its `voice_id`, restart the server on the same
    `--voices-dir`, and synthesize again.
  - Copy that voice file with `codec_fingerprint` altered, restart, and assert:
    - the startup event reports the skip;
    - the server still serves the good voice;
    - a request for the skipped id gets `404` with code `unknown_voice` (spec, User Story 2
      scenario 4, Edge Cases).
- [X] T026 [US2] *(main)* Live gate, quickstart step 3, at both precisions. Record the results in
  `specs/005-mlx-mac-inference/research/live-us2.md`.

---

## Phase 5: User Story 3 - WebSocket streaming on a Mac (P3)

**Goal**: WebSocket clients work against the Mac server unchanged.

**Independent Test**: quickstart step 4.

- [x] T027 [US3] Extend `tests/mlx/test_mlx_server.py` with one real-server WebSocket session.
  Use the message sequence from `tests/ws_helpers.py`: open, send text, receive audio frames, then
  the completion message in the documented order.
- [X] T028 [US3] *(main)* Live gate, quickstart step 4, against the Mac server started by the
  launcher:
  - `node tests/live/sillytavern/run.mjs full`;
  - `.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080`.
  Both must pass with no edits to the checks. Record the results in
  `specs/005-mlx-mac-inference/research/live-us3.md`.

---

## Phase 6: User Story 4 - Linux and Windows users are unaffected (P1)

**Goal**: No change for CUDA users (FR-017, SC-004).

**Independent Test**: quickstart step 9.

- [X] T029 [US4] *(main)* On this Mac, confirm that
  `git diff main --stat -- models/fast_streaming.py models/cudagraph models/stream_runtime breeze_infer/runtime.py infer.py scripts/start_breeze.sh scripts/start_breeze.ps1 docker`
  is empty. Confirm that `requirements.txt` differs from `main` only by the mlx-audio line and its
  comment.
  **Done 2026-10-04 at `cd10c74`:** the protected-path diff is empty; `requirements.txt` adds
  only the comment and the mlx-audio line. The final review (T037) then added `mlx==0.32.3` and
  `mlx-metal==0.32.3` under the same darwin/arm64 marker; a Linux resolve still includes no mlx
  package.
- [X] T030 [US4] *(user)* On the CUDA machine, on this branch:
  - `uv pip install -r requirements.txt` (mlx-audio must **not** install);
  - `.venv/bin/pytest`, which must show 0 failures, including the R10 test fixes still passing on
    Linux;
  - `BREEZE_MODEL=<path> .venv/bin/pytest -m gpu`;
  - `bench_api` against `scripts/start_breeze.sh`.
  Compare against `specs/003-cpp-compatible-api/research/bench-final.md`: time to first audio and
  throughput must be within 5%. Paste the outputs. The main session records them in
  `specs/005-mlx-mac-inference/research/live-us4-cuda.md`.
  Also:
  - `bash docker/build.sh`, then the image's `docker/smoke_check.py`. The pip log must show
    `Ignoring mlx-audio: markers … don't match`, and the pins must match.
  - On Windows, if available: `.\scripts\start_breeze.ps1 -Reinstall`. mlx-audio must not
    install, and the server must reach `/health 200`.
- [X] T031 [US4] *(user)* Rollback drill (Constitution VII). On the CUDA machine, with 2.2.0
  running from `scripts/start_breeze.sh`:
  1. Stop it, `git checkout v2.1.0`, `uv pip install -r requirements.txt`, and start
     `scripts/start_breeze.sh` again.
  2. Time it from stop to `/health` returning `200`; it must be under 5 minutes.
  3. Run the README `curl` POST and `.wav` examples, and confirm `X-Breeze-Version: 2.1.0`.
  4. Return to the branch and confirm 2.2.0 comes back the same way.
  Paste the timings and outputs. The main session records them in
  `specs/005-mlx-mac-inference/research/live-rollback.md`.

---

## Phase 7: Polish & cross-cutting

- [X] T032 *(main)* Speed and memory gate, quickstart step 5:
  - `bench_api` on the Mac at bf16 and at 8-bit, with a browser and an editor open;
  - `sysctl vm.swapusage` before and after.
  Check SC-002 and SC-002a, and set SC-006's number from the measured peak. If bf16 misses
  SC-002, the README recommends 8-bit for 16 GB Macs, with the numbers. Record the results in
  `specs/005-mlx-mac-inference/research/live-perf.md`.
- [ ] T033 *(user)* Listening test, quickstart step 6. Prompts: 3 clone, 3 design, 3 direction
  and 1 plain, on CUDA and on the Mac at both precisions. The main session generates the files
  and a comparison table in `specs/005-mlx-mac-inference/research/live-listening.md`; the user
  fills in the judgements.
- [x] T034 [P] Update `README.md`:
  - a "Quick start (macOS, Apple Silicon)" section: requirements (M1+, 16 GB, macOS), the two
    `hf download` commands with revisions, and `scripts/start_breeze_mac.sh [--precision 8bit]`;
  - the Requirements section lists macOS;
  - "Added in this fork" gains the MLX backend line;
  - the Development table gains `scripts/start_breeze_mac.sh` and
    `BREEZE_MLX_MODEL=<path> .venv/bin/pytest -m mlx`;
  - `BREEZE_MLX_MODEL` is added to the environment-variable list;
  - a note that `infer.py` stays CUDA-only (FR-005a);
  - a note that the MLX weights are an unofficial community conversion and still under the
    BreezeBlue non-commercial licence (FR-011).
- [x] T035 [P] Update `docs/api.md` launch options with `--backend` and the CUDA-only options.
  Add a "Differences on the MLX backend" subsection:
  - CUDA-only options are refused;
  - no kernel eviction of stalled readers (R10);
  - the room limit equals CUDA's exact-length mode (R4);
  - the `model.loaded` fields;
  - measured speed from T032.
- [X] T036 Set `__version__ = "2.2.0"` in `breeze_infer/__init__.py`, and rename `## Unreleased`
  to `## 2.2.0 — <date>` in `CHANGELOG.md`, with Added, Changed (the macOS test fixes) and
  Documentation entries. Do this before any tag or deploy (CLAUDE.md).
- [ ] T037 *(main)* Final review loop over the whole branch (standing rule 5). Confirm every FR
  and SC in spec.md maps to a passing test or a recorded live result, and list the mapping in
  `specs/005-mlx-mac-inference/research/done.md` (Constitution IX "Done means").

---

## Dependencies & execution order

- **T002 gates everything after it.** No code task starts until it reports **Go**.
- Phase 1 (T001, T003–T005) → Phase 2.
- In Phase 2:
  - T006–T008 are parallel, then T009.
  - T010 → T011 → T012.
  - T013 → T014 → T015 → T016 → T017 → T018 → T019. These are sequential: one file, each part
    building on the last.
  - The baseline track (T006–T009) and the settings track (T010–T012) can run beside the runtime
    track, since they touch different files.
- Phase 3 (US1) needs all of Phase 2. US2, US3 and US4 each need US1's T020 and T021. After that,
  US2 (T024–T026), US3 (T027–T028) and US4 (T029–T031) are independent.
- Phase 7 needs the stories it documents. T036 comes after T034 and T035, and T037 comes last.

## Parallel examples

- **Phase 2 kickoff:** T006, T007 and T008 together (three test files, Sonnet). Run T010 beside
  them, since it touches settings tests, not the R10 files.
- **After US1:** T024–T025 (US2), T027 (US3) and T029 (US4) touch different code. T024, T025 and
  T027 all append to `tests/mlx/test_mlx_server.py`, so run them one after another, or give each
  agent its own section and merge them by hand.
- **Polish:** T034 and T035 together.

## Implementation strategy

1. **Gate first (T002).** If the M5 can't stream in real time with stock mlx-audio plus the
   expected speed-ups, we learn it before writing code.
2. **MVP = US1** (through T023): streaming HTTP on the Mac. Stop and demo.
3. **Then US2 → US3**, each verified live.
4. **US4 runs once there is code to regress**, on the CUDA machine.
5. **Polish, version 2.2.0, final review.**
