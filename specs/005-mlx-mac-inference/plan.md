# Implementation Plan: MLX Inference on Apple Silicon Macs

**Branch**: `005-mlx-mac-inference` | **Date**: 2026-10-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/005-mlx-mac-inference/spec.md`

## Summary

Add an MLX inference backend so the server runs on Apple Silicon Macs. It serves the existing
HTTP and WebSocket API unchanged: streaming, every voice mode, saved voices, and the same
validation and errors.

**Approach, following the user's instruction to reuse remote code where it makes sense:**
- **Reused:** the model and codec come from [mlx-audio](https://github.com/Blaizzy/mlx-audio)
  (MIT), pinned to a git commit. The weights are mlx-community's bf16 and mxfp8 8-bit
  conversions, pinned to revisions.
- **Written here:**
  - a frame loop (`models/mlx_streaming.py`) that implements the server's existing runtime seam;
  - a `--backend` launch option, with fail-loud refusals;
  - a macOS launcher.
- **Unchanged:** routes, templates, synthesis, the voice prefix cache, the WebSocket code and the
  Linux and Windows launchers.

The first implementation step is a **measurement gate** (research R7): stock mlx-audio speed on
the 16 GB M5, before any code is written. Version 2.2.0.

## Technical Context

- **Language/Version**: Python 3.12
- **Primary Dependencies**:
  - Existing pins, unchanged: FastAPI/Starlette/uvicorn, torch 2.9.1 and transformers 4.57.3.
  - **New, macOS arm64 only:** `mlx-audio` at git `e1b19b9` (which brings `mlx` 0.32.x, scipy,
    miniaudio and sounddevice), installed with overrides that keep transformers 4.57.3 and
    huggingface-hub 0.36.2 (research R2).
  - `qwen-tts` is unchanged and still installed everywhere. It is measured to coexist with
    mlx-audio (R2).
- **Storage**: N/A. Voice files are unchanged. Weights come from the Hugging Face cache.
- **Testing**: pytest.
  - Model-free suite on every platform.
  - New `mlx` marker (macOS arm64 with `BREEZE_MLX_MODEL`). Existing `gpu` marker.
  - Live gate: SillyTavern `full`, the C++ examples, `bench_api` and a listening test.
- **Target Platform**:
  - macOS on Apple Silicon (M1+, at least 16 GB): new.
  - Linux/WSL and Windows with CUDA: unchanged.
- **Project Type**: Single-process web service (HTTP and WebSocket on one event loop, one GPU
  thread).
- **Performance Goals** (spec SC-002, SC-002a, SC-004):
  - On the M5 16 GB at 8-bit (the default): first audio under 2 s, and RTF ≤ 1.0. A prototype
    measured RTF 0.83 with CFG (research R6). bf16 measured RTF 1.45 and is documented as
    not real time on 16 GB Macs.
  - CUDA within 5% of 2.1.0.
- **Constraints**:
  - API byte-compatible across backends.
  - No change to CUDA behaviour.
  - No silent fallbacks.
  - One process.
  - `infer.py` untouched (FR-005a).
- **Scale/Scope**:
  - New: `models/mlx_streaming.py` (about 500 lines), `scripts/start_breeze_mac.sh`,
    `requirements-mac-overrides.txt`, about 4 test files.
  - Changed: `settings.py`, `api.py`, `model_loading.py`, `requirements.txt` (one added line),
    `pyproject.toml` (marker), README, `docs/api.md`, CHANGELOG, `breeze_infer/__init__.py`.
    `models/fast_streaming.py` is **not** changed (R4).
  - Test fixes so the existing model-free suite passes on macOS: 17 tests in 5 files (R10).

No NEEDS CLARIFICATION items remain. [research.md](research.md) R1–R9 records every decision.
**One open risk:** Mac speed is unmeasured until the R7 gate.

## Constitution Check

*GATE: must pass before Phase 0 research and again after Phase 1 design.*

| # | Principle | Pre-research | Post-design | Notes |
|---|---|---|---|---|
| I | Do not distribute | Pass | Pass | MLX runs in-process on the existing `GpuThread`. The ggml/C++ sidecar was rejected partly on this ground (R1). |
| II | Optimize for deletion | Pass | Pass | Removing the feature means deleting `models/mlx_streaming.py`, the Mac launcher, the overrides file, the `mlx` tests and one `if backend` branch each in `settings`/`api`/`model_loading`. No runtime base class or Protocol is added (R4). The duck-typed seam already has two real implementations, and the contract is written down in [contracts/runtime-seam.md](contracts/runtime-seam.md). About 5,700 lines of vendored code were avoided (R2). |
| III | Explicit dependencies | Pass | Pass | Platform facts are read once in `api.main` into a `Platform` value. The backend is a `Settings` field. The per-request MLX random key replaces global seeding. New dependencies are declared with platform markers, and the overrides are in a checked-in file (R2). |
| IV | Contract at the boundary | Pass | Pass | No wire format changes. `model.loaded` gains two additive fields, and the version goes to 2.2.0. Voice files are unchanged. Template tensors are converted to MLX in exactly one place, the MLX runtime's input adapter. See [contracts/launch-and-events.md](contracts/launch-and-events.md). |
| V | Test the transformation | Pass | Pass | Pure units: settings refusals, `checkpoint_kind`, the MLX room rule. The real adapter runs against real mlx-audio and real weights (`mlx` marker). Nothing we own is mocked. **Recorded deviation (carried from 003):** HTTP tests still use `FakeRuntime` at the GPU edge. |
| VI | Structured events | Pass | Pass | `model.loaded` gains `backend` and `weights`. The refusals print one line to stderr and exit before the event system starts, as existing argparse errors do. No new unstructured logging. |
| VII | Recovery over prevention | Pass | Pass | CUDA users see no behaviour change, so there is nothing to flip. On the Mac, rollback means restarting the 2.1.0 tag. That can't run on a Mac, so the Mac goes from "unsupported" to "unsupported", with no data at risk. No flag is needed: the blast radius is a platform that doesn't work today. Rollback on CUDA is exercised in T031. |
| VIII | Attention is finite | N/A | N/A | No alerts. |
| IX | Value at the user | Pass | Pass | Done means the quickstart live gate passes on the reference Mac and the CUDA regression check passes on the CUDA machine, both recorded in `research/live-*.md`. |
| X | Discoverable commands | Pass (recorded deviation) | Pass (recorded deviation) | `scripts/start_breeze_mac.sh` and `BREEZE_MLX_MODEL=<path> pytest -m mlx` are added to the README Development table, and the launcher fails loudly when weights or platform are wrong. **Recorded deviation (pre-existing, not introduced here):** the repo lists commands in the README table, not in a root `Makefile`/`justfile` that prints its own list, and its model-backed tests read paths from env vars (`BREEZE_MODEL`, now also `BREEZE_MLX_MODEL`). This feature follows the existing convention rather than migrating it. |

**Changes to working code paths** (CLAUDE.md asks for approval before these):
1. ~~Room arithmetic extraction~~: **dropped**. It was approved, but it turned out to be
   unnecessary (R4): the MLX backend uses CUDA's exact-length rule, and
   `FastBreezeStreamingRuntime` is untouched.
2. ~~`--attn-implementation` default~~: **not needed** (decided 2026-10-04 during T011). The
   CUDA-only options record that they were typed through a small argparse action instead, so
   their defaults and `--help` text are unchanged (R5).
3. **Existing tests.** 17 tests fail on macOS today because they assume Linux (R10). They get
   platform-correct assertions, and the harness guards `TCP_USER_TIMEOUT`. They must keep passing
   on Linux.

`requirements.txt` only gains the mlx-audio line. That line is inert on Linux and Windows (its
platform marker), so it is not counted as a change to a working path.

Result: **PASS**, with the three working-path changes above called out for approval.

## Project Structure

### Documentation (this feature)

```text
specs/005-mlx-mac-inference/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/
│   ├── runtime-seam.md          # in-process contract both runtimes implement
│   └── launch-and-events.md     # --backend, refusals, Mac launcher, model.loaded fields
├── checklists/requirements.md
├── research/
│   ├── remote-candidates-2026-10-03.md  # survey of MLX ports (with shas)
│   ├── live-phase0.md                   # R7 gate (implementation step 0)
│   └── live-*.md                        # live gate records
└── tasks.md                     # /speckit-tasks
```

### Source Code (repository root)

```text
breeze_infer/
├── __init__.py          # __version__ → 2.2.0
├── settings.py          # + backend, Platform, checkpoint_kind, refusals; attn default None
├── api.py               # read Platform once; pick device/set_device per backend
├── model_loading.py     # backend branch: load MLX runtime + MLX warmup; report backend/weights
└── (routes, templates, synthesis, voice_*, ws_*: unchanged)

models/
├── fast_streaming.py    # unchanged (MLX imports NoRoomError, prompt_length, frame-cap helpers)
└── mlx_streaming.py     # NEW: MlxBreezeStreamingRuntime, codec adapter, loader

scripts/
├── start_breeze.sh, start_breeze.ps1   # unchanged
└── start_breeze_mac.sh                 # NEW

requirements.txt                 # + one line: mlx-audio (darwin arm64 marker)
requirements-mac-overrides.txt   # NEW: transformers, huggingface-hub pins
pyproject.toml                   # + "mlx" pytest marker

tests/
├── test_settings.py             # + backend default/refusal cases
├── test_speech_abort.py, test_speech_wav_stream.py, test_long_text.py,
│   test_api_main.py, test_voice_store.py   # macOS fixes only (R10)
├── test_checkpoint_kind.py      # NEW: config.json fixtures → kind/refusal
├── test_mlx_room.py             # NEW: MLX room rule (exact length), runs without mlx
├── conftest.py                  # + mlx marker skip rule
└── mlx/                         # NEW: real-weights tests (marker mlx)
    ├── conftest.py
    ├── test_mlx_runtime.py      # tokenizer parity, frame count, seed, CFG, abort, streaming
    └── test_mlx_server.py       # real uvicorn: /health, speech, .wav, WebSocket, voices

README.md, docs/api.md, CHANGELOG.md   # Mac requirements, quick start, options, measured numbers
```

**Structure Decision**: this keeps the existing layout. The MLX runtime sits beside the CUDA one
in `models/`, because both wrap a model. The server package changes only at its composition
points (`settings`, `api`, `model_loading`).

## Implementation order (for /speckit-tasks)

1. **Gate (R7).** Measure stock mlx-audio on the M5 and record `research/live-phase0.md`. Stop
   and report if 8-bit with CFG has RTF > 1.3.
2. **Packaging and a green Mac baseline.** Add the mlx-audio line and the overrides file, and
   confirm a clean Mac install. Fix the 17 macOS test failures (R10), then confirm the model-free
   suite passes on both the Mac and Linux.
3. **Settings.** Add `--backend`, `Platform`, `checkpoint_kind` and the refusals, test-first.
4. **The MLX runtime.**
   - Order: input adapter, prefill, frame loop (CFG, sampling, depth KV cache, random key), codec
     streaming, abort, `build_reference_prefix`, codec encode adapter, warmup.
   - Run the `mlx`-marker tests after each step.
5. **Wiring.** `api`/`model_loading` branches and the `model.loaded` fields. Then the real-server
   `mlx` tests.
6. **The Mac launcher.**
7. **Live gate.** Run quickstart steps 1–7 on the Mac and step 9 on CUDA.
8. **Docs, version 2.2.0, changelog**, before tagging.

## Complexity Tracking

No constitution violations. The working-code-path changes are listed in the Constitution Check
for approval.
