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
  - `qwen-tts` becomes non-macOS only.
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
  - On the M5 16 GB at 8-bit: first audio under 2 s, and RTF ≤ 1.0. bf16 meets the same targets,
    or the docs say why not.
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
  - Changed: `settings.py`, `api.py`, `model_loading.py`, `models/fast_streaming.py` (room
    arithmetic extracted, no behaviour change), `requirements.txt`, README, `docs/api.md`,
    CHANGELOG, `breeze_infer/__init__.py`.

No NEEDS CLARIFICATION items remain. [research.md](research.md) R1–R9 records every decision.
**One open risk:** Mac speed is unmeasured until the R7 gate.

## Constitution Check

*GATE: must pass before Phase 0 research and again after Phase 1 design.*

| # | Principle | Pre-research | Post-design | Notes |
|---|---|---|---|---|
| I | Do not distribute | Pass | Pass | MLX runs in-process on the existing `GpuThread`. The ggml/C++ sidecar was rejected partly on this ground (R1). |
| II | Optimize for deletion | Pass | Pass | Removing the feature means deleting `models/mlx_streaming.py`, the Mac launcher, the overrides file, the `mlx` tests and one `if backend` branch each in `settings`/`api`/`model_loading`. No runtime base class or Protocol is added (R4). The duck-typed seam already has two real implementations, and the contract is written down in [contracts/runtime-seam.md](contracts/runtime-seam.md). The room-arithmetic extraction gets its two callers in the same diff. About 5,700 lines of vendored code were avoided (R2). |
| III | Explicit dependencies | Pass | Pass | Platform facts are read once in `api.main` into a `Platform` value. The backend is a `Settings` field. The per-request MLX random key replaces global seeding. New dependencies are declared with platform markers, and the overrides are in a checked-in file (R2). |
| IV | Contract at the boundary | Pass | Pass | No wire format changes. `model.loaded` gains two additive fields, and the version goes to 2.2.0. Voice files are unchanged. Template tensors are converted to MLX in exactly one place, the MLX runtime's input adapter. See [contracts/launch-and-events.md](contracts/launch-and-events.md). |
| V | Test the transformation | Pass | Pass | Pure units: settings refusals, `checkpoint_kind`, the room arithmetic (pinned to today's CUDA numbers). The real adapter runs against real mlx-audio and real weights (`mlx` marker). Nothing we own is mocked. **Recorded deviation (carried from 003):** HTTP tests still use `FakeRuntime` at the GPU edge. |
| VI | Structured events | Pass | Pass | `model.loaded` gains `backend` and `weights`. The refusals print one line to stderr and exit before the event system starts, as existing argparse errors do. No new unstructured logging. |
| VII | Recovery over prevention | Pass | Pass | CUDA users see no behaviour change, so there is nothing to flip. On the Mac, rollback means restarting the 2.1.0 tag. That can't run on a Mac, so the Mac goes from "unsupported" to "unsupported", with no data at risk. No flag is needed: the blast radius is a platform that doesn't work today. |
| VIII | Attention is finite | N/A | N/A | No alerts. |
| IX | Value at the user | Pass | Pass | Done means the quickstart live gate passes on the reference Mac and the CUDA regression check passes on the CUDA machine, both recorded in `research/live-*.md`. |
| X | Discoverable commands | Pass | Pass | `scripts/start_breeze_mac.sh` and `pytest -m mlx` are added to the README Development table. The launcher fails loudly when weights or platform are wrong. |

**Changes to working code paths** (CLAUDE.md asks for approval before these):
1. **Room arithmetic.** It moves out of `FastBreezeStreamingRuntime` methods into a module
   function. The CUDA numbers are pinned by a test written **before** the move.
2. **`--attn-implementation` default.** It becomes `None`, resolved to `eager` for CUDA, so that
   an explicit use can be detected. A settings test pins the CUDA result.
3. **`requirements.txt`.** `qwen-tts` gains a `sys_platform != "darwin"` marker. Linux and
   Windows install exactly what they install today.

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
├── settings.py          # + backend, Platform, refusals; attn default None
├── api.py               # read Platform once; pick device/set_device per backend
├── model_loading.py     # backend branch: load MLX runtime + MLX warmup; report backend/weights
└── (routes, templates, synthesis, voice_*, ws_*: unchanged)

models/
├── fast_streaming.py    # room arithmetic → module function (no behaviour change)
└── mlx_streaming.py     # NEW: MlxBreezeStreamingRuntime, checkpoint_kind, codec adapter

scripts/
├── start_breeze.sh, start_breeze.ps1   # unchanged
└── start_breeze_mac.sh                 # NEW

requirements.txt                 # + markers: qwen-tts (non-darwin), mlx-audio (darwin arm64)
requirements-mac-overrides.txt   # NEW: transformers, huggingface-hub pins
pyproject.toml                   # + "mlx" pytest marker

tests/
├── test_settings.py             # + backend default/refusal cases
├── test_checkpoint_kind.py      # NEW: config.json fixtures → kind/refusal
├── test_room_arithmetic.py      # NEW: pins CUDA numbers; MLX uses the same function
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
2. **Packaging.** Add the markers and the overrides file, and confirm a clean Mac install. Run
   the model-free suite on the Mac, and confirm the Linux install is unchanged.
3. **Room arithmetic.** Write the pinning test first, then extract.
4. **Settings.** Add `--backend`, `Platform`, `checkpoint_kind` and the refusals, test-first.
5. **The MLX runtime.**
   - Order: input adapter, prefill, frame loop (CFG, sampling, depth KV cache, random key), codec
     streaming, abort, `build_reference_prefix`, codec encode adapter, warmup.
   - Run the `mlx`-marker tests after each step.
6. **Wiring.** `api`/`model_loading` branches and the `model.loaded` fields. Then the real-server
   `mlx` tests.
7. **The Mac launcher.**
8. **Live gate.** Run quickstart steps 1–7 on the Mac and step 9 on CUDA.
9. **Docs, version 2.2.0, changelog**, before tagging.

## Complexity Tracking

No constitution violations. The working-code-path changes are listed in the Constitution Check
for approval.
