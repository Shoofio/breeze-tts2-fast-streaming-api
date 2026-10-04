# Done means (T037, Constitution IX): 2026-10-04

This maps every functional requirement and success criterion in `spec.md` to a passing test or a
recorded live result, as of the branch head after the T037 review fix (`ae6afd8`).

**What "tested" means here:**
- **Tests** run with `.venv/bin/pytest`, the model-free suite: 0 failed on macOS and on Linux.
- **mlx** means `BREEZE_MLX_MODEL=<snapshot> .venv/bin/pytest -m mlx tests/mlx`, which passes at
  8-bit and at bf16 on the reference Mac (Apple M5, 16 GB).
- **The live records** are in this folder: `live-*.md`.

## Functional requirements

| Req | Evidence | Status |
|---|---|---|
| FR-001 MLX on Apple Silicon, no CUDA | Covered at both precisions: <ul><li>`tests/mlx/*`</li><li>`live-us1.md` (launcher)</li><li>`live-perf.md`</li></ul> The Mac venv has no CUDA packages. | ✓ |
| FR-002 backend picked once; default and override | `tests/test_settings.py`: the backend cases and the `platform=None` → cuda case. `tests/test_api_main.py::test_mlx_backend_selects_the_mlx_gpu_device_and_no_cuda_call`. | ✓ |
| FR-003 refuse an Intel Mac or low memory, before weights | `tests/test_settings.py` refusal cases (darwin x86_64, linux, 8 GiB) | ✓ |
| FR-004 CUDA-only options named and refused | `tests/test_settings.py`: every CUDA-only option, including `--no-` forms, in argv order. Live: `live-us1.md` step 7. | ✓ |
| FR-005 same HTTP and WebSocket API; room rule | <ul><li>`tests/mlx/test_mlx_server.py`: PCM, the `.wav` header, 409, abort, long text, voices, WebSocket.</li><li>`tests/test_mlx_room.py` (exact-length rule, equal to the fake runtime's).</li><li>`live-us3.md`: the C++ examples pass in 3 CORS configurations.</li><li>SillyTavern: every server-facing step passed (see SC-003).</li></ul> | ✓ |
| FR-005a `infer.py` unchanged; README says server only | The protected-path diff is empty (T029). README `[!NOTE]` in the macOS quick start. | ✓ |
| FR-006 streaming | <ul><li>`test_speech_streams_pcm_before_generation_ends`: first bytes 0.49 s, response complete 19.9 s.</li><li>`test_audio_streams_in_several_chunks`.</li><li>`live-us1.md`: first byte 0.29–0.34 s.</li></ul> | ✓ |
| FR-007 one at a time; 409; `.wav` queue then 503 | <ul><li>**409:** `test_a_second_request_while_busy_gets_409` and `live-us1.md`.</li><li>**The `.wav` 60 s queue and `503 busy_timeout`:** route code shared with CUDA, covered by the model-free `tests/test_speech_wav.py`. Not exercised live on MLX.</li></ul> | ✓ (queue by shared code) |
| FR-008 abort frees the backend | <ul><li>`test_closing_the_client_mid_stream_frees_the_gate`.</li><li>`test_close_aborts_quickly_and_the_next_request_works`: `close()` takes 0.4 ms.</li><li>`live-us1.md`: freed within about 0.5 s, and the suspend check.</li></ul> | ✓ |
| FR-009 clone, design, direction, cfg, seed, sampling | <ul><li>`test_voice_clone_design_and_direction_return_audio`, with direction at `cfg_scale` 4.</li><li>The `tests/mlx/test_mlx_generation.py` CFG and seed tests.</li><li>`test_mlx_prefix.py`: teacher-forced parity, mismatches are ties only.</li><li>`live-us2.md`.</li></ul> Quality is SC-005. | ✓ |
| FR-010 own fingerprint; mismatch skipped | `test_saved_voices_survive_a_restart_and_a_foreign_fingerprint_is_skipped` and `live-us2.md` (skip event, `404 unknown_voice`). CUDA voice format unchanged. | ✓ |
| FR-011 community weights pinned; no converter; licence note | <ul><li>Pinned revisions in `scripts/start_breeze_mac.sh`, the README and the refusals.</li><li>No converter in the repo.</li><li>README `[!IMPORTANT]` licence and unaffiliated note.</li></ul> | ✓ |
| FR-012 bf16 and 8-bit, 8-bit default; 4-bit refused | <ul><li>`tests/test_checkpoint_kind.py`.</li><li>Launcher default 8bit (`live-us1.md`).</li><li>4-bit refusal in `live-us1.md` step 7.</li></ul> | ✓ |
| FR-013 missing weights print the download command; wrong format refused | <ul><li>Launcher with `HF_HOME=/nonexistent` (`live-us1.md`).</li><li>`tests/test_settings.py` mismatch cases.</li><li>Live refusal, PyTorch checkpoint with `--backend mlx`.</li></ul> | ✓ |
| FR-014 `model.loaded` backend, device, weights; `/health` unchanged | <ul><li>`test_model_loaded_event_reports_the_mlx_backend`.</li><li>CUDA fields asserted in `tests/test_routes_speech.py`.</li><li>`/health` unchanged (the model-free suite).</li></ul> | ✓ |
| FR-015 one-command launcher; `$HF_HOME`; same bind and CORS | `live-us1.md` (both precisions; a later `--host`/`--cors` overrides) | ✓ |
| FR-016 Mac-only dependencies | <ul><li>darwin/arm64 markers on mlx-audio, mlx and mlx-metal.</li><li>The Linux resolve has no mlx package.</li><li>T030 (user): the Linux install, Docker log and Windows launcher skipped mlx-audio.</li></ul> | ✓ |
| FR-017 no CUDA regression; test edits only for R10 | <ul><li>T029: the protected-path diff is empty.</li><li>T030 (user): suite 0 failed, `-m gpu` passed, bench within 5%, Docker and Windows OK.</li><li>T031 (user): rollback drill.</li><li>The test edits are the R10 fixes plus the two approved ones (the port-taken test, one CUDA report assertion).</li></ul> | ✓ |
| FR-018 `mlx` marker; model-free suite green on macOS | `pyproject.toml` marker and the `tests/conftest.py` rule. The model-free suite gave 2561 passed, 0 failed on macOS (it was 17 failed). | ✓ |
| FR-019 docs: requirements, quick start, options, unavailable options, measured speed | README "Quick start (macOS, Apple Silicon)". `docs/api.md`: `--backend`, the CUDA-only markers, "Differences on the MLX backend". | ✓ |

## Success criteria

| SC | Evidence | Status |
|---|---|---|
| SC-001 fresh clone to first audio under 10 min | The pieces are measured: <ul><li>the launcher reaches `/health 200` in 8–10 s with the weights cached;</li><li>the README `curl` returns audio in under 1 s.</li></ul> The user's own testing validated the end-to-end setup as within expectations (2026-10-04). A separate timed fresh-clone run was waived. | ✓ (user-validated) |
| SC-002 first audio under 2 s with a saved voice; real time over about 1 min | `live-perf.md`: `short_voice` first audio 331 ms at 8-bit. RTF 0.84–0.88 over all cases, including about 25 s passages. A 109 s long text ran at RTF 0.83 (T021). | ✓ at 8-bit |
| SC-002a 8-bit meets SC-002; docs give the bf16 numbers | bf16 RTF 1.47–1.50 is published in the README and `docs/api.md`, which recommend 8-bit on 16 GB Macs | ✓ |
| SC-003 SillyTavern `full` and C++ examples pass unchanged | <ul><li>C++ examples: `RESULT: OK` with 0 fail, in 3 CORS configurations.</li><li>SillyTavern: 30/34. The 4 failures are in the test suite (hard-coded `eric`/`vale` ids, a known popup race also seen on CUDA). The user accepted the run; fixing the suite is a follow-up.</li></ul> | ✓ (accepted, suite fix pending) |
| SC-004 CUDA bench within 5%; GPU suite passes | T030 (user-reported pass) | ✓ |
| SC-005 10-prompt listening test | `live-listening.md`: CUDA, Mac 8-bit and Mac bf16, all 30 returned 200. The user judged every Mac output intelligible, artifact-free and matching CUDA. Voice direction sounds better acted in bf16 than in 8-bit. | ✓ |
| SC-006 no swap with a browser and editor; number set by the gate | `live-perf.md`: peak footprint 7.0 GB at 8-bit with no swap growth. SC-006 is set at ≤ 7.5 GB. | ✓ at 8-bit |
| SC-007 every unsupported case gives a clear refusal | `tests/test_settings.py`, `tests/test_checkpoint_kind.py` (including a malformed `quantization`) and `live-us1.md` step 7. Each prints the usage line, the message and exit status 2. | ✓ |

## Open items

1. **Follow-up after this feature:** update the SillyTavern suite:
   - use the current voice ids or make them configurable;
   - harden `acceptPopupIfPresent`;
   - sweep leftover `st_live_tmp` voices.
