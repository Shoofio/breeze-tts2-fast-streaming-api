# Linux and Windows unaffected (T030): 2026-10-04

The user ran T030 on the CUDA machine, on branch `005-mlx-mac-inference`, and reported that every
check passes:
- `uv pip install -r requirements.txt` did not install mlx-audio;
- `.venv/bin/pytest` showed 0 failures, with the macOS test fixes (T006–T008) passing on Linux;
- `BREEZE_MODEL=<path> .venv/bin/pytest -m gpu` passed;
- `bench_api` against `scripts/start_breeze.sh` was within 5% of
  `specs/003-cpp-compatible-api/research/bench-final.md` for time to first audio and throughput;
- the Docker build and `docker/smoke_check.py` passed, with the pip log skipping mlx-audio on its
  marker;
- the Windows launcher, `start_breeze.ps1 -Reinstall`, reached `/health 200` without installing
  mlx-audio.

The outputs were not pasted into this session, so only the user's pass verdict is recorded here.
The main session's own check (T029) confirmed the CUDA files have no diff against `main`.
