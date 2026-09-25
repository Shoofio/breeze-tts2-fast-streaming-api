# Benchmark: Phase 2 (T043, SC-007)

Compared against `research/baseline-2026-09-24.md` with the tasks.md "SC-007 method": 3 warm-ups,
then the median of 10 runs, with nothing else running on the machine.

## Setup

- **GPU**: NVIDIA GeForce RTX 4090 (driver 596.36), WSL2. During the run it stayed in P2 at
  2730–2745 MHz SM (sampled every 2 s).
- **Versions**: the same as the baseline (Python 3.12.12, torch 2.9.1+cu128, CUDA 12.8,
  transformers 4.57.3, fastapi 0.141.1, starlette 1.6.0, uvicorn 0.52.4, python-multipart 0.0.32,
  websockets 17.1).
- **Model**: Breeze-TTS-2 snapshot `c1c8ca18b70b30822735633991d9ebf4898e47d4` (the baseline's).
- **Server**: `enhanced-api` at `bea6767`, version `2.0.0.dev1`. `scripts/start_breeze.sh` ran
  `uv run python -m breeze_infer.api <snapshot c1c8ca18> --host 0.0.0.0 --port 8080 --fast-all`.
  Warmup took 52.8 s with a compile-cache hit.
- **Benchmark**:
  `.venv/bin/python -m breeze_infer.bench_api --api new --url http://127.0.0.1:8080 --cases short_design,medium_design,short_inline,medium_inline`.
  `short_voice` was left out because the voice routes arrive in Phase 7.
- All 40 timed runs returned 200.

## Results against the baseline (10-run medians)

| Case (all gating) | TTFA old → new (ms) | Change | RTF old → new | Change |
|---|---|---|---|---|
| short_design | 66.2 → 49.1 | −26% | 0.369 → 0.368 | −0.3% |
| medium_design | 70.9 → 50.5 | −29% | 0.362 → 0.361 | −0.3% |
| short_inline | 103.3 → 85.8 | −17% | 0.375 → 0.370 | −1.3% |
| medium_inline | 114.5 → 89.4 | −22% | 0.367 → 0.361 | −1.6% |

Audio length per case (median): 5.52, 25.44, 5.44 and 24.08 s, identical to the baseline. The
server log shows all 52 requests (4 cases × 13 including warm-ups) were one piece
(`speech.accepted.pieces == 1`), and all 52 ended in `speech.completed` (no failed or aborted).
So the comparison is like for like: the same output length, with no shorter opening piece.

**SC-007 passes.** No gating case regresses; TTFA improves by 17–29%, and RTF is unchanged within
noise.

Per-run TTFA (ms):
- short_design: 49, 50, 50, 49, 47, 48, 50, 49, 47, 50
- medium_design: 49, 50, 49, 49, 51, 52, 52, 51, 51, 48
- short_inline: 86, 84, 89, 85, 129, 87, 84, 85, 83, 86
- medium_inline: 90, 91, 89, 89, 89, 88, 96, 90, 90, 86

Per-run RTF:
- short_design: 0.369, 0.370, 0.371, 0.369, 0.368, 0.369, 0.367, 0.367, 0.365, 0.367
- medium_design: 0.357, 0.357, 0.357, 0.365, 0.366, 0.367, 0.363, 0.361, 0.360, 0.362
- short_inline: 0.370, 0.370, 0.374, 0.372, 0.380, 0.371, 0.370, 0.370, 0.369, 0.367
- medium_inline: 0.360, 0.362, 0.362, 0.361, 0.362, 0.362, 0.361, 0.360, 0.360, 0.360

## Why TTFA improved (not measured, for context)

The old API's `iter_audio_chunks` ran on anyio worker threads behind a `threading.Lock`, and
prepared inputs on the request path. The new route primes the first chunk on one dedicated GPU
thread. The inline cases also skip the old temporary-file upload path, since the reference is
decoded in memory. Plan risk 6 (an executor hop per chunk costing TTFA) didn't materialize.

TTFA here is measured by the client, from request start to the first body byte.
