# Benchmark: final (T082, SC-007)

Compared against `research/baseline-2026-09-24.md` with the tasks.md "SC-007 method": 3 warm-ups,
then the median of 10 runs, with nothing else running on the machine (no test suites, builds or
agents executing code).

## Setup

- **Date**: 2026-09-29.
- **GPU**: NVIDIA GeForce RTX 4090 (driver 596.36), WSL2. Sampled every 2 s during the run: 167 of
  171 samples in P2 at 2730–2745 MHz SM, the rest at 210 MHz between cases; peak 219 W.
- **Versions**: the same as the baseline (Python 3.12.12, torch 2.9.1+cu128, transformers 4.57.3,
  fastapi 0.141.1, starlette 1.6.0, uvicorn 0.52.4, websockets 17.1).
- **Model**: Breeze-TTS-2 snapshot `c1c8ca18b70b30822735633991d9ebf4898e47d4` (the baseline's).
- **Server**: `enhanced-api` at `16296e3`, version `2.0.0` (`X-Breeze-Version: 2.0.0`).
  `scripts/start_breeze.sh` (`--host 0.0.0.0 --port 8080 --fast-all`, no `--cors`).
- **Benchmark**: `.venv/bin/python -m breeze_infer.bench_api --api new --url http://127.0.0.1:8080`
  (defaults `--warmup 3 --runs 10`, all five cases). `short_voice` registered an unnamed voice
  (`v_7468031581f4ea00`) and deleted it afterwards; the voice list was `eric`, `vale` after the run.
- All 50 timed runs returned 200. The server log shows 65 requests (5 cases × 13), every one a
  single piece (`pieces: 1`) ending in `speech.completed`, and no error-level events.

## Results against the baseline (10-run medians)

| Case | TTFA old → new (ms) | Change | RTF old → new | Change |
|---|---|---|---|---|
| short_design | 66.2 → 48.5 | −26.7% | 0.369 → 0.370 | +0.4% |
| medium_design | 70.9 → 53.3 | −24.9% | 0.362 → 0.368 | +1.8% |
| short_inline | 103.3 → 89.7 | −13.2% | 0.375 → 0.382 | +1.9% |
| medium_inline | 114.5 → 97.3 | −15.0% | 0.367 → 0.377 | +2.7% |
| short_voice (reported only) | — → 53.2 | — | — → 0.389 | — |

RTF is wall time over audio seconds, so higher is slower.

**SC-007 passes.** No gating case regresses by more than 10%: TTFA improves by 13–27%, and RTF is
0.4–2.7% slower, well inside the gate.

TTFA p25 / min (ms), for context: short_design 47.9 / 47.0, medium_design 52.4 / 51.2,
short_inline 86.6 / 85.8, medium_inline 95.1 / 90.7, short_voice 51.1 / 48.6.

## Notes

- **RTF drift since Phase 2.** Phase 2 (`research/bench-phase2.md`) measured RTF 0.361–0.370,
  and dev4 (T054, `research/live-phase2.md`) 0.366–0.376; this run is 0.368–0.382, 1–2.5% above
  dev4 on every case, under the same GPU clocks. It is small next to the gate and was not
  investigated. It is worth watching if a later benchmark adds to it.
- **medium_inline audio length** is 26.48 s against the baseline's 24.08 s. Since `bebe8ce` the
  inline reference encodes to the same codes in every process (research.md R18); dev4 gave
  26.48 s in two separate processes (`research/live-phase2.md`), the same as here. The baseline's
  24.08 s was one of the earlier process-dependent outcomes. RTF is per audio second, so the gate
  still compares like for like, and TTFA does not depend on length. The other gating cases have
  the baseline's lengths.
- **short_voice RTF rose within the case**: 0.372 for the first run to 0.410–0.417 for the last
  four, while TTFA stayed at 49–64 ms. It has no baseline and does not gate; it is recorded for
  the next comparison.

Per-run TTFA (ms):
- short_design: 49, 50, 48, 49, 48, 49, 49, 47, 48, 48
- medium_design: 51, 52, 53, 54, 58, 53, 55, 55, 52, 51
- short_inline: 86, 86, 101, 88, 90, 90, 93, 90, 97, 86
- medium_inline: 103, 107, 91, 91, 95, 98, 95, 101, 96, 99
- short_voice: 51, 49, 53, 52, 50, 54, 64, 55, 55, 54

Per-run RTF:
- short_design: 0.370, 0.373, 0.366, 0.368, 0.372, 0.370, 0.372, 0.371, 0.370, 0.370
- medium_design: 0.369, 0.368, 0.371, 0.371, 0.366, 0.367, 0.370, 0.369, 0.365, 0.366
- short_inline: 0.380, 0.382, 0.382, 0.379, 0.380, 0.382, 0.382, 0.384, 0.384, 0.383
- medium_inline: 0.376, 0.380, 0.376, 0.375, 0.382, 0.377, 0.376, 0.386, 0.381, 0.375
- short_voice: 0.372, 0.373, 0.385, 0.378, 0.383, 0.392, 0.417, 0.410, 0.410, 0.410
