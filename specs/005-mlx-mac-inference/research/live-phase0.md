# Phase-0 speed gate (T002, research R7): 2026-10-03

**Verdict: NO-GO.** 8-bit with CFG measured RTF **2.61** on the passage. The threshold was ≤ 1.3.
Per T002, work stopped here and the result went back to the user.

## Setup

- **Machine:** Apple M5, 16 GB, macOS 26.6.2 (25G83). A browser and an editor were open.
- **Swap before the run:** 3.98 GB used of 5.12 GB. Swap after: 5.13 GB used of 6.14 GB.
- **Code:** mlx-audio `e1b19b9054bf163f5d812221a54fcc346f1890e9` and mlx 0.32.3, installed with
  `requirements.txt` plus the overrides (transformers 4.57.3, huggingface-hub 0.36.2).
- **Weights:**
  - bf16: `mlx-community/Breeze-TTS-2-mlx` @ `3c8829fb`
  - 8-bit (mxfp8): `mlx-community/Breeze-TTS-2-mlx-8bit` @ `c6e4a2ff`
- **Method:**
  - Stock `Model.generate(text, seed=42, stream=True, streaming_interval=0.16, split_pattern=None)`.
    Sampling and repetition penalty stay at mlx-audio's defaults (0.9 / 1.0 / 50, penalty 1.0).
  - CFG runs add `instruct="A calm, warm voice.", cfg_scale=4.0`.
  - Each precision runs in its own process under `/usr/bin/time -l`, after a warmup with and
    without CFG.
  - The script was a throwaway (`bench_gate.py`, in the session scratchpad).
- **Inputs:**
  - sentence: "Hello there, this is a short test of the streaming voice." (57 chars);
  - passage: the README "What this is" section as plain prose (1219 chars).

## Results

| Precision | CFG | Input | First audio | Generation | Audio | RTF | MLX peak |
|---|---|---|---|---|---|---|---|
| 8-bit | no | sentence | 0.29 s | 5.94 s | 4.08 s | 1.46 | 4.95 GB |
| 8-bit | no | passage | 0.74 s | 86.47 s | 60.0 s | **1.44** | 5.11 GB |
| 8-bit | yes | sentence | 0.50 s | 8.81 s | 3.36 s | 2.62 | 4.98 GB |
| 8-bit | yes | passage | 0.66 s | 156.42 s | 60.0 s | **2.61** | 5.25 GB |
| bf16 | no | sentence | 0.43 s | 5.77 s | 3.60 s | 1.60 | 7.91 GB |
| bf16 | no | passage | 0.93 s | 94.17 s | 60.0 s | **1.57** | 8.06 GB |
| bf16 | yes | sentence | 0.63 s | 12.54 s | 4.32 s | 2.90 | 7.94 GB |
| bf16 | yes | passage | 0.69 s | 172.38 s | 60.0 s | **2.87** | 8.21 GB |

**Per process (`/usr/bin/time -l`):**

| Precision | Load | Warmup | Max RSS | Peak memory footprint | Wall time | User CPU | System CPU |
|---|---|---|---|---|---|---|---|
| 8-bit | 3.7 s | 7.4 s | 4.08 GB | 8.78 GB | 269 s | 32.0 s | 10.6 s |
| bf16 | 4.4 s | 5.1 s | 4.47 GB | 11.27 GB | 295 s | 36.9 s | 14.5 s |

## Observations

- **Time to first audio is not the problem.** It is under 1 s in every case (SC-002 wants under
  2 s).
- **Throughput is the problem.** Without CFG, 8-bit takes about 115 ms per frame (86.5 s / 750
  frames). A frame is 80 ms of audio, so real time needs ≤ 80 ms.
- **CFG costs about 1.8×.** Stock `generate()` runs the conditional and unconditional passes one
  after the other, and the depth decoder has no KV cache (research R1). These are exactly the two
  things R6 plans to change.
- **bf16 is only about 9% slower than 8-bit**, while using about 3 GB more memory. Per-frame time
  is therefore dominated by something other than weight bandwidth: kernel launches and per-step
  overhead. A step-level change such as a depth KV cache, batching or `mx.compile` would help
  more than further quantization.
- **The work is GPU-bound.** CPU time is low (32 s user over 269 s wall). The GPU or the dispatch
  path is the limit, not Python.
- **Audio was exactly 60.0 s on the passage in all four runs.** That equals 750 frames, the
  `max_tokens` default. The passage probably hit the frame cap instead of finishing. The RTF per
  frame is unaffected, but the passage audio wasn't checked for completeness.
- **Memory:** the system was already using 3.98 GB of swap before the run, and swap grew by about
  1.1 GB during it. The bf16 footprint peaked at 11.3 GB. On this 16 GB machine, bf16 plus a
  browser and an editor is tight (SC-006).

## What the gate leaves open

The R6 optimizations (batched CFG, a depth-decoder KV cache) have not been measured on this
machine. The remote research cites about 2.4× on an M3 Max, from vanch007's fork with the float32
bug still in. Applied to 2.61, that would give about 1.09: still short of RTF 1.0 with CFG.
Without CFG (1.44), it would clear 1.0 if the depth decoder is most of the per-frame cost, which
is unmeasured. The decision on how to proceed goes back to the user.
