# Speed and memory gate (T032): 2026-10-04

Quickstart step 5 on the reference Mac (Apple M5, 16 GB, macOS 26.6.2), at commit `2449216`.
A browser and an editor were open: 24 browser or editor processes at the start of each run.

## Method

- **Server:** `scripts/start_breeze_mac.sh [--precision bf16] --host 127.0.0.1`, one precision
  per process.
- **Benchmark:** `.venv/bin/python -m breeze_infer.bench_api --api new --url http://127.0.0.1:8080 --ref-audio <server-made reference.wav>`.
  The defaults are `--warmup 3 --runs 10` over all five cases, the same invocation as the CUDA
  baseline in `specs/003-cpp-compatible-api/research/bench-final.md`.
- **Swap:** `sysctl vm.swapusage`, before and after each run.
- **Memory:** `footprint <pid>` on the server's Python process, which runs as the child of
  `uv run`. It was taken after medium-length design and inline requests.
  - `phys_footprint` counts the Metal buffers that hold the weights and KV cache. The process's
    RSS doesn't: it read only about 1.2 GB at both precisions, so it is not used here.

## Results: speed (all 100 timed requests returned 200)

| Case | 8-bit first audio | 8-bit RTF | bf16 first audio | bf16 RTF |
|---|---|---|---|---|
| short_design | 316 ms | 0.859 | 492 ms | 1.501 |
| medium_design (25.6 s / 22.5 s audio) | 359 ms | 0.844 | 500 ms | 1.470 |
| short_inline | 409 ms | 0.877 | 536 ms | 1.502 |
| medium_inline (25.2 s / 25.1 s audio) | 407 ms | 0.854 | 553 ms | 1.471 |
| short_voice | 331 ms | 0.877 | 495 ms | 1.496 |

Values are medians of 10 runs. The spread within each case was small: RTF varied by under 0.01
at 8-bit, and the first-audio maxima were 418 ms at 8-bit and 676 ms at bf16.

## Results: memory and swap

| Precision | Peak `phys_footprint` | Swap before → after the benchmark |
|---|---|---|
| 8-bit | **7.0 GB** (7021 MB peak) | 3136 MB → 3112 MB (no growth) |
| bf16 | 9.7 GB (9697 MB peak) | 3112 MB → 3650 MB (**+539 MB**) |

The machine was already using about 3.1 GB of swap from the open apps before either run.

## Verdict

- **SC-002, met at 8-bit, the default.** First audio is 0.32–0.41 s, under 2 s. RTF is
  0.84–0.88 on every case, under 1.0. The prototype measured 66–68 ms per frame (RTF
  0.82–0.85), so the full server adds little.
- **SC-002a, bf16 is not real time on this machine.** RTF is 1.47–1.50, so audio is produced
  about 1.5× slower than it plays. This matches the prototype's 1.45. The README recommends
  8-bit for 16 GB Macs and publishes these numbers (T034).
- **SC-006, set from this measurement: peak footprint ≤ 7.5 GB at 8-bit.** The measured 7.0 GB
  ran with no swap growth while a browser and an editor were open. bf16 at 9.7 GB pushed the
  system to swap 539 MB more, which is another reason 8-bit is the 16 GB default.
