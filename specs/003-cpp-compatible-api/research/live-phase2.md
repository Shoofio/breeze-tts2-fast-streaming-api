# Live record: Phase 2 gate (T054), speech over HTTP

## Summary

- **Server**: `enhanced-api`, started with `scripts/start_breeze.sh --cors http://127.0.0.1:8000`
  (host 0.0.0.0, `--fast-all`), model snapshot `c1c8ca18`. Two versions ran:
  - `2.0.0.dev3` at `f6e4b68`: Scenario 2, the SillyTavern `speech` phase and the first benchmark.
  - `2.0.0.dev4` at `bebe8ce`: the deterministic reference encode (research.md R18), the SillyTavern
    `speech` phase again, and the benchmark again across two server processes.
- **Version numbering:** these runs reported `2.0.0.dev3` and `2.0.0.dev4`, which took the
  numbers the plan reserves for Phases 3 and 4. With the user's agreement they are renumbered
  `2.0.0.dev2+1` and `2.0.0.dev2+2` (plan.md, live-gate step 1); the code now carries
  `2.0.0.dev2+2`. The values below are what the server reported at the time.
- **Result: the gate passes**, including the user's listening check (one speaker throughout). Voice
  design, the malformed corpus, the mid-stream abort, long text and the CFG values all behave as
  the contract says. SillyTavern's page can call the speech route and read the CORS-exposed
  headers. SC-007 passes on both versions.
- **Expected failures (2 of 11 SillyTavern steps, both runs):** the same Phase 1 gap. The
  extension's `checkReady()` calls `GET /v1/voices`, which arrives in T065, so it logs
  `breeze.check_failed {stage: voices}` and shows the cached-voice-list toast. The T069 voices gate
  re-checks it.
- **Found during the gate and fixed:** inline-reference output wasn't reproducible across server
  restarts (see "Reference encode" below).

## Quickstart Scenario 2

| Check | Result |
|---|---|
| 2.1 voice design, `curl -F text='Hello there.'` | `200`, `content-type: audio/pcm`, `x-sample-rate: 24000`, `x-sample-format: s16le`, `cache-control: no-store`, `x-request-id`, `x-breeze-version: 2.0.0.dev3`, chunked; 1.52 s of speech (peak 11,844, RMS 2,501) |
| 2.2 malformed corpus and 2.3 mid-stream abort | `tests/test_malformed_corpus.py` and `tests/test_speech_abort.py` on a clean `git archive` of `f6e4b68`: 92 passed |
| 2.4 long text (3,126 characters, no reference, 5 runs) and 2.5 CFG values | the full GPU suite at `10f0c29` (the same code as `f6e4b68`), then again at `74c990d`, `6dc966c` and (the encode test plus the HTTP smoke tests) `bebe8ce`: 21 passed each time. Every long-text run: 7 pieces, 208.24 s of audio, no `anchor_skipped` |
| 2.6 SillyTavern `speech` | 9/11 on dev3 and on dev4: `POST /v1/audio/speech` from the page is `200`, `X-Sample-Rate` is readable, the body is even-length PCM, `cfg_scale=banana` is `400 invalid_field`, the settings were restored and verified on disk. The 2 failures are the expected voices gap above. Harness record: `research/live-speech.md` (the dev4 run) |

**Listening check (SC-004, US3's independent test):** `research/long-text-sample.wav` (208 s,
written by the GPU run at `6dc966c`; not committed). **Passed:** the user listened to the whole
file and heard the same speaker throughout (2026-09-25).

## Benchmark (SC-007)

`bench_api --api new --cases short_design,medium_design,short_inline,medium_inline`, 3 warm-ups
then 10 runs, nothing else running. Every request was one piece and ended in `speech.completed`.

| Case | Baseline TTFA / RTF | dev3 TTFA / RTF | dev4 TTFA / RTF |
|---|---|---|---|
| short_design | 66.2 ms / 0.369 | 49.4 ms / 0.371 | 48.4 ms / 0.371 |
| medium_design | 70.9 ms / 0.362 | 52.1 ms / 0.365 | 52.1 ms / 0.366 |
| short_inline | 103.3 ms / 0.375 | 90.2 ms / 0.378 | 85.8 ms / 0.376 |
| medium_inline | 114.5 ms / 0.367 | 93.4 ms / 0.368 | 92.9 ms / 0.368 |

**SC-007 passes on both versions.** TTFA improves by 13–29%, and RTF is within +1.1% (noise).

**Anchor sizing cost (review 31a #8):** the 3,126-character passage against its 129-character piece 0
sent alone, 7 runs each, alternating, 200 responses only: TTFA medians 78 and 69 ms against 72 and
75 ms, within run-to-run noise (58–109 ms). Sizing the later pieces after the lease adds no
measurable time to first audio.

## Reference encode: not reproducible across restarts (fixed in `bebe8ce`)

On dev3 the inline cases' audio lengths differed from the baseline (short_inline 5.44 → 6.16 s,
medium_inline 24.08 → 23.52 s), while the design cases matched exactly. A bisect showed no commit
was responsible:

- With `--fast-all`, the fast codec sets `cudnn.benchmark = True` for the whole process, so each
  server process could autotune a different conv algorithm for the reference encode. About 22 of
  2,064 codes (codebooks 6–15) differed between processes.
- At `bea6767`, the same short_inline request gave 5.20–6.96 s across restarts. Pinning the codes
  gave identical lengths at `bea6767` and at HEAD.
- So the Phase 2 benchmark record's "identical to the baseline" for the inline cases held by
  chance.

With the user's approval, `bebe8ce` runs the encode with `benchmark=False, deterministic=True`
(decode keeps its tuned algorithms). On dev4, two separate server processes gave short_inline
5.44 s and medium_inline 26.48 s both times. The encode cost is within noise (25.6 against 24.9 ms
median). A GPU test now encodes the same wav in two processes and requires identical codes.

## Review loop outcomes (Phases 4–6)

Every Phase 4–6 commit had two `review-agent` passes, with no third. The findings that changed
behaviour:

- **Validation (T044–T049):** huge integer and exponent literals give `400`, never `500`
  (including Python's 4,300-digit limit); an unrepresentable literal keeps its sign; duplicate
  names are checked first, over the raw query and body names, and also when a parser limit trips
  (bounded to 33 names, so a 26 MiB body can't cost about 1 GB); a `ref_audio` in the query is
  rejected before the body is read; blank `instruction` and `ref_text` mean the default, whatever
  their length.
- **Long text (T050–T053):** the CPU room check has its own tokenizer copies and executors, made at
  load, so it never shares the HF fast tokenizer with generation; the later pieces are sized after
  the lease, on their own worker, with a 5 s backstop. A cancel while closing wins over an ordinary
  error but never over Ctrl+C or `SystemExit`. Request ids are stamped by one middleware.
- **Shutdown (T022 follow-ups):** a GPU failure never exits 0 (exit codes are normalised to what the
  OS really exits with, and a drain failure is recorded before any await).
- **Declined or deferred:** the pre-gate check still builds piece 0's full CPU inputs (a
  lengths-only check would need a length API in `templates.py`); the version header's own fallback
  `500` has no `X-Request-Id`; a CPU-executor shutdown maps to `503 gpu_unavailable`, which is
  practically unreachable.

## Decisions made with the user during Phases 4–6

- **Anchor skipping:** skip the anchor whenever it would give any later piece a smaller frame limit
  than it would have without it (`min(cap, room)`), because missing words are worse than a voice
  change. Also skipped when piece 0 was cut off, or when sizing fails, times out or is stopped by
  shutdown; the later pieces then use voice design (CHANGELOG BC-47).
- **Reference encode:** fix the cross-process reproducibility now (research.md R18).
- **Running agents:** for a while the user limited the session to one subagent at a time, with no
  reviews alongside, until the rate limit reset.
