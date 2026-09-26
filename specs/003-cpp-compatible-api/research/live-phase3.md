# Live record: Phase 3 gate (T069), voices

## T068: SillyTavern extension coordination (2026-09-25)

Asked `sillytavern-agent` whether the extension's DELETE-then-POST replace flow and the new delete
wording have landed.

**Answer:** both are implemented in the Breeze TTS extension as **v0.1.3, uncommitted on purpose**:
that user wants every extension change committed together once this server rewrite is done. The
last commit is `7d91d38` (v0.1.2, the old behaviour), in
`<SillyTavern checkout>/extensions/SillyTavern-BreezeTTS` on `main`. Lint is clean and its 103 unit tests
pass. The replace flow has not been checked against a live server yet.

- **Replace flow** (`src/provider.js` ~429–462): the existing-name check ignores case, like the
  server. After the user confirms, it sends `DELETE /v1/voices/{existing.id}`, then a multipart
  `POST /v1/voices` using the existing id's casing, so character voice assignments and per-voice
  settings survive. If the DELETE succeeds but the POST fails, it shows the POST's `body.error`
  and refreshes the voice list.
- **Delete wording** (`src/provider.js:466`): "Its saved file is permanently deleted too." The
  JSDoc (`src/breeze-http.js:82`), the extension CHANGELOG and README are updated to match.

**What the extension relies on (requirements for T059/T065):**

- `GET /v1/voices` lists `id`, `saved` and `seconds` for each voice.
- `POST /v1/voices` returns the voice record (`id`, `seconds`, `saved`), which it reads at once.
- The `id` from `GET /v1/voices`, URL-encoded with `encodeURIComponent`, is exactly what
  `DELETE /v1/voices/{id}` accepts. A `404` there shows an error toast and stops the replace.
- It never reads the DELETE body (only `response.ok`), so it doesn't depend on `file_kept`.
- It never treats a `409` as success: any non-2xx becomes an error toast with `body.error`. If
  another client registers the same name between its DELETE and POST, the user sees "voice already
  exists" and the old voice is gone; that user accepts this.

**Next:** when Phase 7 is ready for live testing, message `sillytavern-agent`. It will run the
replace and delete check through the SillyTavern UI with a throwaway voice (never `eric` or
`vale`).

## T069: voices live gate (2026-09-26)

- **Server**: `enhanced-api` at `167fe36`, version `2.0.0.dev3`, started with
  `scripts/start_breeze.sh --cors http://127.0.0.1:8000` (`--fast-all`), on the repo's empty
  `voices/` directory. Before the gate, the voice and speech GPU tests passed at `8d8c57b` (9 passed),
  and the whole GPU suite passed at `4066051` (28 passed).
- **Result: the gate passes.** Every Scenario 3 step, the SillyTavern `voices` phase (20/20) and
  `sillytavern-agent`'s own UI check (16/16) passed. The Phase 1–2 voice-list gap is gone:
  `checkReady()` now ends in `voices.refreshed`.

### Quickstart Scenario 3 by curl

| Check | Result |
|---|---|
| 3.1 register `eric` and `vale` | `200`, `saved: true` (eric 129 frames, 10.32 s; vale 105 frames, 8.4 s); `eric.voice.json` and `vale.voice.json` written |
| 3.2 `name=ERIC` | `409 {"error":"voice already exists","code":"voice_exists"}` |
| Speech with `voice_id=eric`, twice | `200` both; the first `speech.accepted` has `reference: voice_prefix, warm: false` and `voice.prefix_built` (178 tokens, 20,414,464 bytes), the second `warm: true` |
| Speech with `voice_id=Eric` | `404 unknown_voice` (ids match exactly) |
| 3.3 register `st_live_tmp`, delete, restart | DELETE `200 {"deleted":"st_live_tmp","file_kept":false}`; the file was gone at once and the voice wasn't listed after the restart (BC-28) |
| 3.4 a `.breeze` file, then restart | `voices.loaded {loaded: 2, skipped: 0, breeze_ignored: 1}` (BC-29); the placeholder file was removed afterwards |

### SillyTavern

- **Harness** (`node tests/live/sillytavern/run.mjs voices`), 20/20: upload `st_live_tmp`, replace
  (`voice.replaced`, then re-upload), delete, no error toasts, `eric` and `vale` still listed.
  Record: `research/live-voices.md`.
- **`sillytavern-agent`** (extension v0.1.3 working tree, headless Playwright, throwaway
  `st_live_tmp1` from the vale sample), 16/16:
  - Replace typed as `ST_LIVE_TMP1`: the dialog named `st_live_tmp1` with the permanent-delete
    warning, then `DELETE /v1/voices/st_live_tmp1` (`200`, `file_kept: false`) followed by
    `POST /v1/voices` (`200`, the record with `id`, `seconds`, `saved`), using the existing id's
    casing. Only the success toast appeared, and the voice was listed once.
  - Delete: the dialog reads "Remove the voice "st_live_tmp1" from the Breeze server? Its saved
    file is permanently deleted too."; the voice was gone from the UI and from `GET /v1/voices`.
  - The id round trip through `encodeURIComponent` matched; no `*_failed` events and no console
    errors. `eric` and `vale` were untouched.
- **`ws_port: 0`** in `/health` is expected until Phase 8 builds the WebSocket server. Until then
  the extension can only synthesize in HTTP (buffer) mode; its `test/live/2-http.live.mjs`
  expects 8081 and will fail until Phase 8.

### Findings

- **First encode after start-up is slow**: registering `eric`, the first encode in the process,
  took `encode_ms: 4160`; `vale` right after took 126 ms. The server's warmup decodes but never
  encodes, so the first reference encode pays the encoder's one-off GPU start-up. research.md R18
  assumed the decode warmup already covered it. Follow-up (not done): add a reference encode to the
  warmup.
- **Codec decode variation**: identical tokens decode to PCM that differs by up to about 2,650 int16
  steps across requests (found by T067; research.md R18 open item). The tier-1 GPU test therefore
  compares tokens (user decision).

## Review loop outcomes (Phase 7)

Every Phase 7 commit had two `review-agent` passes (reviews 37–43), with no third; the final fixes
were verified here with the CPU and GPU suites. Findings that changed behaviour:

- **Voice store**: every malformed or oversized file is skipped with an event instead of stopping
  startup; creates commit with a hard link, so nothing is ever overwritten (skipped, late-added or
  case-variant files included); skipped files are tracked and can be deleted; delete and create of
  the same name can't race; DELETE matches the listed id exactly everywhere; clean-up failures are
  structured `voice.cleanup_failed` events.
- **Prefix cache**: bounded by bytes (1 GiB) rather than count; keyed by voice id plus a hash of the
  transcript and codes; a global token read at resolve time keeps a deleted voice out of the cache;
  no locks (callers hold the GPU lease, which a cancelled build takes over); orphaned build failures
  are reported.
- **Routes and speech by voice**: an encode is validated with the scan's rules before commit; one
  voices thread makes every change and schedules the events; an overlong voice is refused at
  registration (`400 voice_too_long`, new); a CUDA OOM in the prefix build frees GPU memory, evicts
  the cache and falls back to the codes path if it fits, else `503 gpu_out_of_memory` (new);
  pre-gate sizing requires only the path the request takes (a reversal of review 42's rule).

## Decisions made with the user during Phase 7

- A voices directory that can't be opened fails startup (`model.load_failed`, `stage: voices`).
- The tier-1 GPU test compares generated tokens and only sanity-checks the PCM.
- Work ran one agent at a time, with no reviews alongside, while the rate limit or the GPU was
  constrained, then in parallel again once they were free.
