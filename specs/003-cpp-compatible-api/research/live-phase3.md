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
