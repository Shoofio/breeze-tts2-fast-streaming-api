# Implementation Plan: Browser-Playable WAV Streaming

**Branch**: `004-browser-wav-stream` | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/004-browser-wav-stream/spec.md`

## Summary

Add `GET /v1/audio/speech.wav`. It streams the same synthesis as `POST /v1/audio/speech`, framed as
a progressive WAV, so SillyTavern can set the URL as `<audio>.src` and play it without gaps. The
route reuses the POST route's pipeline, which differs in only two places:
- it waits up to 60 s for the GPU instead of returning `409`;
- it delivers from a buffer, so the GPU is released as soon as generation ends and the browser
  reads at its own pace.

The only other server change is raising the request-head limit to 192 KiB. Version 2.1.0.

**Scope guidance from the user**: keep it tight. Tests cover the new behaviour's main paths, not
every corner. The live gate carries the browser questions.

## Technical Context

- **Language/Version**: Python 3.12
- **Primary Dependencies**: FastAPI/Starlette 1.6.0, uvicorn 0.52.4 with h11 0.16.0. No new
  dependencies.
- **Storage**: N/A. The buffer lives in memory for one request.
- **Testing**: pytest. `TestClient` with the existing fakes, real uvicorn and httpx for streaming
  behaviour, and `pytest -m gpu` with `BREEZE_MODEL`.
- **Target Platform**: Linux/WSL and Windows, started by hand with the launcher scripts.
- **Project Type**: Single-process web service (HTTP and WebSocket on one event loop).
- **Performance Goals**:
  - Time to first audio on the GET route is within 10% of the POST route's.
  - POST and WebSocket performance stays within 10% of 2.0.0 (SC-008).
- **Constraints**:
  - One generation at a time.
  - The POST route, the WebSocket API and `/health` must not change.
  - No cross-site guard (decided).
- **Scale/Scope**:
  - About 4 changed modules: `routes_speech.py`, `streaming.py`, `api.py` and `limits.py`.
  - One new route and about 6 new tests.

No NEEDS CLARIFICATION items. [research.md](research.md) R1–R6 records the design decisions.

## Constitution Check

*GATE: must pass before Phase 0 research and again after Phase 1 design.*

| # | Principle | Pre-research | Post-design | Notes |
|---|---|---|---|---|
| I | Do not distribute | Pass | Pass | A route in the existing process. No new service or queue. |
| II | Optimize for deletion | Pass | Pass | The existing `_serve_speech` and `SpeechResponse` take two keyword parameters for the two differences (R1, R4). No new class, and no strategy object for two cases. Deleting the feature means deleting one route, `_buffered` and the WAV header function. |
| III | Explicit dependencies | Pass | Pass | The 60 s wait and 600 s send timeout are `limits.py` constants, passed in when the route is installed. The clock is already injected. |
| IV | Contract at the boundary | Pass | Pass | [contracts/http-wav-stream.md](contracts/http-wav-stream.md) is an addendum to the v2.0.0 contract. The version moves to 2.1.0 (additive), and is on the wire as `X-Breeze-Version`. |
| V | Test the transformation | Pass | Pass | Pure units: the WAV header bytes. Real uvicorn for buffered delivery and disconnects. The same fake runtime at the GPU edge as 003 (its recorded deviation still applies). |
| VI | Structured events | Pass | Pass | `speech.generated`, `speech.queued_timeout`, and a `format` field on the existing outcomes ([data-model.md](data-model.md#events)). |
| VII | Recovery over prevention | Pass | Pass | The change is additive, and no existing client changes behaviour until the extension opts in by version. Rollback means restarting the `v2.0.0` tag (seconds). The extension then sees `2.0.0` and falls back to the WebSocket. No flag is needed, since the blast radius is this route only. |
| VIII | Attention is finite | N/A | N/A | No alerts. |
| IX | Value at the user | Pass | Pass | Done means the live gate ([quickstart.md](quickstart.md)) passes on the running server and `st-agent` is told 2.1.0 is live. |
| X | Discoverable commands | Pass | Pass | No new commands. The quickstart uses existing ones. |

Result: **PASS**, with no new deviations.

## Project Structure

### Documentation (this feature)

```text
specs/004-browser-wav-stream/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/http-wav-stream.md
├── checklists/requirements.md
├── research/live-004.md       # live gate record (implementation phase)
└── tasks.md                   # /speckit-tasks
```

### Source Code (repository root)

```text
breeze_infer/
├── __init__.py        # __version__ → 2.1.0.devN, then 2.1.0
├── api.py             # uvicorn.Config: h11_max_incomplete_event_size=MAX_REQUEST_HEAD_BYTES
├── limits.py          # + WAV_GPU_WAIT_SECONDS = 60, WAV_SEND_TIMEOUT_SECONDS = 600,
│                      #   MAX_REQUEST_HEAD_BYTES = 192 KiB
├── streaming.py       # SpeechResponse: + media_type parameter (default audio/pcm)
└── routes_speech.py   # _serve_speech: + gpu_wait / wav parameters; the GET route;
                       #   _wait_for_gpu (R3), _buffered (R4), wav_header (R5)
tests/
├── test_speech_wav.py         # TestClient: header bytes + headers, validation matches POST,
│                              #   503 busy_timeout (shortened wait)
├── test_speech_wav_stream.py  # real uvicorn: GPU released while the client isn't reading, then
│                              #   the full body arrives; disconnect mid-generation frees the GPU;
│                              #   ~90 KB URL accepted; a GET that waits, then streams
└── gpu/test_speech_wav.py     # one GET vs POST comparison on the real model
README.md, CHANGELOG.md        # route, headers, security note, 2.1.0 entry
```

**Structure Decision**: Extend the existing modules. The new code (about 120 lines) sits next to the
POST route it reuses.

## Testing approach (kept tight, per the user)

- **Automated tests**:
  - **Covered**: the two new behaviours that could silently break things, buffered delivery
    freeing the GPU and a disconnect freeing it. Also the wire format, validation parity, and the
    503 path.
  - **Not duplicated**: POST-route validation corners and `SpeechResponse` timeout mechanics. Those
    already have tests.
  - **No slow timer tests**: no 600 s send-timeout test, no 60 s wait test at full length.
- **Browser behaviour and long texts**: checked only at the live gate. These are gaps, playback
  speeds, the `0xFFFFFFFF` header, holding headers, and 10,000-character texts.

## Delivery

1. Bump to `2.1.0.dev1` and add CHANGELOG "Unreleased" entries with the first code commit.
2. Implement in small commits, each followed by `ruff` + `pytest` and the review-agent passes. Run
   `pytest -m gpu` once the route works on the model.
3. Run the live gate per [quickstart.md](quickstart.md) and record it in `research/live-004.md`.
4. Release 2.1.0: final version and CHANGELOG date, then merge to `main`. Tell `st-agent`.

## Complexity Tracking

No new deviations. 003's recorded deviation V (a fake runtime at the GPU edge) still applies.
