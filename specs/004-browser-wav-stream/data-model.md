# Data Model: Browser-Playable WAV Streaming

**Feature**: [spec.md](spec.md) | **Date**: 2026-09-30

Nothing is stored on disk and there are no new records. The only new state lives for the length of
one request.

## Stream buffer (one per GET request, in memory)

| Aspect | Value |
|---|---|
| Holds | PCM chunks generated but not yet handed to the client, plus at most one exception from generation |
| Created | when the first chunk has been sent (the producer task starts) |
| Grows | at generation speed; about 48 KB per second of audio |
| Freed | when the last chunk is sent, when the client disconnects, or on `send_timeout` (600 s blocked) |
| Bound | none (spec Assumptions: one request generates at a time; worst case ~100–115 MB) |

### Life of a GET request

```text
validated ──► queued ──(GPU free)──► generating ──(DONE: GPU released)──► draining ──► completed
                │                      │                                    │
                ├─ 60 s ─► 503         ├─ disconnect ─► aborted             ├─ disconnect ─► aborted
                └─ disconnect ─► gone  └─ error ─► failed                   └─ 600 s blocked ─► aborted
```

- **validated → queued**: all POST-route checks passed (unknown voice, text, ranges, room for
  piece 0 is checked after the GPU is held, as today).
- **generating**: the GPU is held. Headers are sent together with the first chunk.
- **draining**: the GPU is free. Only the buffer remains.

## Events

This table is the record of the route's events; 003's catalog stays as it was for v2.0.0. They use
the same emitter, and every record carries `request_id`.

| Event | When | Fields added |
|---|---|---|
| `speech.completed` / `speech.aborted` / `speech.failed` | as today: exactly one per request that took the GPU | `format: "wav"` on the GET route |
| `speech.generated` | new: the GET route's generation finished and the GPU was released while delivery continues | `audio_seconds` generated |
| `speech.queued_timeout` | new: the GET route waited 60 s for the GPU and answered `503` | `waited_s` |
| `speech.aborted` with `queued: true` | the client disconnected while waiting for the GPU (never took it) | `reason: "client_disconnect"` |

The `speech.aborted` reasons used by the GET route are the existing ones: `client_disconnect`,
`cancelled` and `send_timeout`. `too_slow` never fires on this route.
