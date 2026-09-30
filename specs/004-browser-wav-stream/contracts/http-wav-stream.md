# HTTP API Contract Addendum: `GET /v1/audio/speech.wav` (v2.1.0)

This addendum extends the v2.0.0 contract (`specs/003-cpp-compatible-api/contracts/http-api.md`).
Every existing route, header and error code is unchanged, and so is the WebSocket API. Clients
detect this route by `X-Breeze-Version >= 2.1.0`. `/health` is unchanged.

## Request

```text
GET /v1/audio/speech.wav?text=<pct-encoded>&voice_id=<id>&seed=<n>&...
```

- **Fields**: the same names, types, defaults, limits and validation as `POST /v1/audio/speech`,
  sent in the query string:
  `text`, `voice_id`, `ref_text`, `instruction`, `cfg_scale`, `seed`, `temperature`, `top_k`,
  `top_p`, `repetition_penalty`, `max_new_tokens`, `split_chars`.
- **Reference**: a saved `voice_id`, or none. `ref_audio` in a query gets `400 invalid_field`.
- **Request line**: up to 128 KiB. A longer one gets `400` from the HTTP layer, before the app runs.
- **Ignored**: `Range` and other conditional headers. There is no CORS requirement; an `<audio>`
  element fetches it as a no-cors request.

## Success response

```text
HTTP/1.1 200 OK
Content-Type: audio/wav
Transfer-Encoding: chunked
Cache-Control: no-store
Accept-Ranges: none
X-Sample-Rate: 24000
X-Sample-Format: s16le
X-Breeze-Version: 2.1.0
X-Request-Id: <id>
```

**Body**: a 44-byte header, then s16le mono PCM, flushed chunk by chunk using the POST route's ramp.

| Offset | Bytes | Value |
|---|---|---|
| 0 | 4 | `RIFF` |
| 4 | 4 | `0xFFFFFFFF` (unknown length) |
| 8 | 4 | `WAVE` |
| 12 | 4 | `fmt ` |
| 16 | 4 | 16 |
| 20 | 2 | 1 (PCM) |
| 22 | 2 | 1 (mono) |
| 24 | 4 | sample rate (= `X-Sample-Rate`) |
| 28 | 4 | sample rate × 2 |
| 32 | 2 | 2 (block align) |
| 34 | 2 | 16 (bits per sample) |
| 36 | 4 | `data` |
| 40 | 4 | `0xFFFFFFFF` (unknown length) |

All integers are little-endian.

## Errors

Errors use the same `{"error","code"}` envelope, statuses and order as `POST /v1/audio/speech`,
with one exception: the route never answers `409 busy`.

| Status | Code | When |
|---|---|---|
| `503` | `busy_timeout` | **New.** The GPU stayed busy for 60 s after the request arrived. |
| `503` | `loading` / `gpu_unavailable` | As on every route. |

Every error is sent before any audio, so it is always a real non-2xx response.

## Streaming behaviour

- **GPU queue**: the request waits for the GPU in first-in, first-out order, and holds its status
  line until the first audio chunk exists. If the client disconnects while waiting, it leaves the
  queue.
- **Buffered delivery**: generation runs at full speed and releases the GPU when it ends. The client
  may read the buffered remainder at any pace.
- **Aborts**:
  - A client that disconnects during generation frees the GPU within one chunk.
  - A single send blocked for 600 s aborts the stream (`send_timeout`).
  - There is no minimum read rate.
- **Failure after `200`**: the body ends without the chunked terminator, as on the POST route
  (BC-17).
- **Re-requests**: every GET is an independent synthesis. With the same `seed` it produces the same
  synthesis, within GPU run-to-run tolerance.

## Security note

The route accepts any origin. Browsers send no `Origin` header on media requests, and SillyTavern
suppresses `Referer`. Any web page the user visits can therefore make the server synthesize speech.
This is accepted (spec Clarifications), and the README documents it next to "no authentication".
