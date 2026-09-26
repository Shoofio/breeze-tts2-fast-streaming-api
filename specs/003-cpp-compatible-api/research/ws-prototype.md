# T070: WebSocket prototype (R4 plan check)

Run 2026-09-26 with websockets 17.1, uvicorn 0.52.4 and Python 3.12.12 on WSL2 (Linux 6.18), on
127.0.0.1 with ephemeral ports. The scripts are in the session scratchpad
(`scratchpad/ws-proto/check1_stalled.py` … `check5_protocol.py`), not in the repo.

## Result

| # | Check | Result |
|---|---|---|
| 1 | A peer that stops reading is aborted after `close_timeout`, and the socket is gone | **FAIL** |
| 2 | `process_request` returns JSON `403`/`503`; `X-Breeze-Version` on every handshake response | PASS (two caveats) |
| 3 | Pre-bound socket works; a bind failure is catchable before serving | PASS (one caveat) |
| 4 | Same loop as uvicorn, one shutdown path, clean exit | PASS only with the check 1 workaround; **hangs as designed** |

**Check 1 fails, so the plan needs re-planning (T070: stop and re-plan with the user).** R4's
claim "the `websockets` server aborts the transport when `close_timeout` expires" is only true
while the socket can still be written to. For a peer that has stopped reading while we were
sending audio (the BC-42 case), `ws.close(1008, "client too slow")` never returns, and nor do
the keepalive ping timeout or `Server.close()`. The same stall also hangs the shutdown path of
check 4. The design isn't changed here; possible fixes are listed under check 1.

## Decision (user, 2026-09-26; completed after review 44)

Keep `websockets`, and bound every server-initiated close ourselves (check 1's measured
workaround), with the kernel as a backstop:

- **Bounded close**: cancel the sender, run `ws.close(code, reason)` inside
  `asyncio.timeout(WS_CLOSE_TIMEOUT_SECONDS)` (2 s, the same constant as `close_timeout`), and on
  timeout set `SO_LINGER(1, 0)` and call `ws.transport.abort()`. A client that starts reading again
  in time gets the close code; one that never reads again sees 1006, which no design can avoid.
- **Every close path uses it**:
  1. a slow client: the outbox would overflow `WS_OUTBOX_BYTES` (1008);
  2. a slow client: one `ws.send()` blocked for `WS_SEND_TIMEOUT_SECONDS` (30 s, as for HTTP),
     which catches a client that stalls with less than 2 MiB pending, where nothing overflows and
     the keepalive ping is stuck behind the same full buffer (1008);
  3. shutdown (1001);
  4. every handler exit, normal or by error, in the handler's `finally`, so the library's own
     close after the handler (and its `close(1011)` after an error) is already done and can't
     hang in `drain()`. The keepalive's own `fail(1011)` also goes through `drain()`; it can only
     stall once the buffer is full, which means we were sending, so path 2 reaches it.
- **Our own connection set**: the server tracks every connection from `process_request` until
  its handler's `finally`, in any state, for the 16-connection cap (reserved in
  `process_request`, so concurrent handshakes can't exceed it) and for shutdown. Shutdown walks
  that set, not `server.connections` (OPEN only).
- **Shutdown refusal**: a shutdown flag is set before `server.close(close_connections=False)`;
  `process_request` then refuses new handshakes with our JSON `503 shutting_down` and
  `X-Breeze-Version`, so FR-037a holds with no exception.
- **Backstop**: `TCP_USER_TIMEOUT` (`limits.TCP_USER_TIMEOUT_MS`, 30 s) on the WebSocket listening
  socket too, where the platform has it (Linux). Windows' `TCP_MAXRT` was not evaluated, so there
  the bounded close is the only eviction. The `SO_LINGER` value is packed per platform (two ints on
  Linux, two unsigned shorts on Windows).
- **Library refusals** (its own `400`, `426` and `500`) are rewritten into the JSON envelope in
  `process_response`.
- **Events**: `ws.closed` carries the code we sent, plus `aborted` and `reason`, so a BC-42
  eviction is distinguishable from a client that vanished (both look like 1006 to `close_code`).
- **Rejected**: aborting without a close frame (breaks BC-42's 1008), the kernel option alone
  (holds a connection slot for 30 s, doesn't unblock shutdown, Linux only), and other libraries
  (uvicorn never aborts; `wsproto` means hand-rolled I/O; `aiohttp` and Hypercorn are untested
  and would need the same bounded close).
- **Not yet measured on native Windows**: the prototype ran on WSL2 only.

## Check 1: stalled peer (FAIL)

Setup: a raw-socket client with `SO_RCVBUF=4096` completes the handshake and never reads again.
The server handler puts 64 KiB binary frames into an outbox bounded at 2 MiB, drained by a
sender task doing `await ws.send(chunk)`. When a put would overflow the outbox, the handler
calls `await ws.close(1008, "client too slow")`. Settings: `close_timeout=2`, `ping_interval=20`,
`ping_timeout=20`, `compression=None`.

**As designed (`ws.close(1008)`; my 15 s cap stands in for "forever")**:

```
[  0.039s] server: outbox would overflow (queued=2097152, produced=2162688)
[  0.039s] server: calling ws.close(1008, 'client too slow')
[ 15.055s] server: ws.close STILL BLOCKED after 15.015s (state=CLOSING, fd_alive=True)
```

**Why**: in `websockets/asyncio/connection.py`, `send_context()` sets
`close_deadline = now + close_timeout` (line 911), then calls `self.send_data()` and
`await self.drain()` (lines 913-915). The deadline is set on time, but it is only *enforced*
(`timeout_at(close_deadline)`) after `drain()` returns. So a client that resumes reading at
1.9 s has only about 0.1 s left before the library aborts it.
Once the transport's write buffer is above `write_limit` (32 KiB by default), `drain()` waits for
`resume_writing()`, which never comes while the peer is not reading. So `close_timeout` never
is enforced. The kernel keeps the connection alive indefinitely, because the peer still ACKs
the zero-window probes.

**`ping_timeout` doesn't help either.** With `ping_interval=1` and `ping_timeout=1`, the same
flooded, non-reading peer was still OPEN after 25 s, with 76,074 bytes buffered in the transport.
`keepalive()` calls `ping()`, which goes through the same `send_context()` → `drain()`, so the
pong timer never starts.

```
[ 25.028s] main: STILL 1 connection(s) after 25s: [('OPEN', 76074)]
```

**When it does work**: if nothing is being sent (the transport isn't paused), a peer that never
answers is dropped as documented. With `ping_interval=1` and `ping_timeout=1`, it was dropped
after 1 + 1 + `close_timeout` 2 = 4.0 s: the server sends 1011 and aborts after `close_timeout`,
and the fd is gone.

```
[  4.007s] server: idle peer closed after 4.004s close_code=1006 reason='' fd_alive=False
```

**Workaround measured (not adopted; for the re-plan)**. Cancel the sender, bound `close()`
ourselves, then abort:

```python
sender_task.cancel()
try:
    async with asyncio.timeout(2):
        await ws.close(1008, "client too slow")
except TimeoutError:
    ws.transport.abort()
await ws.wait_closed()
```

```
[  2.041s] server: close timed out; aborting transport
[  2.041s] server: closed after 2.002s
[  2.041s] server: state=CLOSED close_code=1006 fd_alive=False transport_closing=True
ss after: FIN-WAIT-1 0 50860 127.0.0.1:56663 127.0.0.1:58960 timer:(persist,...)
```

- **The fd leaves the process after 2.00 s.** The kernel keeps an orphaned `FIN-WAIT-1` socket
  that still holds the unsent bytes, until the kernel's orphan limits reap it.
- **To free the kernel side too**, set `SO_LINGER` to `(1, 0)` before `abort()`. Then `ss` shows
  no connection at all, and the client gets `ConnectionResetError: [Errno 104]`.
- **A client that resumes reading within the 2 s does receive the 1008 frame.** With reads
  resuming at 1.0 s, the client read 131,111 bytes, ending in
  `b'\x88\x11\x03\xf0client too slow'` (0x03f0 = 1008).
- **A client that never resumes sees 1006.** The close frame is stuck behind the audio in the
  buffer, so a truly stalled client never sees 1008.

**Other options for the re-plan** (not tested here beyond what is noted):
- **Bounded close as above.** Also needed for server shutdown (check 4).
- **`TCP_USER_TIMEOUT` on the accepted sockets (R4's rejected alternative).** It evicts in the
  kernel, but only after the timeout (`bind_http_sockets` already sets this for HTTP).
- **Skip the close frame and `abort()` straight away when the outbox overflows.** This is
  simplest, but no client ever sees 1008, which contradicts BC-42/BC-43.

## Check 2: handshake refusals and `X-Breeze-Version` (PASS)

`process_request(connection, request)` returns a `websockets.http11.Response` built directly,
with `websockets.datastructures.Headers`. The Origin check uses the real `cors.origin_allowed`,
with `CorsPolicy(origins=("http://127.0.0.1:8000",))`.
`process_response(connection, request, response)` deletes and then sets `X-Breeze-Version`.
`server_header=None` drops the `Server:` header. Raw bytes as received:

```
HTTP/1.1 101 Switching Protocols            HTTP/1.1 403 Forbidden
Date: Sat, 26 Sep 2026 21:33:29 GMT         Connection: close
Upgrade: websocket                          Content-Type: application/json
Connection: Upgrade                         Content-Length: 61
Sec-WebSocket-Accept: Sy4SSP0NHKkdDesRRu5E… X-Breeze-Version: 2.0.0.dev3
X-Breeze-Version: 2.0.0.dev3
                                            {"error": "Origin not allowed", "code": "origin_not_allowed"}

HTTP/1.1 503 Service Unavailable            HTTP/1.1 503 Service Unavailable
Connection: close                           Connection: close
Content-Type: application/json              Content-Type: application/json
Content-Length: 65                          Content-Length: 48
X-Breeze-Version: 2.0.0.dev3                X-Breeze-Version: 2.0.0.dev3

{"error": "Too many connections", "code": "too_many_connections"}
{"error": "Model is loading", "code": "loading"}
```

What else each case gave:
- **No `Origin` → 101, and an allowed `Origin` → 101.**
- **Two `Origin` headers → 403**, via `request.headers.get_all("Origin")`.
- **The library's own refusals also pass through `process_response`, so they carry the
  header.** These are `426` (plain GET, "missing Connection header"), `400` (`Sec-WebSocket-Version: 8`)
  and `500` (`process_request` raised).

Caveats:
- **The library's refusals are `text/plain`, not the JSON envelope.** If the contract's "refusals
  get a JSON body" is meant to cover them, `process_response` must rewrite any non-101 response
  whose `Content-Type` is `text/plain` into the envelope.
- **A handshake that arrives after `Server.close()` gets a text/plain `503 Server is shutting
  down.` without `X-Breeze-Version`.** The library swaps that response in *after*
  `process_response` (`asyncio/server.py`, `handshake()`), and only when the status is 101. To
  keep the header, set our own "closing" flag before `server.close()`, and have `process_request`
  return a JSON 503 for it. The library then leaves the non-101 response alone (this last step is
  from reading the code; I didn't run it).

## Check 3: pre-bound socket and bind failure (PASS)

```
pre-bound 127.0.0.1:56287 fd=7
second bind -> OSError errno=98 (EADDRINUSE): [Errno 98] Address already in use
server.sockets=[('127.0.0.1', 56287)] is_serving=True
echo: hi
after close: pre-bound sock fileno=-1 (-1 = closed by asyncio.Server.close)
serve(host, port, sock=...) -> ValueError: host/port and sock can not be specified at the same time
```

The call that works is `server = await serve(handler, sock=sock, ...)`, with host and port left
out. The bind fails in our own `socket.bind()`, before `serve()`, so T078 can catch `OSError`,
emit `ws.bind_failed` and report `ws_port: 0`. `Server.close()` closes the pre-bound socket.

Caveat: **`sock=` takes one socket** (it is passed through to `loop.create_server`). When
`--host` resolves to several addresses (for example `localhost` → `::1` and `127.0.0.1`,
`bind_http_sockets` returns a list), T078 needs one `serve()` per socket, or must bind only the
first address.

## Check 4: same loop as uvicorn, one shutdown path

Child process under `python -X dev -W always`, mirroring `api.serve()`:
- a `uvicorn.Server` subclass with `capture_signals` returning `nullcontext()`;
- `loop.add_signal_handler(SIGTERM/SIGINT)` setting `server.should_exit`;
- pre-bound sockets for both servers;
- `await serve(..., sock=ws_sock)` and then `await server.serve(sockets=[http_sock])` in one
  `asyncio.run`;
- a stopper task that runs the WebSocket close once `should_exit` is set.

The parent opens an HTTP stream, one normal WebSocket client and (optionally) one flooded
stalled peer, then sends SIGTERM.

**No stalled peer: PASS.** The WebSocket client got 1001 in 0.017 s, and the WebSocket server
closed in 0.011 s. uvicorn drained for its `timeout_graceful_shutdown=3` and then cancelled the
stream. Both listening sockets had `fileno=-1`, the exit code was 0, and there was no
"Task was destroyed" warning.

```
parent: WS client saw ConnectionClosedOK: code=1001 reason='' at 0.017s
[child   0.296s] ws server closed in 0.011s
[child   3.441s] uvicorn returned (3.414s after start)
parent: child exit code 0 after 3.193s
```

**With a stalled peer and `Server.close()` as-is: HANGS.** The cause is the same as in check 1.
`Server._close()` awaits `connection.close(1001)` for every OPEN connection, and that call
never returns for the stalled one. My 10 s cap had to abort it:

```
[child  13.867s] WS SHUTDOWN HUNG > 10 s: [('CLOSING', 76072)]
parent: child exit code 0 after 13.216s      (only because the prototype aborted it)
```

**With a stalled peer and a bounded close: PASS.** The shutdown call was
`server.close(close_connections=False)`, then, for each connection,
`asyncio.timeout(2)` around `c.close(1001)` and `c.transport.abort()` on timeout, then
`await server.wait_closed()`. The WebSocket side finished in 2.011 s, no tasks were left, and
the exit code was 0.

One uvicorn behaviour is unrelated to WebSockets: after the forced cancel, uvicorn returns
while its request task is still `cancelling`; `asyncio.run` finishes it. That is today's
behaviour too.

## Protocol probes (raw frames; `max_size=2**20`)

| Probe | Server's close frame |
|---|---|
| client close 1000 "bye" | `1000 'bye'` (echoed) |
| unmasked text frame | `1002 'incorrect masking'` |
| invalid UTF-8 text | `1007 'invalid start byte at position 0'` |
| text of exactly 1 MiB | accepted (`str` of 1,048,576 chars) |
| text of 1 MiB + 1 | `1009 'frame with 1048577 bytes exceeds limit of 1048576 bytes'` |
| 2 × 600 KiB fragments | `1009` (the limit applies to the reassembled message) |
| reserved opcode 3 | `1002 'invalid opcode'` |
| binary frame | delivered to `recv()` as `bytes`; the library does nothing else |

`open_timeout=1` with a silent client, or with a half-sent request: the connection is aborted
after 1.00 s with no HTTP response (the client reads EOF).

## Configuration in 17.1

```python
serve(handler, sock=sock,
      process_request=..., process_response=..., server_header=None,
      open_timeout=10, max_size=2**20, ping_interval=20, ping_timeout=20, close_timeout=2,
      compression=None)   # see below
```

These are all keyword arguments of `serve()`. The defaults differ from the plan in two places:
`close_timeout` defaults to 10, and `compression` defaults to `"deflate"`.

## Gotchas for T077/T078

1. **Never rely on `close_timeout` or `ping_timeout` alone** while the connection may have
   unsent output. Every close path uses the bounded close: the four paths listed under
   "Decision".
2. **Catch `websockets.exceptions.ConnectionClosed` in the handler.** Otherwise every abnormal
   close (1002/1007/1009, aborts) is logged by the library at ERROR as
   "connection handler failed", with a traceback, and it then tries `close(1011)`.
3. **`ws.close_code` is the code *received* from the peer** (1006 when none arrived), not the
   one we sent. For `ws.*` events, log `ws.protocol.close_sent.code` and `.reason`.
4. **`Headers.__setitem__` appends.** `del` a header before setting it, or you get duplicate
   `Content-Type` or `X-Breeze-Version` headers.
5. **`origin_allowed(policy, None)` returns `False`.** The contract allows connections with no
   `Origin`, so `process_request` must treat a missing `Origin` as allowed before calling it.
   Use `headers.get_all("Origin")`: `headers.get` raises `MultipleValuesError` on duplicates.
6. **`server.connections` counts OPEN connections only.** A connection is added after its 101
   has been written, which involves an `await`. Two simultaneous handshakes can therefore both
   pass a `len(server.connections) >= 16` check. Use our own set (see "Decision").
7. **`compression` defaults to permessage-deflate.** PCM compresses poorly and costs CPU, so
   `compression=None` is probably wanted. `max_size` counts decompressed bytes if compression
   stays on.
8. **One socket per `serve()`** (see check 3).
9. **During `Server.close()` the library's own `503` lacks `X-Breeze-Version`** (see check 2).
   Refuse with our own shutdown flag first (see "Decision").
