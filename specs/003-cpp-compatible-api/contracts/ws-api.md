# WebSocket Session Contract — v2.0.0

**Contract version**: `2.0.0`. Behavior is the C++ server's `docs/websocket.md` except where a
breaking change `BC-nn` or an additive change is marked.

## Connection

**Where to connect**
- Listens at `ws://<host>:<ws_port>/`: by default the HTTP port + 1, and only on `--host` (BC-30).
- Any path is accepted, as in C++.
- The port can be discovered with `GET /health`.

**Handshake**: RFC 6455 only. Every handshake response, accepted or refused, carries
`X-Breeze-Version: <server version>` (additive, FR-037a). Refusals get a JSON body `{"error","code"}`:
- `403 origin_not_allowed` if an `Origin` header is present and not allowed by the CORS setting.
  With CORS off, no browser origin is allowed (BC-31).
- `503 loading` while the model loads.
- `503 too_many_connections` above 16 connections.
- `503 shutting_down` once the server has started shutting down.
- The handshake must complete within 10 s.

**Frames**
- Text frames carry JSON messages. Binary frames from the server carry PCM.
- Inbound messages are limited to 1 MiB (close 1009 above that).
- The server pings every 20 s and closes with 1011 if there is no pong within 20 s.

**Close codes** (BC-43)

| Code | When |
|---|---|
| 1000 | echoed on a client close; also on server shutdown (1001). A client that doesn't read the shutdown close frame within 2 s is dropped without one (it sees 1006) |
| 1002 | protocol error (for example an unmasked frame) |
| 1007 | invalid UTF-8 |
| 1008 | `client too slow`: more than 2 MiB of undelivered output, or one send blocked for 30 s; the piece in flight is cancelled first (BC-42). A client that doesn't read the close frame within 2 s is dropped without one (it sees 1006) |
| 1009 | message too big |
| 1011 | ping timeout or internal error |

## Server → client

On connect, before any client message, the server sends
`{"type":"ready","sample_rate":24000,"format":"s16le"}`. `sample_rate` is the loaded model's rate
(BC-33).

| type | Fields | Meaning |
|---|---|---|
| `ready` | `sample_rate`, `format` | Sent once on connect |
| `started` | `voice_id` (string, `""` when none) | A `start` was accepted |
| `queued` | — | The next piece is waiting for the GPU |
| `speaking` | `text` | A piece begins; `text` is the exact piece text (BC-44). Its binary frames follow |
| *(binary)* | — | Mono s16le PCM at `sample_rate` for the current piece |
| `instruction_set` | — | An `instruction` message was applied |
| `cancelled` | — | Exactly one per `cancel`, and per `start` that interrupts pending work, after the last audio frame of the cancelled work (BC-35, BC-36) |
| `done` | — | Exactly one per `end`, after every piece queued before it, even when nothing was left (BC-34). A later `cancel` replaces it with `cancelled` |
| `error` | `message`, `code`, `request_type` | See error codes. `request_type` (additive) is the client message type that caused the error, or `null` |

**Ordering**: all events and audio frames go through one ordered queue.
- Within a session, `speaking`, then that piece's frames, then the next `speaking`, and so on.
- `cancelled`/`done` are emitted strictly in the order of the client messages that caused them.

## Client → server

Messages are JSON objects with a string `type`, parsed with a real JSON parser, so `\uXXXX`
escapes work (BC-32). Rules for every message:
- Unknown fields are ignored.
- Wrong types or out-of-range values produce `error{code: invalid_field}`, and the message has no
  other effect.
- Invalid JSON produces `error{code: invalid_json}`.
- A binary frame produces `error{code: unsupported_binary}` (BC-45).
- The connection stays open after any of these errors.

| type | Fields | Effect |
|---|---|---|
| `start` | `voice_id`, `instruction`, `ref_text`, `cfg_scale`, `seed`, `temperature`, `top_k`, `split_chars`; additive: `top_p`, `repetition_penalty`, `max_new_tokens` | Validates with the same types, ranges and defaults as HTTP. Then, if work is pending, cancels it (one `cancelled`), resets the session, and sends `started`. |
| `text` | `text` (string) | Appends text and speaks each complete sentence. |
| `flush` | `text` (optional) | Appends, then speaks everything buffered. |
| `end` | `text` (optional) | Like `flush`, then one `done` once everything before it has been spoken. The session stays usable afterwards. |
| `instruction` | `instruction` | Applies to pieces that start after this message. Blank means the default instruction (BC-37). Replies `instruction_set`. |
| `cancel` | — | Discards buffered text and queued pieces, stops the piece in flight, replaces a pending `done`, and replies with one `cancelled`, even when idle (BC-35). |

**`start` details**
- `split_chars` is 0–10,000. `0` means no length limit: each drain's ready text becomes one piece, and unpunctuated text
  waits for a sentence end, `flush` or `end` (the 10,000-character buffer limit still applies, so an
  endless unpunctuated stream gets `text_too_long`); absent means the
  server default. A negative value gets `invalid_field` (BC-38).
- `start` is rejected with an `error`, leaving the previous session unchanged, when:
  - `voice_id` is unknown (`unknown_voice`, `unknown voice_id`);
  - `ref_text` comes without `voice_id` (`reference_required`, BC-13).

**`text` details**
- A sentence end at the very end of the buffer waits for more text, `flush` or `end` (BC-39).
- Text is cut at sentence ends and packed into pieces up to the budget, which is soft: a single
  unit heavier than it stays whole. A sentence over the budget is cut into clauses at the first
  break after the clause reaches the budget. Breaks are space, tab, CR, U+3000, NBSP and the other
  fixed-width spaces (U+2000–U+200A, U+202F, U+205F), `，` and `、`; `,` and `:` break only through
  the whitespace after them, so `1,000` and `10:30` stay whole. A clause that would pass 2 × the
  budget is closed at its last break instead, and a run with no break at all (for example CJK
  without punctuation) is hard-cut into chunks within the budget once it is over 2 × the budget,
  never inside a grapheme cluster. Complete sentences and closed clauses are spoken at once; the
  rest waits, so the buffer stays bounded.
- Sentence ends: LF; `.` `!` `?` `;` (after any closers) followed by space, tab, CR, LF or U+3000
  (NBSP doesn't end a sentence, so `Dr.\u00a0Smith` stays whole); and the CJK stops `。！？；…．`,
  which absorb following stops and closers (a `．` between two digits is not a stop). Closers
  absorbed: `" ' ) ] } ” ’ 」 』 ） 》 】 〉 〕 〗 〙 〛 ］ ｝ » › ｣ 〞 〟 ＂ ＇`.
- A piece with no letter or digit (for example emoji-only or punctuation-only) is dropped.
- If the buffer plus the new text would exceed 10,000 characters, the server sends
  `error{code: text_too_long}` and does not append the text (BC-40).
- Control characters are rejected with `invalid_field` (BC-46).

**Other rules**
- Any message other than `start` before a successful `start` gets
  `error{code: not_started, message: "send start first"}`.
- The opening budget of 200 applies only to the first piece of a session that has no reference.
- Piece `i`, counted from `start`, uses seed `(seed + i) mod 2^32`.

**Generation**
- Pieces wait for the GPU in order; `queued` is sent only when a piece actually has to wait.
- The GPU is never held while waiting on the socket (BC-42).
- If generating a piece fails, the server sends `error{code: generation_failed}` and moves on to
  the next item. The session and server keep running (BC-41).

## Error codes

`invalid_json`, `unknown_type` (message `unknown type`), `invalid_field`, `not_started`,
`unknown_voice`, `reference_required`, `text_too_long`, `unsupported_binary`, `generation_failed`,
`internal_error`.

## Compatibility notes for existing clients

Clients written for the C++ server keep working if they wait for `ready`, send
`start → started → text/end`, and handle `queued`, `done`, `cancelled` and `error` with its
`message` key.

The known client is the SillyTavern extension, which sends a single `end`. It gets:
- `\uXXXX` escapes decoded correctly;
- a `cancelled` for every `cancel`, including when idle;
- a clean 1000 close instead of the connection dropping (1006).
