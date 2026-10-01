"""Request and connection limits (specs/003-cpp-compatible-api/data-model.md "Limits").

Constants rather than launch options: nothing needs to change them at run time, and each one is
part of the documented contract.
"""

_MIB = 1024 * 1024

# HTTP bodies: 25 MiB of reference audio plus room for the other form fields.
MAX_BODY_BYTES = 26 * _MIB
MAX_AUDIO_BYTES = 25 * _MIB

MAX_REF_SECONDS = 30
MAX_TEXT_CHARS = 10_000
MAX_REF_TEXT_CHARS = 2_000
MAX_INSTRUCTION_CHARS = 2_000
MAX_NEW_TOKENS_CEILING = 1_500

# Soft opening budget for the first piece when there is no reference: short enough for a quick
# first audio, long enough to give the later pieces a usable anchor.
ANCHOR_CHARS = 200

# How long the GPU thread waits, once piece 0 has finished, for the later pieces' anchor sizing
# (routes_speech._anchor_for_later_pieces) before skipping the anchor instead. A backstop only:
# the sizing has a worker of its own (routes_speech.CpuTokenizer), with nothing else queued on
# it, sizing the largest text (MAX_TEXT_CHARS) takes well under a second on the CPU, and it
# started when piece 0 did. A wait this long means that worker is wedged, or still finishing a
# previous request's abandoned sizing, or the CPU is starved. While it waits, the GPU thread is
# idle and a disconnect's gen.close() queues behind it, so it must stay well under
# gpu.GPU_CLOSE_TIMEOUT_SECONDS (30 s), past which that close poisons the GPU gate.
ANCHOR_SIZING_TIMEOUT_SECONDS = 5.0

UNNAMED_VOICE_CAP = 64

# GPU memory the voice prefix cache (voice_prefix.VoicePrefixCache) may hold, in bytes of KV.
# Bounded by bytes, not by a count of prefixes, since a prefix's size grows with its
# reference: each entry is estimated as prefix_len x bytes per token, where bytes per token is
# 2 (key and value) x layers x KV heads x head_dim x dtype size (voice_prefix
# .kv_bytes_per_token) -- 114,688 B for this checkpoint (28 x 8 x 128, bf16). 1 GiB is about
# what the old server held: 16 prefixes (its --voice-cache-size default) x its 548-token cap
# x 114,688 B = 0.94 GiB. The contract defines no flag for it, and a miss only costs one
# prefix build.
VOICE_PREFIX_CACHE_BYTES = 1024 * _MIB

# The largest *.voice.json the startup scan reads; anything bigger is skipped unread
# (voice_store.scan). A documented generous constant, not computed from the loaded codec:
# the biggest file the server can write is MAX_REF_SECONDS of codes -- at most 376 frames
# (reference_audio.MAX_REF_FRAMES) x 16 codebooks x 2 bytes = 12,032 bytes, 16,044 as
# base64 -- plus MAX_REF_TEXT_CHARS of ref_text, at worst 24,000 bytes if a hand-edited
# file escapes every character as a \uXXXX surrogate pair, plus well under 1 KiB of field
# names and indentation: about 41 KB. 256 KiB leaves six times that. A codec with a different frame
# rate or codebook count also changes the codec fingerprint, so its files are rejected
# anyway; the bound only has to be comfortably above anything valid.
MAX_VOICE_FILE_BYTES = 256 * 1024

WS_MAX_MESSAGE_BYTES = 1 * _MIB
WS_MAX_CONNECTIONS = 16
WS_HANDSHAKE_SECONDS = 10
WS_OUTBOX_BYTES = 2 * _MIB

# A WebSocket client whose socket has stopped draining: the connection's write buffer non-empty
# and not shrinking for this long evicts it with 1008 (ws_server's stall watchdog). The outbox
# limit alone can't catch every such client: one stalled with less than WS_OUTBOX_BYTES pending
# never overflows it, and on an idle session the library's automatic pongs fill the buffer while
# its keepalive ping waits in drain() with its own timeout never started (research/
# ws-prototype.md "Decision"). The same 30 s as HTTP_SEND_TIMEOUT_SECONDS.
WS_SEND_TIMEOUT_SECONDS = 30

# The bound on every server-initiated WebSocket close, and the library's close_timeout. The
# library's own timeout only starts once the close frame has been written into the socket, which
# never happens for a peer that stopped reading, so ws_server bounds each close itself and then
# aborts the connection (T070; research/ws-prototype.md check 1). A client that reads again within
# it gets the close code; one that doesn't sees 1006.
WS_CLOSE_TIMEOUT_SECONDS = 2

HTTP_SEND_TIMEOUT_SECONDS = 30

# How long GET /v1/audio/speech.wav waits for a busy GPU before answering 503 busy_timeout. An
# <audio> element can't retry, so this route waits where POST /v1/audio/speech answers 409 at
# once. 60 s matches the SillyTavern extension's own queue timeout.
WAV_GPU_WAIT_SECONDS = 60

# The WAV route's send timeout: one send blocked this long ends the stream. The route delivers
# from a buffer after releasing the GPU, so a slow or paused reader (browser read-ahead limits,
# playback speed down to 0x) costs only memory, never the GPU. It bounds a reader that stops,
# not one that keeps reading slowly; the buffer's memory is deliberately uncapped (004 R4).
WAV_SEND_TIMEOUT_SECONDS = 600

# The HTTP request-head limit: h11's max incomplete event size, which covers the request line
# and every header (and chunked framing lines). GET /v1/audio/speech.wav carries its fields in
# the query string. 10,000 characters of text outside the basic plane are about 120 KB
# percent-encoded (12 bytes each). `instruction` and `ref_text` share the model's 2,048-token
# context with piece 0, at no more than about 12 bytes per token: about 24 KB more. So the
# largest valid request is about 140-145 KB before headers. h11's default is 16 KiB. Each connection can buffer up to this much
# while its head is still arriving.
MAX_REQUEST_HEAD_BYTES = 192 * 1024

# Minimum delivery rate for a streamed speech response (research.md R3). The send timeout only
# catches a client that stops reading entirely; one that trickles (a few bytes before each
# timeout) could hold the single GPU for hours. So the total time spent blocked in send() may
# not exceed the grace period plus the audio delivered divided by the rate. Only time blocked
# in send() counts, so a slow GPU never trips it; only a slow client does. The grace period
# absorbs slow starts and hiccups; half of real time is far below what any client that plays
# the audio has to read.
MIN_RATE_GRACE_SECONDS = 30.0
MIN_RATE_REAL_TIME = 0.5

# The kernel drops an accepted connection whose peer hasn't acknowledged data, or has kept a
# zero receive window, for this long (api.bind_http_sockets, research.md R5; Linux only). It
# frees only the socket: application timeouts already release the GPU. It must not be shorter
# than the longest application send timeout. Otherwise a browser that stops reading while far
# ahead of playback is reset before the WAV route's own 600 s limit (004 FR-016). It was 30 s
# before 2.1.0.
TCP_USER_TIMEOUT_MS = WAV_SEND_TIMEOUT_SECONDS * 1000
