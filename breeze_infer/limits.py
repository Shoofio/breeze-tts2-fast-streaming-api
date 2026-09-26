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

# KV prefixes kept for registered voices (voice_prefix.VoicePrefixCache). Ported from the
# old server's --voice-cache-size default; the contract defines no flag for it, and a miss
# only costs one prefix build.
VOICE_PREFIX_CACHE_SIZE = 16

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

HTTP_SEND_TIMEOUT_SECONDS = 30

# Minimum delivery rate for a streamed speech response (research.md R3). The send timeout only
# catches a client that stops reading entirely; one that trickles (a few bytes before each
# timeout) could hold the single GPU for hours. So the total time spent blocked in send() may
# not exceed the grace period plus the audio delivered divided by the rate. Only time blocked
# in send() counts, so a slow GPU never trips it; only a slow client does. The grace period
# absorbs slow starts and hiccups; half of real time is far below what any client that plays
# the audio has to read.
MIN_RATE_GRACE_SECONDS = 30.0
MIN_RATE_REAL_TIME = 0.5
TCP_USER_TIMEOUT_MS = 30_000
