"""Request and connection limits (specs/003-cpp-compatible-api/data-model.md "Limits").

Constants rather than launch options: nothing needs to change them at run time, and each one is
part of the documented contract.
"""

MIB = 1024 * 1024

# HTTP bodies: 25 MiB of reference audio plus room for the other form fields.
MAX_BODY_BYTES = 26 * MIB
MAX_AUDIO_BYTES = 25 * MIB

MAX_REF_SECONDS = 30
MAX_TEXT_CHARS = 10_000
MAX_REF_TEXT_CHARS = 2_000
MAX_INSTRUCTION_CHARS = 2_000
MAX_NEW_TOKENS_CEILING = 1_500

# Soft opening budget for the first piece when there is no reference: short enough for a quick
# first audio, long enough to give the later pieces a usable anchor.
ANCHOR_CHARS = 200

UNNAMED_VOICE_CAP = 64

WS_MAX_MESSAGE_BYTES = 1 * MIB
WS_MAX_CONNECTIONS = 16
WS_HANDSHAKE_SECONDS = 10
WS_OUTBOX_BYTES = 2 * MIB

HTTP_SEND_TIMEOUT_SECONDS = 30
TCP_USER_TIMEOUT_MS = 30_000
