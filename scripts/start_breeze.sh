#!/bin/sh
# Starts the Breeze TTS streaming API under WSL.
# Windows counterpart: scripts/start_breeze.ps1.

# Plain sh (see shebang), so no pipefail -- it isn't defined outside bash/zsh.
set -eu

# breeze_infer and models are imported as top-level packages, so the server
# must run from the repo root, regardless of the caller's working directory.
# This also means any relative path passed through in "$@" (e.g.
# --voices-dir ./myvoices) resolves against the repo root, not against the
# directory the caller ran this script from.
cd "$(dirname "$0")/.."

HUB=$HF_HOME/hub/models--BreezeBlue--Breeze-TTS-2
REF_FILE="$HUB/refs/main"

if [ ! -f "$REF_FILE" ]; then
    echo "HuggingFace ref not found: $REF_FILE" >&2
    exit 1
fi

sha="$(cat "$REF_FILE")"
# refs/main has no trailing newline in practice, but guard against a
# whitespace-only file so a blank ref doesn't silently resolve to
# ".../snapshots/" and fail with a confusing "not found" error below.
sha_trimmed="$(printf '%s' "$sha" | tr -d '[:space:]')"
if [ -z "$sha_trimmed" ]; then
    echo "HuggingFace ref is empty: $REF_FILE" >&2
    exit 1
fi

MODEL="$HUB/snapshots/$sha"

if [ ! -d "$MODEL" ]; then
    echo "Model snapshot not found: $MODEL" >&2
    exit 1
fi

# exec replaces this shell with the uv/python process, so a SIGTERM (e.g.
# from systemd or `docker stop`) reaches Python directly instead of being
# absorbed by an intermediate shell that never forwards it.
# Extra arguments pass straight through, e.g. --attn-implementation sdpa.
#
# CORS is on for every origin by default, so browser clients such as
# SillyTavern work without extra flags. That also turns off the server's
# cross-site 403 guard: any web page can upload or delete voices and run
# synthesis. A later --cors wins, so --cors=http://127.0.0.1:8000 narrows it.
exec uv run python -m breeze_infer.api "$MODEL" \
    --host 0.0.0.0 --port 8080 --fast-all --cors '*' \
    "$@"
