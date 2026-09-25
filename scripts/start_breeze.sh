#!/bin/sh
# Starts the Breeze TTS streaming API under WSL.
# Windows counterpart: scripts/start_breeze.ps1.

# Plain sh (see shebang), so no pipefail -- it isn't defined outside bash/zsh.
set -eu

# breeze_infer and models are imported as top-level packages, so the server
# must run from the repo root, regardless of the caller's working directory.
cd "$(dirname "$0")/.."

HUB=$HF_HOME/hub/models--BreezeBlue--Breeze-TTS-2
REF_FILE="$HUB/refs/main"

if [ ! -f "$REF_FILE" ]; then
    echo "HuggingFace ref not found: $REF_FILE" >&2
    exit 1
fi

MODEL="$HUB/snapshots/$(cat "$REF_FILE")"

if [ ! -d "$MODEL" ]; then
    echo "Model snapshot not found: $MODEL" >&2
    exit 1
fi

# Extra arguments pass straight through, e.g. --attn-implementation sdpa.
uv run python -m breeze_infer.api "$MODEL" \
    --host 0.0.0.0 --port 8080 --fast-all \
    "$@"
