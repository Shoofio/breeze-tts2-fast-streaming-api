#!/bin/sh
# Starts the Breeze TTS streaming API under WSL.
# Windows counterpart: scripts/start_breeze.ps1.

HUB=$HF_HOME/hub/models--BreezeBlue--Breeze-TTS-2
MODEL="$HUB/snapshots/$(cat "$HUB/refs/main")"

# Extra arguments pass straight through, e.g. --attn-implementation sdpa.
uv run python -m breeze_infer.api "$MODEL" \
    --host 0.0.0.0 --port 8080 --fast-all \
    "$@"
