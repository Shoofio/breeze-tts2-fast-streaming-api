#!/bin/sh
# Starts the Breeze TTS streaming API on an Apple Silicon Mac (the MLX backend).
# Linux/WSL counterpart: scripts/start_breeze.sh.
#
# Usage: scripts/start_breeze_mac.sh [--precision 8bit|bf16] [server options...]
# 8-bit is the default: bf16 doesn't stream in real time on a 16 GB Mac
# (specs/005-mlx-mac-inference spec FR-012, research R3).

# Plain sh (see shebang), so no pipefail -- it isn't defined outside bash/zsh.
set -eu

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "start_breeze_mac.sh needs an Apple Silicon Mac (this machine: $(uname -s) $(uname -m)); use scripts/start_breeze.sh elsewhere" >&2
    exit 1
fi

precision=8bit
if [ "${1:-}" = "--precision" ]; then
    if [ $# -lt 2 ]; then
        echo "--precision needs a value: 8bit or bf16" >&2
        exit 1
    fi
    precision="$2"
    shift 2
fi

# The pinned community conversions (research R3). A pinned revision, not
# refs/main, so a re-upload can't change the weights under a running setup.
case "$precision" in
    8bit)
        REPO="mlx-community/Breeze-TTS-2-mlx-8bit"
        REVISION="c6e4a2ff6ab9afba68b7853de802273ffe23fb49"
        ;;
    bf16)
        REPO="mlx-community/Breeze-TTS-2-mlx"
        REVISION="3c8829fb7fd335818f085cd2ef49b4100c0e46c8"
        ;;
    *)
        echo "--precision must be 8bit or bf16 (got $precision)" >&2
        exit 1
        ;;
esac

# breeze_infer and models are imported as top-level packages, so the server
# must run from the repo root, regardless of the caller's working directory.
# Relative paths in "$@" therefore resolve against the repo root.
cd "$(dirname "$0")/.."

# mlx-audio declares newer transformers/huggingface_hub than the server pins,
# so a plain install fails on a Mac; the overrides file keeps the server's pins
# (research R2). Only runs when the environment is missing or lacks mlx.
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import mlx.core, mlx_audio" 2>/dev/null; then
    if [ ! -x .venv/bin/python ]; then
        uv venv --python 3.12
    fi
    uv pip install -r requirements.txt --overrides requirements-mac-overrides.txt
fi

# HF_HOME is HuggingFace's own variable for relocating its cache; fall back to
# the default cache location when it is unset.
HUB_DIR="models--$(printf '%s' "$REPO" | sed 's|/|--|')"
MODEL="${HF_HOME:-$HOME/.cache/huggingface}/hub/$HUB_DIR/snapshots/$REVISION"

if [ ! -d "$MODEL" ]; then
    echo "Model snapshot not found: $MODEL (set HF_HOME if the model cache lives elsewhere)" >&2
    echo "Download it with:" >&2
    echo "  uvx --from huggingface_hub hf download $REPO --revision $REVISION" >&2
    exit 1
fi

# exec replaces this shell with `uv run`, which runs the server as its child
# and forwards SIGTERM/SIGINT to it, so stopping this process stops the server.
# (To suspend or inspect the server itself, target the .venv python child.)
# Extra arguments pass straight through; a later --host or --cors wins.
#
# No --fast-all: those options are CUDA-only and the MLX backend refuses them.
# CORS is on for every origin by default, as in start_breeze.sh, so browser
# clients such as SillyTavern work without extra flags. That also turns off the
# server's cross-site 403 guard: any web page can upload or delete voices and
# run synthesis. --cors=http://127.0.0.1:8000 narrows it.
exec uv run python -m breeze_infer.api "$MODEL" \
    --host 0.0.0.0 --port 8080 --cors '*' \
    "$@"
