#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_PATH [API arguments...]" >&2
  exit 2
fi

model_path="$1"
shift
image_tag="${BREEZE_IMAGE:-breeze-pytorch-infer:latest}"

docker run --rm --gpus all \
  --ipc=host \
  --publish 8080:8080 \
  --publish 8081:8081 \
  --volume "$model_path:/models/breeze:ro" \
  "$image_tag" \
  python -m breeze_infer.api /models/breeze \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
