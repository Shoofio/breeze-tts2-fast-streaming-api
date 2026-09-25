#!/usr/bin/env bash
set -euo pipefail

# The container always serves HTTP on 8080 and WebSocket on 8081 -- passing
# --port or --ws-port in API arguments changes what the process inside the
# container binds to, but NOT the --publish mappings below, so it would
# silently stop being reachable on the published host ports. To use a
# different host port, edit the --publish flags instead, e.g. to publish the
# container's fixed 8080 on host port 9000: -p 9000:8080.

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
