#!/usr/bin/env bash
set -euo pipefail

# The container always serves HTTP on 8080. WebSocket on 8081 arrives with
# the WebSocket server (T077); 8081 is published now so it works as soon as
# that lands. Passing --port or --ws-port in API arguments would change what
# the process inside the container binds to, but NOT the --publish mappings
# below, so it would silently stop being reachable on the published host
# ports -- that combination is rejected below instead. To use a different
# host port, edit the --publish flags instead, e.g. to publish the
# container's fixed 8080 on host port 9000: -p 9000:8080.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_PATH [API arguments...]" >&2
  exit 2
fi

model_path="$1"
shift

for arg in "$@"; do
  case "$arg" in
    --port|--port=*|--ws-port|--ws-port=*)
      echo "error: $arg is not supported here -- the container always serves" >&2
      echo "8080 (HTTP) and 8081 (WebSocket). Publish a different host port" >&2
      echo "instead by editing this script's --publish flags, e.g." >&2
      echo "-p 9000:8080." >&2
      exit 2
      ;;
  esac
done

if ! model_path="$(realpath -e "$model_path")"; then
  echo "error: model path not found: $model_path" >&2
  exit 1
fi

image_tag="${BREEZE_IMAGE:-breeze-pytorch-infer:latest}"

# --stop-timeout: docker's default is 10 s before SIGKILL. The server needs up to
# 10 s to let open responses finish plus 10 s to drain the GPU, so give it 30.
docker run --rm --gpus all \
  --stop-timeout 30 \
  --ipc=host \
  --publish 8080:8080 \
  --publish 8081:8081 \
  --volume "$model_path:/models/breeze:ro" \
  "$image_tag" \
  python -m breeze_infer.api /models/breeze \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
