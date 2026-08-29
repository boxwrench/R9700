#!/usr/bin/env bash
set -Eeuo pipefail

environment_file="/ai/lab/experiments/minimax-h3/env/common.env"
set -a
# shellcheck source=/dev/null
source "$environment_file"
set +a

experiment_root="$(cd "$(dirname "$0")/.." && pwd)"
state_root="${H3_R2V_STATE_ROOT:-/tmp/minimax-h3-r2v-runtime}"
output_root="${H3_R2V_OUTPUT_DIR:-/ai/artifacts/runs/minimax-h3-r2v-experiment}"
mkdir -p "$state_root/single-user" "$output_root/single"
cd "$COMFYUI_PATH"
exec "$H3_ENV/bin/python" "$COMFYUI_PATH/main.py" \
  --extra-model-paths-config "$experiment_root/runtime/extra-paths.yaml" \
  --input-directory /ai/comfyui/input \
  --user-directory "$state_root/single-user" \
  --database-url "sqlite:///$state_root/single-user/comfyui.db" \
  --output-directory "$output_root/single" \
  --disable-dynamic-vram \
  --disable-smart-memory \
  --disable-mmap \
  --bf16-vae \
  --reserve-vram 2 \
  --listen 127.0.0.1 \
  --port 8190 \
  --disable-auto-launch
