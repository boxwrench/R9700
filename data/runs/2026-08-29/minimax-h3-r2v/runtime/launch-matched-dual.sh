#!/usr/bin/env bash
set -Eeuo pipefail

environment_file="/ai/lab/experiments/minimax-h3/env/common.env"
set -a
# shellcheck source=/dev/null
source "$environment_file"
set +a
export HIP_VISIBLE_DEVICES=1,0

experiment_root="$(cd "$(dirname "$0")/.." && pwd)"
state_root="${H3_R2V_STATE_ROOT:-/tmp/minimax-h3-r2v-runtime}"
output_root="${H3_R2V_OUTPUT_DIR:-/ai/artifacts/runs/minimax-h3-r2v-experiment}"
mkdir -p "$state_root/matched-user" "$output_root/matched"

"$H3_ENV/bin/python" - <<'PY'
import torch
expected = [("AMD Radeon AI PRO R9700", "gfx1201"), ("Radeon RX 7900 XT", "gfx1100")]
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
for index, (name, arch) in enumerate(expected):
    props = torch.cuda.get_device_properties(index)
    actual_arch = props.gcnArchName.split(":", 1)[0]
    print(f"matched preflight cuda:{index}={torch.cuda.get_device_name(index)}/{actual_arch}", flush=True)
    assert torch.cuda.get_device_name(index) == name
    assert actual_arch == arch
assert torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)
PY

cd "$COMFYUI_PATH"
exec "$H3_ENV/bin/python" "$COMFYUI_PATH/main.py" \
  --extra-model-paths-config "$experiment_root/runtime/matched-extra-paths.yaml" \
  --input-directory /ai/comfyui/input \
  --user-directory "$state_root/matched-user" \
  --database-url "sqlite:///$state_root/matched-user/comfyui.db" \
  --output-directory "$output_root/matched" \
  --disable-dynamic-vram \
  --disable-smart-memory \
  --disable-mmap \
  --bf16-vae \
  --reserve-vram 2 \
  --listen 127.0.0.1 \
  --port 8192 \
  --disable-auto-launch
