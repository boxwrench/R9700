#!/usr/bin/env bash
set -Eeuo pipefail

environment_file="/ai/lab/experiments/minimax-h3/dual-gpu/config/dual-gpu.env"
if [[ ! -r "$environment_file" ]]; then
    printf 'Missing dual-GPU environment: %s\n' "$environment_file" >&2
    exit 2
fi

set -a
# shellcheck source=/dev/null
source "$environment_file"
set +a

python_bin="${H3_ENV}/bin/python"
main_py="${COMFYUI_PATH}/main.py"
[[ -x "$python_bin" ]] || { printf 'Missing H3 Python: %s\n' "$python_bin" >&2; exit 2; }
[[ -f "$main_py" ]] || { printf 'Missing experimental ComfyUI: %s\n' "$main_py" >&2; exit 2; }
[[ -r "$H3_MODEL_PATH_CONFIG" ]] || { printf 'Missing model path config: %s\n' "$H3_MODEL_PATH_CONFIG" >&2; exit 2; }

"$python_bin" - <<'PY'
import sys
import torch

expected = [
    ("AMD Radeon AI PRO R9700", "gfx1201"),
    ("Radeon RX 7900 XT", "gfx1100"),
]

if torch.cuda.device_count() != 2:
    raise SystemExit(f"dual-GPU preflight: expected 2 devices, found {torch.cuda.device_count()}")

for index, (expected_name, expected_arch) in enumerate(expected):
    properties = torch.cuda.get_device_properties(index)
    actual_name = torch.cuda.get_device_name(index)
    actual_arch = properties.gcnArchName.split(":", 1)[0]
    print(
        f"dual-GPU preflight: cuda:{index} = {actual_name} / {actual_arch} / "
        f"{properties.total_memory / 1024**3:.2f} GiB",
        flush=True,
    )
    if actual_name != expected_name or actual_arch != expected_arch:
        raise SystemExit(
            f"dual-GPU preflight: cuda:{index} mismatch; "
            f"expected {expected_name}/{expected_arch}, got {actual_name}/{actual_arch}"
        )

if not torch.cuda.can_device_access_peer(0, 1) or not torch.cuda.can_device_access_peer(1, 0):
    raise SystemExit("dual-GPU preflight: bidirectional peer access is unavailable")

print("dual-GPU preflight: mapping and peer access passed", flush=True)
sys.exit(0)
PY

cd "$COMFYUI_PATH"
exec "$python_bin" "$main_py" \
    --extra-model-paths-config "$H3_MODEL_PATH_CONFIG" \
    --user-directory "$H3_USER_DIR" \
    --database-url "sqlite:///${H3_USER_DIR}/comfyui.db" \
    --output-directory "$H3_OUTPUT_DIR" \
    --disable-dynamic-vram \
    --bf16-vae \
    --reserve-vram 2 \
    "$@"
