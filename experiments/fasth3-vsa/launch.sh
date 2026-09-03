#!/usr/bin/env bash
# Isolated FastH3/VSA ComfyUI. Port 8191 so it never collides with the
# production comfyui-h3 service on 8190. Flags mirror the R9700 golden
# launcher (--disable-mmap is load-bearing on gfx1201).
set -euo pipefail

EXP=/ai/lab/experiments/fasth3-vsa
VENV=/ai/environments/fasth3-vsa

export LD_LIBRARY_PATH=/opt/rocm/lib
export HIP_VISIBLE_DEVICES=1            # R9700 / gfx1201 only
export HF_HOME=/ai/huggingface
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/ai/cache/miopen
export TORCH_HOME=/ai/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/ai/cache/torchinductor
export TRITON_CACHE_DIR=/ai/cache/triton
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

mkdir -p "$EXP/outputs" "$EXP/logs"

exec "$VENV/bin/python" "$EXP/ComfyUI/main.py" \
  --output-directory "$EXP/outputs" \
  --disable-dynamic-vram --disable-smart-memory --disable-mmap \
  --bf16-vae --reserve-vram 2 \
  --listen 127.0.0.1 --port 8191 --disable-auto-launch \
  "$@"
