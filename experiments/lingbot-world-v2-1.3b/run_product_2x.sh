#!/usr/bin/env bash
# LingBot World v2 1.3B — validated product preset (R9700 / gfx1201).
#
# The world model simulates at 368x672 and asynchronously presents at
# 736x1344. Everything in run_product.sh, plus presentation-only 2x Lanczos:
#
#   local_attn_size=12, sink_size=6, 3-step-A (999 -> 908 -> 768),
#   TAEW2.1 decode-first, exact clean-KV commit, timecond memoization,
#   --upscale2 lanczos (async CPU worker; terminal output only).
#
# Measured saturated behavior (seed 42, 31-action matched runs):
#   denoise ~386 ms | first native RGB ~391 ms | first 2x RGB ~456 ms |
#   clean-KV ~127 ms | next-action-ready ~524 ms | VRAM ~6.05 GiB.
# The 2x stage does not move next-action-ready (async, CPU-only, zero GPU
# contention); native PNGs are byte-identical with and without it.
#
# Reference (no upscale, same world state): ./run_product.sh
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export HIP_VISIBLE_DEVICES=1
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1

IMAGE=${IMAGE:-/ai/repos/lingbot-world-v2/examples/03/image.jpg}
PROMPT=${PROMPT:-A serene lakeside scene with a lone tree standing in calm water...}
OUT=${OUT:-/ai/outputs/lingbot-session}

exec /ai/envs/lingbot-world-v2/bin/python interactive_world.py \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --output "$OUT" \
  --size 384*672 \
  --local_attn_size 12 \
  --sink_size 6 \
  --presentation_decoder taew2_1 \
  --schedule decode-first \
  --denoise_schedule 3-step-A \
  --timecond_cache \
  --upscale2 lanczos \
  "$@"
