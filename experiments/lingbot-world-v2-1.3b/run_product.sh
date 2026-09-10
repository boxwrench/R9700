#!/usr/bin/env bash
# LingBot World v2 1.3B — product-path launcher (R9700 / gfx1201).
# Accepted settings: 384x672, TAEW2.1 presentation, decode-first,
# 3-step-A denoise, exact clean-KV. Canonical/4-step stay available
# via the flags below.
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
  --size 384*688 \
  --presentation_decoder taew2_1 \
  --schedule decode-first \
  --denoise_schedule 3-step-A \
  "$@"
