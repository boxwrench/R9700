#!/usr/bin/env bash
# LingBot World v2 1.3B — product-path launcher (R9700 / gfx1201).
# Fast validated stack: 384x672 target (368x672 actual, 966 tok/frame),
# 12-frame window (6 sink), TAEW2.1 presentation, decode-first,
# 3-step-A denoise, exact clean-KV, memoized time conditioning.
# Reference behavior stays available in interactive_world.py defaults
# (4-step, canonical, clean-first, 18-frame window).
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
  "$@"
