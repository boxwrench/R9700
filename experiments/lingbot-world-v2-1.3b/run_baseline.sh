#!/bin/bash
# Reproduce the R9700 baseline from a cold shell.
set -eux
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=1            # device 0 is an RX 7900 XT
# WAN_VAE_CONV3D_TEMPORAL_SPLIT defaults to 1; set 0 for exact upstream behaviour
PY=/ai/envs/lingbot-world-v2/bin/python
CKPT=/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled
OUT=/ai/outputs/lingbot-world-v2-1.3b
HERE="$(cd "$(dirname "$0")" && pwd)"

$PY "$HERE/bench.py" --ckpt_dir "$CKPT" --task i2v-1.3B --size 480*832 \
    --frame_num "${1:-29}" --chunk_size 4 \
    --local_attn_size 18 --sink_size 6 --seed 42 \
    --offload_model 0 --warm_repeats "${2:-2}" \
    --tag "${3:-baseline}" --save_dir "$OUT" \
    --out_json "$HERE/logs/${3:-baseline}.json"
