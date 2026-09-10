#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export HIP_VISIBLE_DEVICES=1
PY=/ai/envs/lingbot-world-v2/bin/python
L=/ai/github/R9700/experiments/lingbot-world-v2-14b-gguf/logs
run () { # $1 split  $2 extra-args  $3 tag
  $PY /ai/lab/lingbot-gguf-r9700/gguf_bench.py \
     --resolution "480x832 (needs tiny window)" --frame_num 29 --chunk_size 3 \
     --local_attn_size 6 --sink_size 2 --shift 3.0 --seed 42 \
     --warm_repeats 1 --vae_split $1 $2 --tag "$3" \
     --out_json $L/$3.json > $L/$3.txt 2>&1
  echo "=== $3 rc=$? ==="
}
run 1 "--profile_dequant" q4km-w6s2-vaesplit1
run 0 ""                  q4km-w6s2-vaesplit0
