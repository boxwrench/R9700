#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib HIP_VISIBLE_DEVICES=1 WAN_VAE_CONV3D_TEMPORAL_SPLIT=1
PY=/ai/envs/lingbot-world-v2/bin/python
IMG=/ai/repos/lingbot-world-v2/examples/03/image.jpg
P="A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds."
SEQ="forward forward turn_right forward turn_right turn_right forward"
for C in 3 2 1; do
  $PY interactive_world.py --image $IMG --prompt "$P" \
    --output /ai/outputs/lingbot-sessions/serial_c$C --chunk_size $C \
    --overlap 0 --script "$SEQ" > logs/serial-c$C.txt 2>&1
  echo "=== chunk $C rc=$? ==="
done
