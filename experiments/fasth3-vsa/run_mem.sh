#!/usr/bin/env bash
# One warm run with torch-level peak memory captured around it.
# usage: run_mem.sh <label> <workflow.json> [seed]
set -uo pipefail
EXP=/ai/lab/experiments/fasth3-vsa
PY=/ai/environments/fasth3-vsa/bin/python
LOG="$EXP/logs/server-mem.log"
label="$1"; wf="$2"; seed="${3:-}"

if [ -n "$seed" ]; then
  tmp="$EXP/workflows/mem/$(basename "${wf%.json}")-s$seed.json"
  mkdir -p "$EXP/workflows/mem"
  "$PY" - "$wf" "$tmp" "$seed" <<'PY'
import json,sys
b,o,s=sys.argv[1],sys.argv[2],int(sys.argv[3])
d=json.load(open(b))
for nid,n in d.items():
    if n.get('class_type')=='RandomNoise': n['inputs']['noise_seed']=s
    if n.get('class_type')=='SaveVideo':   n['inputs']['filename_prefix']=f"fasth3/mem/{s}"
json.dump(d,open(o,'w'),indent=2)
PY
  wf="$tmp"
fi

mark=$(wc -l < "$LOG" 2>/dev/null || echo 0)
curl -s http://127.0.0.1:8191/memprobe/reset >/dev/null

wall=$(LD_LIBRARY_PATH=/opt/rocm/lib "$PY" "$EXP/run_workflow.py" "$wf" --port 8191 \
        --server-log "$LOG" 2>&1 | grep -oE "execution_start->success: [0-9.]+" | grep -oE "[0-9.]+$")

read -r maxa maxr < <(curl -s http://127.0.0.1:8191/memprobe | "$PY" -c \
  "import json,sys;d=json.load(sys.stdin);print(d['max_allocated'],d['max_reserved'])")

new=$(tail -n +$((mark+1)) "$LOG")
samp=$(grep -oE "4/4 \[[0-9]{2}:[0-9]{2}" <<<"$new" | tail -1 | grep -oE "[0-9]{2}:[0-9]{2}" | awk -F: '{print $1*60+$2}')
sit=$(grep -oE "4/4 \[[^]]*\]" <<<"$new" | tail -1 | grep -oE "[0-9.]+s/it" | tr -d 's/it')
vsa=$(grep -oE "topk=[0-9]+/[0-9]+" <<<"$new" | tail -1)
fb=$(grep -ci "dense fallback\|not a MiniMaxH3Model" <<<"$new")

awk -v l="$label" -v sa="${samp:--}" -v si="${sit:--}" -v w="${wall:--}" \
    -v a="$maxa" -v r="$maxr" -v v="${vsa:-NONE}" -v f="$fb" 'BEGIN{
  printf "%-22s sampling=%ss  s/fwd=%s  wall=%ss  max_alloc=%.3f GiB  max_reserved=%.3f GiB  vsa=%s  fallbacks=%s\n",
         l, sa, si, w, a/1073741824, r/1073741824, v, f }'
