#!/usr/bin/env bash
# Repetition protocol: >=3 warm runs per lane at 864x480/124f.
# Seeds vary per repetition because ComfyUI serves an identical graph from cache
# in ~1 ms; a repeated identical run measures nothing.
set -uo pipefail
EXP=/ai/lab/experiments/fasth3-vsa
PY=/ai/environments/fasth3-vsa/bin/python
LOG="$EXP/logs/server-reps.log"
RES="$EXP/logs/rep-results.tsv"
SEEDS=(8112026 8112027 8112028)

printf 'lane\trep\tseed\tsampling_s\ts_per_it\twall_s\tpeak_vram_gib\n' > "$RES"

measure () {  # $1=lane  $2=rep  $3=seed  $4=workflow
  local lane="$1" rep="$2" seed="$3" wf="$4"
  local vram="$EXP/logs/vram-rep.txt"
  local mark; mark=$(wc -l < "$LOG")
  ( while true; do rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\]/&&/Used/{print $NF}'; sleep 1; done ) > "$vram" 2>&1 &
  local s=$!
  local wall
  wall=$(LD_LIBRARY_PATH=/opt/rocm/lib "$PY" "$EXP/run_workflow.py" "$wf" --port 8191 \
          --server-log "$LOG" 2>&1 | grep -oE "execution_start->success: [0-9.]+" | grep -oE "[0-9.]+$")
  kill $s 2>/dev/null; wait $s 2>/dev/null
  local peak; peak=$(awk -v b="$(sort -n "$vram" | tail -1)" 'BEGIN{printf "%.2f", b/1073741824}')
  local new; new=$(tail -n +$((mark+1)) "$LOG")
  local samp sit
  samp=$(grep -oE "4/4 \[[0-9]{2}:[0-9]{2}" <<<"$new" | tail -1 | grep -oE "[0-9]{2}:[0-9]{2}" | awk -F: '{print $1*60+$2}')
  sit=$(grep -oE "4/4 \[[^]]*\]" <<<"$new" | tail -1 | grep -oE "[0-9.]+s/it" | tr -d 's/it')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$rep" "$seed" "${samp:--}" "${sit:--}" "${wall:--}" "$peak" | tee -a "$RES"
}

lane_runs () {  # $1=lane  $2=base workflow  $3=output prefix
  local lane="$1" base="$2" pfx="$3"
  echo "===== lane: $lane ====="
  local i=0
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    local wf="$EXP/workflows/reps/${lane}-s${seed}.json"
    "$PY" - "$base" "$wf" "$seed" "$pfx" <<'PY'
import json,sys
base,out,seed,pfx=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
d=json.load(open(base)); d['6']['inputs']['noise_seed']=seed
d['14']['inputs']['filename_prefix']=f'{pfx}-s{seed}'
json.dump(d,open(out,'w'),indent=2)
PY
    # first iteration doubles as the lane warm-up (model swap); discard it
    if [ "$i" -eq 1 ]; then
      echo "-- warm-up (model load, discarded) --"
      LD_LIBRARY_PATH=/opt/rocm/lib "$PY" "$EXP/run_workflow.py" "$wf" --port 8191 --server-log "$LOG" >/dev/null 2>&1
      # rerun this seed's graph is cached now, so shift: measure the remaining seeds plus one extra
    fi
  done
  # measured warm reps: three distinct, uncached seeds
  local r=0
  for seed in 8112031 8112032 8112033; do
    r=$((r+1))
    local wf="$EXP/workflows/reps/${lane}-s${seed}.json"
    "$PY" - "$base" "$wf" "$seed" "$pfx" <<'PY'
import json,sys
base,out,seed,pfx=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
d=json.load(open(base)); d['6']['inputs']['noise_seed']=seed
d['14']['inputs']['filename_prefix']=f'{pfx}-s{seed}'
json.dump(d,open(out,'w'),indent=2)
PY
    measure "$lane" "$r" "$seed" "$wf"
  done
}

mkdir -p "$EXP/workflows/reps"
lane_runs vsa010 "$EXP/workflows/sweep/vsa-topk0.10.json" "fasth3/reps/vsa010"
lane_runs dense  "$EXP/workflows/sweep/dense.json"        "fasth3/reps/dense"
lane_runs turbov4 "$EXP/workflows/turbov4-864x480-124f.json" "fasth3/reps/turbov4"

echo; echo "raw -> $RES"; column -t "$RES"
