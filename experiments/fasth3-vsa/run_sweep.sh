#!/usr/bin/env bash
# VSA topk_ratio sweep at 864x480/124f. Everything constant except topk_ratio.
# Keeps the FastH3 dense lane separate from VSA@1.00 on purpose: they are
# different code paths, not the same measurement.
set -uo pipefail
EXP=/ai/lab/experiments/fasth3-vsa
PY=/ai/environments/fasth3-vsa/bin/python
LOG="$EXP/logs/server-sweep.log"
RES="$EXP/logs/sweep-results.tsv"

printf 'lane\ttopk_ratio\tblocks_sel\tblocks_tot\tsel_pct\ttokens\tsampling_s\ts_per_it\twall_s\tpeak_vram_gib\n' > "$RES"

run_one () {   # $1=lane label  $2=workflow  $3=topk (or "-")
  local lane="$1" wf="$2" topk="$3"
  local vram="$EXP/logs/vram-$lane.txt"
  local mark; mark=$(wc -l < "$LOG")

  ( while true; do rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\]/&&/Used/{print $NF}'; sleep 1; done ) > "$vram" 2>&1 &
  local sampler=$!

  local wall
  wall=$(LD_LIBRARY_PATH=/opt/rocm/lib "$PY" "$EXP/run_workflow.py" "$wf" \
          --port 8191 --server-log "$LOG" 2>&1 | grep -oE "execution_start->success: [0-9.]+" | grep -oE "[0-9.]+$")
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

  local peak; peak=$(sort -n "$vram" | tail -1)
  peak=$(awk -v b="$peak" 'BEGIN{printf "%.2f", b/1073741824}')

  local new; new=$(tail -n +$((mark+1)) "$LOG")
  local sel tot toks samp sit
  sel=$(grep -oE "topk=[0-9]+/[0-9]+" <<<"$new" | tail -1 | cut -d= -f2 | cut -d/ -f1)
  tot=$(grep -oE "topk=[0-9]+/[0-9]+" <<<"$new" | tail -1 | cut -d/ -f2)
  toks=$(grep -oE "tokens=[0-9]+" <<<"$new" | tail -1 | cut -d= -f2)
  samp=$(grep -oE "4/4 \[[0-9]{2}:[0-9]{2}" <<<"$new" | tail -1 | grep -oE "[0-9]{2}:[0-9]{2}" | awk -F: '{print $1*60+$2}')
  sit=$(grep -oE "4/4 \[[^]]*\]" <<<"$new" | tail -1 | grep -oE "[0-9.]+s/it" | tr -d 's/it')

  local pct="-"
  [ -n "${sel:-}" ] && [ -n "${tot:-}" ] && pct=$(awk -v s="$sel" -v t="$tot" 'BEGIN{printf "%.1f", 100*s/t}')

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lane" "$topk" "${sel:--}" "${tot:--}" "$pct" "${toks:--}" \
    "${samp:--}" "${sit:--}" "${wall:--}" "$peak" | tee -a "$RES"
}

# warm the server first; this run is discarded
echo "== warm-up (discarded) =="
LD_LIBRARY_PATH=/opt/rocm/lib "$PY" "$EXP/run_workflow.py" \
  "$EXP/workflows/sweep/vsa-topk0.10.json" --port 8191 --server-log "$LOG" >/dev/null 2>&1

echo "== sweep =="
for t in 0.10 0.20 0.30 0.50 1.00; do
  run_one "vsa$t" "$EXP/workflows/sweep/vsa-topk$t.json" "$t"
done
run_one "dense" "$EXP/workflows/sweep/dense.json" "-"

echo; echo "results -> $RES"
column -t "$RES"
