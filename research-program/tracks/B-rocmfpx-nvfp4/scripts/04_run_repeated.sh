#!/usr/bin/env bash
# Run a benchmark command N times, saving each repetition's stdout and stderr
# to separate files.
#
# This wrapper deliberately does NOT parse, average, or interpret output. It
# preserves raw logs and records exit codes and wall-clock durations. Metric
# extraction is 05_extract_basic_metrics.py's job; aggregation is
# 06_compare_runs.py's job.
#
# Usage:
#   04_run_repeated.sh --out DIR [--reps N] [--warmup N] [--label NAME] -- <command> [args...]
#
# Defaults: --reps 5 (the benchmark contract's minimum for production-style
# comparisons), --warmup 1.

set -euo pipefail

OUT_DIR=""
REPS=5
WARMUP=1
LABEL="run"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)    OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
        --reps)   REPS="${2:?--reps needs a number}"; shift 2 ;;
        --warmup) WARMUP="${2:?--warmup needs a number}"; shift 2 ;;
        --label)  LABEL="${2:?--label needs a name}"; shift 2 ;;
        --)       shift; break ;;
        *)        printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT_DIR" || $# -eq 0 ]]; then
    printf 'usage: %s --out DIR [--reps N] [--warmup N] [--label NAME] -- <command> [args...]\n' "${0##*/}" >&2
    exit 2
fi

[[ "$REPS"   =~ ^[0-9]+$ ]] || { printf 'error: --reps must be an integer\n' >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { printf 'error: --warmup must be an integer\n' >&2; exit 2; }

mkdir -p "$OUT_DIR"

# Refuse to overwrite an existing result set. Raw logs are evidence.
if compgen -G "$OUT_DIR/${LABEL}.rep*.stdout" > /dev/null; then
    printf 'error: %s/%s.rep*.stdout already exists; choose another --label or --out\n' "$OUT_DIR" "$LABEL" >&2
    exit 3
fi

MANIFEST="$OUT_DIR/${LABEL}.manifest.txt"
{
    printf 'label:    %s\n' "$LABEL"
    printf 'started:  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'reps:     %s\n' "$REPS"
    printf 'warmup:   %s\n' "$WARMUP"
    printf 'command: '
    printf ' %q' "$@"
    printf '\n'
} > "$MANIFEST"

run_once() {
    local tag="$1"; shift
    local out="$OUT_DIR/${LABEL}.${tag}.stdout"
    local err="$OUT_DIR/${LABEL}.${tag}.stderr"
    local start end rc
    start=$(date +%s.%N)
    set +e
    "$@" > "$out" 2> "$err"
    rc=$?
    set -e
    end=$(date +%s.%N)
    local dur
    dur=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')
    printf '%s rc=%s wall_seconds=%s\n' "$tag" "$rc" "$dur" | tee -a "$MANIFEST"
    return 0
}

for ((i = 1; i <= WARMUP; i++)); do
    printf 'warmup %d/%d\n' "$i" "$WARMUP" >&2
    run_once "warmup${i}" "$@"
done

failures=0
for ((i = 1; i <= REPS; i++)); do
    printf 'rep %d/%d\n' "$i" "$REPS" >&2
    run_once "rep${i}" "$@"
    if ! grep -q "^rep${i} rc=0 " "$MANIFEST"; then
        failures=$((failures + 1))
    fi
done

{
    printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'failed_reps: %s\n' "$failures"
} >> "$MANIFEST"

printf '\nraw logs: %s\n' "$OUT_DIR"
printf 'manifest: %s\n' "$MANIFEST"

if [[ $failures -gt 0 ]]; then
    printf 'WARNING: %d of %d repetitions exited non-zero. Do not aggregate these\n' "$failures" "$REPS" >&2
    printf 'results without accounting for the failures.\n' >&2
    exit 4
fi
