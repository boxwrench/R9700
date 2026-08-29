#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "$0")" && pwd)"

printf 'Experiment complete. No render is queued by this script.\n'
printf 'Summary: %s/RESULTS.md\n' "$root"
printf 'Exact data: %s/tables/results.csv\n' "$root"
printf 'Raw logs: %s/logs/\n' "$root"
printf 'Reproduction harness help:\n'
python3 "$root/harness.py" --help
