#!/usr/bin/env bash
# Record exactly what is about to be run, as a JSON object on stdout.
#
# This does NOT execute the command. It captures the invocation so that a run
# record can be built before, not after, the fact.
#
# Usage:
#   03_record_command.sh --repo /ai/scratch/ROCmFPX-audit -- llama-bench -m model.gguf
#
# Environment variables are captured from an explicit allow-list only. A run
# record that claims to list "the environment" while silently omitting a
# behaviour-changing variable is worse than one that lists nothing, so add new
# variables here deliberately as they are introduced.

set -euo pipefail

REPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="${2:?--repo needs a path}"; shift 2 ;;
        --)     shift; break ;;
        *)      printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    printf 'usage: %s [--repo PATH] -- <command> [args...]\n' "${0##*/}" >&2
    exit 2
fi

# Variables known to change behaviour or results. Extend deliberately.
TRACKED_VARS=(
    HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    GGML_VK_VISIBLE_DEVICES GGML_VK_PERF_LOGGER GGML_VK_PERF_LOGGER_FREQUENCY
    GGML_VK_PIPELINE_STATS GGML_VK_FORCE_MUL_MM GGML_VK_DISABLE_F16
    GGML_VK_IQ4XS_TINYN GGML_VK_IQ4XS_ROWS
    LLAMA_FORCE_N_RS_SEQ
    HSA_OVERRIDE_GFX_VERSION AMD_SERIALIZE_KERNEL
    OMP_NUM_THREADS
)

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

printf '{\n'
printf '  "timestamp": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  "hostname": "%s",\n' "$(json_escape "$(hostname)")"
printf '  "cwd": "%s",\n' "$(json_escape "$PWD")"

if [[ -n "$REPO" && -d "$REPO/.git" ]]; then
    printf '  "git_repo": "%s",\n'  "$(json_escape "$(git -C "$REPO" remote get-url origin 2>/dev/null || echo unknown)")"
    printf '  "git_sha": "%s",\n'   "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf '  "git_branch": "%s",\n' "$(json_escape "$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)")"
    if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null || true)" ]]; then
        printf '  "git_dirty": true,\n'
    else
        printf '  "git_dirty": false,\n'
    fi
else
    printf '  "git_repo": null,\n  "git_sha": null,\n  "git_branch": null,\n  "git_dirty": null,\n'
fi

# argv as a JSON array, not a shell string -- quoting is not reconstructable.
printf '  "command": ['
first=1
for arg in "$@"; do
    [[ $first -eq 1 ]] || printf ', '
    printf '"%s"' "$(json_escape "$arg")"
    first=0
done
printf '],\n'

printf '  "env": {'
first=1
for var in "${TRACKED_VARS[@]}"; do
    if [[ -n "${!var:-}" ]]; then
        [[ $first -eq 1 ]] || printf ', '
        printf '"%s": "%s"' "$var" "$(json_escape "${!var}")"
        first=0
    fi
done
printf '},\n'

printf '  "env_tracked_but_unset": ['
first=1
for var in "${TRACKED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        [[ $first -eq 1 ]] || printf ', '
        printf '"%s"' "$var"
        first=0
    fi
done
printf ']\n'
printf '}\n'
