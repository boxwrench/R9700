#!/usr/bin/env bash
# Capture the host, GPU, and toolchain state for a Track B run record.
#
# Read-only. Emits a plain-text report on stdout. Every probe is optional:
# a missing tool is reported as "NOT AVAILABLE", never silently skipped and
# never substituted with a guess.
#
# Usage:
#   01_snapshot_environment.sh [repo_path...]
#
# Any repo paths given are reported with remote, branch, SHA, and dirty state.

set -euo pipefail

section() { printf '\n===== %s =====\n' "$1"; }

# Run a command if it exists; otherwise say so. Never fails the script.
probe() {
    local label="$1"; shift
    if command -v "$1" >/dev/null 2>&1; then
        printf '%s:\n' "$label"
        "$@" 2>&1 | sed 's/^/  /' || printf '  (command failed, rc=%s)\n' "$?"
    else
        printf '%s: NOT AVAILABLE (%s not on PATH)\n' "$label" "$1"
    fi
}

printf 'snapshot_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'hostname: %s\n' "$(hostname)"

section "KERNEL / OS"
uname -a
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    printf 'distro: %s\n' "${PRETTY_NAME:-unknown}"
else
    printf 'distro: NOT AVAILABLE (/etc/os-release unreadable)\n'
fi
printf 'glibc: %s\n' "$(ldd --version 2>/dev/null | head -1 || echo 'NOT AVAILABLE')"

section "CPU"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E '^(Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|CPU max MHz)' || true
else
    printf 'NOT AVAILABLE (lscpu)\n'
fi

section "MEMORY"
free -h 2>/dev/null || printf 'NOT AVAILABLE (free)\n'

section "GPU — PCI enumeration"
if command -v lspci >/dev/null 2>&1; then
    lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' || printf '  no display devices matched\n'
else
    printf 'NOT AVAILABLE (lspci)\n'
fi

section "GPU — ROCm / HIP"
probe "hipconfig --version" hipconfig --version
probe "hipcc --version"     hipcc --version
if [[ -r /opt/rocm/.info/version ]]; then
    printf 'rocm_version_file: %s\n' "$(cat /opt/rocm/.info/version)"
else
    printf 'rocm_version_file: NOT AVAILABLE\n'
fi
printf 'rocm_install_dirs: %s\n' "$(find /opt -maxdepth 1 -name 'rocm*' 2>/dev/null | sort | tr '\n' ' ' || echo none)"

# rocminfo is verbose; keep only the agent identity lines that matter.
if command -v rocminfo >/dev/null 2>&1; then
    printf 'rocminfo (agent names / gfx targets):\n'
    rocminfo 2>/dev/null | grep -E 'Name:|Marketing Name:|gfx[0-9]+' | sed 's/^/  /' || true
else
    printf 'rocminfo: NOT AVAILABLE\n'
fi

section "GPU — Vulkan / RADV"
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null \
        | grep -E 'deviceName|driverName|driverInfo|apiVersion' \
        | sed 's/^/  /' || true
else
    printf 'vulkaninfo: NOT AVAILABLE\n'
fi

printf '\nNOTE: device ORDER above is the enumeration order. On this class of host the\n'
printf 'R9700 is not index 0. Record the index that actually resolved to gfx1201 in\n'
printf 'the run record (field: gpu_index) and pin the device explicitly at run time.\n'

section "REPOSITORY STATE"
if [[ $# -eq 0 ]]; then
    printf 'no repository paths supplied\n'
fi
for repo in "$@"; do
    printf -- '--- %s ---\n' "$repo"
    if [[ ! -d "$repo/.git" ]]; then
        printf '  NOT A GIT REPOSITORY\n'
        continue
    fi
    printf '  remote: %s\n' "$(git -C "$repo" remote get-url origin 2>/dev/null || echo 'none')"
    printf '  branch: %s\n' "$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    printf '  sha:    %s\n' "$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo 'unknown')"
    printf '  shallow: %s\n' "$(git -C "$repo" rev-parse --is-shallow-repository 2>/dev/null || echo 'unknown')"
    local_dirty="$(git -C "$repo" status --porcelain 2>/dev/null || true)"
    if [[ -n "$local_dirty" ]]; then
        printf '  dirty:  YES\n'
        printf '%s\n' "$local_dirty" | sed 's/^/    /'
    else
        printf '  dirty:  no\n'
    fi
done

section "END"
printf 'snapshot complete\n'
