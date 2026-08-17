#!/usr/bin/env bash
# Identify a model file for a run record: size, SHA256, and GGUF metadata.
#
# Read-only. SHA256 of a 15-30 GB file takes a while; that is the point --
# the hash is what makes a run record reproducible.
#
# Usage:
#   02_hash_model.sh <model.gguf> [...]
#   NO_HASH=1 02_hash_model.sh <model.gguf>     # skip SHA256 (size/metadata only)

set -euo pipefail

if [[ $# -eq 0 ]]; then
    printf 'usage: %s <model.gguf> [...]\n' "${0##*/}" >&2
    exit 2
fi

# Read GGUF metadata if a reader is available. We never guess the architecture
# from the filename -- an absent reader is reported as absent.
dump_gguf_metadata() {
    local model="$1"
    if command -v gguf-dump >/dev/null 2>&1; then
        printf '  gguf_metadata (gguf-dump):\n'
        gguf-dump --no-tensors "$model" 2>/dev/null \
            | grep -iE 'general.architecture|general.file_type|general.name|quantization|tensor count|nextn' \
            | sed 's/^/    /' || printf '    (no matching fields)\n'
    elif python3 -c 'import gguf' 2>/dev/null; then
        printf '  gguf_metadata (python gguf):\n'
        python3 - "$model" <<'PY' 2>/dev/null | sed 's/^/    /' || printf '    (reader failed)\n'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for key in ("general.architecture", "general.name", "general.file_type"):
    f = r.get_field(key)
    if f is not None:
        print(f"{key}: {f.contents()}")
print(f"tensor_count: {len(r.tensors)}")
types = {}
for t in r.tensors:
    types[str(t.tensor_type.name)] = types.get(str(t.tensor_type.name), 0) + 1
for k in sorted(types):
    print(f"  type {k}: {types[k]}")
PY
    else
        printf '  gguf_metadata: NOT AVAILABLE (no gguf-dump, no python gguf module)\n'
    fi
}

for model in "$@"; do
    printf -- '--- %s ---\n' "$model"

    if [[ ! -f "$model" ]]; then
        printf '  ERROR: not a regular file\n'
        continue
    fi

    printf '  path:  %s\n' "$(readlink -f "$model")"
    printf '  bytes: %s\n' "$(stat -c %s "$model")"
    printf '  human: %s\n' "$(du -h "$model" | cut -f1)"
    printf '  mtime: %s\n' "$(date -u -r "$model" +%Y-%m-%dT%H:%M:%SZ)"

    if [[ "${NO_HASH:-0}" == "1" ]]; then
        printf '  sha256: SKIPPED (NO_HASH=1)\n'
    else
        printf '  sha256: %s\n' "$(sha256sum "$model" | cut -d' ' -f1)"
    fi

    dump_gguf_metadata "$model"
done
