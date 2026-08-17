#!/usr/bin/env bash
# B1 baseline driver: four arms x (1 warmup + N measured) repetitions.
#
# Runs MIXED/UNIFORM x SERIAL/MTP on a single pinned device with one fixed
# generation configuration. Performs no parsing and computes no statistics --
# it only invokes llama-cli and preserves raw logs. Parsing is 05/b1_parse.py.
set -euo pipefail

BUILD="${BUILD:-/ai/scratch/ROCmFPX-audit/build-vulkan}"
BIN="$BUILD/bin/llama-cli"
MODEL_DIR="${MODEL_DIR:-/ai/models/Qwen3.8-27B-NVFP4-GGUF}"
MIXED="$MODEL_DIR/Qwen3.8-27B-NVFP4.gguf"
UNIFORM="$MODEL_DIR/Qwen3.8-27B-NVFP4-uniform.gguf"
OUT="${OUT:?set OUT to the raw output directory}"
REPS="${REPS:-7}"
WARMUPS="${WARMUPS:-1}"
DEV="${DEV:-Vulkan1}"

# Fixed generation configuration, identical across all four arms.
N_PREDICT="${N_PREDICT:-128}"
CTX="${CTX:-4096}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-512}"
SEED="${SEED:-1234}"
PROMPT="${PROMPT:-Explain why quantization speeds up large language model inference.}"

# Upstream reference MTP settings, taken verbatim from
# scripts/check-rocmfp4-qwen-mtp-regression.sh. Not tuned in B1.
SPEC_N_MAX="${SPEC_N_MAX:-4}"
SPEC_N_MIN="${SPEC_N_MIN:-0}"
SPEC_P_MIN="${SPEC_P_MIN:-0.0}"
SPEC_P_SPLIT="${SPEC_P_SPLIT:-0.10}"

if [[ ! -x "$BIN" ]]; then echo "missing binary: $BIN" >&2; exit 1; fi
for m in "$MIXED" "$UNIFORM"; do
    [[ -f "$m" ]] || { echo "missing model: $m" >&2; exit 1; }
done
if [[ -e "$OUT" ]]; then echo "refusing to overwrite existing results: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

common_args() {
    printf '%s\n' \
        -m "$1" --device "$DEV" -ngl 99 --no-mmap \
        -c "$CTX" -b "$BATCH" -ub "$UBATCH" -fa on -ctk q4_0 -ctv q4_0 \
        --temp 0 --seed "$SEED" -n "$N_PREDICT" --ignore-eos \
        -no-cnv -st --no-display-prompt --simple-io --verbosity 3 \
        -p "$PROMPT"
}

mtp_args() {
    printf '%s\n' \
        --spec-type draft-mtp --spec-draft-device "$DEV" --spec-draft-ngl all \
        --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-n-min "$SPEC_N_MIN" \
        --spec-draft-p-min "$SPEC_P_MIN" --spec-draft-p-split "$SPEC_P_SPLIT" \
        --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
}

run_arm() {
    local label="$1" model="$2" mode="$3"
    local -a args
    mapfile -t args < <(common_args "$model")
    if [[ "$mode" == "mtp" ]]; then
        local -a extra
        mapfile -t extra < <(mtp_args)
        args+=("${extra[@]}")
    fi

    local total=$((WARMUPS + REPS)) i tag start end rc
    for ((i = 1; i <= total; i++)); do
        if ((i <= WARMUPS)); then tag="warmup$i"; else tag="rep$((i - WARMUPS))"; fi
        echo "[$(date -Is)] $label $tag" >&2
        start=$(date +%s.%N)
        set +e
        timeout 1200 "$BIN" "${args[@]}" > "$OUT/$label.$tag.log" 2>&1
        rc=$?
        set -e
        end=$(date +%s.%N)
        printf '%s\t%s\t%s\t%s\n' "$label" "$tag" "$rc" \
            "$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')" \
            >> "$OUT/manifest.tsv"
        if ((rc != 0)); then echo "  WARNING: rc=$rc" >&2; fi
    done
}

printf 'arm\trep\trc\twall_s\n' > "$OUT/manifest.tsv"
{
    echo "bin=$BIN"
    echo "sha=$(git -C "$(dirname "$BUILD")" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
    echo "device=$DEV n_predict=$N_PREDICT ctx=$CTX batch=$BATCH ubatch=$UBATCH seed=$SEED"
    echo "spec: n_max=$SPEC_N_MAX n_min=$SPEC_N_MIN p_min=$SPEC_P_MIN p_split=$SPEC_P_SPLIT"
    echo "prompt=$PROMPT"
    echo "reps=$REPS warmups=$WARMUPS"
} > "$OUT/config.txt"

run_arm mixed-serial   "$MIXED"   serial
run_arm mixed-mtp      "$MIXED"   mtp
run_arm uniform-serial "$UNIFORM" serial
run_arm uniform-mtp    "$UNIFORM" mtp

echo "done -> $OUT" >&2
