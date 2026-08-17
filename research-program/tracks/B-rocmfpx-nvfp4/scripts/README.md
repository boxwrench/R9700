# Track B Reproducibility Harness

Small, defensive, dependency-free helpers for producing run records that satisfy
[`shared/benchmark-contract.md`](../../../shared/benchmark-contract.md).

Bash scripts use `set -euo pipefail` and pass `shellcheck` clean. Python scripts
use the standard library only.

**None of these run inference.** They capture state, wrap someone else's
command, or post-process logs.

| Script | Purpose |
|---|---|
| `01_snapshot_environment.sh` | Kernel, OS, CPU, RAM, GPU enumeration, ROCm/HIP, Vulkan/RADV/Mesa, and git state for any repos given as arguments. |
| `02_hash_model.sh` | Model size, SHA256, and GGUF metadata. `NO_HASH=1` skips the hash for a quick look. |
| `03_record_command.sh` | Emits a JSON record of a command *before* it runs: argv, tracked environment variables, git SHA and dirty state. |
| `04_run_repeated.sh` | Runs a command N times with warmups, saving each repetition's stdout and stderr separately plus a manifest of exit codes and wall times. |
| `05_extract_basic_metrics.py` | Parses metrics out of raw logs, attaching file, line number, and verbatim source line to every value. |
| `06_compare_runs.py` | Aggregates JSONL/CSV run records: n, mean, stdev, median, min, max, p10, p90, and paired deltas. |

## Design commitments

These exist to prevent specific mistakes, most of which this program has already
made once.

**Nothing is silently guessed.** A missing tool prints `NOT AVAILABLE`. A metric
that does not match a known pattern is reported as unmatched. An absent field is
`null`, never a default — the previous campaign lost time to a benchmark harness
whose parser matched the wrong thing and reported it confidently.

**Aggregation is separated from measurement.** `04` never parses; `05` never
averages; `06` never reads a raw log. A mean is a derived quantity and is not
stored as if it were measured.

**Raw logs are evidence.** `04` refuses to overwrite an existing result set and
keeps stdout and stderr apart.

**Statistics stop where the data stops.** `06` reports distributions, warns when
`n < 5`, and warns when a mean delta is smaller than its own standard deviation.
It computes no p-values: a t-test on five GPU benchmark runs would imply more
than the data supports.

**Environment capture is an explicit allow-list.** `03_record_command.sh` has a
`TRACKED_VARS` array covering device selection (`HIP_VISIBLE_DEVICES`,
`GGML_VK_VISIBLE_DEVICES`, …), the Track A probes (`GGML_VK_FORCE_MUL_MM`,
`GGML_VK_IQ4XS_TINYN`, `LLAMA_FORCE_N_RS_SEQ`, …), and ROCm overrides. It also
records which tracked variables were *unset*, so a run record distinguishes
"default" from "not checked". **Add new behaviour-changing variables to that
array as they are introduced** — the list is the contract.

## Device isolation

Every Track B run must pin the R9700 explicitly. It is **index 1**, not 0, under
both backends on this host.

```bash
# Vulkan
llama-bench --device Vulkan1 ...

# HIP - the R9700 becomes ROCm0 once visibility is restricted.
# Without this the HIP backend segfaults on this mixed-architecture host.
HIP_VISIBLE_DEVICES=1 llama-bench --device ROCm0 ...
```

## Example

```bash
./01_snapshot_environment.sh /ai/scratch/ROCmFPX-audit > snap.txt
./02_hash_model.sh /ai/models/<model>.gguf              > model.txt
./03_record_command.sh --repo /ai/scratch/ROCmFPX-audit -- \
    llama-bench --device Vulkan1 -m /ai/models/<model>.gguf   > cmd.json
./04_run_repeated.sh --out ../raw/b1-serial --reps 5 --warmup 1 --label serial -- \
    llama-bench --device Vulkan1 -m /ai/models/<model>.gguf
./05_extract_basic_metrics.py --json ../raw/b1-serial/serial.rep*.stdout > metrics.jsonl
./06_compare_runs.py results.jsonl --metric decode_tok_s --group-by experiment_id
```
