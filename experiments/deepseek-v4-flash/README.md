# DeepSeek V4 Flash serving experiment

This compact record reproduces the parameter matrix described in
[`docs/deepseek-v4-flash-inference.md`](../../docs/deepseek-v4-flash-inference.md).
It intentionally excludes model weights, local absolute paths, API keys, and
large raw logs.

## Selected environment

```bash
export GGML_VK_VISIBLE_DEVICES=1
export CTX_SIZE=32768
export CPU_MOE_LAYERS=41
export CACHE_TYPE_K=q8_0
export CACHE_TYPE_V=q8_0
export DRAFT_CACHE_TYPE_K=q8_0
export DRAFT_CACHE_TYPE_V=q8_0
export KV_OFFLOAD=1
export USE_DSPARK=1
export DSPARK_TOKENS=3
export PARALLEL=1
export THREADS=16
```

`GGML_VK_VISIBLE_DEVICES=1` refers to this workstation's host enumeration. Always
run `llama-server --list-devices` before applying it elsewhere.

## Matrix

[`profiles.tsv`](profiles.tsv) is the ordered screening matrix. Keep the target
quant and drafter fixed while varying one resource decision at a time. Matching
K/V cache types are required for this DeepSeek V4 build.

The campaign used two result roots:

- Screening: short quality and 128-token speed probes for every loadable profile
- Final: long-context retrieval plus two 256-token speed probes for the winner

Each profile should own a status file and immutable outputs. Mark it complete
only after all requested probes are saved. On restart, skip complete profiles
and retry `running`, `load_failed`, or `probe_failed` profiles.

## Recovery contract

The runner used this recovery order after every attempt:

1. Read the saved PID.
2. Verify `/proc/<pid>/cmdline` contains both `llama-server` and the expected
   DeepSeek model name.
3. Send TERM and wait 15 seconds.
4. Send KILL only to that validated PID if it remains alive.
5. Check R9700 VRAM with `rocm-smi -d 1 --showmeminfo vram`.
6. Consider a scoped `sudo -n rocm-smi -d 1 --gpureset` only if allocations
   remain, the card is confirmed as the R9700, and no process holds its render
   node.

Automatic reset should remain off by default. A machine reboot followed by a
resumable run is safer than resetting a GPU with unknown users.

No GPU reset was required for the recorded campaign. llama.cpp load failures
released VRAM normally and the runner continued to the next profile.

## Planned Lucebox host-RAM track

[`lucebox-host-ram.md`](lucebox-host-ram.md) records a separate, disabled
campaign for adapting Lucebox's fused DeepSeek ROCmFPX and DSpark path from
Strix Halo unified memory to the standalone R9700 plus system DDR5. Its matrix
is [`lucebox-host-ram-profiles.tsv`](lucebox-host-ram-profiles.tsv).

This is deliberately not mixed into `profiles.tsv`: it changes the inference
engine and uses a kernel-coupled quant format. It must first prove intentional
host-expert placement rather than ROCm page migration, then pass an exact
six-expert 32K quality/retrieval baseline before sparse prefill or four-expert
approximate execution can be considered.

## Primary agent implementation

DeepSeek Harness is the primary user-facing and agent layer for every promoted
backend. It connects to llama.cpp or Lucebox through a loopback custom
OpenAI-compatible provider. Hermes is retained only as a migration/control
client.

Raw backend probes continue to call the HTTP API directly so Harness prompts,
tools and retries do not contaminate tok/s, memory, retrieval, or DSpark
measurements. Backend finalists then run through the same pinned DeepSeek
Harness headless task suite. See [`deepseek-harness.md`](deepseek-harness.md),
[`deepseek-harness-settings.example.yaml`](deepseek-harness-settings.example.yaml),
and [`agent-harness-profiles.tsv`](agent-harness-profiles.tsv).

## RDNA HIP optimization tracks

[`optimization-plan.md`](optimization-plan.md) now treats the experimental
[`stew675/llama.cpp` `rdna-boosts`](https://github.com/stew675/llama.cpp/tree/rdna-boosts)
branch as a pinned DeepSeek backend candidate. Its phase-4 ladder separates the
old Vulkan control, matched upstream HIP, fork HIP, and—only when needed—a
matched fork Vulkan build. This prevents backend, upstream, and patch-series
changes from being credited to one another.

[`rdna-boosts-experiments.md`](rdna-boosts-experiments.md) is the separate
hardware-wide catalog. It groups reusable R9700/`gfx1201` experiments by Flash
Attention, BF16 KV, quantized decode, MoE, SSM, IMRoPE, and graph overhead, then
defines a distinct RX 7900 XT/`gfx1100` track. The two-GPU speculative-draft
idea remains isolated until both cards pass independent HIP correctness and
performance gates.
