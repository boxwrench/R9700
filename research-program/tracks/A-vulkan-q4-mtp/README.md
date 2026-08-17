# Track A — Vulkan / GGUF Q4 / Native MTP

## Status: PAUSED / PRESERVED

This track indexes and preserves the deeply characterized Vulkan/RADV Qwen3.8-27B research line.

---

## Foundation

* **Model**: `Qwen3.8-27B-UD-Q4_K_XL.gguf`
* **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB VRAM, 64 MB Infinity Cache, ~640 GB/s physical DRAM bandwidth)
* **Backend**: Vulkan / RADV (Mesa)
* **Speculative Decoding**: Native MTP ($n_{\text{max}}=2, p_{\text{min}}=0.3$)

### Historical Reference Performance
* **Serial Decode**: $\approx 29.4\text{ tok/s}$
* **Native MTP Decode**: $\approx 53.2\text{--}53.8\text{ tok/s}$
* **MTP Acceleration Multiplier**: $\approx 1.83\times$

*(Note: The exact, authoritative data points are recorded in the main experiment log.)*

---

## Primary Authoritative Artifacts

* **Main Experiment Log**: [`docs/qwen3-8-27b-experiment-log.md`](../../../docs/qwen3-8-27b-experiment-log.md) (Entries 1 through 17)
* **Proposer Forward Baseline Data**: [`data/experimental/qwen3-8-27b-mtp-proposer.tsv`](../../../data/experimental/qwen3-8-27b-mtp-proposer.tsv)
* **Round Cost Trace**: [`experiments/qwen3-8-27b/round_cost.jsonl`](../../../experiments/qwen3-8-27b/round_cost.jsonl)
* **Experiment Scripts**: [`experiments/qwen3-8-27b/`](../../../experiments/qwen3-8-27b/)

---

## Summary of Closed Branches

1. **Vulkan vs HIP Backend Sweep**: Vulkan/RADV confirmed superior on gfx1201 for this architecture; HIP path closed.
2. **KV Precision Sweep**: `f16` KV established as optimal balance of cache footprint and kernel efficiency; `q8_0` and `q4_0` closed.
3. **MTP Depth & Acceptance Sweep**: $n_{\text{max}}=2, p_{\text{min}}=0.3$ established as stable operating point; higher depths ($N \ge 3$) incur diminishing returns.
4. **Verification Micro-Optimization (Entry 16 — IQ4_XS Dequant Reuse)**: Reduced code size and duplicate dequantization, but yielded $<0.2\text{ ms}$ real wall-time gain. Formally closed.
5. **Verification Micro-Optimization (Entry 17 — Tiny-N MUL_MAT + ADD Fusion)**: Successfully eliminated 80 separate GPU dispatches ($-277\ \mu\text{s}$ elementwise time), but offset by $+219\ \mu\text{s}$ bias load overhead within `mul_mat_vec`, resulting in neutral wall-time impact. Formally closed.

---

## Current Stop Point

* **Accepted Log State**: Entries 1 through 18 are formally logged and locked.
* **Verification Exhaustion**: Verification-side GPU micro-optimizations on the Q4_K_XL trunk are largely exhausted ($41.26\text{ ms}$ verification floor).
* **Proposer Identification**: The $\approx 5.06\text{ ms}$ proposer forward was identified as the largest remaining optimization target, with the Q6_K LM-head matvec ($[248320, 5120]$, $1.04\text{ GB}$) accounting for $1,652\ \mu\text{s}$ per dispatch.
* **Pause Condition**: Broader multi-workload holdout validation and production acceptance were **NOT completed** before pausing to establish Track B.

---

## Experimental Work At Pause — Draft-Vocabulary Trimming

### Status: `CLOSED — NO WIN`

> [!WARNING]
> **This branch is closed.** The true unseen $n_{\text{max}}=2$ speculative
> holdout was run and **failed decisively** — speculation collapsed to zero
> accepted drafts on both trimmed arms and decode throughput roughly halved. See
> **Entry 18** in the
> [experiment log](../../../docs/qwen3-8-27b-experiment-log.md). Do not deploy
> 32K or 64K trimming.
>
> The passages below record what was believed **before** that holdout and are
> retained as history. The microbenchmark wins in "Established" are real; they
> simply did not survive end-to-end.

### Established

* Dedicated reduced draft heads function correctly at both $64\text{K}$ and $32\text{K}$ rows.
* The target model's `model.output` remains **untouched**.
* Major draft-head kernel reduction, directly measured:

  | Draft head | Per-call GPU kernel time |
  |---|---|
  | Full ($248{,}320$ rows, Q6_K) | $\approx 1{,}652\ \mu\text{s}$ |
  | $64\text{K}$ | $\approx 452\ \mu\text{s}$ |
  | $32\text{K}$ | $\approx 234\ \mu\text{s}$ |

* Scatter / `d2t` remap cost is small: $\approx 70\ \mu\text{s}$ per call.
* Isolated / single-step proposer timing improves strongly.
* An independent **non-multitoken** holdout showed roughly **+3–4%** direct throughput.
* ~~On the prompts tested, greedy committed output remained **identical**.~~
  **Contradicted by Entry 18:** on 16 unseen holdout prompts the trimmed arms
  matched the full-vocabulary output **0 / 16** times, because speculation had
  collapsed entirely.

### Unresolved

* ~~**No true unseen $n_{\text{max}}=2$ speculative holdout.** The $+5.7\%$ figure
  under real MTP has not been demonstrated on independent data.~~ **Resolved by
  Entry 18** — the holdout ran and the trimmed arms produced **zero** accepted
  drafts, so the $+5.7\%$ figure did not reproduce in any form. Root cause:
  scattering trimmed logits into a $248{,}320$-element `-INFINITY` buffer
  distorts the candidate softmax, so no candidate clears $p_{\text{min}} \ge 0.3$
  at step 0; the failed proposer graph is then still evaluated every round, at an
  unamortized $28\text{--}33\text{ ms}$ cost.
* **Multi-draft proposer accounting is unreconciled.** The multi-draft
  proposer $\text{dur}(g)$ does not reconcile with the single-step measurements.
  **The cause of this residual is UNKNOWN.** It has not been attributed to CPU
  overhead, dispatch cost, scheduling, or anything else, and it must not be
  described as though it had been.
* **No paired token-level acceptance study.** Whether trimming shifts the
  acceptance distribution — as opposed to leaving it intact — is unmeasured.
* **$32\text{K}$ vs $64\text{K}$ is undecided.** No production winner has been chosen.

### Resumption Gate — satisfied, and failed

The gate required a **true unseen $n_{\text{max}}=2$ holdout** carrying raw round
counters, paired proposal logging, and direct wall-clock throughput. Entry 18
delivered all three across 16 unseen prompts in 6 domains with rotated arm order.
**The result was `NO WIN (≤0%)`**, and the thread is closed rather than resumed.

The gate did its job: component timings and acceptance rates had looked
favourable, and only the end-to-end wall-clock measurement exposed the collapse.
