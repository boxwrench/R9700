# Qwen3.8-27B R9700 Optimization Research Program

## Objective

Maximize real Qwen3.8-27B inference throughput on the AMD Radeon AI PRO R9700 (`gfx1201`), especially native speculative/MTP decode, while preserving model behavior and producing reproducible findings useful to the wider AMD/llama.cpp/ROCmFPX ecosystem.

---

## Research Tracks

The program is formally split into two independent, decoupled research tracks:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │        Qwen3.8-27B R9700 Optimization Program               │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
     ┌───────────────────────────┐                               ┌───────────────────────────┐
     │          Track A          │                               │          Track B          │
     │     Vulkan / GGUF Q4      │                               │     ROCmFPX / NVFP4       │
     │      Native MTP Line      │                               │     Native MTP Line       │
     └─────────────┬─────────────┘                               └─────────────┬─────────────┘
                   │                                                           │
                   │ (STATUS: PAUSED @ E19 GATE)                               │ (STATUS: ARCHIVED / REOPENABLE)
                   │                                                           │
                   ▼                                                           ▼
     ┌───────────────────────────┐                               ┌───────────────────────────┐
     │ Deeply characterized line │                               │ Upstream-first baseline   │
     │ Entries 1–19 logged       │                               │ Pure stock reproduction   │
     │ Authoritative reference   │                               │ Clean gfx1201 benchmarks  │
     └─────────────┬─────────────┘                               └─────────────┬─────────────┘
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Integration Stage      │
                                  │  (No optimization assumed   │
                                  │    portable without A/B)    │
                                  └─────────────────────────────┘
```

### TRACK A — Vulkan / Conventional GGUF / Existing MTP Research

* **Purpose**: Continue and preserve the deeply characterized `llama.cpp` Vulkan/RADV Qwen3.8 line.
* **Established Findings**:
  * Q4/Q6 MTP precision comparison
  * Vulkan vs HIP backend comparison
  * `ctx`, `ubatch`, speculative depth, and `p-min` parameter sweeps
  * KV cache precision study (`f16` vs `q8_0` vs `q4_0`)
  * Qwen3.6 proposer control experiments
  * Recurrent-state ($K=3$) verification cost investigation
  * MTP round-cost decomposition ($41.26\text{ ms}$ verifier $+ 5.06\text{ ms}$ proposer)
  * Vulkan kernel profiling and execution differencing
  * Entry 16: `IQ4_XS` dequant-reuse shader investigation (closed, $<0.2\text{ ms}$ gain)
  * Entry 17: Tiny-$N$ `MUL_MAT + ADD` fusion investigation (closed, neutral wall time)
  * Entry 18: draft-vocabulary trimming via full-vocabulary reconstruction ($64\text{K} / 32\text{K}$ rows) — `FAILED IMPLEMENTATION`; the $1.42\text{ ms}$ kernel reduction was real, but the FILL + `SET_ROWS` reconstruction collapsed speculation end-to-end
* **Current Status**: **`PAUSED AT ENTRY 19 EARLY GATE`** — see [`HANDOFF.md`](../HANDOFF.md).
* **Important**: The **reconstruction** implementation of draft-vocabulary trimming must not be deployed — Entry 18's unseen $n_{\text{max}}=2$ holdout produced **zero accepted drafts** on both trimmed arms and roughly halved decode throughput. The failure localized to the reconstruction and backend-sampling path, **not** to the reduced-vocabulary concept, which remains open and is tested by **Entry 19** (direct reduced-vocabulary sampling). Existing Track A logs and datasets remain at their current repository paths and are authoritative.

### TRACK B — ROCmFPX / Native NVFP4

* **Purpose**: Reproduce and independently characterize Charlie/ROCmFPX's new native NVFP4 execution path on the R9700 (`gfx1201`) before applying any local optimizations.
* **Core Philosophy**:
  ```text
  UPSTREAM FIRST  ──►  REPRODUCE FIRST  ──►  OPTIMIZE SECOND
  ```
* **Initial Baseline Requirements**:
  * Stock/current ROCmFPX upstream commit
  * Native NVFP4 execution
  * Exact model/file SHA256 verification
  * Zero imported Track A kernel modifications
  * Zero Track A vocabulary trimming
  * Zero locally altered speculative policy
* **Goal**: Determine exact native ROCmFPX throughput and scaling on R9700.

---

## Directory Layout

* [`tracks/A-vulkan-q4-mtp/`](tracks/A-vulkan-q4-mtp/README.md): Index to Track A historical logs, benchmarks, and closed branches.
* [`tracks/B-rocmfpx-nvfp4/`](tracks/B-rocmfpx-nvfp4/README.md): Track B workspace, environment snapshots, reproductions, and benchmarks.
* [`shared/`](shared/README.md): Common scientific protocol, [benchmark contract](shared/benchmark-contract.md), and [metrics terminology](shared/metrics.md).
* [`integration/`](integration/README.md): Stage where verified Track A mechanisms are evaluated on Track B.
* [`upstream-rocmfpx/`](upstream-rocmfpx/README.md): Structured lane for preparing minimal reproducers, findings, and upstream PRs.
* [`decisions/`](decisions/0001-two-track-research-program.md): Architectural and programmatic decision records.
