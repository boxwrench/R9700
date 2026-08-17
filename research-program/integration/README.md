# Integration Stage

## Status: BLOCKED / PENDING TRACK B BASELINE

---

## Purpose

The Integration stage serves as the controlled transfer gateway between Track A (Vulkan/Q4) and Track B (ROCmFPX/NVFP4).

Only after Track B has established its own stable, clean, and fully characterized baseline will candidate discoveries from Track A be evaluated on Track B.

---

## Core Rule: NO OPTIMIZATION IS ASSUMED PORTABLE

An optimization that produced gains on Vulkan/Q4 cannot be assumed beneficial or relevant on ROCmFPX/NVFP4. Every transfer must be treated as a new hypothesis and validated through a fresh A/B comparison.

---

## Portability Classification

### 1. Likely Portable / Mechanism-Level
* **Benchmark Contract & Protocol**: Four-stage validation chain, strict holdouts, and measurement classifications.
* **Positional Acceptance Metrics**: $p_0$, $\text{joint-}p_1$, $\text{conditional-}p_1$, accepted drafts/round, and committed tokens/round.
* **Round-Cost Decomposition**: Separating target verification cost from proposer forward latency.
* **Draft-Vocabulary Trimming Mechanism**: Slicing a dedicated $32\text{K} / 64\text{K}$ draft LM head (`nextn.shared_head_head`) with `d2t` remapping while leaving target `model.output` intact.
* **Proposer vs Verifier Accounting**: Decoupling depth scaling ($N \times \text{proposer}$) from verifier width.

### 2. Backend-Specific / DO NOT Transfer Automatically
* **Vulkan Shader Specializations**: `IQ4_XS` dequantization hoisting, loop interchange, and subgroup optimizations.
* **Vulkan Fusion Patches**: `MUL_MAT + ADD` predicate modifications in `ggml-vulkan.cpp`.
* **Vulkan Workgroup & Dispatch Tuning**: RADV-specific concurrency, timeline semaphores, and command buffer batching.

---

## Current Status

```text
DETAILED INTEGRATION PLAN: PENDING
(Awaiting stable Track B baseline characterization)
```
