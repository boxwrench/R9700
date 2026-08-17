# 0001 — Establishment of Two-Track Qwen3.8 R9700 Research Program

* **Date**: 2026-08-16
* **Status**: ACCEPTED
* **Scope**: Research Program Architecture & Execution Strategy

---

## Context

Extensive optimization work on Qwen3.8-27B on the AMD Radeon AI PRO R9700 (`gfx1201`) has established a deeply characterized Vulkan/RADV GGUF Q4 baseline (Entries 1–17 in [`docs/qwen3-8-27b-experiment-log.md`](../../docs/qwen3-8-27b-experiment-log.md)). Verification-side micro-optimizations on the Q4_K_XL trunk have reached an efficiency floor ($41.26\text{ ms}$ verifier forward), and proposer draft-head optimizations have demonstrated initial viability.

Simultaneously, Charlie's upstream `ROCmFPX` project has introduced native NVFP4 weight loading and execution capabilities on AMD hardware, opening a potential path to substantially higher memory-bandwidth efficiency and lower verification latency.

---

## Decision

1. **Pause Track A**: Formally pause and preserve the established Vulkan / Q4_K_XL / MTP optimization line at Entry 17 without discarding or modifying any existing logs, data, or scripts.
2. **Create Track B**: Establish an independent, decoupled research track dedicated to reproducing, characterizing, and benchmarking native NVFP4 on ROCmFPX on the R9700.
3. **Strict Non-Contamination**: Keep Track B completely free of imported Track A local patches (no Vulkan shaders, no fusion modifications, no draft-vocabulary trimming, no altered speculative parameters) until pure stock ROCmFPX baseline behavior is reproduced and frozen.

---

## Rationale

Injecting our existing local Vulkan/Q4 modifications into ROCmFPX prematurely would make it impossible to determine:
* What Charlie's upstream ROCmFPX work independently achieves on `gfx1201`.
* Whether performance deltas originate from NVFP4 tensor formats, ROCmFPX execution kernels, MTP architecture, or local patches.
* Which findings constitute clean, minimal contributions suitable for upstream submission.
* Whether optimizations derived for the Vulkan/Q4 stack remain technically relevant under the ROCm/NVFP4 execution model.

---

## Success Conditions for Integration

Track A mechanisms may only be evaluated against Track B during the Integration stage after Track B satisfies all of the following prerequisites:

1. **Exact Environment Snapshot**: Hardware, OS, ROCm, HIP, and commit SHAs frozen in `snapshot.md`.
2. **Deterministic Reproduction**: Pure stock ROCmFPX builds and passes numerical sanity checks on `gfx1201`.
3. **Matched Serial Baseline**: Clean non-speculative serial decode throughput measured under matched prompt/context conditions.
4. **Matched Native-MTP Baseline**: Stock native-MTP speculative decode throughput measured without local modifications.
5. **Forward-Pass Cost Characterization**: Verifier vs proposer execution time and memory bandwidth cleanly decomposed.
