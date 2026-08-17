# Upstream ROCmFPX Lane

## Status: READY

## Purpose

This lane provides a structured, high-hygiene process for translating useful AMD Radeon AI PRO R9700 (`gfx1201`) discoveries into minimal, production-grade upstream contributions to Charlie's `ROCmFPX` project.

We do **not** submit massive experimental forks or combined patches upstream. Instead, every upstream contribution follows a strict four-step lifecycle:

```text
1. Discovered Finding  ──►  2. Minimal Reproducer  ──►  3. Targeted Patch  ──►  4. Upstream PR
```

---

## Required Artifact Format for Upstream Submissions

Every candidate finding or proposed PR artifact must include:

1. **Exact Upstream Base**: `ROCmFPX` commit SHA and branch.
2. **Hardware Target**: `AMD Radeon AI PRO R9700 (gfx1201, RDNA4)`.
3. **Toolchain Environment**: ROCm version, HIP version, compiler version, and kernel version.
4. **Model Verification**: Exact model repository, filename, and SHA256 hash.
5. **Exact Invocation**: Complete command-line reproduction string.
6. **Quantified A/B Data**: Before vs after performance or numerical delta with statistical confidence.
7. **Step-by-Step Reproduction**: Self-contained instructions to reproduce the result from a clean checkout.
8. **Raw Logs**: Unedited benchmark logs and profiler traces attached.
9. **Concise Rationale**: One-paragraph technical summary of the root cause and proposed resolution.

---

## Contribution Categories

* **For Bug Fixes**:
  * Create the absolute smallest self-contained test case or CLI reproducer in `reproducers/`.
* **For Performance Optimizations**:
  * Provide the smallest isolated patch in `patches/`.
  * Demonstrate numerical correctness invariance.
  * Provide component-level benchmark verification.
  * Provide end-to-end decode throughput verification.

---

## Workspace Structure

* `findings/`: Documented performance cliffs, architectural bottlenecks, or optimization opportunities.
* `reproducers/`: Standalone C++/Python/script reproductions.
* `patches/`: Minimal, clean git patches ready for upstream submission.
