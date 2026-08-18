# ROCm Telemetry Notes for AMD RDNA4 (gfx1201 / Radeon AI PRO R9700)

## Overview

When profiling workloads on RDNA4 (`gfx1201`) hardware under ROCm 7.x, standard Linux GPU monitoring tools can present deceptive signals. This document notes specific telemetry caveats and outlines reliable observation methodologies.

---

## 1. The GPU Utilization Counter (`rocm-smi --showuse`)

### Observed Anomaly
During intense Gemma text encoding passes and certain fused HIP kernel executions:
- **`rocm-smi --showuse` reads `0%` or `1%`**
- **Simultaneous GPU Power:** `299 - 300 W` (TDP ceiling)
- **Simultaneous VRAM Allocation:** `18.7 GiB`
- **Host CPU Load:** `~180 - 200%` (primarily driver dispatch threads)

### Engineering Implication
**Do NOT interpret `0%` GPU utilization as CPU fallback or GPU idleness.**

On `gfx1201`, the utilization register/sampling counter does not accurately capture short-burst asynchronous compute kernels or certain wave dispatch topologies.

### Correct Diagnostic Corroboration
Always corroborate GPU state across multiple telemetry vectors:
1. **Board Power (`rocm-smi --showpower`):** Sustained 250W–300W confirms heavy matrix/ALU activity.
2. **VRAM Footprint (`rocm-smi --showmemuse`):** Confirms model residence in High-Bandwidth GPU memory.
3. **Engine Clocks (`rocm-smi --showclocks`):** Checks whether the GPU is at P-state maximum (e.g. ~2700–2900 MHz) vs idle parked states (e.g. 42 MHz).
4. **PyTorch Profiler / Rocprof Traces:** Inspect real CUDA/HIP API event streams to verify active kernel execution.

---

## 2. Power Consumption vs Kernel Efficiency

While **300W power draw proves that work is executing on the GPU** (and not stuck in host CPU memory), **it does NOT prove that kernels are mathematically optimal.**

- A poorly scheduled kernel spinning on memory barriers or suffering from excessive register spilling can draw near-peak power while achieving a fraction of theoretical FLOPs.
- Component breakdown, per-step timing, and trace analysis remain necessary to evaluate actual kernel efficiency.
