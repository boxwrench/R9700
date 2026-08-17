# Track B — ROCmFPX / Native NVFP4 Closeout

## Status: REPRODUCED / CHARACTERIZED / DEPRIORITIZED

---

## 1. Executive Summary

Track B was established to independently reproduce and characterize Charlie's upstream `ROCmFPX` native-NVFP4 execution path on the AMD Radeon AI PRO R9700 (`gfx1201`).

The upstream native-NVFP4 execution path has been **successfully reproduced, verified, and benchmarked**. However, because uniform NVFP4 achieves **$37.26\text{ tok/s}$ native MTP** (vs Track A's **$\sim 53\text{ tok/s}$** on GGUF Q4_K_XL) and exhibits unresolved greedy MTP divergence, Track B is formally **closed and archived** to return focus to Track A.

---

## 2. What Was Established

* **Upstream Checkpoint**: `RadixArk/Qwen3.8-27B-NVFP4`
* **Hardware & Backend**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB VRAM), Vulkan/RADV
* **Model Variants Evaluated**:
  1. **Original Converted Mixed Model**:
     * File Size: $28,230,539,776\text{ bytes}$ ($28.23\text{ GB}$)
     * Composition: 193 NVFP4 tensors, substantial higher-precision (FP16/Q8_0) remainder
  2. **Uniform NVFP4 Derivative**:
     * File Size: $15,547,030,016\text{ bytes}$ ($15.55\text{ GB}$)
     * Quantization Density: $\approx 4.55\text{ BPW}$
     * Composition: 505 NVFP4 tensors, 1 Q5_K tensor
     * Tensor Invariance: All 193 originally-NVFP4 tensors remained 100% bit-exact through uniformization
     * Scale Handling: All relevant scale tensors preserved; runtime application of `lm_head` scale verified

### Baseline Benchmark Results (B1)

| Configuration | Mixed NVFP4 | Uniform NVFP4 | Classification |
|---|---:|---:|---|
| **Serial Decode Throughput** | 20.32 tok/s | **27.33 tok/s** | **MEASURED** |
| **Native MTP Decode Throughput** | 30.71 tok/s | **37.26 tok/s** | **MEASURED** |
| **MTP Acceleration Multiplier** | 1.511× | **1.363×** | **CALCULATED** |
| **Serial VRAM Footprint** | 25.22 GB | **14.72 GB** | **MEASURED** |
| **MTP VRAM Footprint** | 26.30 GB | **15.80 GB** | **MEASURED** |

*Within Track B, Uniform NVFP4 is the strictly superior artifact.*

---

## 3. Important Semantic Finding: W4A16 vs W4A4

* **Upstream Labeling**: Upstream checkpoints are described in marketing/repository text as `W4A4`.
* **Runtime Profiling Finding**: In the observed `ROCmFPX` runtime, input activation scale tensors are not consumed by the compute graph during execution.
* **Precise Execution Definition**: The reproduced execution is **native NVFP4 weight execution with higher-precision activations (effectively W4A16)**. It must not be described as native W4A4 execution.

---

## 4. Important Correctness Finding: Greedy MTP Divergence

* Under deterministic greedy decoding (`temp = 0.0, top_k = 1`), native MTP output diverges from serial target output.
* The `--spec-mtp-strict-qwen` flag did **not** prevent the first observed divergence; in test traces, the mitigation mechanism became active only *after* the initial divergence occurred.
* **Root Cause**: `UNRESOLVED`. This must not be prematurely attributed to the Track A K3/recurrent-state issue; that relationship remains strictly an unverified hypothesis.

---

## 5. ROCmFP4 FAST Note

* Charlie's public screenshot demonstrating $\approx 72.4\text{ tok/s}$ on the R9700 was explicitly designated as **`ROCmFP4 FAST`**, not the native NVFP4 B1 configuration evaluated here.
* The exact model conversion, kernel specialization, and quantization parameters for that 72.4 tok/s run have not been reproduced.
* `ROCmFP4 / ROCmFP4 FAST` represents a distinct, lossy proprietary representation. It must not be directly compared against or conflated with standard native NVFP4.

---

## 6. Track B Decision & Reopening Gates

* **Action**: Do not continue active NVFP4 kernel micro-tuning, W4A4 activation graph implementations, large-scale quality benchmark sweeps, Track A $\to$ Track B integration, or ROCmFP4 FAST reverse-engineering.
* **Reopening Gates**: Track B may be reopened only if:
  1. The exact $\approx 72\text{ tok/s}$ ROCmFP4 FAST model artifact and command configuration become publicly available.
  2. Upstream `gfx1201` native-NVFP4 kernel throughput materially improves.
  3. A true hardware W4A4 activation compute path is introduced.
  4. A target model strictly requires native NVFP4 precision.
  5. An independent native NVFP4 result exceeds Track A production performance ($\sim 53\text{ tok/s}$).
