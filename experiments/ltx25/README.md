# LTX 2.5 Optimization Campaign on AMD Radeon AI PRO R9700 (gfx1201)

## Overview

This directory contains the experiment records, patches, benchmarking scripts, and workflows from the LTX 2.5 wall-time optimization campaign on AMD RDNA4 hardware.

### Hardware & Environment
- **GPU:** AMD Radeon AI PRO R9700 (32 GB GDDR6, `gfx1201`)
- **CPU:** AMD Ryzen 7 9800X3D (8C/16T, 3D V-Cache)
- **RAM:** ~192 GB DDR5
- **OS / Stack:** Ubuntu 24.04, ROCm 7.2.1, PyTorch 2.9.1+rocm7.2.1, ComfyUI v0.33.2, comfy-kitchen 0.2.31

---

## Key Findings

### 1. MMAP safetensors Transfer Pathology
ROCm virtual memory translation causes `weight.to(device)` on mmap-backed safetensors to take ~0.5 seconds per tensor. Checkpoints with >1,000 tensors took 11 to 60+ minutes to load.
- **Fix:** Launch with `--disable-mmap`.
- **Result:** Loads drop to 1.2–2.0 s (776x faster).

### 2. Encoder Format Selection (INT8 ConvRot vs BF16)
- **INT8 ConvRot:** 14.32 GiB disk, 1.97 s load, 22.56 s cold conditioning pair, 21.42 GiB peak VRAM.
- **BF16:** 24.46 GiB disk, 4.47 s load, 22.80 s cold conditioning pair, 28.58 GiB peak VRAM.
- **Decision:** Keep INT8 ConvRot. Zero compute penalty, saves 7.2 GiB VRAM.

### 3. Gemma Tokenizer Pad Floor Optimization
ComfyUI's Gemma tokenizer enforced `min_length=1024` with `pad_left=True`. A 24-token prompt was padded to 1024, forcing full attention computation across 1,024 tokens (~11 s per pass).
- **Fix:** Apply `patches/gemma-min-length.patch` and set `LTX_GEMMA_MIN_LENGTH=256`.
- **Result:** Single encode drops from 10.93 s to 1.96 s; total conditioning drops from 22.51 s to 5.35 s.
- **End-to-End Wall Time:** Reduced from 43.78 s to 25.69 s (-41.3%) for 768x448, 41 frames, 8 steps.

### 4. DiT Profile & Kernel Execution
Sampling profile breakdown (~6.68 s CUDA time):
- `comfy_kitchen::int8_linear`: 49.1% (3.275 s)
- `gemm_wmma_kernel` (HIP Matrix Core GEMM): 31.1% / 24.1%
- `flash_attention`: 12.1% (0.805 s)
- ConvRot dequant: nested inside int8_linear.
- **Conclusion:** Quantized linear execution natively utilizes RDNA4 WMMA matrix cores.

---

## Status of Hypotheses

### CONFIRMED / SELECTED
- `--disable-mmap` is required.
- `INT8-ConvRot` Gemma encoder and DiT.
- `LTX_GEMMA_MIN_LENGTH=256` environment patch.
- Reusing unchanged negative conditioning (0.00s cache hit).
- `tile_size=1280` workaround for high-res VAE decoding.

### REJECTED / RETRACTED
- Dynamic VRAM as the cause of slow loading (it was mmap).
- BF16 encoder for speed (conditioning speed was identical).
- Warm 11 s conditioning as hidden staging (each Gemma pass is ~11 s).
- Low `rocm-smi` utilization % as CPU fallback (unreliable counter on gfx1201).

### RESEARCH / BACKLOG
- Triton 3.7 vs comfy-kitchen native HIP WMMA.
- BF16 DiT Arm C (requires multi-GPU offload due to 39 GiB weight size).
