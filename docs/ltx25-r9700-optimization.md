# LTX 2.5 Optimization Guide for AMD Radeon AI PRO R9700 (gfx1201 / ROCm 7.2)

## Executive Summary

Through systematic component-level profiling on the AMD Radeon AI PRO R9700 (32 GB VRAM, RDNA4), LTX 2.5 generation wall time for standard previews (768x448, 41 frames, 8 steps) was reduced from **~43.8 s to ~25.7 s** (a **~41% end-to-end reduction**), while cold load latency was cut from **~11 minutes to under 2 seconds**.

```
+------------------------------------------------------------------------------------+
| LTX 2.5 Optimization Progress (768x448, 41 frames, 8 steps)                         |
|                                                                                    |
| Baseline (floor 1024):  [ Conditioning 22.5s ] [ Sampling 10.6s ] [ VAE 10.7s ] -> 43.8s |
| Optimized (floor 256):  [ Cond 5.4s ] [ Sampling 10.7s ] [ VAE 9.6s ]           -> 25.7s |
+------------------------------------------------------------------------------------+
```

---

## 1. What to Change (USE THIS NOW)

| Component | Selected Configuration | Rationale / Benefit |
|---|---|---|
| **Launcher Flag** | `--disable-mmap` | Eliminates ~0.5s/tensor transfer overhead. Cuts load from >11 min to <2 s. |
| **Text Encoder** | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 14.32 GiB vs 24.46 GiB BF16. Saves 7.2 GiB VRAM, 10 GiB disk; identical encode speed. |
| **Diffusion DiT** | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 20.03 GiB fits fully inside 32 GB VRAM; runs native WMMA kernels (49% self-time). |
| **Token Floor** | `LTX_GEMMA_MIN_LENGTH=256` | Cuts encoder sequence length from 1024 to 256. Drops conditioning from 22.5s to 5.4s. |
| **Negative Conditioning** | Reuse unchanged negative prompt | ComfyUI caches untouched CLIPTextEncode nodes (saves ~11s per render). |
| **Spatial Decoding** | `tile_size=1280` workaround | Prevents high-res VAE tiled seam artifacts. |

---

## 2. Why It Helps: Detailed Component Breakdown

### A. Safetensors Transfer Pathology & `--disable-mmap`
- **Observation:** Checkpoints with many small tensors (such as Gemma encoder at 1,342 tensors or DiT at 7,229 tensors) took 11 to 60+ minutes to load under standard PyTorch mmap file-mapping.
- **Root Cause:** ROCm virtual memory mapping incurred ~500 ms per `.to(device)` call when reading from mmap-backed storage.
- **Resolution:** Setting `--disable-mmap` forces pre-reading into system RAM, achieving ~26 GB/s host-to-device PCIe bandwidth (0.61 ms/tensor).

### B. Encoder Format: INT8 ConvRot vs BF16
Comparison on cold model load and prompt conditioning:
- **INT8-ConvRot:** Load = 1.97 s, Cold conditioning pair = 22.56 s, Peak VRAM = 21.42 GiB, Disk = 14.32 GiB.
- **BF16:** Load = 4.47 s, Cold conditioning pair = 22.80 s, Peak VRAM = 28.58 GiB, Disk = 24.46 GiB.
- **Decision:** Keep INT8-ConvRot. BF16 consumes 7.2 GiB more VRAM with zero compute advantage.

### C. Gemma Tokenizer Floor Optimization (`LTX_GEMMA_MIN_LENGTH=256`)
Upstream ComfyUI (`comfy/text_encoders/lt.py`) hardcoded `min_length=1024` with `pad_left=True`. A short 24-token prompt was padded with 1,000 blank tokens, forcing full self-attention across 1,024 positions.

#### Sequence Length Sweep (Single Encoder Pass):
| Token Floor | Seq Length | Encode Time | Output Crop Shape | Mean Abs Err | Max Abs Diff |
|---:|---:|---:|---|---:|---:|
| 1024 | 1024 | 10.93 s | (24, 6144) | 0.0 (baseline) | 0.000 |
| 768 | 768 | 8.26 s | (24, 6144) | ~1e-4 | 0.125 |
| 512 | 512 | 3.62 s | (24, 6144) | ~1e-4 | 0.125 |
| **256** | **256** | **1.96 s** | **(24, 6144)** | **~1e-4** | **0.125** |
| 128 | 128 | 1.62 s | (24, 6144) | ~1e-4 | 0.125 |

#### Full Video End-to-End Render (768x448, 41f, 8 steps):
- **Floor 1024:** Conditioning 22.51 s | Sampling 10.6 s | Total Wall: **43.78 s**
- **Floor 256:** Conditioning 5.35 s | Sampling 10.7 s | Total Wall: **25.69 s** (-41.3%)
- **Floor 128:** Conditioning 4.60 s | Sampling 10.7 s | Total Wall: **25.04 s**

*Selected:* `LTX_GEMMA_MIN_LENGTH=256` balances safety margin for longer prompts with a massive 41% wall-time reduction.

### D. DiT Sampling Profile
Profile of the 8-step sampler (CUDA execution time ~6.68 s):
1. `comfy_kitchen::int8_linear`: **3.275 s (49.1%)**
2. `gemm_wmma_kernel` (Matrix Core GEMM): **~2.08 s (31.1%)** / variant B **~1.61 s (24.1%)**
3. `flash_attention`: **0.805 s (12.1%)**
4. ConvRot quant/dequant kernels: nested inside `int8_linear`.

*Conclusion:* The DiT is executing natively on RDNA4 WMMA matrix cores.

---

## 3. Rejected & Corrected Hypotheses

| Prior Hypothesis | Empirical Finding | Corrected Understanding |
|---|---|---|
| *Dynamic VRAM caused the 15-minute load.* | Load remained 15m31s with dynamic VRAM off until mmap was disabled. | MMAP `.to(device)` per-tensor overhead was the sole root cause. |
| *BF16 encoder would be faster than quantized INT8.* | Conditioning took 22.80s (BF16) vs 22.56s (INT8). | Compute intensity is identical; INT8 saves 7.2 GiB VRAM. |
| *Warm 11s conditioning was hidden staging overhead.* | Run 1 (pos only) = 11.46s; Run 2 (pos+neg) = 22.14s. | Each Gemma encoder pass costs ~11s. One pass was cached. |
| *Low `rocm-smi` GPU% meant CPU fallback.* | Telemetry showed 300W power draw and 18.7 GiB VRAM active. | `rocm-smi --showuse` counter is uncalibrated/unreliable on gfx1201. |

---

## 4. Backlog / Research Items

The following items are deferred until broader model families (MiniMax H3, MiniMax Music) are characterized:
1. **Triton 3.7 vs Comfy-Kitchen WMMA:** Compare Triton 3.7 gemm kernels against native HIP WMMA.
2. **BF16 DiT Arm C:** Requires multi-GPU offload since BF16 DiT (39.13 GiB) exceeds single 32 GB VRAM.
3. **Sub-128 Token Floor:** Minor gains (<0.65s) with prompt truncation risks.
