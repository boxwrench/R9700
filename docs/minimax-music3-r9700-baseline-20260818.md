# MiniMax Music 3 on Radeon AI PRO R9700 — baseline & optimization summary

Date: 2026-08-18

Status: **COMPLETE FOR THIS CAMPAIGN PASS**

This document records the baseline characterization and the conclusive evaluation of potential conditioning optimization paths for MiniMax Music 3 on the AMD Radeon AI PRO R9700 under ROCm/ComfyUI.

## Production configuration

- DiT: `minimax_music3_dit_int8_convrot.safetensors`
- Text / lyrics encoder: `minimax_music3_text_encoder_pruned_int8_convrot.safetensors`
- Audio codec / VAE: `minimax_music3_dav.safetensors`
- Quantization: INT8-ConvRot
- Output duration: 15.0 s (375 frames)
- Sampler / scheduler: `res_multistep / simple`
- Steps: 20
- CFG scale: 1.5
- Top-k: 50
- ComfyUI: v0.33.2, commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`
- Launch flags: `--disable-mmap`

---

## Baseline measurements (15 s output)

| State | Load / stage | Text / lyrics conditioning | DiT generation | Audio decode | Save | Wall | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 2.71 s | 20.40 s | 3.84 s | 0.43 s | 0.32 s | 28.00 s | 10.46 GiB |
| Warm, changed input | 0.00 s | 19.87–20.03 s | 3.64 s | 0.40 s | 0.33 s | 24.42 s | 10.52 GiB |
| Warm, repeated input | — | 19.88 s | 3.66 s | 0.41 s | — | 24.44 s | — |

Telemetry:
- Conditioning power: 23.0 W average / 57.0 W peak
- Generation power: 246.5 W average / 314.0 W peak

---

## Conditioning cost decomposition

A detailed execution trace of `MiniMaxMusic3TextEncode` across 375 frames (15 s audio) revealed:

- Total conditioning: **20.03 s** (81.4% of total wall time)
- Initial prompt prefill / tokenization: **~0.225 s**
- Iterative autoregressive loop: **19.80 s** (~51.6 ms / frame)
  - **Qwen3-8B 36-layer 1-token backbone:** **~12.04 s** (~32.1 ms / frame)
  - **RVQ 4-layer 7-pass depth decoder (`RVQDepthDecoder`):** **~6.82 s** (~18.2 ms / frame)

---

## Evaluated optimization paths & closed routes

### 1. ComfyUI FixedKV / HIP Graph capture
- **Status:** **BLOCKED**
- **Findings:**
  - PyTorch HIP graph capture and replay are functional on ROCm/R9700.
  - However, ComfyUI's internal fixed-KV path unconditionally routes attention through `comfy_kitchen.flash_attention_decode`.
  - That extension is hardcoded for NVIDIA CUDA architectures (SM80+) and throws a runtime exception on ROCm.

### 2. Qwen3-8B one-token backbone `torch.compile`
- **Status:** **REJECTED / BLOCKED FOR CURRENT STRUCTURE**
- **Findings:**
  - Tracing initially succeeds, but ComfyUI's standard KV-cache representation tracks an incrementing Python integer index in `past_key_values[i][2]`.
  - Changing this Python integer each step causes TorchDynamo guard failures on every token iteration, forcing continuous TorchInductor recompilation.

### 3. RVQDepthDecoder `torch.compile`
- **Status:** **REJECTED**
- **Findings:**
  - `RVQDepthDecoder` exhibits strict sequential dependency across passes 1–7 because each pass $N$ consumes the discrete sampled token $c_{N-1}$ to construct the input sequence for pass $N$.
  - Compiling the 4-layer transformer succeeded without graph breaks, but at sequence lengths $S \in [2..8]$, GPU compute time is $<0.3\text{ ms}$ while host launch overhead dominates (~2.3 ms/pass).
  - **Measured Control (Eager):** ~0.91 s / 50 frames (2.60 ms / depth pass)
  - **Measured Test (Compiled):** ~0.88 s / 50 frames (2.51 ms / depth pass)
  - **Speedup:** **1.034x** (3.3% RVQ reduction)
  - **Projected 15 s total wall reduction:** **0.23 s / 0.94%** ($\ll 10\%$).

---

## Campaign conclusion

MiniMax Music 3 optimization is **COMPLETE FOR THIS CAMPAIGN PASS**.

Further meaningful acceleration would require a dedicated, separate engineering project, such as:
1. Developing an AMD-native HIP static-KV flash decode kernel to replace `flash_attention_decode`.
2. Implementing an end-to-end C++ sidecar or graph execution runtime that eliminates Python host dispatch across the 2,625 discrete depth iterations.

Neither path is currently available off-the-shelf on ROCm.
