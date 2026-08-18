# ROCm / ComfyUI Wall-Time Engineering on AMD RDNA4 (R9700 / gfx1201)

## Overview

Optimizing generative AI pipelines on ROCm requires treating the entire timeline from user action ("Queue Prompt") to rendered artifact ("Usable Output") as an integrated system. Focusing exclusively on raw kernel TFLOPs or theoretical memory bandwidth often overlooks large fixed overheads that dominate interactive latency.

This document outlines the architectural bottlenecks, telemetry behaviors, and optimization principles established across the AMD Radeon AI PRO R9700 (32 GB, RDNA4 / gfx1201) optimization campaign.

---

## 1. The Wall-Time Equation

Total turnaround time decomposes into distinct pipeline stages:

$$\text{Total Wall Time} = T_{\text{load/staging}} + T_{\text{text\_encoder}} + T_{\text{prep/latent}} + T_{\text{sampling}} + T_{\text{decode}} + T_{\text{save/encode}}$$

In production sessions, this lifecycle falls into three operational regimes:

1. **Cold Start (Process Fresh):** Weights are mapped/allocated from disk into host memory and transferred to GPU VRAM for the first time.
2. **Warm Run (Changed Prompt):** Model weights remain resident in VRAM or host buffer. Only changed conditioning prompts, seeds, or inputs execute.
3. **Session Cached Run (Seed Change Only):** Both model weights and text embeddings are cached. Only noise initialization, sampling, and decoding run.

### Target Metric: Turnaround Under Modification
The primary metric is **time from user modification to usable artifact**. Optimizing for isolated microbenchmarks is rejected when fixed dispatch overhead or staging latency dominates user wait time.

---

## 2. Fundamental Discoveries & Rules

### Rule 1: Always Launch with `--disable-mmap`
On ROCm 7.x running Linux kernel memory management on `gfx1201`, standard `mmap`-backed safetensors incur a **~0.5 second per-tensor overhead** during `.to(device)` transfers.
- For models with thousands of small parameter tensors (e.g., Qwen3-VL-32B text encoder with 1,836 tensors), default loading took **931 seconds (15.5 minutes)**.
- Passing `--disable-mmap` forces standard bulk RAM reading, collapsing load time to **1.2 seconds (776x speedup)**.
- **Decision:** `--disable-mmap` is a non-negotiable baseline flag for all ComfyUI deployments on ROCm.

### Rule 2: Distinguish Host Caching from Node Staging
When evaluating text conditioning latency:
- If a workflow encodes both a positive and negative prompt with two separate encoder nodes, ComfyUI caches identical inputs.
- If only the positive prompt changes, the negative encoder node takes 0.00s.
- Measuring 11s total conditioning on a single-prompt change vs 22s on a dual-prompt change reflects two ~11s Gemma encoder passes, not an 11s staging phase followed by an 11s compute phase.

### Rule 3: Minimize Tokenizer Floor Padding
Text encoders such as Gemma 3/4 frequently employ default minimum padding lengths (e.g., `min_length=1024`).
- A prompt containing 24 real tokens padded to 1024 forces the attention layers to compute self-attention across 1024 tokens.
- Adjusting the padding floor from 1024 to 256 reduces encoder compute from ~11.0s to ~1.96s per pass without degrading prompt adherence or visual semantics.

### Rule 4: Quantized Weights vs Host Offloading
When GPU VRAM is constrained (e.g. 32 GB):
- Unquantized BF16 DiT models (e.g. LTX 22B BF16 at 39.13 GiB) cannot fit into 32 GB VRAM alongside text encoders and VAEs without aggressive dynamic offload, which destroys throughput.
- Quantized formats (`INT8-ConvRot` at 20.03 GiB) fit fully resident in VRAM and execute via native HIP WMMA matrix cores at peak performance.
- INT8 ConvRot text encoders load in ~1.97s and use ~7.2 GiB less VRAM than BF16, while producing identical conditioning speed.

---

## 3. Recommended Baseline Command Line

For single-GPU Radeon AI PRO R9700 ComfyUI service:

```bash
HIP_VISIBLE_DEVICES=1 \
LTX_GEMMA_MIN_LENGTH=256 \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python /ai/comfyui/main.py \
    --disable-mmap \
    --disable-dynamic-vram \
    --disable-smart-memory \
    --bf16-vae \
    --reserve-vram 2 \
    --listen 127.0.0.1 \
    --port 8190 \
    --disable-auto-launch
```

---

## 4. Methodological Disciplines

1. **Change One Variable at a Time:** Verify the specific impact of every flag or patch before combining.
2. **Never Trust a Single Telemetry Counter:** Low `rocm-smi --showuse` readings on `gfx1201` do not indicate CPU fallback if power draw is 300W and clocks are saturated.
3. **Preserve Semantic Equivalence:** Verify outputs against golden references using visual and numerical diffs.
