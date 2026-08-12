# MiniMax H3 native FP8 vs Turbo v4 single-run smoke comparison — 2026-08-11

## Stack

- Ubuntu 24.04, kernel `6.17.0-42-generic`
- ROCm 7.2.1, PyTorch 2.9.1 ROCm 7.2.1, Triton 3.5.1
- ComfyUI `c2bcbecd82ec5ae66594340b395c24ef0217b238` (0.32.0)
- comfy-kitchen 0.2.30; active backend: HIP
- GPU: Radeon AI PRO R9700 / gfx1201, HIP-visible device 0
- Turbo node `55fee864dd7b2976b1c4ce3c3d5f7968f181409f`

## Shared workload

- Prompt: tiny brass robot watering a glowing mushroom garden at night, synchronized rain/servo/bell audio
- Checkpoint: `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video/audio VAEs: official FP16/FP32 MiniMax H3 VAEs
- 608x352, 39 frames, 24 fps, seed 8112026, simple scheduler
- BF16 VAE execution; 2 GiB VRAM reserve; dynamic VRAM and smart memory disabled

## Results

| Lane | Steps | Sampling | Total | Output | Validation |
|---|---:|---:|---:|---|---|
| Native FP8 | 20 | ~20.8 s | 38.87 s | `native-fp8-smoke_00001_.mp4` | playable H.264, stereo AAC, user visual approval, no GPU errors |
| Turbo v4 EMA | 4 | ~4.5 s | 22.19 s | `turbo-v4-4step-smoke_00001_.mp4` | LoRA active on all forwards, playable H.264, stereo AAC, user visual approval, no GPU errors |

Turbo reduced sampling time by about 4.6x and end-to-end time by about 1.75x (42.9 percent). Shared model loading, prompt encoding, and VAE/audio decoding limit the total speedup.

## Reproducibility

- Native workflow SHA-256: `23353acceadd769c352bc5a2fd367712ca448de1505c0946d570cd4d7d10b277`
- Turbo workflow SHA-256: `da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f`


These are single smoke runs, not the three-run cold/warm/warm2 reference. Further H3 benchmarking was stopped by user direction after the later 864x480 quality validation.
