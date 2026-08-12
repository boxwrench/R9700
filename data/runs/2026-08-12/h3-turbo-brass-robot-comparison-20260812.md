# MiniMax H3 Turbo v4 brass-robot comparison — 2026-08-12

> Superseded by `/ai/benchmarks/cold-video-comparison-20260812.md`. The historical video artifact was removed; use `/ai/artifacts/runs/minimax-h3/minimax-h3/cold-brass-robot-turbo-v4-4_00001_.mp4` as the canonical H3 Turbo benchmark.

> Single-run comparison authorized after the earlier H3 benchmark stop. This is a model-cold/process-warm validation, not a cold/warm/warm2 reference.

## Controlled workload

- Base checkpoint: `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- Encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Turbo adapter: `minimax_h3_turbo_v4_step600_ema.safetensors`, strength 1.0
- Sampler: `MiniMaxH3TurboSampler`; simple scheduler; 4 steps; denoise 1.0
- Official MiniMax H3 video FP16 and audio FP32 VAEs; BF16 VAE execution
- Prompt: `A tiny brass robot carefully waters a glowing mushroom garden at night. Gentle rain, soft mechanical movements, warm lantern light. Audio: quiet rain, small servo sounds, and a soft bell chime. No text or logos.`
- 864x480; 124 frames; 24 fps; seed 8112026

## Result

- Prompt ID: `628ed059-5e53-40fe-a9e0-257882bf5b37`
- ComfyUI history time: 80.387 seconds; service log rounded total: 80.91 seconds
- Output: `/ai/artifacts/runs/minimax-h3/minimax-h3/comparison-brass-robot-turbo-v4-4_00001_.mp4`
- SHA-256: `fab5d52fc3995ab3b59eb9c325d89661a8a55c81e571114d5c1ac3eb1abda4b9`
- Video: H.264, 864x480, exactly 124 frames at 24 fps, 5.166667 seconds
- Audio: AAC, stereo 32 kHz, 5.167 seconds; non-silent (`mean -25.2 dB`, `max -7.4 dB`)
- Full video decode passed without errors.
- Contact-sheet review found coherent non-black frames and clear prompt adherence.
- Run-window kernel scan found no AMDGPU fault, reset, timeout, or OOM.

## Timing context

| Lane | Time | State | Workload difference |
|---|---:|---|---|
| H3 Standard FP8 | 260.91 s | single model-cold validation | 20 standard steps, native 864x480, 124 frames |
| H3 Turbo v4 FP8 | 80.387 s | single model-cold/process-warm validation | 4 Turbo steps, native 864x480, 124 frames |
| LTX-2.5 distilled INT8 | 33.578 s | warm cached | 8+3 stages, native 896x512 then cropped, 121 frames |

- Turbo was 3.25x faster than the matched H3 Standard validation.
- The observed warm LTX time was 2.39x faster than this Turbo run, but this is not a controlled speed ratio because cache state, native geometry, frame count, and model family differ.
- Same numeric seeds do not represent the same latent across H3 and LTX; they only make each lane reproducible.

