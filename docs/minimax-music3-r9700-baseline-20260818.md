# MiniMax Music 3 on Radeon AI PRO R9700 — baseline characterization

Date: 2026-08-18

This is a baseline decomposition, not an optimization result. The purpose is to identify the largest recurring wall-time component before changing the implementation.

## Configuration

- DiT: `minimax_music3_dit_int8_convrot.safetensors`
- text/lyrics encoder: `minimax_music3_text_encoder_pruned_int8_convrot.safetensors`
- audio model/codec/VAE: `minimax_music3_dav.safetensors`
- quantization: INT8-ConvRot
- output duration: 15.0 s
- sampler/scheduler: `res_multistep / simple`
- 20 steps
- CFG scale 1.5
- top-k 50
- ComfyUI v0.33.2, commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`
- `--disable-mmap`

## Measurements

| state | load/stage | text/lyrics conditioning | generation | audio decode | save | wall | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 2.71 s | 20.40 s | 3.84 s | 0.43 s | 0.32 s | 28.00 s | 10.46 GiB |
| warm, changed input | 0.00 s | 19.87 s | 3.64 s | 0.40 s | 0.33 s | 24.42 s | 10.52 GiB |
| warm, repeated input | — | 19.88 s | 3.66 s | 0.41 s | — | 24.44 s | — |

Telemetry:

- conditioning power: 23.0 W average / 57.0 W peak
- generation power: 246.5 W average / 314.0 W peak

## Current conclusion

Text/lyrics conditioning is the dominant recurring cost: 19.87 s of a 24.42 s warm run, about 81.4% of wall time. DiT generation is already fast at roughly 0.18–0.19 s/step, and DAV decoding is about 0.4 s.

The repeated-input run still executed conditioning, so no useful conditioning cache benefit was observed in this workflow.

## Hypothesis boundary

The current working hypothesis is that the `MiniMaxMusic3TextEncode` path contains an expensive autoregressive/token-by-token component, but the baseline data alone does not prove the mechanism. The next justified task is to profile/decompose text/lyrics conditioning before changing kernels, cache behavior, or model format.

## Status

- baseline characterization: COMPLETE
- dominant recurring wall-time component: text/lyrics conditioning
- optimization: NOT YET STARTED
- next target: profile `MiniMaxMusic3TextEncode` / AR path and its cache/kernel behavior
