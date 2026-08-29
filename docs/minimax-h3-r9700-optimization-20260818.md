# MiniMax H3 on Radeon AI PRO R9700 — current optimization record

Date: 2026-08-18

This is the current single-R9700 MiniMax H3 engineering record for ComfyUI/ROCm. It supersedes the older dual-GPU residency study as the operational recommendation, while preserving that study as historical evidence.

## Current selected configuration

- AMD Radeon AI PRO R9700, `gfx1201`, 32 GiB VRAM
- ComfyUI v0.33.2, commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`
- `--disable-mmap`
- FL2VA FP8 scaled checkpoint: `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- Qwen3-VL-32B FP8 text encoder: `qwen3vl_32b_minimax_h3_fp8.safetensors`
- explicit Qwen pre-sampler offload after conditioning

The selected pre-sampler offload uses ComfyUI model management (`comfy.model_management.unload_model_and_clones(clip.patcher)`) rather than a new memory manager.

## 1. T2V baseline and cache states

Benchmark workload:

- 608×352
- 39 frames
- 24 fps
- 20 steps
- `res_multistep / simple`

Initial controlled measurements:

| state | conditioning | sampling | s/step | decode | wall | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| cold prompt A | 2.79 s + 5.58 s loader spans | 25.57 s | 1.279 | 5.51 s | 42.88 s | 26.21 GiB |
| warm changed prompt B | 2.57 s | 25.30 s | 1.265 | 5.50 s | 34.46 s | 26.32 GiB |
| warm same prompt/new seed | cached | 22.41 s | 1.120 | 5.38 s | 28.84 s | — |
| warm changed prompt C/new seed | 2.55 s | 25.15 s | 1.257 | 5.36 s | 34.02 s | 26.32 GiB |

Changed-prompt sampling was stable around 25.2 s while the cached-conditioning state repeatedly sampled near 22.4–22.5 s. This led to the residency discriminator below.

## 2. Qwen residency discriminator

Identical prompt token IDs and seed were used. The only material variable was whether Qwen had executed and remained resident before sampling.

| metric | E1 cached / Qwen absent | E2 conditioning re-executed / Qwen resident |
|---|---:|---:|
| prompt token IDs | 46 | 46, identical |
| conditioning | 0.00 s | 2.82 s |
| VRAM at sampler start | 1.70 GiB | 26.45 GiB |
| Qwen resident entering sampler | no | yes |
| sampling | **22.51 s** | 26.64 s |
| s/step | **1.125** | 1.332 |
| wall | 28.89 s | 50.55 s |

Measured fact: sampler throughput was materially better when the text encoder was not resident. The specific hardware mechanism — cache pressure, memory-bandwidth contention, paging, or some combination — was not proven by this timing test.

## 3. Explicit pre-sampler offload A/B

A direct A/B then explicitly unloaded Qwen after conditioning and before sampling.

| metric | Qwen resident control | explicit Qwen offload |
|---|---:|---:|
| conditioning | 3.05 s | 2.83 s |
| explicit offload | — | 4.03 s |
| VRAM before sampler | 26.18 GiB | **7.19 GiB** |
| sampling | 29.51 s | **22.15 s** |
| s/step | 1.476 | **1.108** |
| decode | 5.78 s | 5.30 s |
| wall | 44.95 s | **41.30 s** |
| peak VRAM | 26.18 GiB | **21.41 GiB** |

Derived from this paired run:

- sampling reduction: 7.36 s, 24.9%
- offload cost: 4.03 s
- net wall reduction: 3.65 s, 8.1%

A production sanity run confirmed the result:

- conditioning/load/prep/offload span: 13.18 s
- VRAM before sampler: 7.19 GiB
- sampling: 22.42 s, 1.121 s/step
- decode: 5.39 s
- wall: 44.91 s
- peak VRAM: 21.41 GiB

**Decision:** explicit Qwen pre-sampler offload is SELECTED for changed-prompt single-R9700 H3 workflows.

Same-prompt/new-seed runs already avoid the Qwen staging path through graph caching, so the benefit is mainly for runs where text/multimodal conditioning actually executes.

## 4. H3 mode matrix

Same broad short-workload class, 20 steps:

| mode | input/ref prep or conditioning | sampling | s/step | decode | warm wall | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| T2V / FL2VA | 2.6 s | 25.2 s before explicit offload selection | 1.26 | 5.5 s | 34.0 s | 26.3 GiB |
| I2V / FL2VA first frame | 12.6 s | 27.3 s | 1.37 | 5.4 s | 46.3 s | 26.7 GiB |
| R2V / Ref2VA | 9.7 s | 30.0 s | 1.50 | 5.5 s | 46.1 s excluding one-time swap | 26.7 GiB |

Ref2VA is a separate checkpoint tree. The measured FL2VA -> Ref2VA model staging cost in the initial mode test was 2.11 s.

### Mode-specific prep decomposition

| component | I2V | R2V |
|---|---:|---:|
| input/reference preprocessing | 0.014 s | 0.010 s |
| VAE encode | 3.719 s | 0.627 s |
| Qwen vision/reference processing | **6.168 s** | **6.390 s** |
| Qwen text/multimodal forward | 2.550 s | 2.550 s |
| connector/keyframe packaging | <0.001 s | <0.001 s |
| total | 12.451 s | 9.577 s |

Packed sequence observations:

| mode | context | packed DiT sequence |
|---|---|---:|
| T2V | `[1, 46, 5120]` | 8,353 tokens |
| I2V | `[1, 249, 5120]` | 8,556 tokens |
| R2V | `[1, 247, 5120]` plus reference latent block | 8,730 tokens |

The larger context/packed sequence correlates with the higher per-step cost, but the campaign did not prove that token count alone causes the entire slowdown.

## 5. Sampler profile with Qwen resident

A two-step steady-state profiler run with Qwen resident showed a very different profile from LTX:

- copies/layout: about 39.9%
- HtoD memcpy alone: about 17.1%
- normalization/elementwise: about 24.0%
- quant/dequant/casts: about 18.6%
- GEMM/linear: about 12.3%
- attention: about 5.3%
- host/launch gaps: <1%

The profiler itself heavily inflated step time, so these percentages are for cost composition rather than normal wall-time prediction. They supported testing residency/offload first instead of immediately rewriting kernels.

Do not present the profile as proof of a specific cache/paging mechanism. The direct residency A/B is the stronger evidence.

## 6. Workflow transition matrix

After selecting clean text-encoder offload, one representative transition per lane measured:

| from -> to | load/stage | condition/prep | sample/gen | decode | wall | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| LTX -> LTX | 0.00 s | 11.55 s | 13.92 s | 3.95 s | 30.22 s | 21.41 GiB |
| H3 -> H3 | 0.00 s | 4.97 s | 22.37 s | 5.03 s | 33.25 s | 21.41 GiB |
| LTX -> H3 | 5.72 s | 4.90 s | 22.09 s | 5.07 s | 38.68 s | 21.41 GiB |
| H3 -> LTX | 5.90 s | 22.17 s | 13.92 s | 4.00 s | 46.68 s | 22.18 GiB |
| FL2VA -> Ref2VA | 4.03 s | 12.39 s | 26.47 s | 5.31 s | 49.19 s | 21.41 GiB |
| Ref2VA -> FL2VA | 2.78 s | 8.21 s | 22.20 s | 5.27 s | 39.49 s | 21.41 GiB |

Cross-family staging cost was about 5.7–5.9 s; intra-H3 checkpoint swaps about 2.8–4.0 s. Once staged, H3 sampling returned to the ~22.1–22.4 s selected single-R9700 range.

**Decision:** workflow switching is not currently large enough to justify a dedicated multi-GPU/residency redesign. Standard single-GPU paging remains the selected default.

## 7. Current status

### Selected

- `--disable-mmap`
- single R9700 as the default H3 execution lane
- explicit Qwen offload after conditioning and before sampling
- ordinary ComfyUI paging for workflow/model-family transitions

### Characterized but not currently targeted

- I2V Qwen vision preprocessing, about 6.2 s
- R2V Qwen reference processing, about 6.4 s
- H3 decode, about 5.4–5.5 s across modes
- deeper H3 sampler kernel optimization

### Historical / superseded operational guidance

The 2026-08-12 RX 7900 XT dual-GPU Qwen residency experiment remains valid as a historical experiment, especially for its host-RAM result and smart-memory failure mode, but it is no longer the recommended next optimization. The selected single-R9700 explicit-offload path is simpler and directly addresses the measured sampler residency penalty.

See [`archive/dual-gpu-residency-20260812.md`](archive/dual-gpu-residency-20260812.md).

## 8. R2V reference-size boundary — 2026-08-29

A follow-up campaign tested the previously failing 960×544 / 124-frame Turbo Ref2VA workload with one high-resolution image reference.

| configuration | reliability | warm wall median | warm sampler median |
|---|---:|---:|---:|
| single R9700, `ref_image_size=match`, Turbo | 4/4 pass | 120.099 s | 95.753 s |
| matched dual, Qwen on RX 7900 XT | 4/4 pass | 120.101 s | 95.870 s |

The original `max` run failed in the Turbo LoRA nested `F.linear` path while requesting 3.30 GiB with 2.97 GiB free. Changing only the reference sizing policy to `match` reduced the controlled graph's relevant activation element count by 52.6% and made the full output workload stable. No spatial or frame reduction was required.

The matched dual run freed exactly 12 MiB of R9700 process allocation at sampler entry and was effectively identical in warm wall time. This confirms the existing operational choice: use one R9700, explicitly offload Qwen before sampling, and use `ref_image_size=match` for reference-heavy R2V unless an identity-fidelity test justifies the extra cost of `max`.

Full cold/warm runs, inside-Comfy allocator captures, device-wide ROCm telemetry, workflows, harness, known OOM evidence, and the abandoned historical-dual attempt are preserved in [`data/runs/2026-08-29/minimax-h3-r2v/`](../data/runs/2026-08-29/minimax-h3-r2v/README.md).
