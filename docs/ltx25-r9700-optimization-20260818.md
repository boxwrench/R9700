# LTX 2.5 on Radeon AI PRO R9700 — wall-time optimization record

Date: 2026-08-18

This document records the current engineering-selected LTX 2.5 configuration on an AMD Radeon AI PRO R9700 (`gfx1201`) under ROCm/ComfyUI. It is intentionally evidence-calibrated: selected settings are separated from hypotheses and backlog work.

## Current selected configuration

- `--disable-mmap`
- Gemma4-12B + projection text encoder: INT8-ConvRot
- LTX 2.5 DiT: INT8-ConvRot
- local Gemma tokenizer floor: `LTX_GEMMA_MIN_LENGTH=256`
- unchanged negative conditioning reused through normal ComfyUI graph caching
- VAE tile workaround: `tile_size=1280` in the tested workflow

Backlog, not required for the selected configuration:

- Triton 3.7 comparison
- BF16 DiT end-to-end test
- deeper WMMA/kernel tuning

## 1. mmap-backed safetensors transfer pathology

The largest loader regression was not storage bandwidth, PCIe bandwidth, Dynamic VRAM, or quantization format. On this workstation, copying tensors from mmap-backed safetensors to the ROCm GPU incurred a large fixed per-tensor penalty.

Representative measurements from the broader ComfyUI campaign:

| Workload | default mmap | `--disable-mmap` | improvement |
|---|---:|---:|---:|
| MiniMax H3 Qwen3-VL-32B encoder, 1,836 tensors | 931 s | 1.2 s | ~776x |
| MiniMax H3 transformer | 419 s | ~20 s | ~21x |
| H3 full warm-up | 1,362 s | 51.07 s | ~27x |

A microbenchmark with 400 tiny tensors measured about 499.8 ms/tensor from mmap-backed storage versus about 0.61 ms/tensor from ordinary RAM. Across several models, observed load time was well predicted by approximately `tensor_count × 0.5 s` on the pathological mmap path.

**Decision:** `--disable-mmap` is the production baseline on this workstation until the upstream/root cause changes.

## 2. INT8-ConvRot text encoder vs BF16

A controlled encoder-format comparison used the same INT8-ConvRot DiT and the same workflow.

| metric | INT8-ConvRot encoder | BF16 encoder |
|---|---:|---:|
| encoder load | **1.97 s** | 4.47 s |
| conditioning, cold pair | 22.56 s | 22.80 s |
| total wall | **43.36 s** | 47.51 s |
| peak VRAM | **21.42 GiB** | 28.58 GiB |
| encoder file size | **14.32 GiB** | 24.46 GiB |

Conditioning differed by about 1%, while INT8-ConvRot loaded faster and used about 7.2 GiB less peak VRAM.

**Decision:** keep INT8-ConvRot. The earlier BF16 swap was reverted.

## 3. Conditioning accounting

Two `CLIPTextEncode` nodes are independent cacheable passes. Controlled same-process measurements showed:

| state | positive | negative | total conditioning |
|---|---:|---:|---:|
| positive changed, negative cached | 11.46 s | 0.00 s | 11.46 s |
| positive and negative both changed | 10.59 s | 11.55 s | 22.14 s |

The earlier interpretation that a warm ~11 s run represented roughly 11 s staging plus 11 s encoding was retracted. It was one cached CLIP node.

**Operational consequence:** keeping an unchanged negative prompt cached saves about 11 s on this short benchmark.

## 4. Gemma minimum-token floor

ComfyUI's LTX Gemma tokenizer was confirmed locally to use `min_length=1024` and `pad_left=True`. A representative prompt contained only 24 real tokens, so 24 tokens of content were padded to 1,024 positions before the final conditioning output was cropped back to the real-token positions.

Direct single-encoder sweep:

| minimum floor | effective sequence | encode time | max abs delta vs 1024 | mean abs delta | cosine |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 10.934 s | 0 | 0 | 0.9999936 self |
| 768 | 768 | 8.256 s | 0.125 | 1.005e-4 | 0.9999937 |
| 512 | 512 | 3.616 s | 0.125 | 1.725e-4 | 0.9999936 |
| 256 | 256 | 1.958 s | 0.125 | 1.346e-4 | 0.9999936 |
| 128 | 128 | 1.615 s | 0.125 | 1.434e-4 | 0.9999935 |

The outputs were not bit-identical. Small numerical changes changed the later diffusion trajectory even with the same seed, so token-floor changes are not a reproduction-preserving switch.

End-to-end video check, same prompt and seed:

| floor | conditioning, 2x CLIPTextEncode | sampling | wall |
|---:|---:|---:|---:|
| 1024 | 22.51 s | 10.65 s | 43.78 s |
| 256 | **5.35 s** | 10.61 s | **25.69 s** |
| 128 | 4.60 s | 10.70 s | 25.04 s |

`1024 -> 256` removed 18.09 s from the tested run, about a 41% wall-time reduction. Two additional spot-check prompts at floor 256 showed no observed quality or prompt-adherence regression. That is an engineering selection, not a publication-grade quality study.

**Decision:** `LTX_GEMMA_MIN_LENGTH=256` is selected for this workstation. Floor 128 gained only about another 0.65 s wall time and was not worth the smaller safety margin for longer prompts.

## 5. DiT profile

A steady-state LTX sampling profile found quantized linear/GEMM execution to be the dominant GPU cost. `comfy_kitchen::int8_linear` alone accounted for 49.1% self CUDA time and executed through native HIP WMMA matrix-core kernels. Attention was about 12% of the profile. Several lower-level WMMA/ConvRot rows were nested and must not be added as independent percentages.

**Decision:** the INT8 DiT is not on an obvious eager fallback. Triton may still beat the current HIP WMMA implementation, but that is a later kernel-comparison project rather than an urgent correctness/performance fix.

## 6. Telemetry warning on gfx1201

During a real Gemma encode the card could draw roughly 299–300 W with about 18.7 GiB resident while `rocm-smi --showuse` reported around 0–1% GPU use. Treat utilization percentage as advisory rather than authoritative for this workload. Correlate power, VRAM residency, node timing, and profiler traces.

High power does not imply kernel optimality; it only disproved the earlier idle-GPU/CPU-execution interpretation.

## Selected / rejected / backlog

### Selected

- `--disable-mmap`
- INT8-ConvRot Gemma encoder
- INT8-ConvRot LTX DiT
- Gemma minimum floor 256
- reuse unchanged negative conditioning
- tested VAE tile workaround

### Rejected or retracted explanations

- Dynamic VRAM was not the cause of the multi-minute model-load regression.
- BF16 text encoding did not improve conditioning throughput.
- the warm ~11 s conditioning result was not hidden staging.
- low `rocm-smi` GPU-use% did not mean the encoder was running on CPU.
- full-power execution does not prove a kernel is optimal.

### Backlog

- Triton 3.7 vs current comfy-kitchen HIP WMMA for representative DiT linear shapes
- BF16 DiT, recognizing the 39.13 GiB model cannot fully reside on a 32 GiB R9700 and therefore confounds kernel speed with offload

## Comparison boundary

Do not compare the ~25.7 s result above directly with the repository's older 67.035 s LTX baseline. The workloads differ. The token-floor campaign used 768×448, 41 frames, 8 steps; the historical baseline used 896×512, 121 frames, 8+3 steps.
