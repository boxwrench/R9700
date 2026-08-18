# ComfyUI wall-time optimization campaign — current status

Date: 2026-08-18

Goal: minimize the time from "I changed what I want" to a usable artifact on an AMD Radeon AI PRO R9700 under ROCm/ComfyUI.

This campaign is deliberately broad. It uses small, decisive experiments, locks large wins after limited sanity checking, and postpones deeper kernel work until the whole workflow landscape is characterized.

## Current operational recommendations

### LTX 2.5

Selected:

- `--disable-mmap`
- INT8-ConvRot Gemma4-12B + projection text encoder
- INT8-ConvRot LTX 2.5 DiT
- `LTX_GEMMA_MIN_LENGTH=256`
- reuse unchanged negative conditioning through normal graph caching
- tested `tile_size=1280` VAE workaround

Key result on the short 768×448, 41-frame, 8-step benchmark:

- floor 1024: 22.51 s conditioning, 43.78 s wall
- floor 256: 5.35 s conditioning, 25.69 s wall
- about 41% wall reduction

See [`ltx25-r9700-optimization-20260818.md`](ltx25-r9700-optimization-20260818.md).

### MiniMax H3

Selected:

- `--disable-mmap`
- single R9700 as the default execution lane
- explicit Qwen3-VL-32B offload after conditioning and before sampling
- normal ComfyUI paging for model-family/checkpoint transitions

Direct pre-sampler offload A/B:

| metric | Qwen resident | Qwen offloaded |
|---|---:|---:|
| VRAM before sampler | 26.18 GiB | 7.19 GiB |
| sampling | 29.51 s | 22.15 s |
| s/step | 1.476 | 1.108 |
| wall | 44.95 s | 41.30 s |

Offload cost was 4.03 s; net wall improvement was 8.1%. A production sanity run reproduced a 22.42 s sampler.

See [`minimax-h3-r9700-optimization-20260818.md`](minimax-h3-r9700-optimization-20260818.md).

The older RX 7900 XT dual-GPU residency experiment remains useful historical evidence but is no longer the main operational recommendation. See [`archive/dual-gpu-residency-20260812.md`](archive/dual-gpu-residency-20260812.md).

### MiniMax Music 3

Baseline only; optimization has not started.

Warm changed-input 15 s output:

- conditioning: 19.87 s
- 20-step generation: 3.64 s
- audio decode: 0.40 s
- wall: 24.42 s
- peak VRAM: 10.52 GiB

Text/lyrics conditioning is 81.4% of warm wall time and is the next justified optimization target.

See [`minimax-music3-r9700-baseline-20260818.md`](minimax-music3-r9700-baseline-20260818.md).

## Cross-workflow switching

One representative transition per lane found staging to be modest relative to generation:

| from -> to | load/stage | condition/prep | sample/gen | decode | wall |
|---|---:|---:|---:|---:|---:|
| LTX -> LTX | 0.00 s | 11.55 s | 13.92 s | 3.95 s | 30.22 s |
| H3 -> H3 | 0.00 s | 4.97 s | 22.37 s | 5.03 s | 33.25 s |
| LTX -> H3 | 5.72 s | 4.90 s | 22.09 s | 5.07 s | 38.68 s |
| H3 -> LTX | 5.90 s | 22.17 s | 13.92 s | 4.00 s | 46.68 s |
| FL2VA -> Ref2VA | 4.03 s | 12.39 s | 26.47 s | 5.31 s | 49.19 s |
| Ref2VA -> FL2VA | 2.78 s | 8.21 s | 22.20 s | 5.27 s | 39.49 s |

Current decision: no urgent multi-GPU/residency redesign for switching alone.

## Important corrected interpretations

The campaign preserves reversals because they are useful engineering evidence:

- multi-minute model loading was traced to an mmap-backed tensor-transfer pathology, not Dynamic VRAM or SSD throughput
- LTX BF16 text encoding did not outperform INT8-ConvRot
- an apparent ~11 s LTX warm conditioning state was one cached CLIP node, not hidden staging
- `rocm-smi --showuse` can report near-zero utilization on a high-power gfx1201 workload
- H3's faster cached state was traced to text-encoder residency, then reproduced with explicit pre-sampler offload
- full GPU power does not imply kernel optimality

## Measurement standard

- report absolute wall time
- distinguish process/model cold, warm changed-input, and cached-input states
- do not add nested profiler percentages as independent costs
- label mechanism explanations as hypotheses unless directly measured
- one or two engineering runs are sufficient for a decision unless variance itself is the question
- do not compare headline times from different geometry/frame/step workloads as if they were the same benchmark

## Next target

MiniMax Music 3 text/lyrics conditioning. The baseline indicates about 20 s of recurring conditioning versus about 3.6 s of DiT generation, so the next task is to decompose/profile `MiniMaxMusic3TextEncode` before changing kernels, cache behavior, or quantization.
