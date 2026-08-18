# R9700 local AI performance lab

Reproducible workflows, measurements, and engineering records for AI workloads on an AMD Radeon AI PRO R9700. The repository is deliberately numbers-first: it records absolute wall time, exact workload shape, software/model provenance, failed hypotheses, and the evidence behind production selections.

> **Something is slow or weird?** Start with [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md). It maps observed symptoms to the first discriminator to run before you reinstall, requantize, or start tuning kernels.
>
> **Setting up the known-good configuration?** Use [`docs/selected-production-configs.md`](docs/selected-production-configs.md).
>
> **AI agent?** Read [`AGENTS.md`](AGENTS.md) first. Do not silently replace selected configurations with newer upstream defaults.

## What this repo has already found

These are measured improvements on this R9700 system, not generic AMD marketing claims. The workloads and timing boundaries differ, so the rows should not be multiplied together or treated as one universal speedup.

| Problem / workload | Unoptimized or pathological path measured here | Selected / corrected path | Result |
|---|---:|---:|---:|
| H3 Qwen3-VL model load, mmap-backed | 931 s | 1.2 s with `--disable-mmap` | **~776x faster load** |
| H3 full warm-up path | 1,362 s | 51.07 s with mmap disabled | **~27x faster warm-up** |
| LTX short 768x448 / 41-frame / 8-step workload | 43.78 s wall | 25.69 s wall | **41% less wall time / ~1.70x throughput** |
| H3 paired changed-prompt T2V | 44.95 s wall | 41.30 s with Qwen pre-sampler offload | **8.1% less wall time** |
| H3 sampler inside that paired A/B | 29.51 s | 22.15 s | **24.9% faster sampling** |
| MiniMax Music 3 | ~24.42 s warm wall | complete for this pass | **AR bottleneck characterized; FixedKV/compile closed** |

The spectacular mmap numbers are **model-loading fixes**, not generation speedups. Once models are warm, they do not compound with the recurring LTX/H3 improvements.

The other value of this repository is avoiding dead ends. This campaign has already ruled out or scoped several tempting paths: storage/PCIe tuning for the mmap pathology, BF16 LTX encoder as a speed fix, generic H3 dual-GPU residency as the default, unreliable `rocm-smi` utilization conclusions on gfx1201, H3 kernel tuning before encoder residency, Music DiT tuning while conditioning dominates, ComfyUI's NVIDIA-only FixedKV Music graph path, and naive `torch.compile` on Music's dynamic Qwen KV-cache loop.

## If you see this, check this first

| Symptom | First check |
|---|---|
| Model load takes many minutes; one CPU core busy; GPU mostly idle | Run with `--disable-mmap`; see [catastrophically slow model loading](TROUBLESHOOTING.md#1-catastrophically-slow-model-loading) |
| H3 changed-prompt sampling is ~25-30 s instead of ~22 s | Check whether Qwen3-VL is still resident before the sampler |
| H3 sampler starts around 26 GiB allocated | Check pre-sampler encoder offload; selected path drops to roughly 7 GiB before DiT staging |
| LTX short prompt spends ~22 s conditioning | Check Gemma minimum sequence floor; selected value is `256` |
| `rocm-smi` says ~0-1% GPU while board power is near 300 W | Do not diagnose CPU fallback from that counter alone |
| MiniMax Music spends ~20 s conditioning before a ~3.6 s DiT | Inspect `MiniMaxMusic3AR.generate`; the AR loop is the bottleneck |
| Music FixedKV/graph path crashes on AMD | Do not force it; installed flash decode path is NVIDIA-only |
| Music `torch.compile` continually recompiles | Check the changing Python KV-cache index guards |

The full symptom map, measurements, dead ends, and quick discriminators are in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Automation and support

Almost all of this repository is generated and assembled automatically. If you find an error, need information or clarification, or have a testing request, please submit it as a [GitHub Issue](https://github.com/boxwrench/R9700/issues). I’ll try to address issues and requests promptly.

## Current ComfyUI wall-time campaign — 2026-08-18

The current video/audio campaign optimizes the practical latency from **“I changed what I want” to a usable artifact**. It favors small decisive experiments over exhaustive benchmarking and separates measured facts from mechanism hypotheses.

Start with the [current campaign index](docs/comfyui-walltime-campaign-20260818.md).

### Current selections

| workload | selected result / next action | current status |
|---|---|---|
| LTX 2.5 | `--disable-mmap`, INT8-ConvRot encoder+DiT, `LTX_GEMMA_MIN_LENGTH=256`, reuse unchanged negative conditioning | **selected** |
| MiniMax H3 | `--disable-mmap`, single R9700, explicitly offload Qwen3-VL after conditioning and before sampling | **selected** |
| MiniMax Music 3 | baseline characterized (~19.8 s AR / ~3.6 s DiT); FixedKV/compile evaluated and closed | **complete for this pass** |

### Exact files used in the current ComfyUI campaign

**LTX 2.5**

```text
Transformer: ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
Text encoder: gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
E2B: gemma4_e2b_it_bf16.safetensors
```

Selected LTX behavior:

```text
--disable-mmap
LTX_GEMMA_MIN_LENGTH=256
tile_size=1280
INT8-ConvRot text encoder + DiT
reuse unchanged negative conditioning when cached
```

**MiniMax H3**

```text
FL2VA: minimax_h3_fl2va_pruned_fp8_scaled.safetensors
Ref2VA: minimax_h3_ref2va_pruned_fp8_scaled.safetensors
Text/vision encoder: qwen3vl_32b_minimax_h3_fp8.safetensors
```

Selected H3 behavior:

```text
--disable-mmap
single R9700
explicitly unload Qwen3-VL after conditioning and before sampling
comfy.model_management.unload_model_and_clones(clip.patcher)
```

Reference T2V config:

```text
608x352
39 frames
24 fps
20 steps
res_multistep / simple
```

**MiniMax Music 3**

```text
DiT: minimax_music3_dit_int8_convrot.safetensors
Text/lyrics encoder: minimax_music3_text_encoder_pruned_int8_convrot.safetensors
DAV: minimax_music3_dav.safetensors
```

Current benchmark config:

```text
--disable-mmap
15.0 s output
20 steps
res_multistep / simple
CFG 1.5
top_k 50
```

The complete runtime versions, launcher, environment, caveats, and reference results are in [`docs/selected-production-configs.md`](docs/selected-production-configs.md).

### For an AI agent

If the user asks to **set up H3**, **restore the optimized workflow**, **use the fastest current config**, or otherwise reproduce the known-good R9700 state:

1. Treat the selected configurations in this README and [`docs/selected-production-configs.md`](docs/selected-production-configs.md) as authoritative.
2. Do not substitute a newer model, quant, workflow, ComfyUI commit, Torch/ROCm stack, or default merely because it is newer.
3. Verify the exact model filenames before changing anything.
4. Preserve the selected local behavior: LTX Gemma floor `256` and H3 Qwen pre-sampler offload.
5. Consult [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) before diagnosing a performance problem from scratch.
6. Treat updates as **candidates** until they reproduce the relevant canary.
7. If an update would overwrite a selected local modification, report exactly what would be lost before proceeding.

More detailed agent rules live in [`AGENTS.md`](AGENTS.md).

Key current records:

- [LTX 2.5 R9700 optimization record](docs/ltx25-r9700-optimization-20260818.md)
- [MiniMax H3 R9700 optimization record](docs/minimax-h3-r9700-optimization-20260818.md)
- [MiniMax Music 3 baseline](docs/minimax-music3-r9700-baseline-20260818.md)
- [H3 measurements](data/experimental/h3-walltime-20260818.tsv)
- [LTX token-floor measurements](data/experimental/ltx25-token-floor-20260818.tsv)
- [LTX wall-time measurements](data/experimental/ltx25-walltime-20260818.tsv)
- [workflow transition measurements](data/experimental/workflow-transitions-20260818.tsv)
- [MiniMax Music 3 measurements](data/experimental/minimax-music3-baseline-20260818.tsv)

### Two large current wins

**LTX 2.5 token floor.** On the short 768×448, 41-frame, 8-step benchmark, reducing the Gemma minimum sequence floor from 1024 to 256 cut conditioning from 22.51 s to 5.35 s and wall time from 43.78 s to 25.69 s, about a 41% wall-time reduction. This is not a pixel-reproduction-preserving switch: the conditioning remains numerically very close but the later diffusion trajectory changes.

**H3 pre-sampler Qwen offload.** In a paired identical-prompt/seed A/B, leaving Qwen3-VL resident produced 29.51 s sampling at 1.476 s/step. Explicitly unloading it before sampling reduced sampling to 22.15 s at 1.108 s/step. The offload cost 4.03 s, leaving a net 8.1% wall-time improvement. A production sanity run reproduced a 22.42 s sampler.

These are workload-specific engineering results, not universal ROCm claims. Exact geometry, frames, steps, cache state, and model formats are disclosed in the linked records.

## Historical baseline: process-cold/model-cold, 2026-08-12

The original repository baseline is retained for provenance. Do not compare these headline times directly with the newer short optimization workloads; geometry, frame count, steps, model versions, and cache state differ.

| Lane | Prompt → saved artifact | Delivered video | Wall seconds / output second | Native workload |
|---|---:|---:|---:|---|
| MiniMax H3 Standard FP8 | **261.038 s** | **5.167 s** | **50.52** | 864×480, 124 frames, 24 fps, 20 steps |
| MiniMax H3 Turbo v4 FP8 | **80.927 s** | **5.167 s** | **15.66** | 864×480, 124 frames, 24 fps, 4 Turbo steps |
| LTX-2.5 distilled INT8 | **67.035 s** | **5.042 s** | **13.30** | 896×512, 121 frames, 24 fps, 8+3 steps |

These are absolute wall-time measurements, not headline percentages. Each lane was one successful fresh-process/model-cold run. Linux filesystem caches and persistent compiled-kernel caches remained warm, so “cold” here is not disk-cold. The model-native LTX geometry differs from H3 and is disclosed rather than silently cropped in the headline.

## Historical H3 dual-GPU residency experiment

The August 12 experiment placed Qwen3-VL on an RX 7900 XT and kept H3 sampling on the R9700. The corrected lane improved changed-prompt wall time by 7.9% and reduced observed host-RAM peak from 56.8 GB to 35.5 GB, but remained below the adoption threshold.

It also exposed an important failure mode: `--disable-smart-memory` defeated intended encoder residency in that pinned build.

The study remains useful evidence, but the current operational recommendation is the simpler single-R9700 explicit Qwen pre-sampler offload described above.

- [historical dual-GPU record](docs/archive/dual-gpu-residency-20260812.md)
- [compatibility pointer at the old path](docs/dual-gpu-residency.md)

## DeepSeek V4 Flash local inference

A separate llama.cpp/Vulkan campaign ran DeepSeek V4 Flash UD-Q4_K_XL plus the Q8_0 DSpark drafter on the R9700 and system DDR5. The selected configuration uses a 32,768-token context, Q8 target/draft KV caches on GPU, one slot, and 41 target MoE expert layers in system RAM. The RX 7900 XT is hidden from llama.cpp.

| Validation | Result |
|---|---:|
| Two-run mean decode | **8.14 tokens/s** |
| DSpark acceptance, final code prompt | **78.9%** |
| Long-context retrieval | **Pass at 24,603 input tokens** |
| Deterministic sanity check | **5/5** |
| R9700 allocation after full run | **32.156 GB / 34.209 GB** |

The 32K Q8 GPU-KV profile was faster than Q4 GPU KV and substantially faster than Q8 KV in system RAM. Device isolation was essential: selecting the R9700 for the target while both GPUs remained visible caused the DSpark shared-output tensor assertion. `GGML_VK_VISIBLE_DEVICES=1` exposed only the R9700 to both contexts and fixed the load.

The [full DeepSeek V4 inference record](docs/deepseek-v4-flash-inference.md) documents placement, commands, screening results, the 24.6K-token retrieval test, limitations, and guarded crash recovery. Normalized measurements are in [`data/experimental/deepseek-v4-flash.tsv`](data/experimental/deepseek-v4-flash.tsv), with the compact profile matrix under [`experiments/deepseek-v4-flash/`](experiments/deepseek-v4-flash/).

## Qwen3.8-27B quantization comparison

Two Unsloth dynamic quantizations of Qwen3.8-27B were measured on the R9700 under llama.cpp/Vulkan, fully offloaded, using the model's built-in MTP head for speculative decoding. Q4_K_XL is the selected profile for routine local serving.

| | UD-Q4_K_XL | UD-Q6_K_XL |
|---|---:|---:|
| Context size | 163,840 | 65,536 |
| Decode, ten-sample mean | **51.30 tokens/s** | **39.08 tokens/s** |
| Prefill, ten-sample mean | **705.99 tokens/s** | **700.05 tokens/s** |
| MTP draft acceptance | **71.6%** | **70.4%** |
| R9700 allocation | 29.784 GB | 30.409 GB |

Prefill is unchanged between the two, inside run-to-run spread, because prompt processing is compute-bound. Decode falls 23.8% because it is bandwidth-bound and Q6 weights are 1.446× larger. Draft acceptance is effectively identical, so Q6 returns nothing through speculation. The context sizes differ by necessity: Q6 weights leave too little VRAM for 163,840 tokens. This is a speed measurement only — no quality or perplexity comparison was run.

Two negative results are worth the space. `--models-max` defaults to 4 and evicts by model count rather than VRAM, so the router held two large models on one 32 GB card, spilled weights to host RAM, and dropped decode to 3.85 tokens/s with nothing in the logs explaining it. Separately, varying only the sampler seed between benchmark samples yields prompt-cache hits and a meaningless prefill figure; prompts must be prefixed uniquely.

The quantization table above holds thinking disabled on both sides. The gateway that serves this host actually sends `reasoning_effort: low`, and Q4 re-measured in that mode decodes slightly *faster*, at 52.93 tokens/s with 74.8% draft acceptance, most likely because chain-of-thought text is more formulaic and the MTP head predicts it better. That is not a claim of faster answers: token rate counts thinking tokens, and reasoning mode emits more of them before the response begins.

The [full Qwen3.8-27B record](docs/qwen3-8-27b-quant-comparison.md) covers the hybrid Gated DeltaNet architecture and its unusually small KV footprint, router preset precedence, the speculative-decoding sweep, and the null results for batch sizing, host scheduling flags, and GPU clock control. Normalized measurements are in [`data/experimental/qwen3-8-27b-quant.tsv`](data/experimental/qwen3-8-27b-quant.tsv), with the harness and preset under [`experiments/qwen3-8-27b/`](experiments/qwen3-8-27b/).

The ongoing R9700 inference-optimization research program lives under [`research-program/`](research-program/README.md), with the authoritative experiment log at [`docs/qwen3-8-27b-experiment-log.md`](docs/qwen3-8-27b-experiment-log.md). It is currently **paused at the Entry 19 early gate**; [`HANDOFF.md`](HANDOFF.md) is the entry point for resuming it.

## Qwen3.8-27B runtime parameter sweep

Twenty-nine configurations were measured against the production Qwen3.8-27B UD-Q4_K_XL profile. **None beat it.** The production point measured 50.91 tokens/s across five repeats (sd 0.22), confirming it sits on a real local optimum rather than merely appearing fast.

| Parameter | Optimum | Result |
|---|---|---|
| `spec-draft-n-max` | **2** | Decisive; 44.46 at depth 1 and 28.41 at depth 8 |
| `spec-draft-p-min` | 0.3 | Flat 0.0–0.3, monotonic decline above |
| `ctx-size` | 163840 | Flat; free for decode |
| `ubatch-size` | 512 | Flat for decode, but 25% of prefill below 384 |

The instructive result is that acceptance rate moves *opposite* to throughput. Draft depth 1 has the highest acceptance in the study at 0.811 and nearly the worst decode; confidence gating at 0.8 reaches 0.886 acceptance and is 12% slower. Acceptance is a diagnostic of the proposer, not a tuning target. The production point's 0.7068 cannot be improved with ordinary llama.cpp knobs.

A ROCm/HIP build at the same commit was measured against Vulkan on identical weights: prefill **+41.6%**, decode **−11.6%**. Backend choice depends on workload shape, and this host serves short-prompt/long-answer agent traffic, so Vulkan is retained. HIP additionally required `HIP_VISIBLE_DEVICES` to load at all — the same device-isolation requirement already recorded for Vulkan in the DeepSeek campaign.

The [full parameter sweep record](docs/qwen3-8-27b-parameter-sweep.md) includes per-stage tables, positional draft-acceptance survival curves for depths 1 through 8, the metric definitions, the HIP backtrace, and limitations. Normalized measurements are in [`data/experimental/qwen3-8-27b-sweep.tsv`](data/experimental/qwen3-8-27b-sweep.tsv).

## Qwen3.8-27B KV-cache precision and speculative-decoding equivalence

Quantizing the KV cache does not buy throughput on this card, but it buys a lot of memory. Against f16, `q8_0` frees 4.69 GiB for 2.3% decode and `q4_0` frees 7.19 GiB — 21% of the card — for 2.6%. Draft acceptance is flat to three digits across all three, so KV precision does not touch the proposer. KV precision is closed as a throughput lever and retained as a memory lever.

The correctness gate in that experiment surfaced a larger finding. MTP speculative decoding changes greedy output relative to ordinary decode, while n-gram speculation on the same target is bit-identical. The cause is neither the acceptance rule nor speculative batch shape: selecting a model-based draft type sets `n_rs_seq = n_max`, which routes Gated DeltaNet through a recurrent-state snapshot kernel for every token. Forcing that configuration with no drafter at all reproduces the divergence.

The [running experiment log](docs/qwen3-8-27b-experiment-log.md) indexes every Qwen3.8-27B experiment in this campaign, including two retracted conclusions and the controls that overturned them. KV measurements are in [`data/experimental/qwen3-8-27b-kv-cache.tsv`](data/experimental/qwen3-8-27b-kv-cache.tsv).

## Qwen3.8-27B MTP verification and proposer optimization

A multi-stage optimization campaign targeted the 46.3 ms end-to-end MTP round (41.3 ms verification, 5.06 ms proposer forward) on R9700/Vulkan.

| Phase / Investigation | Strategy & Scope | Measured Result | Status |
|---|---|---|---|
| **IQ4_XS Dequant Reuse** | Eliminating redundant SPIR-V dequantization in FFN down | Kernel −2.4% (−0.18 ms), in-graph wall time flat (<0.2 ms) | Closed |
| **IQ4_XS Occupancy Variant** | Halving rows-per-workgroup (`ROWS=2`) for 24 subgroups/SIMD | +3.6% kernel regression from doubled dispatch overhead | Closed |
| **Tiny-N MUL_MAT + ADD Fusion** | Generalizing `mm_add_ok` predicate to multi-column N=2..8 | Eliminates 80 dispatches (−277 µs), but offset by +219 µs bias fetch | Closed |
| **MTP Proposer Decomposition** | Full kernel & tensor breakdown across 556 rounds | Proposer scales linearly at 2.70 ms/token (5.24 ms/round at depth 2) | Complete |

The proposer decomposition revealed that **69.2% of proposer GPU execution** (0.99 ms / step, 1.92 ms / round) is concentrated in the single 1.04 GB `output.weight` full-vocabulary LM Head (`Q6_K`, M=248320, K=5120), while the entire 8-matmul transformer block takes only 0.32 ms. Proposer execution scales strictly linearly with draft depth (depth 1: 2.94 ms, depth 2: 5.61 ms, depth 3: 7.54 ms, depth 4: 10.09 ms).

Normalized proposer measurements are in [`data/experimental/qwen3-8-27b-mtp-proposer.tsv`](data/experimental/qwen3-8-27b-mtp-proposer.tsv), pipeline and path metrics are in [`data/experimental/qwen3-8-27b-iq4xs-pipeline-stats.tsv`](data/experimental/qwen3-8-27b-iq4xs-pipeline-stats.tsv) and [`data/experimental/qwen3-8-27b-iq4xs-path.tsv`](data/experimental/qwen3-8-27b-iq4xs-path.tsv), with full narrative logs in [`docs/qwen3-8-27b-experiment-log.md`](docs/qwen3-8-27b-experiment-log.md).

## Hardware and backend

Current ComfyUI wall-time records in this branch use ComfyUI v0.33.2, commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`.

The hardware remains:

- AMD Radeon AI PRO R9700, 32 GB VRAM, `gfx1201`
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads; 188 GiB host RAM
- Ubuntu 24.04.4 LTS
- ROCm 7.2.x / PyTorch ROCm

The older canonical baseline was pinned to:

- Linux `6.17.0-42-generic`
- ROCm 7.2.1 / HIP 7.2.53211; PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`
- Triton `3.5.1+rocm7.2.1.gita272dfa8`; comfy-kitchen `0.2.30`
- ComfyUI `0.32.0`, commit `c2bcbecd82ec5ae66594340b395c24ef0217b238`

The full historical stack and launch behavior are in [`docs/hardware-software.md`](docs/hardware-software.md).

## Visual examples

These poster frames are extracted from the canonical historical artifacts. Click any frame to open the GitHub Pages gallery with the full video:

<table>
<tr>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/h3-standard-fp8.jpg" alt="MiniMax H3 Standard FP8 poster frame" width="320"></a><br><sub>Historical H3 Standard FP8 — 261.038 s → 5.167 s</sub></td>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/h3-turbo-v4-fp8.jpg" alt="MiniMax H3 Turbo v4 FP8 poster frame" width="320"></a><br><sub>Historical H3 Turbo v4 FP8 — 80.927 s → 5.167 s</sub></td>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/ltx-2.5-distilled-int8.jpg" alt="LTX-2.5 distilled INT8 poster frame" width="320"></a><br><sub>Historical LTX-2.5 INT8 — 67.035 s → 5.042 s</sub></td>
</tr>
</table>

The gallery source is [`docs/index.html`](docs/index.html). Its MP4 sources are GitHub Release assets, keeping video binaries out of ordinary Git history. See [`docs/publishing-video.md`](docs/publishing-video.md) for the one-time Pages and release setup.

## Start here

1. To verify the live system against the machine-checkable production manifest:
   ```bash
   python3 scripts/production-preflight.py
   ```
2. If something is wrong or slow, use [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) first.
3. For the known-good ComfyUI files/settings, inspect [`docs/selected-production-configs.md`](docs/selected-production-configs.md) and [`production/manifest.json`](production/manifest.json).
4. For candidate software/model update procedures, follow [`docs/update-gate.md`](docs/update-gate.md).
5. Read the [current ComfyUI wall-time campaign](docs/comfyui-walltime-campaign-20260818.md) for active video/audio recommendations.
6. Golden workflows are in [`production/workflows/`](production/workflows/), with canaries in [`production/canaries/`](production/canaries/).
7. Run the historical baseline repository verification with:
   ```bash
   python3 scripts/verify.py
   ```

## Scope and safety

Model weights, caches, environments, credentials, and video binaries are not tracked. Canonical artifact paths and SHA-256 values are recorded in [`data/artifacts.tsv`](data/artifacts.tsv) so another operator can validate a local copy without inflating normal Git history. The social-post directory contains text records only; the original MP4 attachments are intentionally omitted.

No repository license has been selected yet. Model and prompt-asset licensing must be checked at their upstream sources. Private authorization correspondence is not included. Do not treat a workflow or a measurement as permission to redistribute model weights.

## Public source links

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [MiniMax H3 on Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [MiniMax H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [LTX-2.5 on Hugging Face](https://huggingface.co/Lightricks/LTX-2.5)