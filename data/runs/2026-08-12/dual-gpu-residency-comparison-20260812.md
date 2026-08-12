# MiniMax H3 dual-GPU encoder-residency engineering result

Date: 2026-08-12

Classification: controlled engineering decision run; single pass per state, not a publication-grade variance study.

## Decision

**Retain as an experimental/long-job topology; keep the single-R9700 service as the default.**

The corrected dual-GPU lane is stable and consistently faster, but its decisive changed-prompt improvement is 7.9%, below the plan's 10% adoption threshold. The isolated implementation remains useful when lower host-RAM pressure or longer prompt-changing sessions matter.

## Controlled workload

- MiniMax H3 FP8 diffusion model and Turbo v4 EMA LoRA
- Qwen3-VL 32B NVFP4/AWQ encoder
- 864x480, 124 frames, 24 fps, 5.167 seconds delivered
- Four Turbo steps, simple scheduler, denoise 1.0
- Prompt A seed 8112026 for process/model-cold
- Prompt A seed 8112027 for the same-prompt cached path
- Prompt B seed 8112027 to force text encoding while holding the seed fixed

## Wall-time result

| State | Single R9700 | Initial dual (aggressive offload) | Corrected dual residency | Corrected gain vs single |
|---|---:|---:|---:|---:|
| Process/model-cold, prompt A | 80.673 s | 80.044 s | **76.867 s** | **4.7%** |
| Same prompt, new seed | 70.436 s | 70.969 s | **68.003 s** | **3.5%** |
| Changed prompt B, same seed | 75.456 s | 74.119 s | **69.506 s** | **7.9%** |

Wall seconds per delivered output second:

| State | Single R9700 | Corrected dual |
|---|---:|---:|
| Cold prompt A | 15.61 | **14.88** |
| Same prompt/new seed | 13.63 | **13.16** |
| Changed prompt B | 14.60 | **13.45** |

The earlier 82.55-second dual render was a setup shakedown and is not used in this table.

## Kink discovered and corrected

The first implementation copied `--disable-smart-memory` from the golden launcher. In this pinned ComfyUI build, that flag explicitly forces aggressive model offload to system RAM, contradicting the experiment's residency objective.

The isolated dual launcher was corrected by removing only `--disable-smart-memory`. It retains `--disable-dynamic-vram`, BF16 VAE execution, the 2 GiB reserve, exact ComfyUI/custom-node pins, and the two-device preflight. The golden launcher and service were never modified.

Initial dual telemetry showed the 7900 XT falling to about 1.5 GB VRAM between prompts and reloading Qwen for prompt B. Corrected telemetry showed:

- RX 7900 XT / physical AMD SMI GPU 0: Qwen stayed around 16.9-18.3 GB through the changed-prompt run.
- R9700 / physical AMD SMI GPU 1: H3 stayed around 24.3-27.7 GB through the warm and changed-prompt runs.
- No second `Requested to load MiniMaxH3TEModel_` event occurred for prompt B.
- A 2.44 GB partial unload was the video VAE on the R9700 between decode and the next sampling pass, not Qwen or H3.

## Resource observations

Corrected dual peak observations across the three-run sequence:

| Physical GPU | Role | Peak VRAM | Edge | Hotspot | Memory | Peak power |
|---|---|---:|---:|---:|---:|---:|
| RX 7900 XT | Qwen encoder | 18,547 MB | 55 C | 82 C | 78 C | 310 W |
| R9700 | H3 + sampling/VAE | 27,659 MB | 72 C | 100 C | 88 C | 324 W |

- Corrected dual host-RAM peak: 35.5 GB.
- Single-GPU host-RAM peak: 56.8 GB.
- The corrected sequence therefore reduced observed host-RAM pressure by about 21.2 GB.

These are one-second telemetry peaks from one engineering sequence, not thermal or power-limit certification.

## Validation and health

- All six accepted comparison artifacts are H.264 864x480, 24 fps, exactly 124 frames, with stereo AAC audio.
- Every artifact decoded fully; every audio track was non-silent.
- Automated black-interval checks passed.
- Six-panel visual review found coherent prompt-matched frames and no obvious corruption.
- Qwen and H3 each initially reported `loaded completely` and `full load: True` on their assigned GPU.
- No AMDGPU VM fault, ring timeout, GPU reset, KFD error, OOM, or NaN appeared in either bounded kernel window.
- The golden `comfyui-h3.service` was restored on port 8190; the experimental service was stopped.

## Provenance

- Single and initial-dual raw bundle: `/ai/benchmarks/minimax-h3/dual-gpu-residency-20260812-101311`
- Corrected-dual raw bundle: `/ai/benchmarks/minimax-h3/dual-gpu-residency-20260812-102534`
- Visual contact sheet: `/ai/benchmarks/minimax-h3/dual-gpu-residency-20260812-102534/contact-sheet-single-vs-corrected-dual.jpg`
- Corrected isolated launcher SHA-256: `6b4300a3c96486df8ae7e975b5ee5ba1448cc372993544fb95536a2aaec53447`
- Preserved initial launcher SHA-256: `9efdad6d27bbd17a14ccf412c78419287244d4244450257d482302359303f88a`

The Hugging Face workflow guidance was applied by keeping the existing verified local weights immutable. No model, quantization, or Hub revision was downloaded or changed during the benchmark.
