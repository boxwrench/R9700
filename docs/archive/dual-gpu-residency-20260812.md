# Archived: MiniMax H3 dual-GPU residency experiment (2026-08-12)

> Historical record. This experiment remains useful evidence, but it is no longer the current H3 operational recommendation. See [`../minimax-h3-r9700-optimization-20260818.md`](../minimax-h3-r9700-optimization-20260818.md) for the selected 2026-08-18 single-R9700 path using explicit Qwen pre-sampler offload.

This is an experimental long-job lane for the MiniMax H3 Turbo v4 workflow.
It keeps the H3 diffusion model, sampling, and video/audio VAE on the Radeon AI
PRO R9700 while placing the Qwen3-VL 32B NVFP4/AWQ text encoder on a Radeon RX
7900 XT.

The result is useful, but it does not replace the single-R9700 default. The
corrected lane was consistently faster across this three-state engineering
sequence, while the changed-prompt improvement was 7.9%, below the 10%
adoption threshold. It also reduced observed host-RAM peak from 56.8 GB to
35.5 GB. These are one-sequence observations, not a variance study or thermal
certification.

## The important failure mode

The first isolated launcher copied the `--disable-smart-memory` flag from the golden
launcher. In this pinned ComfyUI build that flag forces aggressive model offload
to system RAM, which defeats encoder residency. The corrected launcher removes
only that flag; it retains dynamic-VRAM disabling, BF16 VAE execution, a 2 GiB
reserve, exact software pins, and a two-device preflight.

The initial dual run therefore belongs in the record as a warning: it produced
valid media but the 7900 XT dropped to about 1.5 GB VRAM and Qwen reloaded for
the changed prompt. The corrected run kept Qwen around 16.9–18.3 GB on the
7900 XT and H3 around 24.3–27.7 GB on the R9700.

## Controlled result

All states used 864×480, 124 frames, 24 fps, 5.167 delivered seconds, four
Turbo steps, and the same prompt/seed sequence.

| State | Single R9700 | Initial dual | Corrected dual | Corrected gain |
|---|---:|---:|---:|---:|
| Process/model-cold, prompt A | 80.673 s | 80.044 s | **76.867 s** | **4.7%** |
| Same prompt, new seed | 70.436 s | 70.969 s | **68.003 s** | **3.5%** |
| Changed prompt B, same seed | 75.456 s | 74.119 s | **69.506 s** | **7.9%** |

The earlier 82.55-second dual shakedown is excluded from the comparison table.
All six comparison artifacts decoded fully, had non-silent stereo AAC audio,
and passed the black-interval checks. No GPU reset, OOM, VM fault, or NaN was
observed. The golden service was restored after the run.

## Reproduction record

- [Normalized nine-row measurement table](../../data/experimental/dual-gpu-residency.tsv)
- [Full decision report](../../data/runs/2026-08-12/dual-gpu-residency-comparison-20260812.md)
- [Implementation plan and provenance](../../data/runs/2026-08-12/DUAL-GPU-IMPLEMENTATION-PLAN.md)
- [Corrected UI workflow](../../workflows/minimax-h3/MiniMax-H3-Turbo-v4-FP8-DualGPU-Qwen-on-7900XT.json)
- [Corrected API workflow](../../workflows/minimax-h3/h3-dualgpu-shakedown-5s-turbo-v4-api.json)
- [Single-R9700 control API workflow](../../workflows/minimax-h3/minimax-h3-turbo-v4-4-control-api.json)
- [Launcher snapshot](../../experiments/dual-gpu/scripts/h3-dualgpu.sh)
- [Benchmark harness snapshot](../../experiments/dual-gpu/scripts/benchmark-residency.py)
- [Pinned experiment notes](../../experiments/dual-gpu/README.md)
- [Visual contact sheet](../assets/dual-gpu-residency-contact-sheet.jpg)

The source benchmark bundles and MP4s remain on the workstation and are referenced by path and SHA-256 in the normalized record; video binaries are not required for the operational recommendation.
