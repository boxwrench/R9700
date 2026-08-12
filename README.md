# R9700 local video benchmark

Reproducible ComfyUI workflows, prompts, measurements, and stack records for
video generation on an AMD Radeon AI PRO R9700. The repository is deliberately
numbers-first: it records the time from accepted prompt to saved artifact, the
delivered video duration, and the wall-time cost per output second.

## Baseline: process-cold/model-cold, 2026-08-12

| Lane | Prompt → saved artifact | Delivered video | Wall seconds / output second | Native workload |
|---|---:|---:|---:|---|
| MiniMax H3 Standard FP8 | **261.038 s** | **5.167 s** | **50.52** | 864×480, 124 frames, 24 fps, 20 steps |
| MiniMax H3 Turbo v4 FP8 | **80.927 s** | **5.167 s** | **15.66** | 864×480, 124 frames, 24 fps, 4 Turbo steps |
| LTX-2.5 distilled INT8 | **67.035 s** | **5.042 s** | **13.30** | 896×512, 121 frames, 24 fps, 8+3 steps |

These are absolute wall-time measurements, not headline percentages. Each lane
was one successful fresh-process/model-cold run. Linux filesystem caches and
persistent compiled-kernel caches remained warm, so “cold” here is not disk-cold.
The model-native LTX geometry differs from H3 and is disclosed rather than
silently cropped in the headline.

## Hardware and backend

- AMD Radeon AI PRO R9700, 32 GB VRAM, `gfx1201`
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads; 188 GiB host RAM
- Ubuntu 24.04.4 LTS; Linux `6.17.0-42-generic`
- ROCm 7.2.1 / HIP 7.2.53211; PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`
- Triton `3.5.1+rocm7.2.1.gita272dfa8`; comfy-kitchen `0.2.30`
- ComfyUI `0.32.0`, commit `c2bcbecd82ec5ae66594340b395c24ef0217b238`

The full stack and launch behavior are in
[`docs/hardware-software.md`](docs/hardware-software.md).

## Visual examples

These poster frames are extracted from the canonical local artifacts. Click any
frame to open the GitHub Pages gallery with the full video:

<table>
<tr>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/h3-standard-fp8.jpg" alt="MiniMax H3 Standard FP8 poster frame" width="320"></a><br><sub>H3 Standard FP8 — 261.038 s → 5.167 s</sub></td>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/h3-turbo-v4-fp8.jpg" alt="MiniMax H3 Turbo v4 FP8 poster frame" width="320"></a><br><sub>H3 Turbo v4 FP8 — 80.927 s → 5.167 s</sub></td>
<td><a href="https://boxwrench.github.io/R9700/"><img src="docs/assets/ltx-2.5-distilled-int8.jpg" alt="LTX-2.5 distilled INT8 poster frame" width="320"></a><br><sub>LTX-2.5 INT8 — 67.035 s → 5.042 s</sub></td>
</tr>
</table>

The gallery source is [`docs/index.html`](docs/index.html). Its MP4 sources are
GitHub Release assets, keeping video binaries out of ordinary Git history. See
[`docs/publishing-video.md`](docs/publishing-video.md) for the one-time Pages
and release setup.

## Start here

1. Read [`docs/methodology.md`](docs/methodology.md) for the timing boundary and
   the cold-state definition.
2. Inspect the normalized measurements in
   [`data/results.tsv`](data/results.tsv) and the field definitions in
   [`docs/data-schema.md`](docs/data-schema.md).
3. Load the exact JSON workflows under
   [`workflows/`](workflows/). Their SHA-256 values are checked by
   [`scripts/verify.py`](scripts/verify.py).
4. Use the neutral prompt in
   [`prompts/neutral-brass-robot.txt`](prompts/neutral-brass-robot.txt), or
   choose a separate I2V/T2V card from the
   [`Boxwrench v1 prompt suite`](prompts/boxwrench-v1/README.md).

Run the local checks with:

```bash
python3 scripts/verify.py
```

## Scope and safety

Model weights, caches, environments, credentials, and video binaries are not
tracked. Canonical artifact paths and SHA-256 values are recorded in
[`data/artifacts.tsv`](data/artifacts.tsv) so another operator can validate a
local copy without inflating normal Git history. The social-post directory
contains text records only; the original MP4 attachments are intentionally
omitted.

No repository license has been selected yet. Model and prompt-asset licensing
must be checked at their upstream sources. Private authorization correspondence
is not included. Do not treat a workflow or a measurement as permission to
redistribute model weights.

## Public source links

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [MiniMax H3 on Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [MiniMax H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [LTX-2.5 on Hugging Face](https://huggingface.co/Lightricks/LTX-2.5)
