# LTX-Desktop-ROCm — cross-GPU wall time, 7900 XT vs R9700

Date: 2026-08-23

Status: **EXPERIMENTAL**. This is a different application stack from the rest of this repository — [LTX-Desktop-ROCm](https://github.com/boxwrench/LTX-Desktop-ROCm) (`rocm-port` branch), a community AMD/ROCm bring-up of [Lightricks/LTX-Desktop](https://github.com/Lightricks/LTX-Desktop)'s FastAPI backend, not ComfyUI. It does not touch, replace, or compete with the selected ComfyUI production configuration in [`selected-production-configs.md`](selected-production-configs.md); see [`AGENTS.md`](../AGENTS.md).

## What this is

LTX-Desktop-ROCm is a from-scratch backend boot: fork, ROCm PyTorch pin, dependency fixes, and a milestone-zero text-to-video generation, brought up independently on this workstation's two AMD cards. This record captures the first cross-GPU wall-time comparison once the backend was working end-to-end on both.

## Two bugs found and fixed during bring-up

Both are backend-agnostic correctness bugs, not ROCm-specific hacks, and are candidates for a future upstream PR.

1. **`device_supports_fp8()` misidentified ROCm as CUDA-capable.** ROCm PyTorch reports `torch.cuda.is_available() == True` and `device.type == "cuda"` identically to real CUDA — the only reliable discriminator is `torch.version.hip` (set on ROCm, `None` on CUDA). The original check (`services_utils.py`) tested `device.type == "cuda"` directly, which would have made the backend attempt FP8 quantization on hardware that doesn't support it. Fixed with a new `accelerator_backend()` helper (`runtime_config/accelerator.py`) that checks `torch.version.hip` first.
2. **`runtime_policy.py`'s VRAM-threshold mode selection assumed FP8 is always available.** The `>=31 GiB -> full_models_loading` threshold was calibrated around CUDA's FP8-halved transformer footprint (~23 GiB). Confirmed live: the R9700 (31.86 GiB reported) crossed that threshold, attempted to hold the full ~42 GiB bf16 transformer resident, and produced a real `HIP out of memory` crash (`Tried to allocate 32.00 MiB ... 0 bytes is free`). Fixed by adding an `fp8_capable` parameter (default `True`, preserving existing CUDA behavior) that forces `streaming_models_loading` whenever FP8 isn't available — which on the current ROCm build is always, so **every ROCm run in this record uses `streaming_models_loading` regardless of VRAM headroom**. That is a real, measured consequence of the fix, not an oversight in the benchmark.

## Method

Each row is one cold process: fresh `uv run python ltx2_server.py`, `HIP_VISIBLE_DEVICES` pinned to a single card, wait for `Uvicorn running`, one `POST /api/generate` (T2V, `model=fast`, `fps=24`, `audio=false`, `seed=42`, fixed prompt), wall time measured from request send to response, process killed, port cleared, repeat for the next cell. No warm/repeat runs — each number is one clean sample, not a mean.

**A first pass at 540p was discarded.** The per-run server failed to bind on both 540p attempts because a stale server process from earlier interactive Gate-testing was still listening on the shared port; `curl` silently answered from that leftover R9700 process instead of the intended fresh per-GPU server, so the original "7900xt_540p" and "r9700_540p" numbers (89.10 s, 87.35 s) were actually two requests to the same warm R9700 process. Caught by checking each log for `Uvicorn running` vs `address already in use` before trusting a result. The 540p cells below are a clean rerun after clearing the port; the 720p cells were clean on the first pass.

Full data: [`ltx-desktop-rocm-walltime-20260823.tsv`](../data/experimental/ltx-desktop-rocm-walltime-20260823.tsv).

## Results

| Resolution | 7900 XT (`gfx1100`, 20 GiB) | R9700 (`gfx1201`, 32 GiB) | R9700 vs 7900 XT |
|---|---:|---:|---:|
| 540p (1024x576) | 131.37 s | 106.39 s | **19.0% faster** |
| 720p (1280x704) | 188.98 s | 223.03 s | **18.0% slower** |

Both cards ran `streaming_models_loading` at both resolutions; all four runs generated a valid 5.04 s / 24 fps H.264 MP4 (verified with `ffprobe`).

Resolution scaling within each card (540p -> 720p):

- 7900 XT: 1.44x
- R9700: 2.10x

## Interpretation

The R9700 is faster at 540p and slower at 720p than the 7900 XT — a crossover, not a flat win or loss for either card. This is one clean sample per cell, not a variance study; no component-level breakdown (encoder load, conditioning, sampling, decode) was captured, since `ltx2_server.py`'s own timing log reports `load=0.00s, text=0.00s` for both cards under `streaming_models_loading` and puts everything under `inference`. No profiling was done to explain *why* the R9700 scales worse with resolution here — candidates worth checking before drawing a hardware conclusion include RDNA4 kernel/tuning maturity in this ROCm build for the shapes involved, streaming-mode staging overhead, and memory bandwidth at the larger tile size, but none of that has been measured. Treat this as a discriminator that a difference exists, not an explanation of its cause.

## Comparison boundary

Do not compare these numbers against this repository's ComfyUI LTX 2.5 records ([`ltx25-r9700-optimization-20260818.md`](ltx25-r9700-optimization-20260818.md)). Different application (LTX-Desktop-ROCm FastAPI backend vs ComfyUI), different model entry point and request shape, no INT8-ConvRot / Gemma-floor tuning applied here, and `streaming_models_loading` is forced on ROCm regardless of VRAM as described above. These are not directly reducible to a single "LTX is Nx faster/slower on AMD" number.

## Selected / rejected / backlog

### Not yet selected (no production config exists for this app on this repo)

This is bring-up telemetry for a separate application, not a configuration decision for the ComfyUI production stack.

### Rejected

- The original 540p pair (89.10 s / 87.35 s) — invalidated by the stale-server port collision described above.

### Backlog

- Component-level timing breakdown (load / encode / sample / decode) instead of one `inference` bucket.
- Repeat runs per cell to establish variance, not just one clean sample.
- Profile the 540p->720p scaling gap between the two cards to find a mechanism.
- I2V, longer durations, and additional resolutions once `LTX-Desktop-ROCm`'s own README/Gate work is finished.

## Environment

- ROCm 7.2.1, HIP `7.2.53211-e1a6bc5663`, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`
- LTX-Desktop-ROCm, `rocm-port` branch, commit `d0b65f0` ("ROCm milestone zero: boot backend and generate one video on AMD") plus the two fixes above
- Same host as the rest of this repository: Ubuntu 24.04.4 LTS, Ryzen 7 9800X3D, 188 GiB host RAM

Full source: [`boxwrench/LTX-Desktop-ROCm`](https://github.com/boxwrench/LTX-Desktop-ROCm), `rocm-port` branch.
