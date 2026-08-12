# MiniMax H3 dual-GPU encoder-residency experiment

Status: completed 2026-08-12. Decision: retain the corrected dual-GPU lane as an experimental/long-job topology; keep the single-R9700 service as default. Final report: `/ai/benchmarks/minimax-h3/dual-gpu-residency-comparison-20260812.md`.

Date: 2026-08-12

## Outcome

Keep the H3 FP8 diffusion model and all sampling/video/audio work on the 32 GB Radeon AI PRO R9700 while keeping the Qwen3-VL AWQ text encoder resident on the 20 GB Radeon RX 7900 XT.

This is a whole-component placement experiment. It does not split layers, use DisTorch, change model quantization, update ComfyUI, or introduce GGUF.

```text
Logical cuda:0 — Radeon AI PRO R9700 / gfx1201 / 31.86 GiB
├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
├── MiniMax H3 Turbo v4 LoRA and sampler
├── video and audio VAEs
└── latent/sampling work

Logical cuda:1 — Radeon RX 7900 XT / gfx1100 / 19.98 GiB
└── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

The logical order will be created deliberately with `HIP_VISIBLE_DEVICES=1,0`. Physical HIP device 1 is the R9700 and physical HIP device 0 is the 7900 XT. This preserves the R9700 as ComfyUI's default `cuda:0` device and makes the 7900 XT the selectable `cuda:1` device.

## Verified starting state

| Item | Verified value |
|---|---|
| OS/kernel | Ubuntu 24.04; `6.17.0-42-generic` |
| ROCm/PyTorch | ROCm 7.2.1; PyTorch `2.9.1+rocm7.2.1.gitff65f5bc` |
| ComfyUI | `c2bcbecd82ec5ae66594340b395c24ef0217b238` / 0.32.0 |
| Turbo node | `55fee864dd7b2976b1c4ce3c3d5f7968f181409f` |
| Golden service | `comfyui-h3.service`, port 8190 |
| Golden device mask | `HIP_VISIBLE_DEVICES=1`, exposing only the R9700 as `cuda:0` |
| Golden launch flags | `--disable-dynamic-vram --bf16-vae --reserve-vram 2 --disable-smart-memory` |
| Golden Turbo workflow | `/ai/comfyui/user/default/workflows/MiniMax-H3-Turbo-v4-FP8.json` |
| Golden workflow SHA-256 | `da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f` |
| Encoder file | `15687142551` bytes; prior full load reported `14960.20 MiB` |
| Diffusion file | `20958205608` bytes; prior full load reported about `19984.52 MiB` |
| GPU peer access | R9700 to 7900 XT and 7900 XT to R9700 both reported `True` |
| `comfy-kitchen` binary | Includes `gfx1100` and `gfx1201` code objects |

The installed core has vendor-neutral multi-device discovery helpers, but the pinned `CLIPLoader` still hard-codes its choices to `default` and `cpu` and only handles the CPU override. Merely exposing the second GPU will not place the encoder there.

## Isolation and rollback design

The golden installation will not be edited.

- Keep `/ai/comfyui`, `comfyui-h3.service`, `common.env`, and every golden workflow unchanged.
- Create a detached ComfyUI worktree at the exact existing ComfyUI commit.
- Give the experiment its own custom-node directory, user directory, output directory, environment file, launcher, service, workflow, logs, and results.
- Use the same read-only shared model files through the existing extra-model-path configuration; do not duplicate weights.
- Pin `pollockjj/ComfyUI-MultiGPU` to commit `b51c99a525e9607e43545ee2a8b7694c74a4775a` (project version 2.6.4 at planning time).
- Load only `CLIPLoaderMultiGPU` from that extension in the experimental workflow. Do not use its DisTorch, UNET, VAE, checkpoint, GGUF, or model-splitting nodes.
- Prevent the golden and experimental services from running simultaneously.

Proposed paths:

```text
/ai/lab/experiments/minimax-h3/dual-gpu/
├── README.md
├── config/
│   ├── dual-gpu.env
│   └── comfyui-h3-dualgpu.service
├── runtime/ComfyUI/               # detached worktree at the golden SHA
├── user/                           # isolated ComfyUI user data
├── workflows/
├── logs/
└── results/

/ai/artifacts/runs/minimax-h3-dualgpu/
```

## Implementation checklist

- [x] **1. Freeze and verify the control**
  What to build: Record the current kernel, package versions, GPU PCI addresses, physical HIP order, service file, launch environment, workflow/model hashes, and the latest valid H3 Turbo control result.
  Acceptance: The golden workflow still hashes to `da892ad...a1d23f`; both model hashes match their existing records; the golden Git worktree is clean.
  Verify: SHA-256 checks, `git status --short`, `systemctl --user cat comfyui-h3.service`, and the two-device PyTorch inventory.

- [x] **2. Create the isolated runtime**
  What to build: Add a detached ComfyUI worktree at `c2bcbecd...7b238`, an isolated user directory, and shared-model configuration pointing to `/ai/models`.
  Acceptance: No files under the golden `/ai/comfyui/user/default/workflows` or golden custom-node directory change; no model weight is copied.
  Verify: Compare golden hashes before/after, inspect worktree status, and confirm the experiment sees the four H3 model components.

- [x] **3. Install and pin only the experimental MultiGPU extension**
  What to build: Clone `pollockjj/ComfyUI-MultiGPU` inside the experimental worktree and checkout `b51c99a525e9607e43545ee2a8b7694c74a4775a`. Make the existing Turbo node available to the experiment at its current pinned revision.
  Acceptance: The experimental runtime reports `CLIPLoaderMultiGPU`; the golden runtime does not load the new extension; no new Python dependency is installed unless the pinned repository proves one is required.
  Verify: Git SHA checks and a startup-only node inventory from `/object_info`.

- [x] **4. Add the dual-GPU environment and service**
  What to build: Create a separate launcher/service using `HIP_VISIBLE_DEVICES=1,0`, port 8191, the isolated user/output paths, and the same four material memory flags as the golden service. Add a preflight that requires logical `cuda:0 = gfx1201 R9700` and logical `cuda:1 = gfx1100 7900 XT` before ComfyUI starts.
  Acceptance: R9700 remains ComfyUI's default device; both GPUs are visible; the two H3 services cannot be active together.
  Verify: PyTorch device inventory, `/object_info/CLIPLoaderMultiGPU`, service status, and an explicit check that the golden service is inactive while the experiment runs.

- [x] **5. Duplicate the Turbo workflow without altering the golden graph**
  What to build: Copy the golden Turbo v4 FP8 workflow to `MiniMax-H3-Turbo-v4-FP8-DualGPU-Qwen-on-7900XT.json`. Replace only node 128 (`CLIPLoader`) with `CLIPLoaderMultiGPU` and set its device to `cuda:1`. Leave node 127 (`UNETLoader`) on `default`, which resolves to R9700 `cuda:0`. Leave both VAEs, Turbo LoRA strength, sampler, scheduler, step count, prompt, seed, dimensions, frames, and output settings unchanged.
  Acceptance: A normalized graph diff shows only the loader node type/device and experiment output prefix changed.
  Verify: JSON parse, live workflow validation, node-by-node diff, and a saved API graph.

- [x] **6. Run a loader-placement gate before the comparison**
  What to build: Start the isolated service, submit one short 608x352, 39-frame Turbo smoke render, and collect per-GPU telemetry and the full service log.
  Acceptance: The console identifies R9700 as `cuda:0`; the Qwen encoder loads and computes on `cuda:1`; H3 diffusion reports full load on `cuda:0`; both report `full load: True`; neither is actually partially loaded or moved to host RAM. A logged CPU *offload policy* is acceptable only when telemetry and `full load: True` prove that no offload occurred. No OOM, NaN, VM fault, ring timeout, or GPU reset occurs; the MP4 and audio decode successfully.
  Verify: Log assertions, GPU-memory telemetry by PCI device, `ffprobe`, decode-to-null, audio-level check, contact sheet, kernel/journal scan, and artifact SHA-256.

- [x] **7. Run the controlled residency comparison**
  What to build: Compare the existing single-GPU service and the isolated dual-GPU service using the canonical H3 Turbo workload: 864x480, 124 frames, 24 fps, Turbo v4, 4 steps, seed 8112026, prompt enhancement disabled, and the exact neutral brass-robot prompt. For each topology, run in this order:

  1. Process-cold/model-cold first render with prompt A.
  2. Same process, prompt A unchanged, seed changed.
  3. Same process, fixed prompt B substituted so text encoding must re-run.

  Acceptance: The paired graphs differ only in encoder loader/device and output prefix; every execution has a complete artifact and state label; no result is silently averaged.
  Verify: ComfyUI history timestamps, service PID/start timestamps, graph cache records, workflow hashes, model hashes, and per-run logs.

- [x] **8. Validate quality and system health**
  What to build: Validate every measured video and capture both GPU memory/temperature/power plus host RAM.
  Acceptance: Correct H.264 geometry/frame count/frame rate/duration, non-silent synchronized AAC audio, no corrupt or black/grey frames, no NaNs, no CPU fallback, and no new AMDGPU fault/reset/OOM messages in each run window.
  Verify: `ffprobe`, full decode-to-null, audio analysis, contact sheet, artifact hashes, telemetry, and bounded journal scan.

- [x] **9. Analyze the result using wall time first**
  What to build: Report each topology's prompt-to-saved-artifact wall time, delivered clip duration, wall seconds per output second, service startup, and the three cache states. Report model-load and prompt-encode sub-times only if the collected events support those boundaries; do not infer missing component times.
  Acceptance: The report separately answers cold-start behavior, same-prompt behavior, and changed-prompt behavior. It explains that sequential stages do not become parallel; the expected win is avoided unload/reload churn.
  Verify: Recalculate every ratio from raw timestamps and preserve raw records.

- [x] **10. Apply the decision gate and document rollback**
  What to build: Classify the experiment as adopt, retain-for-long-jobs, or reject.
  Acceptance: Adopt as the preferred H3 interactive topology only if placement is stable and changed-prompt wall time improves by at least 10% without more than a 5% same-prompt regression. Retain as experimental if it only helps long/new-prompt jobs. Reject on OOM, partial/offloaded models, instability, corrupt output, or no practical wall-time benefit.
  Verify: Stop the experiment, start `comfyui-h3.service`, confirm port 8190 and the golden workflow hash, then add the final report and exact rollback commands.

## Benchmark record

Primary metric:

```text
prompt accepted / execution_start
        to
saved MP4 / execution_success
```

Report in this order:

1. Prompt-to-artifact wall time.
2. Delivered clip duration.
3. Wall seconds per output second.
4. Process startup and restart-to-artifact time.
5. Peak VRAM per GPU, host RAM, temperature, and power.
6. Placement/full-load status and health checks.

The first pass is an engineering decision run, not a publication-grade benchmark. If it passes and the user wants publishable variance, repeat the process-cold lane three times per topology and report the median while preserving all six individual cold results.

## Stop conditions and fallbacks

Stop immediately on any of the following:

- Logical GPU order differs from the required R9700 `cuda:0`, 7900 XT `cuda:1` mapping.
- Encoder or diffusion model reports partial loading or is actually moved to host RAM. A CPU offload-policy label alone is not a failure when `full load: True` and telemetry prove residency.
- 7900 XT OOMs during Qwen prompt encoding.
- A kernel reports an unsupported/invalid device operation on `gfx1100`.
- Any AMDGPU VM fault, ring timeout, GPU reset, NaN, corrupt media, or persistent queue failure appears.

Fallback order:

1. Return to the untouched golden single-GPU service.
2. Inspect placement and allocator logs; do not add model splitting.
3. If the encoder alone is too tight for 20 GB, evaluate a smaller compatible encoder quant as a separate experiment.
4. Only after this experiment is closed should GGUF, VAE relocation, or layer splitting be considered.

## Source basis

- `comfy-kitchen` HIP backend and architecture matrix: <https://github.com/Comfy-Org/comfy-kitchen/blob/main/README.md#hip-backend-amd-rdna2--rdna3--rdna35--rdna4>
- MultiGPU loader inventory and placement intent: <https://github.com/pollockjj/ComfyUI-MultiGPU>
- `CLIPLoaderMultiGPU` device parameter: <https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/web/docs/CLIPLoaderMultiGPU.md>
- ComfyUI multi-device helper implementation: <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py>
- AIMDO secondary-device failure and `--disable-dynamic-vram` workaround: <https://github.com/Comfy-Org/ComfyUI/issues/13792>
