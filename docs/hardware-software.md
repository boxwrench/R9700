# Hardware and software stack

This is the validated local stack recorded on 2026-08-12. Values are copied
from the stack record, install records, benchmark reports, and canonical
system-state capture. A value not supported by those records is not extended
here.

| Layer | Validated value |
|---|---|
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| Host memory | 188 GiB reported by the LTX run context |
| Primary GPU | AMD Radeon AI PRO R9700, 32 GB class VRAM, `gfx1201` |
| Secondary GPU | Radeon RX 7900 XT, `gfx1100` |
| GPU selection | R9700 physical HIP index 1, exposed with `HIP_VISIBLE_DEVICES=1` |
| OS | Ubuntu 24.04.4 LTS, x86_64 |
| Kernel | `6.17.0-42-generic` |
| ROCm userspace | 7.2.1 |
| HIP runtime | 7.2.53211 |
| PyTorch | `2.9.1+rocm7.2.1.gitff65f5bc` |
| Triton | `3.5.1+rocm7.2.1.gita272dfa8` |
| comfy-kitchen | `0.2.30`, HIP backend |
| ComfyUI | `0.32.0`, `c2bcbecd82ec5ae66594340b395c24ef0217b238` |
| Turbo custom node | `55fee864dd7b2976b1c4ce3c3d5f7968f181409f` |

The BIOS iGPU was disabled for the validated ROCm topology. Secure Boot was
disabled. The production H3 lane retained the distribution `amdgpu` module;
the record does not use `amdgpu-dkms`.

## Launch behavior

The benchmark lane used a 2 GiB VRAM reserve, disabled dynamic VRAM, and
disabled smart memory. The environment also recorded experimental ROCm
AoTriton enablement and `MIOPEN_FIND_MODE=FAST`; reproduce the lane-specific
environment from the operator's local setup rather than copying private cache
paths into a public shell profile.

The repository records the stack, workflows, and measurements. It does not
install ROCm, ComfyUI, custom nodes, or model weights.
