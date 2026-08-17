# Track B Baseline Snapshot

*This document freezes the exact system, toolchain, repository commit, and model
hashes for the pure stock Track B reproduction.*

**Status: PARTIAL.** Source, toolchain, host, GPU, and build state were captured
during preparation on 2026-08-17. **Model and runtime fields remain `PENDING`** —
no native NVFP4 model is present locally, so B1 has no reproduction target yet.

**B0 is not complete until every `PENDING` below is resolved.** Regenerate the
environment section with
[`scripts/01_snapshot_environment.sh`](scripts/01_snapshot_environment.sh) and the
model section with [`scripts/02_hash_model.sh`](scripts/02_hash_model.sh) at the
time of the actual run — the values below describe preparation, not a benchmark.

---

## Source

```yaml
ROCmFPX remote:     https://github.com/charlie12345/ROCmFPX.git
ROCmFPX branch:     main
ROCmFPX commit:     f4b2c5a3edfd183274641094d0db0fcc8092c0ad
fetched (UTC):      2026-08-17T03:53:08Z
checkout path:      /ai/scratch/ROCmFPX-audit
clone method:       git clone --filter=blob:none   (full commit history)
working tree:       clean - no source file modified
shallow:            false
```

> **Unresolved.** A second ROCmFPX exists at `/ai/github/ROCmFPX`, pointing at
> `ciru-ai/ROCmFPX` (HEAD `0d313da`), shallow and dirty. It was deliberately not
> used or touched. Which remote is canonical is an open question for the user.

## Host

```yaml
kernel:             Linux 7.0.0-28-generic
OS:                 Ubuntu 24.04.4 LTS
CPU:                AMD Ryzen 7 9800X3D (8 cores / 16 threads)
RAM:                186 GiB
```

## GPU

```yaml
GPU:                AMD Radeon AI PRO R9700
gfx arch:           gfx1201 (RDNA4, Navi 48)
VRAM:               32624 MiB
enumeration index:  Vulkan1 / ROCm1        # NOT index 0 - see note
other devices:      index 0 = Radeon RX 7900 XT (gfx1100)
                    index 2 = AMD Radeon Graphics iGPU (gfx1036)
GPU clocks/power:   PENDING BASELINE SNAPSHOT
```

> **Device isolation is mandatory.** The R9700 is index 1 under both backends.
> Vulkan runs must pin `Vulkan1`. **HIP runs must set `HIP_VISIBLE_DEVICES=1`**
> — without it the HIP backend segfaults on this mixed-architecture host
> (diagnosed in [reproduction/](reproduction/2026-08-17-build-and-nonmodel-tests.md));
> with it set, the R9700 becomes `ROCm0`.

## Toolchain

```yaml
ROCm version:       7.2.1
HIP version:        7.2.53211-e1a6bc5663
ROCm install:       /opt/rocm-7.2.1
Vulkan loader:      1.3.275 (libvulkan.so)
Vulkan API:         1.4.318
RADV / Mesa:        25.2.8-0ubuntu0.24.04.2
glslc extensions:   GL_KHR_cooperative_matrix       supported
                    GL_EXT_integer_dot_product      NOT supported
                    GL_NV_cooperative_matrix2       NOT supported
                    GL_EXT_bfloat16                 NOT supported
device int dot:     0        # MMVQ / Q8_1 path unavailable on Vulkan
```

## Builds

```yaml
Vulkan build dir:   /ai/scratch/ROCmFPX-audit/build-vulkan
Vulkan configure:   cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
                          -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
Vulkan build:       cmake --build build-vulkan -j 14
Vulkan duration:    2 m 16 s
Vulkan result:      PASS - 0 errors, 114 binaries

HIP build dir:      /ai/scratch/ROCmFPX-audit/build-hip
HIP command:        CMAKE_HIP_ARCHITECTURES=gfx1201 BUILD_DIR=build-hip JOBS=14 \
                        bash scripts/build-rdna4.sh
HIP result:         PASS - 0 errors, 32 binaries
code object arch:   gfx1201  (VERIFIED by upstream's own post-build check:
                    "Verified gfx1201 code objects in libggml-hip.so")

ggml version:       0.11.1
ggml commit:        f4b2c5a
LLAMA_BUILD_NUMBER: b240
```

## Model

```yaml
model repo:         PENDING BASELINE SNAPSHOT
model filename:     PENDING BASELINE SNAPSHOT
model SHA256:       PENDING BASELINE SNAPSHOT
model bytes:        PENDING BASELINE SNAPSHOT
model arch:         PENDING BASELINE SNAPSHOT
quant:              PENDING BASELINE SNAPSHOT
model provenance:   PENDING BASELINE SNAPSHOT   # upstream-published | locally-converted
output.scale:       PENDING BASELINE SNAPSHOT   # present / absent
FFN bias present:   PENDING BASELINE SNAPSHOT   # build_ffn ASSERTS on NVFP4 + bias
```

> **Blocked.** No native NVFP4 Qwen3.8-27B model exists on this machine.
> See [reproduction/2026-08-17-model-inventory.md](reproduction/2026-08-17-model-inventory.md).

## Runtime

```yaml
backend:            PENDING BASELINE SNAPSHOT   # vulkan | hip - genuinely open, see PLAN B4
launch command:     PENDING BASELINE SNAPSHOT
ctx / batch / ubatch: PENDING BASELINE SNAPSHOT
parallel:           PENDING BASELINE SNAPSHOT
flash attention:    PENDING BASELINE SNAPSHOT
KV type:            PENDING BASELINE SNAPSHOT
MTP config:         PENDING BASELINE SNAPSHOT   # n-max, p-min
environment vars:   PENDING BASELINE SNAPSHOT
benchmark harness:  research-program/tracks/B-rocmfpx-nvfp4/scripts/ @ PENDING repo SHA
prompt id / seed:   PENDING BASELINE SNAPSHOT
notes:              Initial stock reproduction on R9700 prior to any local modifications.
```

---

## Non-model validation performed during preparation

Recorded here because it constrains B1, not because it is a result.

| Backend | Test | Result |
|---|---|---|
| Vulkan1 | `MUL_MAT type_a=nvfp4` | 26/26 pass |
| Vulkan1 | `MUL_MAT_ID type_a=nvfp4` | 73/73 pass |
| Vulkan1 | `GET_ROWS nvfp4` | 4/4 pass |
| Vulkan1 | `CPY nvfp4` | 3/3 pass |
| HIP (isolated) | `MUL_MAT type_a=nvfp4` | 41/41 pass |
| HIP (isolated) | `MUL_MAT_ID type_a=nvfp4` | 73/73 pass |

**No performance conclusion is drawn from any of the above.** These are
correctness gates against a CPU reference; none of them loads a model, and none
exercises the lm_head scale path, which lives in `llama-graph` rather than in a
ggml op.
