# Track B Preparation — Build and Non-Model Validation

**Date:** 2026-08-17
**Stage:** pre-B0 preparation. **This is not B1.** No model was loaded and no
inference was run, so nothing here is a throughput result.

**Purpose:** establish that ROCmFPX builds for `gfx1201` and that its NVFP4
kernels are numerically correct on the R9700, before any performance work
begins.

---

## Checkout

```yaml
path:      /ai/scratch/ROCmFPX-audit
remote:    https://github.com/charlie12345/ROCmFPX.git
branch:    main
sha:       f4b2c5a3edfd183274641094d0db0fcc8092c0ad
clone:     git clone --filter=blob:none      # blobless; full commit history
shallow:   false
dirty:     no
modified:  none - the tree was not edited at any point
```

A checkout already existed at `/ai/github/ROCmFPX`. It was **not used and not
touched**: it points at a *different remote* (`ciru-ai/ROCmFPX`), is a depth-1
shallow clone in detached HEAD, and has five locally modified script files.
See the [upstream audit](../upstream-audit/2026-08-17-upstream-audit.md) §0.

---

## Host

```yaml
kernel:  Linux 7.0.0-28-generic (Ubuntu 24.04.4 LTS)
cpu:     AMD Ryzen 7 9800X3D (8C/16T)
ram:     186 GiB
rocm:    7.2.1 (hipconfig 7.2.53211-e1a6bc5663)
mesa:    25.2.8-0ubuntu0.24.04.2 (RADV), Vulkan API 1.4.318
gpus:    Vulkan0/ROCm0 = Radeon RX 7900 XT   (gfx1100)
         Vulkan1/ROCm1 = AMD Radeon AI PRO R9700 (gfx1201)   <-- target
         Vulkan2/ROCm2 = AMD Radeon Graphics  (gfx1036, iGPU)
```

**The R9700 is index 1 under both backends.** Every command below pins it.

---

## Build 1 — Vulkan

**Status: PASS.**

```bash
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build-vulkan -j 14
```

```yaml
configure:  0.8 s
build:      2 m 16 s   (03:57:14Z -> 03:59:30Z UTC)
errors:     0
binaries:   114 in build-vulkan/bin/
ggml:       0.11.1, commit f4b2c5a
```

Notable configure-time output — **relevant to Track A's MMVQ finding**:

```
-- GL_KHR_cooperative_matrix supported by glslc
-- GL_EXT_integer_dot_product not supported by glslc
-- GL_NV_cooperative_matrix2 not supported by glslc
-- GL_EXT_bfloat16 not supported by glslc
```

Track A established that the R9700 under RADV reports `int dot: 0`, ruling out
the MMVQ/Q8_1 path for every quantization type. This build confirms the same
constraint one level up: **the shader compiler in this toolchain cannot emit
`GL_EXT_integer_dot_product` at all.** Runtime device report, from
`llama-bench --list-devices` on this build:

```
ggml_vulkan: 1 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | uma: 0 | fp16: 1 |
             bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 |
             matrix cores: KHR_coopmat
```

So NVFP4 decode on Vulkan necessarily runs the float DMMV path.

## Build 2 — HIP / ROCm, gfx1201

**Status: PASS.**

```bash
CMAKE_HIP_ARCHITECTURES=gfx1201 BUILD_DIR=build-hip JOBS=14 bash scripts/build-rdna4.sh
```

```yaml
build:     ~2 m 40 s total across two invocations (see note)
errors:    0
binaries:  32 in build-hip/bin/
```

Upstream's own safeguard fired correctly on real Navi 48 hardware:

```
Verified gfx1201 code objects in libggml-hip.so
Built for gfx1201:
  build-hip/bin/llama-cli
  build-hip/bin/llama-server
  build-hip/bin/llama-quantize
  build-hip/bin/test-backend-ops
```

This is worth recording: `docs/BUILD-AMD-ARCHITECTURES.md` warns that a
`gfx1200` build on a `gfx1201` card *"links and loads a model, then segfaults or
hangs with no error message"* (issues #18, #37). The detection and
code-object verification in `scripts/build-rdna4.sh` **work as documented here.**

> **Process note.** The first HIP build attempt was terminated at rc=143
> (SIGTERM) roughly two minutes in — killed by a `pkill` I issued to clean up an
> unrelated stuck test process, not by any toolchain problem. Its log contains
> zero compile errors. The build was restarted and completed cleanly. The
> ~2 m 40 s figure spans both invocations; the second reused the first's objects
> and so is not a clean-build time.

---

## Non-model validation

All via `test-backend-ops test`, which checks each op against a CPU reference
under an NMSE gate. **No model file is involved.**

### Vulkan — R9700 (`-b Vulkan1`)

| Test | Filter | Result |
|---|---|---|
| `MUL_MAT` | `type_a=nvfp4` | **26/26 passed** |
| `MUL_MAT_ID` | `type_a=nvfp4` | **73/73 passed** |
| `GET_ROWS` | `nvfp4` | **4/4 passed** |
| `CPY` | `nvfp4` | **3/3 passed** (permuted nvfp4→nvfp4 variants report "not supported" and are skipped, not failed) |
| `ADD` | — | 99/99 passed (control) |
| `test-quantize-fns` | — | ran, NVFP4 exercised, rc=0 |

### HIP — R9700 (`-b ROCm0` with `HIP_VISIBLE_DEVICES=1`)

| Test | Filter | Result |
|---|---|---|
| `MUL_MAT` | `type_a=nvfp4` | **41/41 passed** |
| `MUL_MAT_ID` | `type_a=nvfp4` | **73/73 passed** |
| `GET_ROWS` | `nvfp4` | 0/0 — no NVFP4 case is registered for this op on the HIP backend |
| `ADD` | — | 99/99 passed (control) |

**HIP runs 41 NVFP4 `MUL_MAT` cases to Vulkan's 26.** Consistent with the audit
finding that HIP has an NVFP4 Q8_1 vec-dot (`mmvq.cu:162`) that Vulkan on RADV
structurally cannot use — though the case-count difference alone does not prove
which kernels ran. **INFERRED**, to be confirmed in B3.

### Conclusion

**NVFP4 matmul is numerically correct on gfx1201 on both backends.** Given that
upstream's docs state published numbers and regression guards assume gfx1151,
this is the first evidence in this program that the NVFP4 kernels are sound on
RDNA4 at all. It says nothing about their speed.

---

## Finding: HIP segfaults on this multi-GPU host without device isolation

**Not an NVFP4 bug. Not a gfx1201 bug.** Recorded because it will otherwise cost
Track B a day.

**Symptom.** With the stock environment, *every* `test-backend-ops` invocation
against *any* ROCm device segfaults (SIGSEGV, rc=139) — including the trivial
`ADD` op, and including the RX 7900 XT.

```
Thread 1 "test-backend-op" received signal SIGSEGV
#0-#4  ?? () from /opt/rocm-7.2.1/lib/libamdhip64.so.7
#5  ggml_cuda_op_bin_bcast<bin_bcast_cuda<&(op_add(float, float)), 1>>(...)
#6  ggml_cuda_op_add(ggml_backend_cuda_context&, ggml_tensor*)
#7  ggml_cuda_graph_evaluate_and_capture(...)
```

The crash is inside the ROCm runtime at the first kernel dispatch.

**Isolation performed.**

| Configuration | Result |
|---|---|
| `-b ROCm0` (gfx1100), `-b ROCm1` (gfx1201), `-b ROCm2` (gfx1036), op `ADD` | all SIGSEGV |
| `-b CPU`, same binary | 99/99 passed |
| Vulkan build, `-b Vulkan1`, op `ADD` | 99/99 passed |
| NVFP4, `q4_0`, `f16`, `mxfp4` type filters | all SIGSEGV — not type-dependent |
| **`HIP_VISIBLE_DEVICES=1`, `-b ROCm0`, op `ADD`** | **99/99 passed** |

**Cause.** The build emits code objects for `gfx1201` only, while three devices
of three different architectures (`gfx1100`, `gfx1201`, `gfx1036`) are visible
to the HIP runtime. Restricting visibility to the R9700 alone resolves it
completely. **INFERRED** from the isolation table; the precise failure inside
`libamdhip64` was not traced further, and the symbols there are stripped.

**Operational consequence for Track B: every HIP run on this host must set
`HIP_VISIBLE_DEVICES=1`,** after which the R9700 becomes `ROCm0`. This must be
captured in the run record (`env`, and `gpu_index`), because the same command
without it does not merely run slower — it crashes.

**Not upstream-reportable as-is.** A single-arch build on a mixed-arch host is a
user configuration, and a multi-arch build was not attempted. Whether ROCm
*should* fail this way is a separate question that would need a reproducer
against stock llama.cpp before it could be attributed to ROCmFPX.

---

## What this does NOT establish

* **No throughput number.** No model was loaded; no decode, prefill, or MTP
  measurement exists for Track B.
* **No NVFP4 model has been run**, or is present locally.
* **Correctness at op level is not correctness end-to-end.** The lm_head scale
  path — which `7b02624` warns can *"silently degrade"* logits by a constant
  factor — is not exercised by `test-backend-ops`, because it lives in
  `llama-graph`, not in a ggml op.

---

## Raw logs

| Description | Path |
|---|---|
| Vulkan configure | `/ai/scratch/rocmfpx-vulkan-configure.log` |
| Vulkan build | `/ai/scratch/rocmfpx-vulkan-build.log` |
| HIP build | `/ai/scratch/rocmfpx-hip-build.log` |
| Vulkan NVFP4 `MUL_MAT` | `/ai/scratch/tbo-nvfp4-mulmat.log` |
| Vulkan NVFP4 extras | `/ai/scratch/tbo-extra.log` |
| HIP NVFP4 | `/ai/scratch/tbo-hip-nvfp4.log` |
| HIP crash | `/ai/scratch/tbo-hip-mulmat.log` |

These are in scratch, not in the repository. Promote any that back a published
claim into `raw/` before citing them.
