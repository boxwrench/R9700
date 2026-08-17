# ROCmFPX Upstream Audit — Native NVFP4 on gfx1201

**Audit date (UTC):** 2026-08-17T03:53:08Z
**Method:** read-only source and history audit of a fresh blobless clone. No source modified, no build, no inference run.
**Audit checkout:** `/ai/scratch/ROCmFPX-audit` (clone of `https://github.com/charlie12345/ROCmFPX.git`, branch `main`)

Every claim below is tagged:

| Tag | Meaning |
|---|---|
| **SOURCE FACT** | Read directly out of the tree at the recorded SHA. Cited by file and line. |
| **COMMIT CLAIM** | Asserted in a commit message or doc. Not independently verified here. |
| **INFERRED** | My reading of how the pieces connect. Not directly stated upstream. |
| **UNKNOWN** | Not determinable from source; needs measurement or an upstream answer. |

---

## 0. Repository identity — a discrepancy to resolve

The work package named `charlie12345/ROCmFPX`. The pre-existing checkout on this
machine points somewhere else.

| | Remote | HEAD at audit time |
|---|---|---|
| Work package / audited here | `github.com/charlie12345/ROCmFPX` | `f4b2c5a3edfd183274641094d0db0fcc8092c0ad` |
| Pre-existing local checkout `/ai/github/ROCmFPX` | `github.com/ciru-ai/ROCmFPX` | `0d313da1849f73c5a7f8c5f7e5b8d7d278fbb69d` |

**SOURCE FACT.** Both remotes resolve and both serve a repository named
`ROCmFPX`, but their `HEAD`s differ. `ciru-ai` additionally publishes ~16 named
branches (`ciru/*`, `agent/*`); `charlie12345` is the one carrying PRs #70–#78.

**UNKNOWN.** Which is canonical, and whether one is a fork of the other. The
local checkout is a depth-1 shallow clone in detached HEAD state at
`a71e6c8 feat(hip): add ActiveFPX PromptForge routes for Qwen3.8-27B`, which has
no history to compare against. **This is a question for the user** (see the
final report). This audit proceeds against `charlie12345`, as specified.

---

## 1. Current upstream HEAD

**SOURCE FACT.**

```
repository : github.com/charlie12345/ROCmFPX
branch     : main
HEAD       : f4b2c5a3edfd183274641094d0db0fcc8092c0ad
subject    : Merge pull request #78 from charlie12345/fix/rocm-werror-completion
date       : 2026-08-16
fetched    : 2026-08-17T03:53:08Z
```

Upstream has moved past every SHA named in the work package. Recent history:

| SHA | Date | Subject |
|---|---|---|
| `f4b2c5a` | 2026-08-16 | Merge PR **#78** — `fix/rocm-werror-completion` |
| `1eb6102` | 2026-08-16 | Merge PR **#77** — `fix/test-scale-tensor-mean` |
| `7e076cb` | 2026-08-16 | Merge PR **#75** — `fix/lm-head-scale-gating` |
| `248376b` | 2026-08-16 | `fix(completion): let the ROCm clang build compile with -Werror` |
| `a087b5d` | 2026-08-16 | `fix(test): centre fabricated scale tensors on 1, not 0` |
| `5466f3b` | 2026-08-16 | `fix: only claim lm_head scales for NVFP4, and save them` |
| `840e343` | 2026-08-16 | Merge PR **#74** — `fix/dflash-unused-params` |
| `30b7019` | 2026-08-16 | Merge PR **#73** — `fix/example-pos-next` |
| `1781bd0` | 2026-08-16 | `fix(examples): pass pos_next in speculative-simple's draft params` |
| `b41ba87` | 2026-08-16 | Merge PR **#72** — `fix/mrope-pos-next-scoped` |
| `7d43016` | 2026-08-16 | `fix(spec): give draft-mtp an M-RoPE position instead of overloading n_past` |
| `98de747` | 2026-08-16 | Merge PR **#71** — `revert-mrope-n_past` |
| `06b38c6` | 2026-08-16 | Revert `fix(spec): pass an M-RoPE position…as draft n_past` |
| `82357de` | 2026-08-16 | Merge PR **#70** — `nvfp4-native-support` |
| `61b71b5` | 2026-08-16 | `fix(spec): pass an M-RoPE position, not a KV token count, as draft n_past` |
| `5290625` | 2026-08-16 | `feat(quantize): expose NVFP4 as a quantize target` |
| `7b02624` | 2026-08-16 | `fix(nvfp4): claim and apply the lm_head scale side-tensors` |

### Status of the named PRs

**SOURCE FACT.** All three work-package SHAs are ancestors of `HEAD`
(`git merge-base --is-ancestor` → true for each).

* **PR #70** (`nvfp4-native-support`) — **MERGED** at `82357de`.
  Title: *"NVFP4: native load + quantize support, and fix vision with draft-mtp"*.
  Content commits: `7b02624` (lm_head scale side-tensors), `5290625` (quantize
  target), `61b71b5` (draft-mtp M-RoPE position).
* **PR #75** (`fix/lm-head-scale-gating`) — **MERGED** at `7e076cb`, content `5466f3b`.
* Three further merges land after #75: **#77** (`a087b5d`) and **#78**
  (`248376b`), both repairs to #70/#75 fallout.

**INFERRED.** The NVFP4 line is four days old at most and is still actively
being repaired — #71 reverts #70's own M-RoPE change, then #72 re-lands it
differently; #75, #77, and #78 all fix consequences of #70's scale handling.
Pinning a SHA for reproduction matters more than usual here.

---

## 2. The ten source questions

### Q1 — Is NVFP4 executed natively, or converted at load/runtime?

**Executed natively. SOURCE FACT.**

NVFP4 is a first-class `ggml` type with its own block layout, not a load-time
conversion to something else.

`ggml/include/ggml.h:430`
```c
GGML_TYPE_NVFP4 = 40, // NVFP4 (4 blocks, E4M3 scale)
```

`ggml/src/ggml-common.h:211-217`
```c
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
```

**36 bytes per 64 weights = 4.50 bpw** (CALCULATED from the struct). Contrast
Track A's IQ4_XS at 136 B / 256 weights = 4.25 bpw and Q5_K at 176 B / 256 =
5.50 bpw.

`ggml/src/ggml.c:794-801` registers full type traits (`to_float`,
`from_float_ref`, `is_quantized = true`), so the type participates in the normal
quantized-tensor machinery rather than being widened at load.

### Q2 — Which backends actually execute NVFP4?

**SOURCE FACT.** NVFP4 appears in the CPU, CUDA/HIP, Vulkan, and SYCL backends:

| Backend | Evidence |
|---|---|
| CPU | `ggml/src/ggml-cpu/quants.c`, `ops.cpp`, `arch/arm/quants.c` |
| CUDA / HIP | `ggml-cuda/{quantize,convert}.cu`, `mmvq.cu`, `mmq.cu`, `template-instances/mmq-instance-nvfp4.cu` |
| Vulkan | `ggml-vulkan.cpp` + `vulkan-shaders/{dequant_nvfp4.comp, dequant_funcs.glsl, dequant_funcs_cm2.glsl, mul_mm_funcs.glsl, types.glsl, copy_from_quant.comp}` |
| SYCL | `ggml-sycl/{convert,mmvq}.cpp`, `vecdotq.hpp`, `dequantize.hpp` |

Metal is **absent** from the NVFP4 file list — **SOURCE FACT**, by omission.

### Q3 — What handles NVFP4 on Vulkan?

**SOURCE FACT.** NVFP4 is wired into every Vulkan matmul family, not just a
dequant fallback. From `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

| Pipeline family | Line | Shader |
|---|---|---|
| `mul_mm` (GEMM, f16 acc) | 4178 | `matmul_nvfp4_f16` |
| `mul_mm` (GEMM, f32 acc) | 4285, 4315, 4443, 4635 | `matmul_nvfp4_f32` |
| `mul_mm_id` (MoE GEMM) | 4215, 4356, 4497, 4550, 4686, 4720 | `matmul_id[_subgroup]_nvfp4_f32/f16` |
| **`mul_mat_vec` (DMMV) f32×f32** | **4820** | **`mul_mat_vec_nvfp4_f32_f32`** |
| `mul_mat_vec` (DMMV) f16×f32 | 4852 | `mul_mat_vec_nvfp4_f16_f32` |
| `mul_mat_vec_id` | 4916 | `mul_mat_vec_id_nvfp4_f32` |
| standalone dequant | 4986 | `dequant_nvfp4` |
| `get_rows` | 5019, 5052 | `get_rows_nvfp4[_f32]` |

**Decode goes through the DMMV path.** `ggml-vulkan.cpp:4820` — **SOURCE FACT**:

```cpp
ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],
    "mul_mat_vec_nvfp4_f32_f32", ..., {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, ...);
```

Three specialization constants — `{BLOCK_SIZE, NUM_ROWS, NUM_COLS}` — and
`rm_iq` rows per workgroup. **This is the same generic `mul_mat_vec.comp`
template, with the same `rm_iq` row count, that Track A profiled for IQ4_XS in
entries 15–16.**

The dequant itself is the generic hook, `dequant_funcs.glsl:642-658` — **SOURCE FACT**:

```glsl
#if defined(DATA_A_NVFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    ...
    return vec2(float(kvalues_mxfp4[qs0]), float(kvalues_mxfp4[qs1])) * d * 0.5;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) { ... }
#endif
```

and `get_dm()` returns `vec2(1.0, 0.0)` (line 713-717), i.e. NVFP4 folds its
scale inside `dequantize()` and carries no separate block min/mult.

**INFERRED, high confidence.** Because NVFP4 uses the generic
`dequantize4()` inside `mul_mat_vec.comp`, it inherits the *identical structural
property* Track A documented in entry 15: the column loop is outer, so at
`NUM_COLS = N` each weight fragment is dequantized `N` times. Track A entry 16
then showed that removing that redundancy on IQ4_XS bought under 0.2 ms because
the kernel was bandwidth-and-latency bound. **Whether that same conclusion holds
for NVFP4 is UNKNOWN and must be re-measured, not assumed** — NVFP4 moves 4.50
bpw vs IQ4_XS's 4.25, and `ue4m3_to_fp32` is a LUT lookup rather than IQ4_XS's
arithmetic unpack, so both the byte count and the ALU cost differ.

`ggml-vulkan.cpp:3462-3463` — **SOURCE FACT** — notes NVFP4's LUT budget:
*"Same kvalues budget as MXFP4 plus `ue4m3_fp32_lut[128]` (types.glsl, DATA_A_NVFP4)."*

### Q4 — What handles NVFP4 on HIP?

**SOURCE FACT.** llama.cpp's HIP backend *is* `ggml-cuda` compiled through
hipify, so the CUDA sources are the HIP implementation. NVFP4 is present in all
three matmul strategies:

* **MMVQ** (quantized vec-dot, decode): `ggml-cuda/mmvq.cu:162`
  `case GGML_TYPE_NVFP4: return vec_dot_nvfp4_q8_1;`, with
  `VDR_NVFP4_Q8_1_MMVQ` (line 202) and a dispatch at line 1387.
* **MMQ** (quantized GEMM, prefill): dedicated instantiation
  `ggml-cuda/template-instances/mmq-instance-nvfp4.cu`.
* **dequant/convert**: `convert.cu`, `quantize.cu`.

Note this is a real divergence from Vulkan. On RADV the R9700 reports
`integer_dot_product = false`, which Track A established rules out MMVQ for
*every* type; the HIP path has no such restriction, so **HIP may reach NVFP4
through a Q8_1 vec-dot that Vulkan structurally cannot use**. That makes
backend choice a genuinely open Track B variable rather than a settled one.
**INFERRED**; to be measured in B1/B4.

### Q5 — Are gfx1151 and gfx1201 routed identically?

**No. SOURCE FACT.** They are different architecture classes in the HIP backend.

`ggml/src/ggml-cuda/common.cuh:100-109`
```c
#define GGML_CUDA_CC_RDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x1200) // RX 9000
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
```

gfx1151 (Strix Halo) satisfies `IS_RDNA3_5`; gfx1201 (Navi 48 / R9700) satisfies
`IS_RDNA4`. They take **different branches** at every `GGML_CUDA_CC_IS_RDNA4`
site, including the MMVQ parameter-table selection at `mmvq.cu:245` and
`mmvq.cu:442`.

This matters directly, because upstream's own docs say
(`docs/BUILD-AMD-ARCHITECTURES.md:62-63`) — **SOURCE FACT**:

> Published benchmark numbers and regression guards assume **Strix Halo / `gfx1151`**.

**INFERRED, and the single most important finding of this audit:** the NVFP4
work was developed and measured on gfx1151, which is *not* the code path
gfx1201 takes on HIP. gfx1201 NVFP4 is therefore effectively **untested
upstream**, and Track B's B1 reproduction is genuinely novel measurement rather
than a confirmation exercise.

### Q6 — Are there gfx1201-specific branches or heuristics?

**SOURCE FACT.** There are **no gfx1201-specific *compute* heuristics** in the
NVFP4 kernels. `gfx1201` appears only in build tooling and documentation:

* `scripts/build-rdna4.sh` — selects the target via
  `rocmfpx_select_hip_arch gfx1200 '^gfx120[01]$'`
* `scripts/rocmfpx-hip-arch.sh:64` — code-object verification after build
* `scripts/build-rocmfp4-rocm714-local.sh:21` — `HIP_ARCH="gfx1200;gfx1201"`
* `docs/BUILD-AMD-ARCHITECTURES.md` — the target table and the hazard below

Everything gfx1201 does at runtime, it does as generic `RDNA4`.

**A documented correctness hazard, SOURCE FACT** (`docs/BUILD-AMD-ARCHITECTURES.md:57-59`):

> **RDNA4 is the exception: `gfx1201` is not interchangeable with `gfx1200`.** A
> `gfx1200` build on a `gfx1201` card links and loads a model, then segfaults or
> hangs with no error message (issues #18 and #37).

`scripts/build-rdna4.sh` autodetects Navi 48 and verifies emitted code objects
against the requested target, failing loudly on mismatch. **B0 must record the
actual emitted code-object arch, not just the requested one.**

### Q7 — Which NVFP4 tensors carry scale / input_scale side tensors?

**SOURCE FACT.** Two families, both optional (`TENSOR_NOT_REQUIRED`), created in
`src/llama-model.cpp`.

*Per-layer* (`.scale` and `.input_scale`, shape `{1}`, or `{n_expert}` for MoE),
lines ~1300–1421 — attention `Q/K/V/OUT/QKV/GATE`, FFN `GATE/DOWN/UP`, MoE
`*_EXPS`, shared-expert `*_SHEXP`, and recurrent/linear-attention
`SSM_IN/OUT/ALPHA/BETA`.

*lm_head* (non-layer, added by `7b02624`, gated by `5466f3b`),
`src/llama-model.cpp:1426-1450` — **SOURCE FACT**:

```cpp
// Only ask for them when lm_head is itself a scale-carrying quant. The
// request must not be unconditional: callers that synthesize a model
// rather than read a file (llama_model_init_from_user, as used by
// test-llama-archs) materialize every tensor that is asked for, so an
// unconditional request invents a random lm_head scale for models that
// have none.
const bool output_has_scales = output && output->type == GGML_TYPE_NVFP4;
if (!output_s    && output_has_scales) { output_s    = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"),       {1}, TENSOR_NOT_REQUIRED); }
if (!output_in_s && output_has_scales) { output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED); }
```

The scale is applied as a post-matmul multiply, `src/llama-graph.cpp:1188-1192`
— **SOURCE FACT**:

```cpp
ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
if (w_s) { res = ggml_mul(ctx0, res, w_s); }
```

**INFERRED, and relevant to Track A entry 17.** Every scale-carrying NVFP4
matmul emits an *extra elementwise `GGML_OP_MUL` node* after the matmul. Track A
entry 17 measured exactly this class of node — MUL_MAT followed by a separate
elementwise op — and found the fusion predicate is gated on
`ggml_nrows(mul) != 1`, so it does not fuse at N>1. Whether NVFP4's scale
multiplies fuse on gfx1201 is **UNKNOWN** and is a concrete B3 question.

**COMMIT CLAIM** (`7b02624`): without threading `model.output_s` to the logits
matmul *"the logits would be off by a constant factor and silently degrade."*
Correctness-relevant; worth an explicit B1 sanity check.

### Q8 — Does MTP use the same NVFP4 execution machinery?

**Yes. INFERRED, high confidence.**

**SOURCE FACT** on the surrounding structure: `draft-mtp` is one speculative
implementation among several (`common/speculative.cpp:29`,
`common_speculative_state_draft_mtp` at :1322), and the MTP block is exposed as
ordinary model layers via `hparams.nextn_predict_layers`
(`src/llama-model.cpp:2305-2306`, and the `n_layer - nextn_predict_layers` split
at :2098, :2162, :2166, :2173).

**INFERRED.** Because the nextn layers are built by the same
`build_lora_mm` / `build_ffn` helpers and NVFP4 execution is dispatched on
`ggml_tensor::type` inside `ggml_mul_mat`, the MTP block runs through identical
NVFP4 kernels. There is no MTP-specific NVFP4 code path.

**SOURCE FACT — NVFP4 has hard restrictions in `build_ffn`**
(`src/llama-graph.cpp:1381-1402`):

```cpp
// NVFP4 support is currently restricted to
// 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
// 2) bias absense (*_s would be applied after bias addition, which is incorrect)
// TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
...
GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));
```

These are **assertions, not graceful fallbacks**: a scale-carrying NVFP4 FFN
combined with a bias or a LoRA will **abort**. Track B must therefore not apply
LoRA adapters, and any Qwen3.8 FFN bias would be a hard blocker. Record this in B0.

Separately, `61b71b5` / `06b38c6` / `7d43016` show the draft-mtp position
handling was changed, reverted, and re-landed within PR #70's window —
**a freshly churned area** directly under Track B's feature of interest.

### Q9 — What model types / architectures are explicitly tested?

**SOURCE FACT.** NVFP4 test coverage in-tree is thin and type-level, not
model-level:

* `tests/test-backend-ops.cpp:7586, 7604` — NVFP4 is in the quantized-type
  sweep lists alongside `MXFP4`, `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`.
* `tests/test-backend-ops.cpp:3981, 4170` — a `BLACKWELL_NATIVE_FP4` backend-feature
  guard for MXFP4/NVFP4. **INFERRED:** NVIDIA-Blackwell-specific; will not
  trigger on RADV or on ROCm/RDNA4.
* `tests/test-quantize-fns.cpp:161, 186` — NVFP4 uses the looser
  `MAX_QUANTIZATION_TOTAL_ERROR_FP4` threshold.
* `tests/test-quant-type-selection.cpp:59` — CLI name `"NVFP4"` maps to
  `LLAMA_FTYPE_MOSTLY_NVFP4`.
* `tests/test-llama-archs` — exercises the scale-claiming path (that is what
  `5466f3b` regressed and repaired), but with *synthesized* models.

**SOURCE FACT.** The only real model named anywhere in the NVFP4 commits is
`RadixArk/Qwen3.8-27B-NVFP4`, cited in both `7b02624` and `5290625`.

**UNKNOWN.** There is no in-tree end-to-end NVFP4 model test. Nothing verifies
NVFP4 decode correctness against a reference on any backend or architecture.

### Q10 — Which upstream benchmark claims are source-backed?

**None are measurements this audit can verify.** All are **COMMIT CLAIM** or
doc claim, and — critically — **all are on gfx1151, not gfx1201**.

From `5290625` (`feat(quantize): expose NVFP4 as a quantize target`) —
**COMMIT CLAIM**:

> Measured on RadixArk/Qwen3.8-27B-NVFP4 (gfx1151, Vulkan):
> 28.2 GB mixed -> 15 GB uniform 4-bit; all 193 original NVFP4 tensors verified
> bit-exact (0 changed, 0 missing); decode 18.3 -> 29.5 t/s with MTP.

Reading this carefully: **18.3 → 29.5 t/s is a mixed-precision → uniform-4-bit
comparison on gfx1151**, i.e. the gain is attributed to shrinking the BF16
remainder, *not* to NVFP4 kernel speed. It is **not** an R9700 number and **not**
an NVFP4-vs-Q4_K_XL comparison. It must not be quoted as either.

From `7b02624` — **COMMIT CLAIM**: *"Verified on RadixArk/Qwen3.8-27B-NVFP4
(Vulkan, gfx1151): previously failed to load, now loads and generates
correctly."* A correctness claim, not a performance one.

From `docs/BUILD-AMD-ARCHITECTURES.md:62` — **SOURCE FACT** (a doc statement
about scope): *"Published benchmark numbers and regression guards assume Strix
Halo / gfx1151."*

From `README.md:544-547` — **COMMIT/doc CLAIM**, and about ROCmFP4 rather than
NVFP4: a 9B NVFP4→ROCmFP4 conversion on gfx1151 landed "within noise" of source
perplexity at 4.50 bpw.

**UNKNOWN — and this is the load-bearing gap.** The work package refers to a
separate *"~72 tok/s R9700 result"*. **Nothing in this tree substantiates it.**
No gfx1201 throughput figure, no R9700 benchmark, and no model identification
for such a run appears in the source, the docs, or the commit messages I read.
Its model, backend, quantization, and MTP settings are all unknown. It must be
treated as an unverified external report, and Track B must not adopt it as a
target to reproduce until the user supplies its provenance.

---

## 3. Build, toolchain, and runtime requirements

**SOURCE FACT**, from `docs/BUILD-AMD-ARCHITECTURES.md` and `scripts/`:

* **HIP build for R9700:** `CMAKE_HIP_ARCHITECTURES=gfx1201 env JOBS=16 scripts/build-rdna4.sh`.
  Plain `scripts/build-rdna4.sh` autodetects Navi 48 and selects `gfx1201`; the
  default when no RDNA4 GPU is visible is `gfx1200`, which is the hazardous value.
* `build-rdna4.sh` delegates to `scripts/build-rocmfp4.sh` with
  `BUILD_DIR=build-rdna4`.
* Changing target **wipes** the build directory unless `ROCMFPX_KEEP_STALE_BUILD=1`.
* Post-build code objects are compared against the requested target; mismatch fails the build.
* ROCm must ship device libraries for the target.
* **Vulkan-only path exists and needs no HIP arch** — referenced as
  "[Vulkan-only (no HIP arch needed)]" for when HIP is not ready.

**Local toolchain, measured on this machine 2026-08-17 — SOURCE FACT:**

```
kernel : Linux 7.0.0-28-generic (Ubuntu 24.04.4 LTS)
CPU    : AMD Ryzen 7 9800X3D (8C/16T)
RAM    : 186 GiB
ROCm   : 7.2.1  (hipconfig 7.2.53211-e1a6bc5663, /opt/rocm-7.2.1)
Vulkan : RADV, Mesa 25.2.8-0ubuntu0.24.04.2, API 1.4.318
GPUs   : [0] Radeon RX 7900 XT (RADV NAVI31)
         [1] AMD Radeon AI PRO R9700 (RADV GFX1201)   <-- target
         [2] AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)
         [3] llvmpipe
```

**Device isolation is mandatory.** The R9700 is **not** device 0 under Vulkan.
Track A established that both Vulkan and HIP enumerate the 7900 XT first;
every Track B run must pin the device explicitly and record which index it used.

---

## 4. Consequences for the Track B plan

1. **B1 is novel measurement, not confirmation.** Upstream's NVFP4 numbers are
   gfx1151 (RDNA3.5); gfx1201 is RDNA4 and takes different HIP branches. Nothing
   upstream reports NVFP4 on an R9700.
2. **Pin the SHA.** The NVFP4 line is days old and three of the last five merges
   repair it. Record and freeze `f4b2c5a` (or a deliberately chosen successor).
3. **Resolve the remote question before B0.** `charlie12345` vs `ciru-ai` HEADs differ.
4. **Backend choice is a real open variable.** HIP has an MMVQ NVFP4 vec-dot;
   Vulkan on RADV cannot use MMVQ at all. Track A's "Vulkan beats HIP" result
   was established for Q4_K/IQ4_XS and does **not** transfer.
5. **Verify the code-object arch,** given the documented silent gfx1200/gfx1201 hazard.
6. **No LoRA, and check for FFN bias** — `build_ffn` *asserts* rather than falls back.
7. **Do not quote 18.3 → 29.5 t/s** as an NVFP4 speedup. It is a
   mixed→uniform-4-bit size reduction on different silicon.
8. **The ~72 tok/s R9700 figure is unsubstantiated here** and is not a Track B target.

---

## 5. Audit provenance

```yaml
audit_date_utc:     2026-08-17T03:53:08Z
audited_repo:       https://github.com/charlie12345/ROCmFPX.git
audited_branch:     main
audited_sha:        f4b2c5a3edfd183274641094d0db0fcc8092c0ad
audit_checkout:     /ai/scratch/ROCmFPX-audit   (blobless clone, read-only)
source_modified:    none
build_attempted:    see reproduction/ build log
inference_run:      none
auditor_note:       All line numbers are relative to f4b2c5a. Re-verify after any rebase.
```
