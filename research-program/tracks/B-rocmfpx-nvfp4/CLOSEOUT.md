# Track B Closeout — ROCmFPX / Native NVFP4

**Status:** `REPRODUCED / CHARACTERIZED / DEPRIORITIZED`
**Date:** 2026-08-17

---

## Research question

Does ROCmFPX native NVFP4 provide a sufficiently compelling new Qwen3.8-27B
inference foundation on Radeon AI PRO R9700 / gfx1201 to **replace** the
established Track A Vulkan / Q4_K_XL optimization line?

## Result

**Native NVFP4 execution on gfx1201 is confirmed.**

The exact upstream `RadixArk/Qwen3.8-27B-NVFP4` checkpoint was downloaded,
converted to GGUF, loaded through ROCmFPX, executed through Vulkan/RADV on the
R9700, validated for NVFP4 `lm_head` scale application, and exercised with both
serial and native-MTP inference. A uniform NVFP4 derivative was also produced
and benchmarked.

Evidence: [B1 gates](reproduction/2026-08-17-b1-gates.md) ·
[B1 results](reproduction/2026-08-17-b1-results.md) ·
[raw logs](raw/b1-2026-08-17/)

## B1 measured performance

| | MIXED | UNIFORM |
|---|---:|---:|
| Model bytes | 28,230,539,776 (28.23 GB) | 15,547,030,016 (15.55 GB) |
| Composition | 193 NVFP4 + higher-precision remainder | 505 NVFP4 + 1 Q5_K, ≈4.55 BPW |
| Serial decode | 20.32 tok/s | **27.33 tok/s** |
| Native MTP | 30.71 tok/s | **37.26 tok/s** |
| MTP multiplier | 1.511 | 1.363 |
| Serial VRAM | 25.22 GB | 14.72 GB |
| MTP VRAM | 26.30 GB | 15.80 GB |

**The uniform derivative is the preferred ROCmFPX NVFP4 inference artifact.**

## Established findings

1. ROCmFPX genuinely loads and executes the checkpoint's native NVFP4 weight
   representation on AMD gfx1201.
2. The 193 original NVFP4 tensors remain **bit-exact** through the
   uniformization process.
3. NVFP4 `lm_head` scaling is actually applied at runtime.
4. Quantizing the remaining higher-precision tensors to NVFP4 materially
   improves speed and memory footprint inside ROCmFPX.
5. The resulting uniform model is a **derived inference quant** and must not be
   described as bit-identical to the released mixed checkpoint.
6. The source checkpoint contains **W4A4 calibration metadata**, but the
   observed llama.cpp/ROCmFPX execution does not consume the activation input
   scales. Observed runtime behavior is therefore effectively **W4A16**.
   ([detail](upstream-audit/2026-08-17-w4a4-vs-w4a16.md))
7. Native MTP greedy output **diverges** from serial target output under
   deterministic decoding (`temp=0.0`, `top_k=1`). `--spec-mtp-strict-qwen` does
   not eliminate the first observed divergence in the tested configuration — in
   the traces, the mitigation becomes active only *after* the initial divergence
   has already occurred. **Cause: `UNRESOLVED`.** This must not be attributed to
   the Track A recurrent-state / K3 issue; that relationship is an unverified
   hypothesis.

## Cross-track interpretation

Track A historical performance is approximately **~29.4 tok/s serial** and
**~53 tok/s native MTP**.

> [!IMPORTANT]
> These values were obtained under a different pinned implementation and
> benchmark configuration and are therefore **NOT a formal matched B2
> comparison.**

Nevertheless, B1 provides no evidence of an advantage large enough to justify
replacing Track A with ROCmFPX/NVFP4 as the primary optimization program. A
formal B2 comparison remains available if needed.

## ROCmFP4 FAST note

The separately observed **~72.4 tok/s** R9700 result from Charlie was reported
for **ROCmFP4 FAST**, not the native NVFP4 configuration tested in B1. Its exact
model and configuration have not yet been reproduced.

ROCmFP4 / ROCmFP4 FAST involves a **different, lossy quantization path** and must
not be treated as equivalent to executing the released NVFP4 checkpoint. This
supersedes the earlier framing of the ~72 tok/s figure as an unexplained gap
against B1 — the two numbers were never measuring the same thing.

Reopen this branch if the exact ~72 tok/s model and configuration become
available.

## Decision

**Do not continue** NVFP4 kernel optimization, W4A4 activation work, quality
benchmarking, or integration of Track A optimizations at this time.

**Return primary optimization effort to Track A.**

Preserve Track B as:

* a successful gfx1201 native-NVFP4 reproduction
* a reference implementation / benchmark
* a potential source of upstream ROCmFPX findings
* a branch that can be reopened if materially new performance evidence appears

## Reopen conditions

Reopen Track B if any of the following occurs:

1. A comparable native-NVFP4 result materially exceeds Track A.
2. Charlie provides the reproducible ~72 tok/s R9700 configuration.
3. ROCmFPX adds a substantially faster gfx1201 NVFP4 kernel.
4. True W4A4 activation execution becomes available.
5. A model of interest is distributed **only** in NVFP4 and preserving that
   representation becomes a project requirement.
