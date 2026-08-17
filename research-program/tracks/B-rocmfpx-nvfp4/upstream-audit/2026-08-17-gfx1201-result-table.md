# ROCmFPX NVFP4 on gfx1201 (Radeon AI PRO R9700) — measured baseline

Prepared for possible sharing with ROCmFPX upstream. **Not submitted. Not pushed.**

Upstream's published NVFP4 numbers and regression guards assume Strix Halo /
`gfx1151` (`docs/BUILD-AMD-ARCHITECTURES.md:62`). These are independent
measurements on **Navi 48 / gfx1201**, which follows the `GGML_CUDA_CC_IS_RDNA4`
path rather than `RDNA3_5`. They are offered as a data point on hardware
upstream does not appear to test, not as a correction to anything.

## Setup

| | |
|---|---|
| Model | `RadixArk/Qwen3.8-27B-NVFP4` → GGUF (`convert_hf_to_gguf.py`, `--outtype bf16`) |
| Uniform | `llama-quantize --pure --token-embedding-type q5_K … NVFP4` |
| ROCmFPX | `f4b2c5a3edfd183274641094d0db0fcc8092c0ad` |
| GPU | AMD Radeon AI PRO R9700, gfx1201, 34 GB |
| Backend | **Vulkan / RADV**, Mesa 25.2.8 (ROCm 7.2.1 present but unused here) |
| Kernel | Linux 7.0.0-28-generic |
| Config | `-c 4096 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0 --temp 0 --seed 1234 -n 128 --ignore-eos` |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0.0 --spec-draft-p-split 0.10` (your `check-rocmfp4-qwen-mtp-regression.sh` values, untuned) |
| Protocol | 1 warmup + 7 measured reps per arm, 32 runs, all rc=0 |

## Results — mean ± sample stdev, n=7

| | MIXED (28.2 GB) | UNIFORM (15.5 GB) |
|---|---|---|
| Serial decode tok/s | 20.32 ± 0.03 | **27.33 ± 0.02** |
| MTP decode tok/s | 30.71 ± 0.04 | **37.26 ± 0.09** |
| MTP multiplier | 1.511 | 1.363 |
| Serial PP tok/s | 135.91 ± 1.71 | 193.96 ± 6.13 |
| VRAM serial (MiB, device) | 25,222 | 14,718 |
| Accepted drafts / round | 1.442 | 1.470 |
| p0 / joint-p1 / cond-p1 | 0.635 / 0.442 / 0.696 | 0.725 / 0.392 / 0.541 |
| acceptance per position | 0.635, 0.442, 0.250, 0.115 | 0.725, 0.392, 0.216, 0.137 |

**Uniform NVFP4 conversion is worth +34.5% serial decode and +21.3% MTP decode
on gfx1201, and saves 10.5 GB.** Every acceptance counter had stdev exactly 0
across all 7 reps.

## Confirmations of your claims, independently reproduced

* **All 193 original NVFP4 tensors survive `llama-quantize` bit-exact.** Verified
  by SHA256 of raw tensor data, mixed vs uniform: 193 bit-exact, 0 different,
  and all 386 `.scale`/`.input_scale` tensors identical. (Your `5290625`.)
* Tensor count 1252 as expected by `7b02624`; 28.2 GB → 15.5 GB at 4.55 BPW.
* **The gfx1201 code-object guard works on real Navi 48** — a HIP build prints
  `Verified gfx1201 code objects in libggml-hip.so`.
* NVFP4 is numerically correct on gfx1201 on both backends via
  `test-backend-ops`: Vulkan `MUL_MAT` 26/26, `MUL_MAT_ID` 73/73, `GET_ROWS`
  4/4, `CPY` 3/3; HIP `MUL_MAT` 41/41, `MUL_MAT_ID` 73/73.
* The lm_head scale is loaded and applied correctly on Vulkan — a graph dump
  shows `result_output = MUL(MUL_MAT(output.weight, result_norm), output.scale)`
  taking logits from ~8.4e4 to ~10.7, matching the stored scale `0.00012716`.

## Observations that may be worth your attention

Offered as observations, not bug reports — each is scoped to what was actually
measured.

**1. Greedy MTP output diverges from serial on Vulkan, and
`--spec-mtp-strict-qwen` does not close the gap.**
At `-n 128` the flag changes nothing at all — byte-identical output, identical
counters, same throughput. Your `server-context.cpp:2485` comment explains why:
the mitigation caps drafts at 256-cell KV block boundaries, and a 64+128 = 192
position run never straddles one. At `-n 400` the flag does take effect
(strict vs default diverge at char 2356), **but the first MTP/serial divergence
is at char 1017 in both cases** — before any block straddle. So there appears to
be a second divergence source on Vulkan that the block-boundary cap does not
address. We have not isolated it and are not claiming a cause; the comment's
reduction-width mechanism is described for ROCm, and RADV's kernel shapes differ.

**2. The MTP draft head does not apply the lm_head scale.**
`src/models/qwen35.cpp:227` passes `model.output_s` into the target head;
`:629` calls `build_lora_mm(head_w, cur)` with no scale. With no
`nextn.shared_head_head` in the checkpoint, `head_w` resolves to `model.output`,
which is NVFP4 and does carry a scale. This is inert under greedy sampling with
`p-min 0.0` — the scale is a positive scalar and argmax is scale-invariant — so
it does not affect the numbers above. It may not be inert under your own
reference settings (`--temp 0.2 --top-k 20 --top-p 0.9`), where unscaled logits
of ~8.4e4 would saturate the softmax. We have not measured that case.

**3. Prompt processing is ~21% slower with MTP enabled** on both models
(135.91 → 114.60 and 193.96 → 152.96 tok/s). Not investigated.

**4. `.input_scale` tensors are loaded but never used by any graph.** This is
presumably intentional — it is what makes a W4A4 ModelOpt checkpoint execute as
W4A16 — but it may be worth a note, since the model card advertises W4A4 and the
two labels look contradictory until you find that activations stay in f32.

## Not claimed

* Nothing here measures the ROCm/HIP backend. A single-arch gfx1201-only HIP
  build segfaults on this mixed-GPU host unless `HIP_VISIBLE_DEVICES` restricts
  visibility; that is a local configuration matter, not an upstream issue.
* We could not reproduce, and make no claim about, a ~72 tok/s R9700 figure.
  Our fastest measured arm is 37.26 tok/s.
