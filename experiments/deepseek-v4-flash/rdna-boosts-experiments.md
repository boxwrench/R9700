# RDNA llama.cpp optimization experiments

## Purpose and provenance

This is the hardware-wide companion to `optimization-plan.md`. It catalogs
reusable experiments from Stew Forster's llama.cpp `rdna-boosts` branch without
mixing them into the controlled DeepSeek V4 campaign.

Initial source pin:

- Fork: <https://github.com/stew675/llama.cpp/tree/rdna-boosts>
- Head: `ed89854b2aeb0e333dd61424f14af2aedaca126e` (2026-08-13)
- Upstream base: `a94d563ed801d1da1b8c2432946de07d0231bb3d`
- Reviewed delta: 23 commits, 19 files, approximately 1,925 insertions and 294
  deletions, concentrated in llama.cpp's CUDA/HIP backend

The branch is experimental and its published commit measurements are mostly on
Qwen and Gemma, not this workstation's complete workload set. A commit message
is evidence that a path was measured by its author, not proof of a gain on our
models. Preserve an upstream binary, pin every candidate, and require correctness
before timing.

## Shared experiment rules

For each card and workload:

1. Build an upstream HIP control and fork HIP candidate from the same base with
   the same compiler and CMake configuration.
2. Compile only the intended GPU target: `gfx1201` for the R9700 or `gfx1100`
   for the RX 7900 XT.
3. Save binary/source hashes, ROCm version, complete command, model and quant,
   context occupancy, KV formats, batch sizes, GPU clocks/power, and thermals.
4. Run backend correctness tests, then deterministic output or perplexity checks.
5. Measure prompt processing, target-only decode, speculative total decode where
   relevant, VRAM, host RAM, PCIe traffic, power, and at least three repetitions.
6. Treat less than 3% as noise until a longer run confirms it. Reject crashes,
   NaNs, incorrect backend tests, quality regression, or unexplained fallback.

Use separate build and result directories per GPU and commit. Never replace the
known-good Vulkan server or share a status file between architectures.

## RDNA4 / R9700 (`gfx1201`) cards

### R4-1: RDNA4 WMMA Flash Attention

Relevant commit: `18fc188f`.

Useful for long-context prefill and attention-heavy dense or MoE models,
especially head sizes above 128. The branch enables its RDNA4 WMMA path through
head size 576 and reports direct `gfx1201` verification.

Compare fork default against `GGML_CUDA_FA_WMMA_256=0`, using F16, BF16, and the
deployment's normal quantized KV format where supported. Sweep short, 8K, 32K,
and the largest safe context. Include models with head sizes 128, 256, 320, and
512 when locally available; do not extrapolate one head shape to all models.

### R4-2: native BF16 Flash Attention

Relevant commits: `cb05a6ae` through `24b08992`, plus `444fea6b`.

Useful when BF16 KV precision or deep-context numerical behavior matters. Test
BF16 against F16 and the currently deployed Q8 KV at the same actual context
occupancy. Record perplexity or deterministic quality as well as speed: the
branch deliberately changes accumulation and intermediate precision.

### R4-3: Q6_K prefill and decode

Relevant commits: `18fc188f` and `d4aaf63b`.

Useful for Q6_K dense models and Q6_K non-expert tensors. Separate pp512/prompt
throughput from one-token decode because the fork reports a large prefill change
but only a modest memory-bound decode gain. This card does not justify changing
the DeepSeek target quant.

### R4-4: quantized decode launch reduction

Relevant commits: `ff6fde50`, `9b279620`, `5024d787`, and `ba9e339e`.

Useful for Q8_0, MXFP4, Q4/Q5/Q6, and other mmvq-eligible weights when several
projections share an activation. Compare per-token kernel count and GPU time in
addition to tok/s. Candidate workloads include Qwen 3.5/3.6, DeepSeek V4,
GPT-OSS, and other projection-heavy models.

### R4-5: MoE graph fusions

Relevant commits: `2f0d3c56` and `817cb6ba`.

Useful for models with shared experts and routed expert weighting. Test one
small MoE that fits fully in VRAM and one hybrid CPU/GPU MoE. The full-VRAM case
isolates kernel work; the hybrid case reveals whether host expert time hides the
gain. Require bit-identical or quality-equivalent output for the chosen sampling
mode.

### R4-6: SSM/gated-delta-net fusion bundle

Relevant commits: `36270950`, `9961c06d`, and `3666525e`.

Useful for Qwen 3.5 MoE and related hybrid SSM architectures. It is not a
DeepSeek V4 card. Compare kernel counts, decode latency, and backend tests with
the bundle applied, then bisect individual commits only if correctness or
attribution requires it.

### R4-7: BF16 IMRoPE/KV write fusion

Relevant commit: `6e478a11`.

Useful for multimodal models using IMRoPE with BF16 KV. Test image/video prompt
prefill separately from text decode and verify multimodal output correctness.

### R4-8: single-GPU graph-wrapper overhead

Relevant commit: `a6e774d4`.

Useful only when a single-GPU invocation would otherwise create llama.cpp's Meta
device, notably tensor-split paths. Compare graph launches with and without the
wrapper under identical placement. A normal `--split-mode none` run may receive
no benefit and should not be presented as a positive result.

## RDNA3 / RX 7900 XT (`gfx1100`) cards

The RX 7900 XT is a discrete RDNA3 GPU, not the `gfx1151` RDNA3.5 APU used for
several branch measurements. Compile explicitly for `gfx1100` and validate each
path. Do not enable both workstation GPUs during these single-card tests unless
a card explicitly studies multi-GPU behavior.

### R3-1: native BF16 Flash Attention at supported head sizes

Relevant commits: `cb05a6ae` through `24b08992`, plus `444fea6b`.

The source includes RDNA3 BF16 dot-product support and a native-BF16 tile path
covering all tested head sizes. Its separate WMMA Flash Attention expansion is
limited to head size 128 on RDNA3/3.5; the larger head-size expansion is RDNA4
only. Test head sizes 64, 112, and 128 first, then exercise a larger head size as
a BF16-tile test without attributing it to the RDNA4 WMMA work. Compare BF16,
F16, and Q8 KV for long-context quality, prefill, decode, and VRAM.

### R3-2: generic quantized decode fusions

Relevant commits: `ff6fde50`, `9b279620`, `5024d787`, and `ba9e339e`.

Test Q8_0 and one Q4/Q6 model that fits substantially or entirely in 20 GiB.
Collect kernel counts so a neutral end-to-end result can be distinguished from
an inactive pattern. This is the best general-purpose RX 7900 XT inference card
in the patch series.

### R3-3: MoE shared-expert and top-k fusions

Relevant commits: `2f0d3c56` and `817cb6ba`.

Use a smaller MoE that fits the RX 7900 XT before attempting a system-RAM hybrid.
Then test one hybrid placement to quantify the PCIe/DDR5 ceiling. These fusions
are plausible on `gfx1100`, but the commit messages do not provide RX 7900 XT
measurements, so label results unverified until our correctness suite passes.

### R3-4: SSM/gated-delta-net workloads

Relevant commits: `36270950`, `9961c06d`, and `3666525e`.

Use Qwen 3.5 MoE or another architecture that actually emits the fused graph
patterns. Do not benchmark this bundle with a conventional transformer and then
conclude that it is ineffective.

### R3-5: Q6_K decode portability

Relevant commit: `d4aaf63b`.

The kernel change is generic, but its recorded validation is on `gfx1201`.
Run `test-backend-ops` for Q6_K first, followed by a long enough decode to expose
the expected memory-bandwidth limit. Promote only an RX 7900 XT result, not the
author's R9700 number.

### R3-6: RX 7900 XT as a dedicated speculative-draft device

This is a system integration experiment, not a claim made by the fork. After
the single-card RDNA3 cards pass, evaluate whether a HIP DSpark or other small
drafter on the RX 7900 XT can overlap usefully with an R9700 target. First prove
that llama.cpp can assign target and draft contexts to explicit HIP devices
without peer-copy or global-device-selection errors. Compare:

1. R9700 target plus R9700 draft control;
2. R9700 target plus RX 7900 XT draft;
3. target-only control.

Record target stalls, draft latency, acceptance, PCIe peer traffic, aggregate
power, and total verified tok/s. Stop if separate devices serialize through the
host or the draft cannot outrun target verification. Do not mix this with the
R9700-only DeepSeek promotion gate.

## Explicit non-candidates

- `9f807c21` adds a `gfx1151`-specific Q8_0 mmvq table; it does not target either
  the `gfx1201` R9700 or `gfx1100` RX 7900 XT.
- `85a9069a` works around integrated-GPU host buffers on HIP; both workstation
  cards are discrete GPUs.
- `ed89854b` repairs tensor-parallel split granularity for three or more GPUs;
  it is not a single-card speed optimization.
- The RDNA4 WMMA head-size expansion in `18fc188f` is not an RX 7900 XT upgrade.
- SSM-specific commits are not DeepSeek V4 improvements.

## Suggested order

Run R4-1, R4-4, and R4-5 first on the R9700 because they overlap the most useful
DeepSeek and general-model paths. Run R3-2 and R3-1 first on the RX 7900 XT.
Only after both cards have independent correctness and performance baselines
should R3-6 attempt a two-GPU speculative pipeline.
