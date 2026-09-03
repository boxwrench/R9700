# FastH3 four-forward + FastVideo Video Sparse Attention on R9700 / gfx1201 — 2026-09-02

**Status: ACCEPTED 2026-09-02.** Bring-up is complete and the configuration is
accepted for normal use. Optimization and benchmarking are closed — this is no
longer an active experiment.

Production H3 in `production/` remains Turbo v4 FP8 with pre-sampler Qwen offload
and is unchanged; ACCEPTED here means accepted for use, not promoted through the
update gate. The frozen experiment and the VSA patch are preserved as-is.

- AMD Radeon AI PRO R9700, `gfx1201`, 32 GiB (31.86 GiB reported), 32 CU
- Ubuntu 24.04.4 LTS, kernel `7.0.0-28-generic`, Mesa 25.2.8
- ROCm 7.2.1, HIP `7.2.53211-e1a6bc5663`
- PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, Triton `3.5.1+rocm7.2.1.gita272dfa8`
- ComfyUI `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa` (0.33.2), isolated copy on port 8191

## Headline result

FastH3's four-forward distillation **and** FastVideo's Video Sparse Attention run
on gfx1201, with the transformer fully resident and no denoiser offload.

| Lane | Sampling | s/forward | Warm wall | Peak VRAM |
|---|---:|---:|---:|---:|
| H3 Turbo v4 FP8 (current production model path) | 48 s | 12.18 | 71.48 s | 24.82 GiB |
| FastH3 four-forward, dense attention | 34 s | 8.69 | 71.02 s | 26.58 GiB |
| **FastH3 four-forward + VSA @ topk 0.10** | **26 s** | **6.55** | **50.94 s** | **27.82 GiB** |

Sampling is **1.86x faster than Turbo v4**; VSA contributes **1.33x** of that on
top of the four-step schedule. Both effects are real and separable. These are
medians of three warm runs each (section 8); the single-run figures this record
originally carried have been superseded.

Workload: 864x480, 124 frames, 24 fps, seed 8112026, 4 steps, `euler`/`simple`,
CFG-free guider, the standard brass-robot prompt, FP8 Qwen3-VL encoder held
constant across every lane.

## Architecture — no FastVideo quantized loader was required

The initial assumption was that a ConvRot INT8 loader would have to be ported
into FastVideo. That was wrong, and the wrong branch was expensive. The working
decomposition keeps each stack doing what it already does:

```text
pruned INT8 ConvRot H3 base        (ComfyUI quant path, already working here)
  -> FastH3 four-step VSA LoRA     (fasth3_vsa_4-steps-v5.safetensors)
  -> FastH3 trained gates          (fasth3_vsa_gate.safetensors, 50 blocks)
  -> vsa.video_sparse_attn         (FastVideo Triton kernel)
  -> ComfyUI                       (barelymining/ComfyUI-MiniMax-H3-FastVideo)
```

Quantized **weights** stay ComfyUI's responsibility; only the sparse **kernel**
comes from FastVideo. Nothing needed porting.

Do not confuse the two sparse-attention paths. ComfyUI-side Sol-Attention
(`ComfyUI-sol-attn`) is vendored NVIDIA source dispatching SM86/89/90/100/120/121
and is **not** usable here. The path that works on AMD is FastVideo's Triton VSA.

## 1. FastVideo's Triton VSA kernel is correct on gfx1201

Standalone probe, no ComfyUI and no H3 weights
(`experiments/fasth3-vsa/probe_vsa_gfx1201.py`):

- `vsa/__init__.py` branches on `torch.cuda.get_device_capability(0)`; gfx1201
  reports `(12, 0)`, so the `major == 9 and minor == 0` H100 branch is skipped
  and the Triton path is selected. `vsa_cuda` is never imported and
  `block_sparse_fwd is None`.
- **Correctness, not just absence of crash:** at `topk == n_blocks` VSA is
  mathematically required to equal dense attention. Measured relative L2 vs a
  dense reference: `4.775e-03` without gate, `5.419e-03` with gate — bf16
  rounding, not a broken kernel.
- Executed clean at H3-representative shapes (56 heads, head dim 128, bf16) for
  4096 / 8192 / 16384 tokens at topk ratio 0.10. Peak 3.05 GiB.

The `vsa` PyPI package builds a Hopper `-arch=sm_90a` extension and fails on
non-Hopper cards. Installing the Python sources only (`vsa==0.0.3` sdist,
`vsa/` directory onto `sys.path`, plus `pytest`) activates the Triton fallback.
No compilation, no local patch.

**The RDNA `num_stages` workaround was not needed.** That fix applies to a
different sparse-attention kernel. This kernel autotuned and ran clean on
Triton 3.5.1 (`BLOCK_M`/`BLOCK_N` fixed at 64; `num_stages` 2..7 x `num_warps`
4,8). No gfx1201-specific change was required anywhere in this bring-up.

## 2. VSA is workload-dependent — sparse is slower on small workloads

| Workload | Tokens | Blocks | Dense sampling | VSA @0.10 sampling |
|---|---:|---:|---:|---:|
| 608x352 / 39f | 2,683 | 42 | 4 s | **8 s** |
| 864x480 / 124f | 15,444 | 242 | 34 s | **27 s** |

At 2,683 tokens VSA is about **2x slower** than dense: selection, the compress
branch and per-block gate streaming cost more than the attention they save. At
15,444 tokens VSA wins clearly.

**It is also a quality failure, not only a speed failure.** The forced small-workload
run (`min_tokens` lowered to 256 so VSA would engage at 2,683 tokens, giving
topk 5/42) was **rejected at the operator's visual quality gate**. Every
864x480/124f VSA output passed the same gate. With only 42 blocks to choose from,
a 10% top-k keeps 5 blocks, and the approximation visibly breaks down.

The bridge node's `min_tokens=4096` guard is therefore justified on **two**
independent grounds — throughput and output quality — and should be treated as a
correctness-relevant guard, not a tuning knob. The exact gfx1201 crossover has
not been measured and 4096 should not be inherited as a tuned value; whatever
replaces it must be validated visually, not only on sampling time.

This nearly produced a false negative: the first 608x352/39f run emitted a valid
video and reported success while logging `[H3-VSA] dense fallback: below
min_tokens`. Any FastH3/VSA run must assert on the `[H3-VSA] ACTIVE` line;
`run_workflow.py` fails the run when it is absent or a fallback line appears.

## 3. Sparsity sweep

864x480/124f, seed 8112026, everything constant except `topk_ratio`. Warm runs.

| Lane | topk | Blocks selected | Sampling | s/forward | Warm wall | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| VSA | 0.10 | 25/242 (10.3%) | 27 s | 6.69 | 60.21 s | (see note) |
| VSA | 0.20 | 49/242 (20.2%) | 28 s | 7.10 | 54.25 s | 27.82 GiB |
| VSA | 0.30 | 73/242 (30.2%) | 30 s | 7.68 | 55.64 s | 27.82 GiB |
| VSA | 0.50 | 121/242 (50.0%) | 35 s | 8.85 | 60.39 s | 27.82 GiB |
| VSA | 1.00 | 242/242 (100%) | 47 s | 11.77 | 72.19 s | 27.82 GiB |
| FastH3 dense | — | — | 34 s | 8.68 | 72.02 s | 26.58 GiB |

The 0.10 row's VRAM figure is withheld here: that single run was contaminated by
a preceding dense-lane warm-up. Section 9 supersedes it with repeated
measurements. Sampling scales smoothly with selected blocks. Warm wall time does
not, because
model load, conditioning and decode dominate; the 0.10 wall figure above came
from a differently-warmed server than the 0.20-1.00 rows and should not be read
as a 0.10-vs-0.20 regression. Sampling seconds are the trustworthy column.

**VSA @1.00 is not the dense lane.** 47 s vs 34 s sampling for the same four
forwards. VSA at full selection still pays for block-mean compression, scoring,
top-k and the `out_c * gate + out_s` combination. Keep the lanes separate.

## 4. Approximation diagnostics

Diagnostic only. Aesthetic judgement is the operator's; these numbers do not
rank videos.

Against the FastH3 dense lane, SSIM stays near 0.50-0.56 and does **not** rise
toward 1.0 as topk approaches 1.00 — because the VSA path is a different
computation, not a sparsified dense one. Dense is therefore the wrong reference
for isolating sparsity error.

Against VSA @1.00, which shares the code path, the curve behaves as expected:

| topk | SSIM vs VSA@1.00 | PSNR vs VSA@1.00 |
|---:|---:|---:|
| 0.10 | 0.524 | 14.81 dB |
| 0.20 | 0.522 | 15.98 dB |
| 0.30 | 0.576 | 17.70 dB |
| 0.50 | 0.612 | 18.84 dB |

Temporal delta (mean absolute frame-to-frame luma change, 216x120 grayscale):
dense 0.852; VSA lanes 2.33 / 2.90 / 2.68 / 2.72 / 2.68 for 0.10 / 0.20 / 0.30 /
0.50 / 1.00. Higher may indicate more motion or more flicker; the metric does
not distinguish them.

### Operator visual quality gate — 2026-09-02

| Set | Verdict |
|---|---|
| A Turbo v4, B FastH3 dense, C FastH3 VSA @0.10 (864x480/124f) | **PASS** — all acceptable |
| Sweep: dense, 0.10, 0.20, 0.30, 0.50, 1.00 (864x480/124f) | **PASS** — all acceptable |
| VSA forced below `min_tokens` (608x352/39f, 2,683 tokens, topk 5/42) | **REJECTED** |

Quality is acceptable across the entire tested sparsity range at 864x480/124f,
including the trained 0.10 policy. No ratio was judged better than another, so
there is no quality argument for paying for a denser setting: **topk 0.10 is both
the fastest tested configuration and the sparsity the LoRA was distilled at**
(`vsa_sparsity=0.9`).

The only rejected output is the one where VSA was forced to run below its token
guard.

Matched contact sheets (identical timestamps, configurations adjacent):
[`assets/fasth3-vsa/ABC-864x480-124f.png`](assets/fasth3-vsa/ABC-864x480-124f.png)
and [`assets/fasth3-vsa/sweep-864x480-124f.png`](assets/fasth3-vsa/sweep-864x480-124f.png).

## 5. Validation

All seven outputs passed: exact geometry, 124 frames at 24 fps, 5.167 s duration
(39 frames / 1.625 s for the small lane), H.264 `yuv420p`, AAC stereo 32 kHz with
matching audio duration, non-silent audio (mean volume -17.4 to -45.9 dB), and a
clean full decode to null with no errors. SHA-256 of every artifact is recorded.
No GPU reset, VM fault, ring timeout or OOM occurred.

## 6. Residency

The transformer stayed resident for all four forwards. No denoiser offload at any
point. Steady-state peak is **27.82 GiB of 31.86 GiB** at topk 0.10 — about
**4.0 GiB headroom**, roughly 3 GiB above Turbo v4's 24.82 GiB. (An earlier
single-run 29.26 GiB figure was an allocator/lifecycle artifact; see section 9.) Gates are held CPU-pinned and streamed
per block per step (~73 MiB x 50 blocks = ~3.6 GiB not held resident), which is
part of why headroom survives.

Text encoder and VAEs offload between stages, which is the accepted pattern here
and is independently faster on this system.

## 7. Provenance

- Base: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20.97 GB),
  `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`
- LoRA: `fasth3_vsa_4-steps-v5.safetensors` (2.20 GB),
  `70831f37ad1431f0686c8976d3ec949a4359ae67316d654b453d8809517bb268`
- Gate: `fasth3_vsa_gate.safetensors` (3.85 GB),
  `bf95408e335c4b0d6a1a44946a428a3e12849e6eab2de9566ecedae4a0043420`
- Bridge node `barelymining/ComfyUI-MiniMax-H3-FastVideo` `be8f1ef72e3dc430c005e10923c2a0863aa6a9e8` (2026-08-30)
- FastVideo reference checkout `0bd19a976b2e88ccd0d10b687b238bc1fa28c52a`
- `vsa==0.0.3`, Python sources only, Triton path
- Full hash list: [`../experiments/fasth3-vsa/asset-sha256.txt`](../experiments/fasth3-vsa/asset-sha256.txt)
- Stack capture: [`../experiments/fasth3-vsa/stack.txt`](../experiments/fasth3-vsa/stack.txt)
- Media: `/ai/artifacts/runs/fasth3-vsa/`

Launcher flags mirror the golden H3 launcher, including the load-bearing
`--disable-mmap`, on port 8191 so the production service on 8190 is never
contended.

## 8. Repetition protocol — medians over three warm runs per lane

864x480/124f, seed varied per repetition because ComfyUI serves an identical
graph from cache in ~1 ms. Median (min-max) of three warm runs.

| Lane | Sampling | s/forward | Warm wall | Peak VRAM |
|---|---:|---:|---:|---:|
| H3 Turbo v4 FP8 | 48 s | 12.18 (12.17-12.19) | 71.48 s (71.38-71.50) | 24.82 GiB |
| FastH3 dense | 34 s (34-35) | 8.69 (8.68-8.97) | 71.02 s (70.83-72.09) | 26.58 GiB |
| FastH3 VSA 0.30 | 30 s | 7.74 (7.73-7.74) | 56.08 s (56.06-56.18) | 27.84 GiB |
| FastH3 VSA 0.20 | 28 s | 7.15 (7.14-7.15) | 53.78 s (53.74-56.18) | 27.84 GiB |
| **FastH3 VSA 0.10** | **26 s** | **6.55 (6.54-6.55)** | **50.94 s (50.87-51.00)** | 27.82 GiB |

Variability is negligible: per-forward spread is under 0.5% within every lane
except FastH3 dense, whose third run drifted to 8.97 s. `p95` is not reported —
three observations cannot support it.

Median speedups on per-forward time:

- VSA 0.10 vs Turbo v4: **1.86x**
- VSA 0.10 vs FastH3 dense: **1.33x**
- FastH3 dense vs Turbo v4: 1.40x

End-to-end warm wall, VSA 0.10 vs Turbo v4: 50.94 s vs 71.48 s, **28.7% less
wall time**.

## 9. VRAM repeatability, and the image-reference workload

**Peak VRAM is invariant to `topk_ratio`.** Nine warm runs, three per ratio:

| Ratio | Peak VRAM (3 runs) |
|---:|---|
| 0.10 | 27.82 / 27.82 / 27.82 GiB |
| 0.20 | 27.84 / 27.84 / 27.84 GiB |
| 0.30 | 27.84 / 27.84 / 27.84 GiB |

Stable to the hundredth of a GiB and flat across a 5x change in selected blocks.
This empirically confirms that **VSA is a speed optimization, not a memory
optimization** — sparsity removes attention compute, not activation footprint.

An earlier single-run figure of 29.26 GiB at ratio 0.10 was an
**allocator/lifecycle artifact**, not a real working-set difference: that run
followed a dense-lane warm-up whose allocation was still resident. True VSA
steady state is **~27.82 GiB and is effectively independent of `topk_ratio`**.
Do not treat single-run `rocm-smi` peaks as steady state.

### Ref2VA image-reference workload — fits comfortably

First image-reference configuration, VSA 0.20, 864x480/124f, one reference image,
`ref_image_size=match`, on `minimax_h3_ref2va_pruned_int8_convrot`. Three warm
runs:

| Metric | T2VA (FL2VA base) | Ref2VA, one ref image |
|---|---:|---:|
| Tokens | 15,444 | 16,260 |
| Blocks selected @0.20 | 49/242 | 51/255 |
| Sampling | 28 s | 30 s |
| s/forward | 7.15 | 7.56 |
| Warm wall | 53.78 s | 57.03 s |
| Peak VRAM | 27.84 GiB | **28.15 GiB** |
| Headroom of 31.86 GiB | 4.02 GiB | **3.71 GiB** |

The reference image costs **+816 tokens, +0.31 GiB and +0.41 s per forward**.
It fits with room to spare; **no memory optimization is warranted for this
workload.** The cold first run peaked at 29.65 GiB including the model-load
transient, which is still inside budget.

`ref_image_size=match` was used. `max` uses a 2048px short edge and, per the
node tooltip, reference tokens ride through every sampling step — the earlier
R9700 finding that `max` OOMed at 960x544/124f under Turbo still stands as the
reason to default to `match`. `max` was not retested here.

Output validated: 864x480, 124 frames, 24 fps, 5.167 s, H.264 + AAC stereo
32 kHz, mean volume -27.1 dB, clean decode.

**Caveat, unresolved:** the FastH3 LoRA and gates were converted for the **FL2VA**
base and were applied here to a **ref2va** base. They bound without error — all
50 blocks patched, gates loaded, VSA active at 51/255 — because the LoRA targets
`blocks.N.attn.*` and `mlp.*`, which are identical across variants. The upstream
bridge states that Ref2VA "would need matching FastVideo LoRAs (not yet
released)". Timing and memory figures above are therefore trustworthy;
**identity fidelity is not yet validated** and awaits the operator's visual gate.

API note: the Ref2VA reference image is wired with the dotted autogrow key
`ref_images.ref_image_0`, not `ref_image_0`. The latter raises
`TypeError: MiniMaxH3ReferenceToVideo.execute() got an unexpected keyword
argument`. The production `h3_r2v.json` predates the current node signature: it
uses singular `ref_image` and lacks `audio_vae` and `ref_image_size`.

## 10. Memory optimization — DEFERRED, do not implement

**Decision (2026-09-02): no invasive memory optimization.** The primary practical
target — one Ref2VA image at `ref_image_size=match`, 864x480/124f — fits
comfortably at **28.15 GiB peak with ~3.71 GiB headroom**, and T2VA sits at
27.82-27.84 GiB with ~4.0 GiB headroom. There is no memory problem to solve, so
solving one would add risk to a working configuration for no measured benefit.

The following ideas were identified while characterizing the workload and are
**preserved as future candidates only**. Do not implement any of them unless it
is independently useful on its own merits, or a larger workload actually
exhausts headroom:

| Candidate | Idea | Trigger to reconsider |
|---|---|---|
| Early release of reference/embedding construction tensors | Free ref-image and conditioning construction buffers as soon as the conditioning stage completes, rather than holding them through sampling | Multi-reference Ref2VA, or `ref_image_size=max` |
| MLP / FinalLayer chunking | Process the FFN and final projection in sequence chunks to cap peak activation | Higher resolution or longer clips |
| Remove VSA coarse-output `repeat()` materialization | The compress branch expands per-block output back to full sequence length via `repeat()`, materializing a full-size tensor that could be fused or broadcast instead | Longer sequences where the coarse tensor becomes a material fraction of peak |

Any of these must be justified by a measurement, benchmarked before and after,
and validated against the visual quality gate — not adopted because it sounds
like a saving.

Two workloads that are *not* yet characterized and could change this decision:
`ref_image_size=max` (reference tokens ride through every sampling step; the
prior R9700 finding is that `max` OOMed at 960x544/124f under Turbo), and
multi-reference Ref2VA (the node accepts up to 9 images).

## 11. Final bounded memory pass — 2026-09-02 (CLOSED)

One bounded pass, then experimentation stopped. Measured with torch's own
counters (`max_memory_allocated` / `max_memory_reserved`) via a probe route, not
`rocm-smi`; these are a different basis from the device-wide figures in sections
9 and 10 and are not directly comparable to them.

### Accepted: VSA broadcast combine (`patches/vsa-broadcast-combine.patch`)

The 64-token VSA path expanded its block-level coarse output with `repeat()` into
a full `[B,H,S,D]` tensor, then allocated two more full-size tensors for the
gate product and the sum. Replaced with a `[B,H,n_blocks,1,D]` broadcast that
accumulates into the sparse result via `addcmul_` under `no_grad`, mirroring the
128/256-block paths. The fp32 block means also switched from `.float().sum()` to
`.sum(dtype=torch.float32)`, which accumulates identically without materializing
an fp32 copy of q/k/v.

| Workload | s/forward | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Ref2VA @0.20 before | 7.55 | 25.885 GiB | 27.301 GiB |
| Ref2VA @0.20 after | **7.23** | **25.473 GiB** | **26.863 GiB** |
| T2VA @0.20 before | 7.14 | 25.748 GiB | 27.100 GiB |
| T2VA @0.20 after | **6.84** | **25.358 GiB** | **26.686 GiB** |

**-0.41 GiB allocated, -0.44 GiB reserved, and 4.2% faster** — no speed tradeoff
to weigh. Equivalence verified against the pre-patch implementation
(`verify_vsa_patch.py`): bit-exact without a gate; with a gate, within one bf16
ULP (max abs 9.77e-4, rel L2 ~1e-3) because `addcmul_` fuses the multiply-add.
Top-k routing identical in every case. VSA active, zero fallbacks.

Structured to upstream to FastVideo unchanged.

### Kept but delivers nothing measurable: early release (`patches/h3-early-release-conditioning.patch`)

Drops the embedding and reference/conditioning construction tensors
(`video_embed`, `audio_embed`, `all_video_rows`, `cond_*_rows`, `text_states`,
update masks) as soon as `h` is packed, instead of holding them across the
50-block loop. Nothing is recomputed and nothing moves between CPU and GPU.

**Measured effect: zero.** 25.473 / 26.863 GiB before and after, identical to
three decimals on both workloads.

**Informational only.** The patch is retained as a record of what was tried and
what it was worth, not as part of the accepted configuration. It buys no headroom
and would need re-applying after every ComfyUI update. Do not treat it as
required, and do not carry it forward on its own account.

### Skipped: MLP / FinalLayer chunking

Not attempted. Two independent reasons:

1. **No existing implementation to incorporate.** No H3-Optimizations chunking
   exists in this ComfyUI; it would have to be written from scratch, which is the
   "integration is becoming complicated, stop" condition.
2. **A hard floor caps the possible gain below the acceptance bar.** A memory
   trajectory through a full Ref2VA run shows two separate phases each reaching
   ~25 GiB:

```text
t=3.5 s    25.010 GiB   text-encoder phase (Qwen3-VL FP8 is a 25 GB encoder)
t=20.7 s    0.149 GiB   encoder offloaded
t=34-63 s  23.3-25.0 GiB DiT resident (20 GB) + sampling activations
t=66-84 s   5.1 GiB     VAE decode
```

The text-encoder phase sets a floor at **25.01 GiB**. Shaving DiT activations can
therefore recover at most ~0.45 GiB before peak becomes encoder-bound — below the
0.5 GiB bar set for this candidate. Any future memory work should target the
encoder phase first, not the block loop.

This also explains the null result above: freeing ~0.2 GiB of embedding tensors
cannot move a peak that is set by a different high-water mark.

### Final frozen configuration

Ref2VA, `ref_image_size=match`, 864x480/124f, VSA topk 0.20, patched:
**7.19-7.23 s per forward, 28 s sampling, ~53.6 s warm wall, 25.473 GiB peak
allocated, 26.863 GiB peak reserved.** Output validated: 864x480, 124 frames,
24 fps, 5.167 s, H.264 + AAC stereo, mean volume -22.3 dB, clean decode.

**Experimentation is closed. The next phase is normal use.** Not attempted, by
explicit scope: streamed QKV,
CPU offload, transformer offload, new sparse kernels, top-k tuning, min-token
tuning, larger-reference optimization, further quality metrics, further
benchmark matrices.

## Engineering record

```text
TASK:       Determine whether FastH3 four-forward + FastVideo VSA can run on
            gfx1201 with the transformer resident, and whether it beats the
            selected H3 Turbo v4 model path.
HYPOTHESIS: A ConvRot INT8 loader must be ported into FastVideo first.
CONTROL:    H3 Turbo v4 FP8, same harness, same encoder, same prompt/seed/geometry.
CHANGE:     Rejected the port. Used the ComfyUI bridge so quantized weights stay
            in ComfyUI and only the Triton sparse kernel comes from FastVideo.
RESULT:     VSA correct on gfx1201 (rel L2 ~5e-3 vs dense at full selection).
            Sampling 48 s -> 34 s (four-step) -> 27 s (VSA 0.10). Resident,
            steady-state peak 27.82 GiB, no denoiser offload. No gfx1201 change.
            VSA is ~2x slower below ~2.7k tokens.
DECISION:   ACCEPTED. Bring-up complete and accepted for normal use. Operator
            quality gate PASSED for every 864x480/124f lane and every sparsity
            ratio tested; the only rejection was VSA forced below its token
            guard. Ref2VA @ topk 0.20, ref_image_size=match is the accepted
            working configuration; topk 0.10 remains the fastest T2VA setting
            and the trained policy. Production manifest unchanged.
NEXT:       Normal use. No further optimization or benchmarking. The VSA
            broadcast-combine patch is retained as an upstream candidate for
            FastVideo; the early-release patch is informational only. Re-apply
            and re-verify the VSA patch after any ComfyUI or vsa update.
```

## Closed scope

Bring-up is complete and accepted; the items below were deliberately not pursued
and are not outstanding work. Revisit only if a genuinely new requirement
appears, and treat any of them as a fresh, separately-scoped task:

- cold-lane triples per the three-run reference protocol (warm triples are done)
- `ref_image_size=max` and multi-reference Ref2VA characterization
- sampler profiling and any kernel, launch-parameter or fusion work
- the exact gfx1201 sparse/dense crossover threshold behind `min_tokens`
- the deferred memory candidates in section 10
  ref2va base — binds cleanly, fidelity unvalidated)
- `ref_image_size=max` and multi-reference Ref2VA characterization
- Cold-lane triples per the three-run reference protocol (warm triples are done)
- Sampler profiling — where the remaining 27 s goes
- Exact gfx1201 sparse/dense crossover threshold
- No kernel, launch-parameter, gate-caching or fusion work attempted, by design
