# Two-GPU MiniMax H3 Inference on Mixed AMD Radeon GPUs (R9700 gfx1201 + RX 7900 XT gfx1100): Feasibility & Implementation Plan

## 1. Executive Verdict

**Two-way tensor parallel (TP) of the H3 denoiser on this specific mixed Radeon pair is currently NOT practical for reducing single-video wall time, and should be treated as "do not pursue for latency" today. Sequence/context parallelism (SP/CP) is architecturally the correct approach and is "prototype-first," but is blocked by the same hardware-layer defect. The single blocking fact is that RCCL collectives across two consumer Radeon GPUs on the current ROCm 7.2.1 / RCCL 2.27.7 stack currently deadlock or error on the first collective — before any H3 code matters.**

- **Is two-way H3 TP practical now?** No. Megatron-style TP needs an all-reduce (or reduce-scatter/all-gather) twice per transformer block — ~100 collectives per denoise step across 50 blocks. On a PCIe-only consumer pair with ~10-48 GB/s links and a currently-broken RCCL collective path, TP is the *worst-fit* method. Even if RCCL worked, the communication-to-compute ratio at this sequence length makes 2-way TP marginal at best. **Uneconomic.**
- **Is H3 SP/CP practical now?** Prototype-worthy in principle — H3 is a single-stream dense transformer where (per diffusers PR #14355 author apolinario) "One packed token sequence carries text, conditioning, audio and video rows through a shared 33B transformer… full self-attention over all of it. There is no separate vocoder and no audio post-hoc pass." That is exactly what DeepSpeed-Ulysses/Ring-Attention/USP target, and MiniMax itself ships H3 with `--ulysses-degree` sequence parallelism. But (a) it needs the same RCCL collectives that are currently broken on consumer RDNA, (b) all public tooling (xDiT/xFuser, SGLang, Raylight) is validated on 8× datacenter GPUs or homogeneous consumer pairs, never on a mixed RDNA3+RDNA4 pair, and (c) the mixed-architecture process group is completely unproven publicly. **Prototype-first, gated on a Phase-0 RCCL preflight that will most likely fail on today's stack.**
- **Best probability of cutting the 5-second Turbo wall time?** None of the true-parallel methods beats the current component-placement topology *today*. If/when RCCL is fixed for consumer RDNA, **Ulysses-style sequence parallelism (via Raylight/xFuser) is the highest-probability latency win**, but the Amdahl ceiling (below) caps realistic end-to-end gains at ~15-22% even in the optimistic case, and only if collective overhead stays small.

The honest recommendation: **Do not invest in H3 TP for latency. Run the Phase-0 RCCL/P2P preflight as the cheap falsification test. If and only if it passes, prototype Ulysses SP via Raylight. Otherwise, redirect the second GPU to capacity/throughput wins (concurrent jobs, encoder offload, VAE placement), which already work.**

---

## 2. Verified H3 Architecture Relevant to Parallelism

All values below are from the AMD "Day 0 Support for MiniMax-H3" engineering blog (Aug 2, 2026), the official MiniMaxAI/MiniMax-H3 model card, and the HuggingFace diffusers PR #14355 — cross-checked. Where the ComfyUI native path differs, it is noted.

**H3-Omni-Transformer (the `MiniMaxH3DiTModel` denoiser):**

| Property | Value | Divides by 2 cleanly? |
|---|---|---|
| Parameters | 33.1B dense, single-stream (~13B in AdaLN branches, cacheable at inference) | — |
| Hidden size (d_model) | 5376 | Yes (2688) |
| Transformer layers | 50 (+2 token-refiner layers) | Yes (25) |
| Attention heads | 56, head dim 128 (attention dim 7168) | Yes (28 heads/rank) |
| FFN hidden size | 14336 | Yes (7168) |
| Video latent channels | 24 (f16t4d24 VAE: 16× spatial, 4× temporal) | Yes |
| Audio latent channels | 32 (per-channel, 40 Hz temporal rate at 32 kHz) | Yes |
| Patch size (T,H,W) | (1, 2, 2) → effective 32× spatial, 4× temporal downsample | — |
| Text conditioning dim | 5120 (from Qwen3-VL-32B layer-50 hidden states) | Yes |
| Position encoding | 3D MM-RoPE over (t, h, w) | — |
| Conditioning | modality-specific AdaLN from diffusion timestep; separate video/audio flow-shift | — |
| Weights | BF16 native | — |
| Attention | **dense full self-attention** in the open release (native sparse attention trained but NOT released; "future update") | — |

**Data flow / packing (from ComfyUI `comfy.ldm.minimax.model.MiniMaxH3Model._forward` and diffusers PR #14355):**
- One packed sequence: **`[text | optional keyframe/reference segments | target audio | target video]`**. Target audio and target video are the final two contiguous segments, in that order (confirmed via `PackedLayout.segments`).
- Text encoded by Qwen3-VL-32B (layer 50); visual refs by both encoder and Visual VAE; audio by Audio VAE only.
- Modality behavior confined to **two input projections, a per-row AdaLN modality tag, and two output heads** — the attention and FFN layers themselves have NO modality-specific structure. This is what makes SP "easy" (per the diffusers maintainer on the H3 discussion: "we do not have to deal with any weird packing… only odd-length padding is necessary").
- Cross-modal full attention over the entire packed sequence is what keeps motion and sound aligned — meaning any sequence split MUST preserve global attention (Ulysses all-to-all or Ring), not naive independent chunks.

**Workload shapes (measured benchmark: 864×480, 124 frames, 24 fps, 4 Turbo steps):**
- Video tokens: latent grid = (124 padded to 17k+5 grid) temporal /4 → ~32 latent frames; spatial 864/32 × 480/32 = 27 × 15 = 405 tokens/frame → order ~13,000 video tokens (exact count depends on padding to the 17-frame block grid; **evidence gap** — ComfyUI snaps length to the 17k+5 grid so exact token count needs instrumentation).
- Audio tokens: 40 Hz × ~5.17 s ≈ 207 per channel.
- Text tokens: prompt-dependent, typically 5,650-33,000 in the official Context-IR examples (but local ComfyUI uses raw prompt, far smaller).
- Net: the target video segment dominates the packed sequence — good for SP amortization at higher resolution/longer duration, marginal at 864×480/5s.

**Turbo LoRA:** `minimax_h3_turbo_v4_step600_ema` is a standard low-rank update (W_eff = W + B@A, alpha=rank, no extra scaling), ~744 MB BF16, targeting the base transformer. It changes **model-forward weights** (must be sharded/applied identically on both ranks in TP) AND scheduling (4-step dual-clock Euler: video flow_shift=12, audio flow_shift=3). The 4-step sampler changes scheduling but the per-step forward is the same dense transformer — so 4 steps means **4× fewer opportunities to amortize collective setup**, hurting parallel efficiency versus the 50-step default. (Note: the pruned `fp8_scaled` variants use a different time-conditioning layer and are documented as incompatible with the non-pruned Turbo LoRA; the LoRA requires a BF16 or full INT8-convrot base.)

**Concise data-flow diagram:**
```
Qwen3-VL-32B (enc, layer50) ─┐
Visual VAE (refs) ───────────┤
Audio VAE (refs) ────────────┤
                             ▼
        PackedLayout: [text | refs | tgt-audio | tgt-video]
                             │  + 3D MM-RoPE, AdaLN(timestep, modality)
                             ▼
     50× {AdaLN → full self-attn (56h×128) → AdaLN → FFN(5376→14336→5376)}
                             │   (×4 Turbo steps)
                             ▼
        two output heads → video latent + audio latent
                             ▼
   Visual VAE decode (ViT decoder) + Audio VAE decode → MP4 (video+stereo)
```

---

## 3. Fundamental Theory

**Compute partitioning options and what each collective costs:**

- **Tensor parallelism (Megatron):** column-parallel QKV + FFN-in, row-parallel attn-out + FFN-out. Requires **1 all-reduce after attention + 1 all-reduce after FFN per block** = 100 all-reduces/step for 50 blocks. Each all-reduce moves an activation tensor of shape [seq_len × d_model]. At ~13k tokens × 5376 × 2 bytes (BF16) ≈ 140 MB per tensor; a 2-GPU ring all-reduce moves ~2×(N-1)/N × size ≈ 140 MB on the wire per collective → **~14 GB of PCIe traffic per denoise step**, ~56 GB across 4 Turbo steps. At ~48 GB/s bidirectional best-case that's ~1.2s pure comms; at the realistically-observed ~10 GB/s RCCL busbw it's ~5.6s — already larger than the entire 48s H3 stage benefit budget once you account for latency per collective (100 collectives/step × 4 steps = 400 synchronizations).
- **Sequence/context parallelism (Ulysses):** split the packed sequence across ranks; **2 all-to-all per attention layer** (before/after) to switch between sequence-sharded and head-sharded layouts. All-to-all moves the QKV activations once each way; total bytes per step scale with sequence×d_model but the *count* of collectives is far lower and each rank does real FFN work on its shard with no comms. This is why USP/Ulysses is the DiT-standard.
- **Ring Attention:** keeps sequence split, streams K/V peer-to-peer around the ring; overlaps comms with compute. Good for PCIe but load-imbalance-prone with masks.
- **CFG parallelism:** trivially splits the two CFG branches across GPUs (constant factor 2). **Not applicable here** — H3's released weights are CFG-distilled and Turbo runs effectively CFG=1, so there is no second CFG branch to peel off.
- **Pipeline / PipeFusion:** shard blocks across GPUs, pipeline patches with stale-activation reuse; async P2P, PCIe-friendly. But needs many steps to fill the pipeline — **4 Turbo steps is near-worst-case** for pipeline warmup amortization (PipeFusion's own paper notes warmup-step sensitivity at low step counts).
- **Weight/layer offload (DisTorch, ComfyUI-MultiGPU):** NOT parallel compute — it stages layers on a second device/RAM and computes them serially on the main GPU. This is the current topology; it saves model churn, not denoise time. (The maintainer explicitly frames it as memory management: "the extension allows different components of your models to be loaded onto different devices… the steps in your workflow still execute one after the other.")

**Amdahl ceiling for this workstation:**
- Measured stage split: H3 sampling ≈ 48 s; decode/save ≈ 18 s; plus encode + overhead. Cold end-to-end ≈ 80.7 s (single R9700).
- If ONLY the 48 s H3 stage is accelerated and everything else is fixed at ~32.7 s:
  - Max theoretical speedup if H3→0: 80.7 → ~32.7 s = **59% ceiling** (unattainable).
  - With ideal 2× on H3 (48→24): 80.7 → 56.7 s = **30% end-to-end**.
  - With realistic 1.5× on H3 (48→32): 80.7 → 64.7 s = **~20%**.
  - With 1.3× on H3 (48→37): **~14%**.
- **H3-stage speedup required for a given end-to-end gain** (cold, 80.7 s base, 32.7 s non-H3 floor):

| End-to-end gain target | Allowed total time | Required H3 stage time | Required H3 speedup |
|---|---:|---:|---:|
| 10% | 72.6 s | 39.9 s | 1.20× |
| 15% | 68.6 s | 35.9 s | 1.34× |
| 20% | 64.5 s | 31.8 s | 1.51× |
| 25% | 60.5 s | 27.8 s | 1.73× |
| 30% | 56.5 s | 23.8 s | 2.02× |

- **The break-even / negative-return point:** if collective + sync overhead added to the H3 stage exceeds the compute saved, two GPUs become *slower* than the single R9700. With 4 Turbo steps and TP's 400 synchronizations, plus consumer-PCIe collective bandwidth, the overhead term almost certainly exceeds the ~24 s that a perfect 2× split would save — i.e., **TP is expected to be net-negative at this workload**.
- **20-step vs 4-step:** the standard schedule spends ~5× longer in the H3 stage (raising the Amdahl ceiling toward ~40-45%) AND amortizes collective/pipeline setup over 5× more steps. **Every parallel method looks better at 20-50 steps than at 4 Turbo steps.** The user's Turbo lane is the hardest case for parallel speedup.
- **Longer/higher-res sequences:** SP benefit scales with sequence length. At 864×480/124f the video token count is modest; at 2K-class or 15 s the sequence grows ~5-20×, attention (quadratic) dominates, and SP amortization improves markedly — this is where SP delivers a **capacity win** (fitting/accelerating longer clips) even if the 5-s case stays marginal.

---

## 4. Current Ecosystem & Community Support Matrix

| Project | Parallel method | Inf/Train | Video DiT | H3 support | ROCm support | Mixed RDNA3/RDNA4 evidence | Reusable value |
|---|---|---|---|---|---|---|---|
| **ComfyUI native** (`comfy.ldm.minimax`) | none (single-GPU) | Inf | Yes (H3 native) | **Yes, first-class** | Yes (torch-rocm) | None | Golden reference; the model impl to wrap |
| **ComfyUI-MultiGPU / DisTorch2** (pollockjj) | component/layer offload (NOT parallel compute) | Inf | Yes | via loaders | Yes (device-agnostic) | Author tests "mixed GPU configs" (unspecified) | VRAM relief only; not a latency win |
| **Raylight** (komikndr) | USP (Ulysses+Ring), FSDP, CFG, DP via Ray | Inf | Yes (Wan, Qwen, Flux, HunyuanVideo) | **"Minimax H3" listed in changelog** | Partial (MI210/MI300X user-confirmed; consumer RDNA untested) | None found | **Highest-value reusable path** for SP |
| **xDiT / xFuser** | USP, PipeFusion, DistriFusion, CFG, TP | Inf | Yes (many DiTs) | not H3-specific yet | **Yes** (rocm/pytorch-xdit, "MI300X or newer") | None (validated 8-24× MI300X only) | USP engine underneath Raylight |
| **SGLang (diffusion)** | SP/Ulysses (8-way), TP | Inf | Yes (H3) | **Yes — AMD Day-0 H3 recipe** | **Yes** (MI355X/MI300X) | None (8× Instinct only) | Proves H3 SP works on ROCm at scale |
| **diffusers** | ContextParallel (Ring+Ulysses via `ContextParallelConfig`) | Inf | Yes | **Yes (PR #14355, Modular blocks)** | Yes | None | Cleanest H3 SP reference impl |
| **vLLM-omni** | xFuserLongContextAttention (USP) | Inf | Yes | H3 served via vLLM-Omni | Yes (Instinct) | **Negative: dual-R9700 TP=2 deadlock (#40980)** | Primitive reference; RCCL cautionary tale |
| **PipeFusion** | displaced-patch pipeline | Inf | Yes | no | CUDA-first | None | PCIe-friendly theory; 4-step hurts it |
| **DistriFusion** | displaced patch parallel | Inf | Yes | no | CUDA/NCCL-assuming | None; assumes homogeneous power-of-2 GPUs | Not suited to heterogeneous pair |
| **stable-diffusion.cpp** | per-component backend assignment; NO single-model split | Inf | H3 GGUF denoiser (Unsloth) | partial | Yes (ROCm/Vulkan) | None | CPU encoder trick only |
| **city96 ComfyUI-GGUF** | GGUF loader (single-GPU compute) | Inf | Yes | via GGUF | Yes | None | Quant loading, not parallelism |

**H3-specific findings:** MiniMax officially ships H3 with sequence parallelism. Per the official model card the SGLang recipe is `sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 --performance-mode speed … --model-variant fl2va`. AMD's Day-0 blog goes further with `--num-gpus 8 --sp-degree 8 --ulysses-degree 8 --trust-remote-code --warmup-mode off --attention-backend aiter`, verified on MI355X/MI300X. The diffusers integration explicitly exposes `ContextParallelConfig` (ring + ulysses). Raylight lists "Minimax H3" support and its maintainer notes H3's single-stream design makes SP porting "a breeze." **So H3 SP is real and works — but exclusively demonstrated on homogeneous multi-GPU (8× MI355X, or NVIDIA consumer pairs). The vLLM Recipes 2× RTX 5090 TP config exists (`--tensor-parallel-size 2 --usp 1 --ring 1 --text-encoder-tp-size 2 --vae-patch-parallel-size 2 --enable-distributed-layerwise-offload`), and the vLLM ROCm USP path is "Verified on 4× AMD Instinct MI300X (gfx942)" — never a consumer Radeon pair.**

**Negative evidence (critical):**
- **Dual R9700 (gfx1201/RDNA4) TP=2 deadlocks** on ROCm 7.2.1 / RCCL 2.27.7. Per vllm #40980 (kyuz0, Apr 27 2026), the stack is "vLLM: v0.19.1+rocm721 … PyTorch: 2.11.0 (built from source for gfx1201) ROCm: 7.2.1, RCCL: 2.27.7 (built from source with --amdgpu_targets gfx1201)" and "both GPUs immediately sit at 100% utilization with no active requests." TP=1 works and the same build runs TP=4 on MI25 (gfx900), isolating the fault to gfx1201 multi-GPU. Per rocm-systems #5480, "The deadlock occurs during the first multi-GPU operation (CUDA graph capture or first forward pass), not during RCCL initialization… The same hardware and TP=2 configuration worked correctly with RCCL 2.27.3 on a previous ROCm nightly build (December 2025)" — i.e., a regression between RCCL 2.27.3 → 2.27.7. Both issues remain OPEN as of Aug 2026.
- **Dual RX 7900 XTX (gfx1100/RDNA3)** first collective errors `HIP failure: 'the operation cannot be performed in the present state'` on the same stack (ROCm/ROCm #6074).
- **No public report exists of a single RCCL process group spanning mixed gfx1100 + gfx1201** in one host — success or failure. This is the largest single evidence gap.

---

## 5. Hardware Feasibility Assessment (mixed RDNA3/RDNA4, PCIe P2P, RCCL)

- **Both GPUs are ROCm-supported individually.** ROCm 7.2 officially lists gfx1201 (RDNA4) and gfx1100 (RDNA3) as supported compute targets. The R9700 (AMD Radeon AI PRO R9700: RDNA 4, full Navi 48 die, 64 CUs / 4096 SP, 128 2nd-gen AI accelerators, 32 GB GDDR6 on a 256-bit bus at 640 GB/s, PCIe 5.0 ×16, ~95.7 TFLOPS FP16 / 1531 TOPS INT4; native gfx1201 support added in ROCm 7.2) and the RX 7900 XT (gfx1100, 20 GiB) each work standalone.
- **RCCL across two consumer Radeons is currently broken on the exact stack in use (ROCm 7.2.1 / RCCL 2.27.7).** Both same-arch cases (2× gfx1201 deadlock; 2× gfx1100 error) fail on the first collective. This regressed from RCCL 2.27.3, which worked. Since the user's workstation runs ROCm 7.2.1, **this is a direct, present-tense blocker.**
- **Mixed gfx1100 + gfx1201 in one process group is entirely unproven.** A mixed-arch build additionally requires a "fat" RCCL/PyTorch binary containing BOTH gfx1100 and gfx1201 code objects; no found report demonstrates this for a live 2-rank collective. Even the compiled Triton/kernel caches would need architecture isolation.
- **PCIe P2P on consumer Radeon is unreliable and often absent.** Measured 2× 7900 XTX rocm-bandwidth-test: ~28 GB/s unidirectional, ~48 GB/s bidirectional GPU↔GPU — but only after enabling large-BAR/P2P kernel paths, and several reports show ROCm *reporting* peer access available while transactions never complete. On the R9700, ROCm/ROCm #5571 ("Radeon AI PRO R9700 support for Peer-2-Peer GDRMA using rccl-tests") found all_reduce busbw ~10.8 GB/s and states "all_reduce_perf is the same when P2P GRDMA is enabled or disabled. Given the bandwidth numbers, it appears P2P is not being used." The task states PyTorch reports bidirectional peer access between these devices — **that flag must not be trusted without a measured rocm-bandwidth-test + rccl-tests all_reduce**, given documented false-positives.
- **One-process-per-GPU is required.** AMD's own H3 ROCm recipe uses the multiprocessing **spawn** start method (HIP-safe) with a `__main__` guard + `mp.freeze_support()`; forking inside ComfyUI's process is unsafe. This means any real H3 parallelism must run as an **external inference service / separate processes**, not in-process ComfyUI threads.
- **Heterogeneity penalty:** RCCL collectives (all-reduce, all-gather, reduce-scatter, all-to-all) require **equal shapes across ranks**. The R9700 (RDNA4, 32 GiB, faster AI throughput) and 7900 XT (RDNA3, 20 GiB) have different compute; with equal shards the faster R9700 idles waiting for the 7900 XT at every sync — so equal-split TP/SP wastes the R9700. Weighted/asymmetric splits (Design B / STADI-style) are possible but no RCCL collective natively supports variable shard sizes without custom all-to-all-v, and no consumer-Radeon implementation exists.

**Verdict:** Hardware+RCCL is the binding constraint. Even before H3 architecture matters, the collective layer must be fixed and proven on this pair. This is why Phase-0 is a hard gate.

---

## 6. Concrete TP Designs

### Design A — Classic Megatron 2-way TP
- **Sharding:** column-parallel QKV (56 heads → 28/rank) and FFN-in (14336 → 7168/rank); row-parallel attn-out and FFN-out. Norms, embeddings, AdaLN, RoPE, input/output heads replicated.
- **Fit to H3 dims:** all key dims divide by 2 cleanly (28 heads, 2688 hidden shard, 7168 FFN shard). LoRA: each B@A update must be sharded consistently with its target matrix (column-parallel targets shard one dim; row-parallel shard the other) — non-trivial but mechanical.
- **Per-rank weight memory:** ~33B params, but ~13B AdaLN cacheable → ~20B active; BF16 → ~40 GB total, ~20 GB/rank for the sharded matrices + replicated norms/heads. **Problem: the 7900 XT has only 20 GiB** — replicated components + activations + LoRA push it over. FP8 storage (current `fp8_scaled`) halves this but reductions must accumulate in BF16/FP32, and FP8 sharded GEMM kernels for RDNA3 are not guaranteed.
- **Collectives:** 2 all-reduce/block × 50 = **100 all-reduce/step; 400/gen** at 4 steps. ~140 MB/all-reduce → ~14 GB/step wire traffic.
- **Bottleneck:** PCIe collective bandwidth + 400 synchronizations + faster-rank idle. **Expected net-negative at 4 steps.**
- **Numerics:** all-reduce in BF16 introduces small ordering differences; acceptable if accumulation is FP32.
- **Integration complexity:** Very high (custom sharded loader, LoRA sharding, RCCL service).

### Design B — Weighted/asymmetric partition (R9700-heavy)
- **Idea:** give the R9700 ~60-65% of heads/FFN and the 7900 XT ~35-40%, matching compute and the 32/20 GiB VRAM split; or keep whole "hot" blocks on R9700 and TP only a subset.
- **Reality:** standard RCCL all-reduce/all-gather need equal shapes. Asymmetric requires all-to-all-**v** / custom reduce with padding, or a STADI-style temporal-adaptive scheduler — none implemented for ROCm consumer GPUs. Uneven head counts also complicate attention kernels.
- **Where imbalance helps:** it prevents the R9700 idling; where it hurts: every collective must still barrier, and variable-count collectives are not a first-class RCCL primitive.
- **Verdict:** Theoretically the *right* shape for this heterogeneous pair, but **no reusable implementation exists**; this is research-grade effort. Memory: fits better (R9700 ~13 GB shard, 7900 XT ~7 GB shard) but same collective-overhead problem.

**Both TP designs share the fatal issue:** at 4 Turbo steps with consumer-PCIe RCCL, collective+sync overhead is projected to exceed the compute saved.

---

## 7. Concrete SP/CP Designs

### Design C — DeepSpeed-Ulysses / USP all-to-all (recommended if RCCL is fixed)
- **Sharding:** split the packed sequence [text|refs|audio|video] across the 2 ranks (each holds ~half the tokens). Before attention, **all-to-all** to re-shard from sequence→head layout (each rank gets all tokens for 28 heads); after attention, all-to-all back. FFN runs fully local per shard.
- **Collectives:** **2 all-to-all per block × 50 = 100 all-to-all/step**, but each rank does genuine FFN + half-attention compute concurrently, and all-to-all on 2 GPUs is a single paired exchange. Communication volume per step ≈ sequence×d_model×few — smaller *effective* overhead than TP's all-reduce because compute overlaps and there's no parameter replication pressure.
- **Fit:** H3's modality-agnostic attention/FFN and single packed sequence make this clean (diffusers maintainer: "only odd-length padding necessary"). 56 heads / 2 = 28 per rank — divides cleanly, satisfying Ulysses' head-divisibility constraint.
- **Memory:** activations halved per rank; weights replicated (needs FSDP to shard — Raylight supports FSDP+USP). With FSDP2, the 33B BF16 weights shard ~20 GB/rank — fits R9700 (32) comfortably, tight on 7900 XT (20) but feasible with CPU offload.
- **Heterogeneity:** equal sequence split idles the R9700; a ~60/40 token split would balance but Ulysses all-to-all wants equal counts (pad to equal).
- **Numerics:** attention is mathematically identical (exact), only floating-point reduction ordering differs — **can preserve near-exact output**, unlike stale-activation methods.
- **Best case for the 5-s workload:** modest (sequence is short); **strong case for longer/2K clips** (capacity + latency).

### Design D — Ring Attention (streaming K/V)
- **Sharding:** sequence split; each rank streams its K/V blocks around the 2-GPU ring while queries stay local; overlaps comms with compute — **PCIe-friendly**, no all-to-all.
- **Collectives:** peer-to-peer send/recv per block; on 2 GPUs the "ring" is just a bidirectional exchange. Communication ≈ K/V bytes (2/3 of full activation).
- **Fit/complications:** H3 uses full (non-causal) attention over the packed sequence, so Ring has NO causal load-imbalance problem here — a plus. But 3D MM-RoPE and the modality tag must be applied to the correct global positions on each shard (bookkeeping).
- **Known ROCm issue:** Raylight reports "VRAM leakage when using Ring > 1 instead of Ulysses; increase Ulysses degree for now" — so on ROCm, **Ulysses is currently more reliable than Ring**.
- **Verdict:** viable hybrid partner but secondary to Ulysses on ROCm today.

### Design E — Hybrid USP (Ulysses×Ring 2D) — not justified at 2 GPUs
With only 2 GPUs, a 2×1 mesh is just Ulysses-2 or Ring-2; the hybrid buys nothing until ≥4 GPUs. Recommend pure Ulysses-2.

**Memory & comm scaling estimates:**
- 864×480/124f (~13k video tokens): per-rank activation ~halved; all-to-all volume per step ~O(13k×5376×2B) ≈ 140 MB × few exchanges. Marginal amortization at 4 steps.
- Longer clip (~362 frames, 15 s, H3's trained max per ComfyUI node tooltip): ~3× tokens → attention cost ~9× (quadratic) → SP amortizes far better; likely the **only way to fit/accelerate 15-s clips** on 20-32 GiB cards.
- Higher-res (representative 1280×720, no canonical local 2K since 2K routes through the hosted Regenerate-2K API): ~2.8× video tokens vs 864×480 → attention ~8× → SP strongly favored; **capacity win**.

---

## 8. Encoder-Placement Interaction

The Qwen3-VL-32B encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq`, ~14,960 MiB) currently sits on the 7900 XT; the denoiser (~19,984 MiB) on the R9700. On AMD the accelerated path is **AWQ W4A16, not Blackwell NVFP4 execution** (the nvfp4_awq quant is documented by Comfy-Org as not requiring a Blackwell GPU).

Four topologies:
1. **CPU Qwen (stable-diffusion.cpp `--backend te=cpu`):** encode once on Ryzen 9800X3D + RAM, freeing **both** GPUs entirely for H3 shards. Best complement to two-GPU H3 compute — but CPU encode latency must be measured (33B-class encoder on CPU may add seconds; only paid when conditioning is recomputed).
2. **7900-resident Qwen (current):** encode fast, but the 7900 XT is then occupied and cannot hold an H3 shard concurrently — incompatible with SP/TP that needs both GPUs for denoise.
3. **Phase-switched Qwen:** encode on 7900 XT, save conditioning, **unload Qwen, then load an H3 shard onto the 7900 XT**. This is the most promising for SP: it time-shares the 7900 XT between encode and denoise phases. Adds load/unload latency (model churn — exactly what the current dual-component setup avoids), so it trades churn for concurrent denoise.
4. **Precomputed conditioning:** cache the encoded conditioning tensor; recompute only on prompt change. The measured "changed prompt" case (7.9% improvement) shows conditioning recompute is a meaningful slice — precompute + reuse is a cheap, always-valid optimization independent of any parallelism.

**Recommendation:** For any SP prototype, use **CPU Qwen (option 1)** or **phase-switched (option 3)** so both GPUs are free during denoise. Option 1 is cleanest if CPU encode latency is acceptable.

---

## 9. Format / Runtime Decision

| Format | Sharded matmul kernels? | Dequant before shard? | Quant metadata partitionable? | Reduction dtype | Comms volume | Impl difficulty | Verdict |
|---|---|---|---|---|---|---|---|
| **Native FP8 `fp8_scaled` (current)** | Not for TP; FP8 GEMM on RDNA3 unproven | FP8 shards need BF16 compute; scales per-tensor | risky (per-tensor scale must follow shard) | BF16/FP32 | as BF16 activations | High | Correctness ref only if kept whole |
| **BF16 native safetensors** | Yes (standard) | No | n/a | FP32 accum | full BF16 activations | Medium | **Best correctness ref for SP prototype** |
| **INT8 convrot** | INT8 GEMM per-rank possible | INT8→BF16 for reductions | per-channel scale follows shard | BF16 | as BF16 | High | Better quality than fp8 per Comfy-Org note |
| **Unsloth GGUF Q6/Q4** | GGUF kernels are single-device; no sharded matmul | must dequant to shard across ranks | GGML block scales don't partition across ranks cleanly | BF16 | high (dequant traffic) | Very high | Not suited to TP/SP sharding |
| **Qwen AWQ (7900 XT) / GGUF Qwen (CPU)** | encoder only | — | — | — | one-time | Low | Encoder placement, not denoiser split |

Key points:
- **GGUF cannot be tensor-sharded cleanly** — GGML block-quant scales are per-block and don't partition across ranks; you'd dequantize first, defeating the purpose. GGUF is for single-device VRAM relief, not parallel compute.
- **For an SP prototype, sequence parallelism does NOT require sharding the weight matrices at all** (Ulysses replicates weights, shards activations; FSDP optionally shards weights by parameter, not by matmul dimension) — so **BF16 native is the correct starting format** for correctness, with FSDP handling the 20 GiB card. Quantize only after the reference SP path passes. Note the Turbo LoRA requires a non-pruned base (BF16 or full INT8-convrot), which aligns with a BF16 SP prototype.
- **stable-diffusion.cpp does NOT currently split one H3 denoiser across two local GPUs** — it assigns separate *components* to separate backends (e.g., `te=cpu`), which is component placement, not tensor/sequence parallelism.

---

## 10. Theoretical Performance Table (PROJECTIONS — not measured)

Base: cold 80.7 s (H3 48 s, decode/save 18 s, encode+overhead ~14.7 s). All figures are **projections**, clearly labeled.

| Scenario | H3 method | H3 speedup (proj.) | Collective overhead (proj.) | End-to-end (proj.) | Gain vs single R9700 |
|---|---|---|---|---:|---:|
| **Pessimistic** | TP-2, 4 Turbo steps, RCCL ~10 GB/s | 0.8× (net slower) | +8-12 s | ~88-92 s | **−9 to −14% (worse)** |
| **Pessimistic** | SP Ulysses-2, 4 steps, RCCL broken | n/a | deadlock | no result | fails Phase 0 |
| **Realistic** | SP Ulysses-2, 4 steps, RCCL fixed, ~30 GB/s | 1.3-1.4× | +3-5 s | ~66-70 s | **+13 to +18%** |
| **Optimistic** | SP Ulysses-2 + FSDP, 4 steps, ~48 GB/s, overlap | 1.6× | +2 s | ~62-64 s | **+21 to +23%** |
| **Realistic (20-step lane)** | SP Ulysses-2, 20 steps | 1.6-1.8× | amortized | proportionally larger H3 share | **+25-32%** (H3 dominates) |
| **Capacity (15 s / 2K-class)** | SP Ulysses-2 | enables clip that OOMs on 1 GPU | — | — | **capacity win, not latency** |

Interpretation: **No TP scenario clears the 10% latency gate; the best-case SP scenario reaches ~15-22% but only after RCCL is fixed and only if collective overhead stays small.** The 4-step Turbo lane is the hardest; the 20-50-step lane is where SP shines. The single most valuable *guaranteed* result is a capacity win (longer/higher-res clips) via SP.

---

## 11. Staged Experiment & Implementation Plan

**Preserve the golden environment:** copy the pinned ComfyUI (commit `c2bcbecd82ec5ae66594340b395c24ef0217b238`) into an isolated venv/container; never modify the working lane. Run all experiments as separate processes/service.

### Phase 0 — Topology & communication preflight (HARD GATE)
Tools (pin exact versions): `rocminfo`, `rocm-bandwidth-test`, `rccl-tests` (all_reduce/all_gather/reduce_scatter/all_to_all), `rocm-smi`, `amd-smi`, `dmesg`/journal watch.
- Confirm logical mapping under `HIP_VISIBLE_DEVICES=1,0` (cuda:0=R9700/gfx1201, cuda:1=7900XT/gfx1100).
- `rocm-bandwidth-test` bidirectional P2P — record actual GB/s; **verify P2P is truly used (not falsely reported)**.
- Build RCCL with BOTH `--amdgpu_targets gfx1201 gfx1100` (fat binary); run a 2-rank `torch.distributed` all_reduce + all_to_all across the mixed pair over the message sizes H3 uses (8B-256MB sweep).
- Watch for the known deadlock (both GPUs 100%, no progress) / `HIP failure: operation cannot be performed in present state`.
- Try mitigations from community reports: `NCCL_P2P_DISABLE=1`, `RCCL_NET=Socket`, raise `/dev/shm` to 16 GiB, disable AITER-equivalent paths.
- **No-go criteria (abort project):** any RCCL collective deadlocks/errors on the mixed pair and cannot be worked around; OR host-staged fallback bandwidth <~8 GB/s (comms will dominate); OR GPU reset/VM fault during the sweep. Given current evidence, **this phase is likely to fail on ROCm 7.2.1 / RCCL 2.27.7** — that is the cheap falsification test.

### Phase 1 — One-block correctness prototype (no ComfyUI)
- Construct/load ONE H3 transformer block (BF16). Single-GPU reference forward.
- Implement Ulysses all-to-all SP across 2 ranks for that block; compare output vs reference at justified BF16 tolerance (e.g., rtol 1e-2, atol 1e-2 given the attention-kernel noise floor cited in diffusers' bit-for-bit verification of the converted transformer at 30 steps).
- Record compute vs collective time separately; **prove both GPUs compute concurrently** (rocprof timeline overlap).

### Phase 2 — Full H3 forward as a service (one process/GPU, spawn-safe)
- Start from BF16 native (correctness). Preserve real conditioning, packing, dual-clock schedules.
- Minimal frame grid (e.g., 5-frame) before 124 frames. Add FSDP for the 20 GiB card. Quantize only after reference passes.

### Phase 3 — Turbo + ComfyUI integration
- Add Turbo v4 LoRA + 4-step dual-clock sampler (video shift 12, audio shift 3).
- Integrate as an **external inference service / ComfyUI server node** (NOT global monkey-patch); clean fallback to single-GPU on any failure; golden workflow untouched.

### Phase 4 — SP/CP prototype proper
- Only after Phase 0 collective numbers are known. Compare TP vs Ulysses-SP vs hybrid at identical prompt/seed/shape. Include 362-frame and 1280×720 memory-scaling tests.

### Phase 5 — Controlled benchmarks
- For each lane: process-cold, warm×2, same-prompt/new-seed, changed-prompt. Record full config (resolution, frames, fps, sampler, scheduler, steps, denoise, checkpoint, LoRA), total wall, prompt-exec, H3 time, encode, decode/save, per-GPU util/VRAM/temp/power/clocks, collective time+bytes, CPU RAM/util, output hashes, `ffprobe`/decode-to-null validation, A/B audio-video review, and any new AMDGPU/RCCL kernel errors. Do not average cold/warm; do not invent component timings telemetry can't support.

---

## 12. Risk Register & Abort Conditions
- **RCCL mixed-arch collective deadlock/error (HIGH, likely):** primary abort. Evidence: #5480, #40980, #6074 all open.
- **False-positive P2P → GPU reset / VM fault (MED-HIGH):** documented on consumer RDNA; abort on any reset.
- **7900 XT 20 GiB OOM under replicated weights (MED):** mitigate with FSDP/CPU offload; else asymmetric shard.
- **Faster-rank (R9700) idle from equal split (MED):** wastes the better GPU; needs weighted split (research-grade).
- **4-step amortization failure (MED):** collective setup not amortized → net-negative; mitigate by validating on 20-step first.
- **Numerical drift / audio instability (MED):** dual-clock schedule + 4-step is fragile (documented 4-step audio clipping/high noise floor); require stereo-audio validation + A/B.
- **Kernel/cache cross-contamination between gfx1100/gfx1201 (MED):** isolate Triton/HIP caches per architecture.
- **Fork-unsafety in ComfyUI (MED):** must use spawn + separate processes.
- **Failure conditions = GPU reset, VM fault, OOM, corrupt frames, NaNs, unstable audio, thermal throttling.**

---

## 13. Recommendation

**Decision: Prototype-first, but expect "do not pursue for latency" — and run the cheap falsification test before any H3 work.**

1. **First prototype = Phase-0 RCCL/P2P preflight on the mixed pair.** This is the earliest, cheapest falsification test and it gates everything. It costs a day, touches no H3 code, and — given the open dual-R9700 (#5480) and dual-7900XTX (#6074) collective regressions on ROCm 7.2.1/RCCL 2.27.7 plus the total absence of mixed-arch evidence — it is **more likely to fail than pass** on the current stack. A negative result here is a legitimate, valuable project outcome: it says "wait for an RCCL fix (or downgrade to an RCCL 2.27.3-class build) before spending effort."
2. **If Phase 0 passes, the first real H3 prototype = Ulysses-2 sequence parallelism via Raylight** (or a diffusers `ContextParallelConfig` service), BF16 + FSDP, CPU or phase-switched Qwen encoder. **Why it beats the alternatives:** H3 is a single-stream dense transformer with full attention over one packed sequence — the exact shape Ulysses/USP is built for; MiniMax and AMD both ship H3 with sequence parallelism; it preserves near-exact numerics; it shards activations (not fragile quant metadata); and it degrades gracefully to a capacity win for long/high-res clips even if the 5-s latency gain is modest. TP is rejected because its 400 synchronizations at 4 Turbo steps over consumer PCIe are projected net-negative. Pipeline/PipeFusion is rejected because 4 steps can't fill the pipeline. GGUF sharding is rejected because block-quant scales don't partition.
3. **Engineering effort estimate:** Phase 0: ~1-2 days. Phases 1-2 (BF16 SP one-block → full forward service): ~2-4 weeks. Phase 3 (Turbo + ComfyUI service integration): ~1-2 weeks. Realistically **~6-8 weeks** to a benchmarked SP lane, contingent on RCCL working.
4. **Earliest cheap falsification test:** the Phase-0 mixed-arch `rccl-tests all_reduce/all_to_all` run. If it deadlocks/errors, stop and redirect.

**If Phase 0 fails (expected on current stack):** redirect the second GPU to the wins that already work — **concurrent-job throughput** (two independent videos), **encoder/VAE placement** to free the R9700, and **precomputed conditioning** — and revisit SP after an RCCL release fixes consumer-RDNA collectives.

**Decision framing (final):** **Do not pursue two-way TP for single-video latency** — communication cost, the 4-step Turbo amortization problem, and current RCCL runtime gaps make a TP latency win unlikely. **Prototype-first for SP/Ulysses**, strictly gated on the Phase-0 RCCL preflight; until that passes on this mixed pair, redirect the second GPU to capacity and concurrent-throughput wins.

---

## 14. Open Questions & Evidence Gaps
- **Exact packed-sequence token count** for 864×480/124f after 17k+5 grid padding — needs instrumentation (I estimated ~13k video tokens).
- **Mixed gfx1100+gfx1201 single process-group behavior** — zero public evidence; must be measured.
- **Whether an RCCL 2.27.3-class (last-known-working) build can be paired with ROCm 7.2.1** on this pair as a workaround.
- **CPU Qwen encode latency** on the 9800X3D — determines whether option-1 encoder placement is viable.
- **FP8/INT8 sharded GEMM correctness on RDNA3** — unverified; affects whether quantized SP is possible.
- **Real per-rank compute imbalance** (R9700 vs 7900 XT at H3 shapes) — needs standalone GEMM/attention benchmarks to size a weighted split.
- **Raylight's actual H3 code path on ROCm consumer GPUs** — "Minimax H3" is listed but only NVIDIA/Instinct-validated.

---

## 15. Annotated Sources
- **MiniMaxAI/MiniMax-H3 model card** (huggingface.co/MiniMaxAI/MiniMax-H3) — official architecture, VAEs (f16t4d24 visual, 40 Hz audio), sparse-attention-not-released note, SGLang `--ulysses-degree 4`. Primary architecture source.
- **AMD "Day 0 Support for MiniMax-H3 on AMD Instinct GPUs"** (amd.com, Aug 2, 2026) — the definitive DiT config table (5376 hidden, 50 layers, 56×128 heads, 14336 FFN, BF16), ROCm SP/Ulysses recipe (`--sp-degree 8 --ulysses-degree 8`), AITER attention, spawn requirement. Primary.
- **huggingface/diffusers PR #14355 + minimax_h3.md** — packed-sequence layout, two-input-proj/two-output-head/AdaLN structure, bit-for-bit verification at 30 steps, ContextParallelConfig. Primary implementation.
- **ComfyUI `comfy_extras/nodes_minimax_h3.py` + `comfy.ldm.minimax.model`** — native node defaults (length 124, 17k+5 grid, trained 124-362), packing order `[text|refs|audio|video]`. Primary (golden path).
- **ROCm/rocm-systems #5480** (Apr 27, 2026, OPEN) — dual-R9700 gfx1201 RCCL TP=2 deadlock, first-collective failure, regression from RCCL 2.27.3. Critical negative evidence.
- **vllm-project/vllm #40980** (OPEN) — same deadlock, vLLM side (v0.19.1+rocm721, RCCL 2.27.7, gfx1201), community workaround (P2P off, Socket, /dev/shm 16G). Critical negative.
- **ROCm/ROCm #6074** — dual-7900XTX gfx1100 first-collective `HIP failure: operation cannot be performed in present state`. Critical negative (RDNA3 side).
- **ROCm/ROCm #5571** (Oct 24, 2025) — R9700 rccl-tests all_reduce busbw ~10.8 GB/s, "P2P not being used." Bandwidth evidence.
- **ROCm/ROCm #2253** — 2× 7900 XTX rocm-bandwidth-test ~28 GB/s unidir / ~48 GB/s bidir. Bandwidth evidence.
- **komikndr/Raylight** (GitHub) — USP+FSDP+CFG for ComfyUI, "Minimax H3" listed, Ring VRAM-leak note, MI210/MI300X confirmations. Highest-value reusable SP tooling.
- **xdit-project/xDiT + rocm/pytorch-xdit** — USP/PipeFusion/DistriFusion engine; ROCm images "MI300X or newer." Underlying SP engine.
- **AMD ROCm Blogs: "Accelerating Video Generation on ROCm with USP"** — USP Ulysses/Ring mechanics, Ulysses-8/Ring-1 optimal on MI300X. Theory + ROCm validation.
- **vLLM Recipes (recipes.vllm.ai/MiniMaxAI/MiniMax-H3)** — 2× RTX 5090 TP config and 4× MI300X USP verification; shows H3 multi-GPU only on datacenter/NVIDIA. Reference for TP/USP primitives.
- **PipeFusion (arXiv 2405.14430) / DistriFusion (arXiv 2402.19481)** — PCIe-friendly pipeline theory; homogeneous/power-of-2 assumptions; warmup-step sensitivity. Adjacent-work theory.
- **STADI (arXiv 2509.04719)** — heterogeneous-GPU straggler problem + temporal adaptation; directly relevant to the R9700/7900XT imbalance.
- **larryvrh/MiniMax-H3-Turbo-Lora + Larryvrh/ComfyUI-MiniMax-H3-Turbo** — Turbo LoRA (v4 step600 EMA), dual-clock 4-step sampler (video shift 12, audio shift 3), LoRA-as-standard-low-rank-update, non-pruned base requirement. Primary for Turbo behavior.
- **Diffusers distributed-inference docs (ContextParallelConfig)** — Ring vs Ulysses API, mesh dims. SP reference.
- **fangpenlin.com (Jun 2025)** — 2× 7900 XTX P2P only works after enabling large-BAR/HSA_AMD_P2P; PCIe-shared bandwidth caveat. Community P2P evidence.