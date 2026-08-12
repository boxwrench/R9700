# 1. Executive Verdict

Two-way multi-GPU H3 inference **is technically conceivable but highly challenging** on your AMD setup today. In principle, splitting the 4.3K×124-frame joint video/audio diffusion transformer across the 32 GB R9700 and 20 GB 7900 XT could reduce *single-video* latency, since both cards have free compute when operating concurrently. However, ROCm 7.2.1 on RDNA3/4 currently **fails basic collectives** between GPUs, and no existing toolkit cleanly implements H3-specific TP or SP for AMD.  Sequence-parallel methods (e.g. Raylight’s USP) or tensor-parallel methods (Megatron-style head-splitting) **theoretically** harness both GPUs, but each incurs heavy PCIe/collective overhead, and the failing RCCL issue suggests a high risk of broken runs. The only low-friction multi-GPU gains currently come from **component placement** – e.g. moving the ~12 GB Qwen3-VL encoder to the 7900 while the R9700 does the DiT – which your dual-card tests already show as a small (~4–8%) speedup. In short:

- **Tensor-parallel H3 (TP)**: *Possible but prototype-level.* Two designs (head-splitting and Megatron-column/row splitting) could run, but both need all-reduce/all-gather operations per block. Without proven ROCm NCCL reliability on dual Radeons, TP is **risky** until the communication bug is fixed. Even if working, we must carefully shard the 7168-dim attention and 14336-dim MLP projections, sync layer outputs, and handle AdaLN/LoRA merges. Engineering cost is very high.

- **Sequence/context-parallel H3 (SP/CP)**: *Likely more promising*, since H3’s 12M+ token sequence can be split. Raylight’s Unified Sequence Parallelism (USP) already handles H3 on NVIDIA. Two GPUs could each process half the tokens (e.g. half the frames). But full attention requires exchanging K/V across GPUs each block and then assembling outputs, a huge PCIe cost. Without NCCL, one would resort to manual HIP peer copies or blocking gathers – very slow and hacky. A hybrid (half heads + half sequence) might amortize communication but greatly complicates correctness.

**Bottom line**: Given current evidence, a *two-GPU H3 solution could become useful but is not yet turnkey.* The safest path is to **prototype** carefully: first measure raw PCIe bandwidth/latency and a toy two-block PyTorch test, then try a minimal two-rank split. We should not bet on a large (>10%) *latency* win until we see strong empirical results; at best we may gain ~10–15% wall-time (per Amdahl’s Law) while risking model instability. However, if the goal were *maximum clip size* rather than speed, multi-GPU is more obviously worthwhile. For your top priority (short-video latency), I recommend **“Prototype First”**: do controlled two-GPU tests per the plan below, but be prepared to bail if multi-GPU overheads or ROCm bugs dominate.  

# 2. Verified H3 Architecture

According to MiniMax documentation and AMD’s Day-0 report, the **MiniMax-H3 DiT transformer** has:

- **Hidden dimension:** 5376  
- **Layers:** 50 (plus 2 extra “refiner” blocks)  
- **Attention:** 56 heads, each head dimension 128 (total 56×128=7168)  
- **Feed-Forward (MLP):** inner size 14336  
- **Video latent channels:** 24 (patch size 2×2 spatial)  
- **Audio latent channels:** 32  
- **Tokenization:** The video (and audio) latents are flattened into one sequence for a joint denoising pass. With patch (T,H,W)=(1,2,2), a 864×480×124 workload yields ~432×240 tokens per frame×124≈**12.85M tokens** (each token is a 24-vector) plus audio tokens.  

Internally, each layer does: query/key/value projections (5376→7168 each), multi-head self-attention, a 14336→5376 MLP (GELU, etc), and an output projection. AdaLN (adaptive LayerNorm) conditioned on timestep is used, and sinusoidal Rotary position embedding (3D RoPE) is applied. The model outputs *24-channel video latents* and *32-channel audio latents*, which are later decoded by the VAE components. All H3 weights are stored BF16 (or INT8 in optimized kernels). The 4-step **Turbo LoRA** is a post-training weight patch: it does not change the network shape (it adds low-rank matrices to layers). In summary, H3 is a *single-stream transformer* processing millions of tokens, with no built-in “expert” branches – all modalities intermix at every layer.

*(Data-flow diagram:)* Video/audio frames → VideoVAE (24-ch latent) + AudioVAE (32-ch latent) → Flatten spatial/temporal into token sequence → add conditioning (text encoder, keyframes, etc) → 50-layer transformer (self-attn + MLP) → split outputs to video and audio heads → decode via VAEs to 24-ch video + 32-ch audio waveforms. Key dims for 864×480×124 (24fps) are: token sequence ≈12.8M, hidden=5376, head=128, output latents (24×432×240 per frame). These numbers dictate any parallel split. 

# 3. Fundamental Theory of Parallelism

We consider every plausible parallel strategy and its impact on one-video latency:

- **Component placement (model parallelism):** e.g. put the text encoder on GPU B and DiT+VAEs on GPU A. *Latency:* Not reduced for DiT, but saves VRAM on GPU A so larger clips fit. We already use that: 7900 XT encodes prompts, R9700 does diffusion. This adds ~4–8% speedup (from removing encoder reloading) but still left R9700 doing 100% of denoising. It does *not* split computation of one block; it only offloads independent parts. (We should test also moving VAEs or Audiovae to 7900, but their compute is minor vs DiT). CPU offload (all CPU encoder, or block swap) likewise moves work off GPU but *slows* inference, not speeds it (except if CPU runs concurrently, which it doesn’t effectively).

- **Layer/block offload (sequential model partition):** e.g. GPU0 runs DiT layers 1–25, GPU1 runs 26–50. This is a form of pipeline parallelism: each block of layers is on a different card. The GPUs alternate: after layer 25, GPU0 must send its last hidden state (size [total_tokens×5376]) to GPU1, which then finishes the forward pass. In an *ideal* 2-stage pipeline, total compute ~same as one GPU (minus overlapping overhead). But in reality, you must wait after layer 25, incurring a synchronization + PCIe memcpy of ~hidden_size×seq_len. For our test (12.8M tokens × 5376 dims × 2 GPUs ≈ 110B numbers ≈ 440 GB) – utterly impossible to transfer every block. Even dividing sequences reduces size but still huge. So pure pipeline would give at most ~2× compute but ~2× latency from transfer, essentially no gain. It *does* use both GPUs’ arithmetic, but only sequentially, so minimal concurrency. Pipeline is better suited for many GPUs but expensive here. So for **latency** it’s poor (each frame still waits all layers sequentially).

- **Tensor parallelism:** Shard each transformer *layer* across GPUs so both do work for the *same block at the same time*. Two approaches:

  - *Head-parallel (Megatron “tensor parallel”):* Split the 56 attention heads roughly in half. Each GPU has its own Wq/Wk/Wv for 28 heads (5376×3584 weight shards). Each GPU computes attention for its 28 heads (Q,K,V of shape [seq×3584]) and yields [seq×(28×128)] outputs. To combine, one must gather these partial outputs into the full [seq×7168] hidden (simple concatenation along feature dim). Similarly, the MLP: each GPU can hold half of the FFN’s output weights (5376×7168 for first linear, 7168×5376 second linear) and compute half of the activations, then gather. Communication: one all-gather of hidden features at each layer (two GPUs = each sends/receives  half the hidden state). **Latency impact:** In theory compute per GPU halves, but after each layer a full-feature gather of size ~hidden_size×seq_len (~5376×12.8M ≈ 69B numbers, ~276 GB!) is needed. Even if sequence is split, communication is enormous. If PCIe yields ~12 GB/s, the cross-GPU gather would take ~23 seconds per layer (out of scope). So pure head-splitting is infeasible for our large sequence. It does preserve exact output, but too much sync.  

  - *Megatron column/row parallel:* Shard the linear weights themselves. E.g., split each Q/K/V matrix’s columns: GPU0 has Wq[:,:3584], GPU1 Wq[:,3584:]. Then X @ Wq yields two partial Q pieces which must be concatenated. This similarly ends up splitting Q/K/V half and then requiring all-gather of Q or all-reduce of outputs. The MLP could be split similarly (one GPU computes X @ W1 parts, then all-gather, then each does its part of W2). Essentially the same communication cost as head-splitting (allreduce/all-gather each layer). 

  *Invariant:* Any attempt to shard within-layer demands heavy collectives (all-gather hidden states or reduce outputs each block) to assemble the complete result for the next layer. If collectives fail or are slow, both TP designs break. **If everything worked**, TP could cut raw compute per GPU ~2×, but (Amdahl’s) end-to-end speedup is limited by the overhead of O(#layers×communication). For 50 layers, even modest communication kills the speedup.

- **Sequence/context parallelism:** Split the *token sequence* or input frames between GPUs. For example, GPU0 processes tokens 0–half, GPU1 the rest. Two variants:

  - *Sequence Parallel (Unified Sequence / Ulysses):* Each GPU holds half the token dimension. For each self-attention layer, each GPU needs the other’s K/V (all-to-all): e.g. GPU0 computes K0,V0 from tokens0 and sends them to GPU1; GPU1 sends K1,V1 to GPU0. Then each GPU can compute attention for its local Q (since it now has full global K/V). After attention and MLP, each GPU outputs new hidden for half the tokens. Finally, to feed next layer, we must gather the full hidden sequence (or equivalently partition tokens again). This is exactly what Raylight’s USP does. *Latency:* Two PCIe transfers (exchange of half-sequence activations) per layer plus an all-gather of hidden states. Communication ∼ O(seq_len×head_dim) per layer ~ tens of gigabytes per layer. With tens of layers, this again dwarfs compute. If somehow fused or overlapped, maybe partial benefit, but with ROCm NCCL bug, this is currently unsafe.  

  - *Context Parallel (frame-block split):* Use each GPU to generate different parts of the video or different noise timesteps (like expert branches). For example, split the frames into two mini-videos and decode each separately. This breaks the single-video constraint, so not allowed. Alternatively, one could try cross-attention splitting (conditioning split), but H3 uses no CFG in forward (Gaussian guidance), so no trivial split.

  - *Ring Attention:* A streaming approach where GPUs pass K/V “round robin” instead of full all-gather per layer. For 2 GPUs, this reduces to a send/receive per layer per GPU anyway. Not a big gain. 

In summary, **none of these textbook parallelisms offer a free lunch**: they either require multi-GPU collectives (which currently fail) or huge PCIe bandwidth far beyond practical. All *reduce-latency* methods end up requiring something like an all-gather at each block. As a result, they will only show net benefit if the compute saved (almost 2× shorter critical path) outweighs the communication cost. With a 32 GB / 20 GB machine lacking NVLink, that threshold is very high.

- **Data-parallel (multi-seed)**: Splitting multiple prompts or seeds across GPUs can improve *throughput* (e.g. batch=2 with different seeds), but by definition does *not* speed up a single output. It is outside this target of one video’s latency, though it is trivial with ComfyUI-Distributed (not GPU-sharing). We will label any such gain a “throughput” win, not a “latency” win.

- **Prompt/Conditioning parallelism:** You could compute multiple prompts or multi-modal inputs in parallel threads, but H3’s forward pass consumes one set of tokens, so this again only helps if batching seeds (throughput). Asynchronous encoding (overlap prompt encoding and first step) could hide encoder time, but in practice our text encoder is already on 7900 and just takes seconds, far less than the ~48s diffusion time. 

**Preserving output exactness:** Note all parallel sharding *must* reconstruct exactly the same floating outputs as the reference. Head-parallel/column-parallel preserve exactness (ignoring FP16/FP8 rounding) because they just redistribute computations. Sequence-parallel with splits will also produce identical results (aside from added floating errors) if done correctly. Only “expert splitting” (like pipeline partial steps) could change the math order slightly, which we’ll avoid. Multi-GPU code should accumulate in FP32 where needed (e.g. FP8→FP32).

**Batch size=1, Turbo=4 steps:** All our schemes assume one forward pass per transformer block, as with batch=1. Distributed layers (like P2P) should work with batch1. The 4-step sampler means 4 full transformer runs per clip; this reduces overall compute vs 20-step, but does not change how parallelism scales (less total work to parallelize). 

In sum, the **only robust, latency-reducing approach** without exotic collective tricks is *pure component placement* (e.g. encoder on GPU0, model on GPU1) plus **distributing the heavy tensor ops themselves if HPC connectivity were solved**. We must now check what existing tools can actually realize any of these strategies in ROCm.

# 4. Ecosystem and Community Support Matrix

We surveyed existing projects for multi-GPU or distributed inference, focusing on H3 or similar large diffusion models (video/XL). Summarized:

| Project                          | Method                              | Inference vs Train | Video/DiT Support          | H3 Support    | ROCm Support     | Mixed AMD (RDNA3/4)  | Reusable Value                               |
|----------------------------------|-------------------------------------|---------------------|----------------------------|---------------|------------------|----------------------|---------------------------------------------|
| **ComfyUI (native)**             | Component placement (single GPU)    | Inference          | Native H3 pipeline, DiT   | Native (v0.30+) | Yes (ROCm)       | N/A (single GPU)     | Baseline; can put encoder on 2nd GPU       |
| **ComfyUI-MultiGPU (pollockjj)** | DisTorch weight-split, virtual VRAM | Inference          | Not specialized for video  | Unclear       | Likely (CUDA/ROCm?) | Possibly (Torch)      | Shards model weights (UNet etc), uses CPU offload for VRAM |
| **Raylight (komikndr)**          | USP (sequence), FSDP (tensor)       | Inference          | H3 explicitly listed | Claims H3      | CUDA only (NCCL) | CUDA-only (NCCL)     | Implements unified sequence parallelism (USP) + FSDP weight split |
| **ComfyUI-Distributed (robertvoy)** | Data-parallel (multi-seed)      | Inference          | Video (tiles)               | No (NVIDIA only)| NO (NVIDIA)     | No (NVIDIA only)     | Multi-machine, throughput focus       |
| **ComfyUI-Networked (netdist)**  | Data-parallel (jobs)                | Inference          | No                         | No            | NO              | No                   | Similar to Distributed, NVIDIA only        |
| **ComfyUI-WorkSplit**            | CFG-level parallel (batch split)    | Inference          | Not relevant               | No            | Possibly CUDA   | No                   | Limited use, NVIDIA-oriented               |
| **xDiT / AITER (MIT)**           | Custom attention kernels (variable) | Inference (pack)   | DiT multi-head attention   | Maybe (via SD) | CUDA-only (NCCL) | No (NVIDIA only)     | AMD blog uses AITER as backend on MI300 |
| **Diffusers (Accelerate)**       | Data-parallel (accelerate)          | Inference          | No (API doc)              | No (H3 in HF)  | No (Torch DP)    | Theoretically yes    | Batch inference on multiple GPUs   |
| **SGLang (AMD)**                 | Ulysses/Sequence parallelism        | Inference (AMD)    | Yes (was tested on MI355X) | Yes (official) | Yes (ROCm)       | N/A (Homog AMD)      | Used 8-way SP on AMD Instinct  |
| **stable-diffusion.cpp**         | CPU offload, non-batched CUDA       | Inference (CPUs)   | No (tools for H3 exist)    | H3 support via GGUF | Partial (HIP backend on Linux) | Not multi-GPU | Can run encoder on CPU, denoiser on one GPU |
| **PyTorch DP / DDP**             | Data-parallel (multi-seed)          | Inference/Train    | N/A                        | N/A           | Yes (Torch distributed) | Yes                 | Only batches images, not single video     |
| **DTensor / DeviceMesh**         | (Speculative)                     | ??? (proposed)      | None (no video example)    | Unlikely      | No (not released on ROCm) | No               | Torch distributed CT kernels, unlikely working |
| **TorchPipeline/FSDP**           | Model parallel (FSDP)               | Train              | No                         | No            | Partially (patchy) | No (not supported on ROCm fully) | FSDP splits layers across GPUs, but ROCm lacks it for inference |
| **NVIDIA Megatron-LM**           | Tensor parallel (heads/split)       | Train              | Custom (LLMs, no video)    | No            | No (CUDA)        | No                   | Approach can inspire H3 design              |
| **DeepSpeed-Ulysses / DSPipe**   | Sequence-based alltoall            | Inference          | No (LLM-focused)           | No            | No (CUDA)        | No                   | Concept of splitting KV (like USP) but CUDA |
| **HuggingFace Accelerate**       | Data-parallel                        | Inference          | No                         | No            | No (Torch DDP)   | N/A (multi-node)     | Only parallel prompts      |
| **community experiments**        | (Rosen DiT, LTX2, etc)            | Inference          | Some on video (LTX, H2)   | No H3 yet     | Mixed (some ROCm) | Mixed (some RDNA)    | Empirical: no known multi-GPU H3 success; primarily single-GPU runs |

**Key observations:** 

- **H3-specific**: AMD’s official SGLang and ComfyUI both support H3, but *no existing tool for mixing H3 across GPUs on ROCm*. Raylight and MultiGPU target other models. ComfyUI’s default H3 node is single-GPU only. Unsloth’s GGUF H3 is CPU-centric.

- **ROCm support**: Very few projects support AMD. ComfyUI itself runs on ROCm, as does SGLang on AMD. However, Raylight and Distributed explicitly assume NCCL/CUDA, so they cannot currently run on ROCm. Pollock’s MultiGPU doesn’t mention AMD but may work on Torch-ROCm’s CUDA API. SGLang uses a different stack (SGLang on ROCm is current).

- **Mixed AMD**: No project explicitly tests RDNA3+4 mixtures. The ROCm bug suggests even two RDNA3’s fail; RDNA3+4 is untested. There is no documented AMD-Primus or ROCm tuning for mixed consumer cards.

- **Data vs Tensor vs Sequence**: Almost all existing multi-GPU Diffusion tools (Accelerate, Distributed) target *data parallelism*. Raylight is an exception using USP. No known library does in-place tensor-parallel diffusion on AMD GPUs yet.

This matrix shows we’re in mostly uncharted territory: no turnkey pathway is documented. At best we can repurpose Raylight’s ideas or MultiGPU offloading. But those need serious adaptation for AMD. The AMD blog confirms H3 multi-GPU scaling (8 GPUs, sequence-splitting, custom kernels), but via their own SGLang, not accessible to us. 

# 5. Hardware Feasibility Assessment (AMD specifics)

**Mixed RDNA3/4 & ROCm:** ROCm’s `torch.cuda` API will see R9700 (gfx1201) and 7900XT (gfx1100) as `cuda:0`/`cuda:1` if you set `HIP_VISIBLE_DEVICES=1,0`. Hardware-peer access is “bidirectional” as you noted, meaning logically `hipPeerEnable()` works. However, actual *ROCm collectives* currently **fail on consumer Radeons**. A concrete AMD/ROCm GitHub issue shows even a minimal PyTorch 2.9.1+ROCm7.2 test with two RX7900s fails at `dist.barrier()`. In that test, the first collective barfs with a HIP error on rank1. The topology was reported as PHB (PCIe). Disabling P2P in NCCL did not help. This means **NCCL/RCCL doesn’t work reliably on ROCm 7.2.1** for these cards. The exact error: 

> *“DistBackendError: NCCL error in ProcessGroupNCCL.cpp ... HIP failure: 'the operation cannot be performed in the present state'”*. 

In other words, any approach needing `dist.init_process_group` with NCCL/RCCL (e.g. torch.distributed or Raylight) will likely crash or deadlock. Until a fix arrives, **avoid using torch.distributed on this dual-AMD setup**. 

**PCIe bandwidth:** Both GPUs are on PCIe (likely Gen4 ×16). The ROCm report said “path bandwidth 6.0” (presumably 64 GB/s per direction). Practical throughput is lower (bidirectional ~100 GB/s). Communication of tens of gigabytes per layer (see Sec.3) thus costs on the order of seconds. This dwarfs the ~1s raw compute per layer (48s for 50 layers). Without NVLink or XGMI, all data goes through PCIe, so high-latency. We should actually measure P2P copy speed and latency (see phase 0 below).

**Process vs Threads:** ROCm does not require separate processes per GPU; a single process can `cuda.set_device(i)`. However, torch.distributed often spawns one process/GPU. If bypassing torch.distributed, one can write a single-process multi-device code. That avoids inter-process sync but is tricky to code by hand. ComfyUI usually runs in one process controlling multiple GPUs, which is feasible. Both GPUs have HIP-visible device IDs and can share a context. However, some custom multi-GPU nodes assume separate processes/workers.

**Hip/ROCM kernels:** HIP kernels (compiled by Triton or Torch) may be built per architecture. If JIT'ed on RDNA3 vs RDNA4, PyTorch likely compiles separately. Running a Mixed graph (one op on gfx1201, one on gfx1100) may require two separate kernel binaries. Usually PyTorch’s caching handles this transparently, but fallback might recompile on first use of the other GPU. Should be fine if all shape-compatible.

**Library support:** Triton 3.5.1+ROCm is our stack. Raylight’s README warned about NCCL needing a special build on ROCm; it uses NVIDIA NCCL 2.28.9. We won’t attempt NCCL. TorchDP/GPU offload may work (DisTorch uses torch functions, not NCCL). HIP P2P copies (`hipMemcpyPeer`) should work for equal-size contiguous tensors. But aggregated all-gather need either NCCL or manual copies. HIP peer copy exists, but not sure if we can call it manually from Python (torch.cuda might do it if we use `x.cuda(device=1)` etc). We can test it (phase 0).

**No thermal/power issues expected** beyond normal. 7900XT is 300W, R9700 is ~200W. Running them together is safe.

**Summary:** There is no hardware-level blocker (P2P is present), but *software-level blockers* (RCCL bug) and *bandwidth limits* are. We will need to carefully gate feasibility: if simple distributed tests with torch.distributed fail, that’s an immediate abort criterion.

# 6. Concrete Tensor-Parallel Designs

We explore **two candidate TP shardings** of H3’s DiT blocks, each with detailed memory and comm estimates. The bench sequence is 864×480×124 (video tokens ≈12.85M). Hidden size H=5376, head=128, L=50 layers.

## Design A: *Head-parallel (50/50 attention heads)*

**Sharding plan:** Split the 56 attention heads equally (28 on each GPU). Each GPU holds weight slices for those heads:

- **Self-attention:** On GPU0: Wq0 (5376×3584), Wk0, Wv0, Wo0 (7168×5376 output matrix covering heads0-27). Similarly GPU1 holds Wq1 (5376×3584) etc for heads28-55. Each GPU receives the full input hidden X ([seq×5376]) and computes local Q0=X·Wq0 (seq×3584), K0, V0. It then performs attention per-head independently and produces Y0 (seq×3584). No communication needed *during* attention (heads are independent). 

- **Combine outputs:** We need to form the full output [seq×5376] = concat(Y0,Y1). This requires each GPU to obtain the other’s Y. Instead of all-gather, we could e.g. have GPU0 send Y0 to GPU1 and GPU1 send Y1 to GPU0 (bidirectional peer copy). Then each constructs [Y0;Y1] locally. That costs transferring 2×(seq×3584 floats) total. Seq≈12.85M, floats=4B: transfer ~12.85M×3584×4 ≈ 184 GB (total for both directions per layer!). At 100 GB/s combined, ~1.8 s per layer *just to assemble outputs*, i.e. ~90s per 50-layer pass (far above baseline ~48s). That alone dwarfs compute.

- **MLP projection:** Next, each GPU must do the FFN. If we also split the FFN output in half, similar logic: First linear (X_full×W1, W1 split [5376×7168] into two [5376×3584] each). GPU0 and GPU1 each compute [seq×3584] intermediate, send halves, combine to [seq×7168], apply activation, then second linear split [7168×5376] similarly. Communication ~same magnitude. 

- **Synchronizations:** We must sync before each attention and MLP so that both have the same X input (given we just split last output? Actually both GPUs will have the full output after gather). If we keep both GPUs updated with full [seq×5376] hidden at each layer’s start, we need to broadcast the new hidden from one to the other every layer. But since we already all-gathered outputs, each GPU *does* have the full hidden locally for the next layer; no extra step. But that combined gather is extremely heavy as shown.

- **Memory:** Each GPU stores ~half the parameters. The full DiT size (~4.3B weights) is halved ~2.15B per GPU. That’s about *2.15e9* × 2 bytes (BF16) ≈ 4.3 GiB; plus overhead (LayerNorm, etc, say ~5 GiB each). Activation: each holds X and some intermediates. Hidden state [seq×5376] = 12.85e6×5376×2 bytes ≈ 138 GB; obviously that cannot fit! We must tile tokens, but inference sequentially? Actually, in practice X is never fully resident at once due to block processing. But if it were, it’s impossible. We would have to fold some activations (layer-by-layer) or memory-swap (which kills speed). So activation memory is a nonstarter for full sequence. Possibly this is why H3 is diffusing flows (it might not actually load all 12.8M tokens at once, maybe it only processes windows). But assuming worst-case, neither GPU can hold the full hidden (R9700 has 32 GB, far below 138 GB). So Design A as described needs either offloading or streaming, which again kills latency.  

*Estimates:* Even ignoring memory, communication is ~100+ GB/layer ⇒ ~180s total just for gather, vs 48s original compute. So no speedup. Numeric differences: Since each GPU computes exactly half the attention, combining is exact (aside FP rounding). Complexity: Very high, requires custom code or NCCL (impossible here). 

## Design B: *Column/Row parallelism (Megatron style)*

**Sharding plan:** Split each large linear in half by output features (“column” parallel). For Q/K/V: Wq (5376×7168) is split as [5376×3584] on each GPU. So GPU0’s Wq0 yields first 28 heads, GPU1’s Wq1 the other 28 heads. This is actually equivalent to Design A’s Q-split. Then each GPU computes partial Q0, Q1. But to do attention, each needs *all* K and V for every head. So each GPU also needs to compute full K and V for its weight portion, then **exchange** partial K/V: GPU0 sends K0,V0 (size seq×3584×2) to GPU1 and vice versa. After exchange, each has complete (K,V) for all heads, but only its own Q. Then each GPU can compute attention *for its own Q* and produce a partial output [seq×5376] (512 bytes?). Actually, each GPU’s attention outputs only 28 heads, but let’s say each computes its 28-head output as [seq×3584]. Then they add their partial outputs with e.g. an all-reduce (sum) or gather: if true megatron, attention output is sum of both GPUs’ contributions. Here since heads are disjoint, it's like concatenation (same as design A). So same gather as before. Then FFN split similarly: W1 (5376×14336) is split columns 5376×7168 each; both compute partial MLP outputs [seq×7168], then all-gather to get [seq×14336], GELU, split W2 (14336×5376) rows or columns similarly, and all-gather final [seq×5376]. 

**Communication:** In addition to merging Q outputs like Design A, we need to exchange K/V at start of each attn. That’s sending ~2*(seq×3584×4B) = 2*(12.85M×3584×4) ≈ 368 GB per layer. Then merging outputs ~184 GB/layer (like A). Total ~552 GB/layer, or ~7.5 seconds/layer. MLP also needs exchanges (~similar scale). Essentially *even worse* than Design A.

**Memory:** Each GPU stores weights ~half of each projection and MLP (total model ~4.3B/2 weights ~2.15B weights, ~4.3 GiB). Hidden activations needed on each GPU is the full [seq×5376] at start to compute attention. As before, practically impossible. One could tile tokens and synchronize per tile, but that drastically increases sequential steps.

**Summary:** This design gains no speed, likely ends up slower. The required collectives (all-to-all K/V, all-gather features) per block are unachievable here. 

**Conclusion for TP:** Both concrete TP schemes fail our hardware test: the R9700 cannot hold the activation volume, and RCCL cannot perform required collectives. If one had NVLink or faster interconnect, one might imagine some benefit. As a theoretical exercise, even perfectly parallelizing halves of compute would give at most ~2× speedup on the ~48s H3 stage (24s), yielding ~8–10% end-to-end. But our environment cannot sustain the communication needed to approach that. We will *not* try implementing these until simpler methods are ruled out.

# 7. Concrete Sequence/Context-Parallel Designs

We outline **two sequence-parallel** designs, focusing on dividing the *token* or *frame* dimension across GPUs. 

## Design C: *Frame-split (temporal tiling)*

**Idea:** Split the clip’s frames into two continuous chunks, e.g. frames 0–61 on GPU0 and 62–123 on GPU1. Each GPU runs the entire DiT model on its subset of frames (as if smaller clip), but this breaks H3: frames are interdependent via global attention! To fix that, one could devise a sliding-window scheme: e.g. GPU0 processes frames0–61 with truncated cross-frame attention, GPU1 processes frames62–123 simultaneously, then possibly swap or refine. This is essentially splitting time (like a pipeline along time). It’s not clear how to get the correct joint distribution (attention) with only partial context. One could do two passes (e.g. GPU0: forward from start, GPU1: backward from end, then combine), but this deviates from the single forward pass logic. Alternatively, each GPU could temporarily treat the other's frames as “masked” or cached memory, but current H3 code does not support any recurrent or partial-latent input for external context. 

**Assessment:** A pure frame-split is likely *incorrect* (it ignores cross-temporal attention) or requires a custom scheme. One could try a pipeline: GPU0 runs layers 1–L on frames0–61 (with partial cross-attn), then GPU1 runs layers L/2–L on frames0–61, while GPU0 does layers L/2–L on frames62–123, then GPU1 runs layers 1–L/2 on frames62–123, etc – a 4-stage pipeline. But this is essentially a complicated pipeline parallelism, not true parallel compute. It might speed up if done carefully (like a pipeline interleaving), but such scheduling is custom and very complex. Also data must move: after GPU0 finishes first half of clip's 1st half layers, it must send hidden states to GPU1 for next layers, etc. This is as heavy as the earlier pipeline case. 

Given the complexity and no existing code base, we do not recommend this. It *could* yield correct results with complicated engineering, but risk of bugs (context integrity) is high. Abandon as practical path.

## Design D: *Unified Sequence Parallelism (USP) / Token partition*

**Idea:** Each GPU gets half of the sequence tokens. For simplicity, divide along the flattened token sequence (which loosely corresponds to half the spatiotemporal volume). That is, split the 12.85M tokens into two ~6.4M token groups. Each GPU only stores/updates its half of the hidden sequence. On each transformer layer:

1. Each GPU computes Q,K,V for *its* tokens using the full weight matrices (so no weight splitting needed).
2. GPUs exchange K,V: GPU0 sends its K0,V0 to GPU1, GPU1 sends K1,V1 to GPU0. Now each has the full global K/V (12.85M×7168 combined).
3. Each GPU computes attention for *its own Q* against the full K/V, producing new hidden for its token subset.
4. GPUs then exchange nothing at layer end; they carry on to next layer with just their local hidden. (Since each only computes half the tokens, we do not need to combine hidden across GPUs—each GPU keeps its portion separate).

At the end, we have two halves of the final hidden sequence. To reconstruct the full video latents, we would need to combine or interleave them back. But if we simply concatenate the token outputs back, we get the correct complete output (again neglecting necessary ordering details). Essentially, the output video is same, just we had two halves of it in parallel.

**Communication:** At each layer, the K/V exchange is *all-to-all of tokens of size half* per GPU: each sends ~6.4M tokens×7168 dims of Floats (6.4e6×7168×4B ≈ 175 GB) to the other. That’s 175GB/layer per GPU, or 350GB total traffic. No additional gather is done (since each GPU keeps partial output). Over 50 layers, >17 TB moved. At 100 GB/s, one layer ≈1.75 s, all 50 ≈87 s. 

**Memory:** Each GPU holds half the hidden (6.4M×5376×2B ≈ 64 GB) – already too large for 32 GB! This suggests one cannot even store the local tokens fully. In practice, one would tile or stream tokens through, but that adds more steps. However, note: the AMD article’s demo ran on MI300 with 8 GPUs; they must not store all tokens at once, likely streaming. In our case, 64 GB/ GPU is impossible. We’d need to split further (e.g. each GPU sequentially processes subsequences of its half). That doubles passes (2× more sync), halving each sub-pass size to ~32 GB, still barely. Actually, 6.4M×5376×2 = 68.8e9 bytes = 64 GiB, so splitting further to 2 passes of ~32GiB each might fit in 32GB with minimal overhead, but cutting sequence length per layer increases layers runs to 100 (double). If each pass also needs 175GB comm, *total comm doubles*, so 174s. 

**Summary:** This USP-like approach was done on 8 NVIDIA GPUs (with AITER kernels). It splits the 3D RoPE in two dimensions, too. But on 2 consumer PCIe GPUs, **it appears impossible to hold** the required hidden segments without either extra swapping or DRAM spill, and communication is enormous. One could attempt *framewise splitting* in smaller chunks, but each chunk still needs K/V exchange from entire sequence – no win. 

Given the hardware, if we tried design D, we would need to heavily tile within layers (virtual memory), which likely slows more than helps. It is conceptually *valid* – GPUs concurrently do half the sequence’s attention – but effectively identical to running whole model twice sequentially with overhead. It would not shorten wall time unless overlapped in a pipeline, which seems impractical.

**Alternative: Local K/V caching:** A compromise: treat K/V from the other GPU as “cache” and not update every layer. For instance, fix keys/values from initial noise (no diffusion steps yet) on GPU1, let GPU0 attend them without update. This breaks model correctness (it’s not what H3 does). So not allowed.

**Conclusion for SP/CP:** Neither design offers a clear win. Design D (USP) is known in theory but utterly bandwidth-bound here and requires splitting the graph into micro-steps. Without a clever AMD kernel or hardware links, it’s likely slower. Design C (time-tiling) breaks the model’s semantics. We can **implement a small test** of USP on a tiny synthetic sequence to see if the communication bottleneck is indeed fatal (phase 2 below), but we should be skeptical.

# 8. Encoder Placement and Scheduling

The H3 workflow has two main components: the *text encoder* (Qwen3-VL 32B AWQ) and the *diffusion DiT*. Currently the encoder (~15 GiB AWQ) sits on the 7900 XT and DiT (~20 GiB FP8) on R9700. This avoids R9700 having to load and unload the encoder on each prompt. We should consider other placements:

- **CPU encoder:** Running Qwen3 on CPU (via stable-diffusion.cpp or unsloth) is possible. This frees nearly 15 GB GPU VRAM. But CPU inference of a 14.9B model is *very slow* (seconds to tens of seconds) and uses DDR memory. It also ties up the CPU core(s) during generation. The AMD blog approach kept encoder on GPU. For our latency target, CPU encoder is likely *slower* than GPU encoding, given our fast 12-core CPU.

- **Encoder on 7900 XT (current):** Good choice. Qwen AWQ can run on 7900 sufficiently fast (two nodes presumably). The 7900 remains mostly idle during the 48s diffusion. We should ensure `--gpu 1` assignment (HIP_VISIBLE_DEVICES=1,0) is correct.

- **Encoder on R9700:** We tried this (single GPU baseline) and found runs ~4–8% slower than dual-run, due to extra swapping. Likely not desirable.

- **VAEs on 7900:** VideoVAE and AudioVAE are small (maybe 0.5–1 GB each). If R9700 VRAM gets tight, we could move them to 7900 or even CPU (if decode speed is acceptable). They run at the end of diffusion (decode stage ~18s), so offloading VAE decoding to 7900 might reduce R9700 memory footprint at the end of sampling. We should try setting VAE on cuda:1 in ComfyUI. But currently, most time is DiT, VAEs are minor on latency.

- **Swapping model parts:** Pollock’s MultiGPU allows explicit placement. We can use his UNET/VAELoaderMultiGPU node or Force device in ComfyUI. For now, confirm Qwen on 7900, DiT on 9700, VAE optionally 7900.

- **Overlap potential:** The encoder runs only once per prompt, while DiT runs many steps. We could overlap one encode with loading DiT on R9700 if we batched seeds, but with batch=1 no real overlap. Possibly pipeline: start DiT on R9700 after encoder sends first few conditioning tokens? Unlikely to overlap meaningfully.

*Conclusion:* Keep encoder on 7900, model on R9700 (we already do). Optionally, move VAEs to 7900 if needed for VRAM. CPU encoder is ruled out for latency.

# 9. Format and Runtime Decisions

We have several ways to represent/run H3:

- **Native FP8 safetensors (comfy-kitchen):** ComfyUI currently uses `minimax_h3_fl2va_pruned_fp8` and `qwen3vl_32b_minimax_h3_awq_fp8` via comfy-kitchen. These run on Triton on ROCm. They require Triton 3.5.1 (ours) and PyTorch 2.9.1+ROCm. They produce bit-exact results as the weights. **Pros:** Highest performance on GPU, no dequant overhead. **Cons:** Weight storage consumes ~20 GB, likely no extra support for splitting. 

- **Native BF16 safetensors:** The original H3 weights are BF16. We could load them on GPU and run. This would increase memory (20 GB→ maybe similar, as FP8 is int8 compression). But as a reference check, we can do one BF16 run on R9700 to verify quantized results. It’s useful for comparing output correctness. Multi-GPU splitting could be easier in BF16, but still limited by memory. However, PyTorch has no specialized kernels beyond Triton fused ops for H3; they run similar.

- **GGUF quant (stable-diffusion.cpp or unsloth):** Unsloth provides H3 in GGUF Q(2,3,4,5,6,8) with AWQ for Qwen. stable-diffusion.cpp can load those, running the encoder on CPU if needed. However, stable-diffusion.cpp currently only uses one GPU (or CPU). It cannot split one model across GPUs; it can only run separate tasks on separate devices. So this does not directly help with two-GPU DiT. GGUF *is* portable across CPU/GPU, but vectorization is limited by that engine. It is not multi-GPU except by forging two processes. The main benefit of GGUF is smaller memory footprint (Q6 ~15GB, fits R9700 with margin). It also uses FP16 or FP32 compute under the hood. But it’s single-device. It could be used as a fallback: if our PyTorch plan fails, one could run encoder on CPU and DiT on GPU through stable-diffusion.cpp, but that’s still one GPU for DiT. 

- **PyTorch quant kernels (AWQ/BF16):** PyTorch has no out-of-the-box support for H3 AWQ like CUDA has (no NVFP4 path on AMD). So comfy-kitchen INT8 convrot is our target. No alternative quant format in PyTorch for two-GPU mode. We must use Safetensors.

- **Model size for design choices:** If memory is an issue, we could try loading the pruned smaller version. Currently using `fl2va_pruned_fp8` and the FL2VA pipeline. If memory is tight, consider a pruned version (with fewer tokens, e.g. a “Xl trimmed”), but not needed for 124f.

- **Summary:** We will use the existing FP8 comfy-kitchen weights (we already have on R9700). We should keep the VAE, transformer, encoder all in INT8 for consistency. For debugging, we can validate one run with BF16 (if memory permits) to check outputs. We will not pursue GGUF for multi-GPU: it cannot utilize GPU0 for encoder (it's CPU by default) and GPU1 for DiT, since stable-diffusion.cpp is monolithic. The only hybrid path is: CPU encode via GGUF, GPU decode or shuffle, but again that yields no dual-GPU DiT compute. So **no additional formats to download for dual-GPU**. Our dual-GPU experiments will reuse the same safetensors.

# 10. Theoretical Performance Table

Let T_total = T_encode + T_H3 + T_decode + overhead. From our measurements: roughly 80–76 s cold with single vs dual (H3 stage ~48 s, decode+save ~18 s). Let T_H3=48s, T_encode ~5s, T_decode+save~18s, overhead~9s.

If we could perfectly parallelize the H3 stage on two GPUs, ideal speedup = 2× (but plus comm cost). Amdahl’s law: if H3 is 48s of 80s (60%), the *ceiling* is 1/(0.40+0.60/2) = ~1.26× end-to-end (i.e. 26% faster). In practice, we must subtract communication. Let “Comm” be extra per-layer transfer time. Even optimistic 4s overhead in H3 yields only ~1.17× speedup (approx 15% less).  

**Projected E2E speedups needed:**

- **10% gain:** From 80s to 72s. H3 needs ~10% faster (~43.2s vs 48s).
- **15% gain:** To 68s. H3 ~41s (15% faster).
- **20% gain:** To 64s. H3 ~38.4s (20% faster).
- **25% gain:** To 60s. H3 ~36s (25% faster).
- **30% gain:** To 56s. H3 ~33.6s (30% faster).

Clearly, to hit 10–15% total, we need about 15–20% faster H3 stage. Half-shaving (50%) is impossible (ceiling 1.26×). But even a modest ~15% is challenging if comm costs are nonzero. (E.g., if one extra 10s of comm was incurred, we'd get hardly any net gain.)

**Sequence length effect:** If longer or higher-res (say short-edge 768 → ~1344×768, 15s clip), T_H3 ~ scaling ~frames×pixels. If T_H3 doubles, our parallel fraction increases (Amdahl less severe). However, comm also doubles roughly. In any case, more compute helps amortize overhead.

We will use these numbers to judge success: e.g. hitting ≥10% end-to-end (≥7.2s saved) requires noticeably faster H3.

# 11. Staged Experiment and Implementation Plan

We need to do this carefully, preserving a working single-GPU path at every step.

## Phase 0: Topology & Communication Preflight

Perform on the *idle workstation* (no code change yet) using ROCm tools:

1. **Device query:** `torch.cuda.get_device_name(0/1)` in Python to confirm mapping (likely 0=7900,1=R9700 with HIP_VISIBLE_DEVICES=1,0). Check `torch.cuda.device_count()` and device names to document assignment.

2. **PCIe topology:** Use `lspci` and `nvidia-smi topo` (or ROCm equivalent) to see link (PHB reported was “bandwidth 6.0” meaning Gen4×16). Confirm actual bus numbers of each card. Ensure whether R9700 is above or below chipset (d000 likely for R9700).

3. **Peer-to-peer test:** Use PyTorch or HIP: For a large tensor (~1 GiB), measure `start = torch.cuda.Event(); a = torch.randn(1<<27, device=0); b = a.to('cuda:1')` and time it. Do reverse direction, with and without `cupy` pinned memory. The ROCm `rocm-smi --showhw` should confirm P2P enabled. If `torch.cuda.PeerTensorCopy` fails or stalls, abort (since distributed comm is impossible). Also test `torch.cuda.comm.reduce_add` or do a trivial `dist.all_reduce` with 1-element (it will likely fail as seen).

4. **Collectives test:** Run the PyTorch script from [14] (`torch.distributed` barrier/all_reduce) on 2 GPUs exactly as shown, confirming failure. Try with variants: NCCL_P2P_DISABLE=1 (was in repro). If it fails, we *already know abort criterion is met*. Document error.

5. **Bandwidth measurement:** Use HPC benchmarks: e.g. `rocblas-benchmark` or write a quick HIP peer memcpy loop of various sizes (e.g. 64MiB to 1GiB) to get actual GB/s in each direction. Possibly use `RCCL-tests` (if available) or `rccl_stream_perf` from RCCL repo. We want actual latency and bandwidth vs message size.

6. **RCCL communication:** Try a simple torch.distributed script for a small all_gather of a moderate tensor (e.g. 10M floats) to see if error occurs. If fails, skip heavy DP code.

7. **Per-GPU compute test:** Run a simple transformer block or large matmul on each GPU separately to record raw GFLOP/s (to calibrate compute cost in our perf model).

8. **Thermal:** While stressing, check `rocm-smi` temperatures/clock to ensure no throttling.

**No-go criteria:** If any collective or P2P copy fails or is extremely slow (<1GB/s), then **halt** two-GPU experiments. Without basic rank-to-rank comm, there's no hope. If P2P ~20–50 GB/s (somewhat slow) we can proceed but temper expectations. If bandwidth is poor, parallelism is less promising.

## Phase 1: One-block PyTorch Prototype

Goal: A minimal 2-GPU proof of concept with a *single transformer block* to verify overlapping compute and communication.

1. **Select shape:** Use a toy sequence length, e.g. [seq=4096, H=5376]. Not full H3 size (too big), but modest (≈22M elements, 44MB per tensor). Hidden=5376, heads=56.

2. **Implement two variants** in pure PyTorch (no ComfyUI):
   - *Head-split block:* Construct Wq0/Wq1 (split 7168→3584), Wk0,Wk1,... etc in FP32. In a 2-process group (spawn 2 ranks or manual with `torch.cuda.set_device`). Each rank does local Q/K/V, does attention, then uses `dist.all_gather` (if it even works) or manual send to combine. Actually, since RCCL broken, use **manual peer-to-peer**: e.g. rank0 does `K0 = K0.to(1)`, rank1 sends `K1.to(0)`. (Torch may not have easy peer copy; fallback: `torch.cuda.current_device()` and `cudaMemcpyPeer`). See if this is implementable in Python. Then assemble full K, then compute local attn. Time separate steps.
   - *Sequence-split block:* Simpler: each GPU holds part of hidden. We’ll artificially keep hidden tensors local for half of Q dimension. To simulate sequence-split, do: Rank0 has X0 (half seq×5376), Rank1 X1. Each makes Q0,K0,V0 = X0·W etc. Exchange K/V fully and compute. This is just a simpler all-to-all. Or use A=N token, split at N/2 for local Q. 

3. **Timing:** Measure compute and communication separately. E.g. measure time for `attn = softmax(QK^T)V` and time for the `K send/receive`. If we see both GPUs doing compute concurrently and the sum of times plus transfers > one GPU time, note overhead.

4. **Correctness:** Compare the merged 2-GPU output to a single-GPU run of the same block (assemble final [seq×H] and do same block on it). Check maximum error (should be near 0 for FP32 aside from roundoff).

5. **Variance:** Try smaller heads or different splits to see if logic breaks. 

*Goal result:* Confirm that both GPUs indeed compute something (e.g. overlap K/V compute and Q?). Measure GPU occupancy (like `nvidia-smi`-equivalent for AMD). This isn’t ComfyUI, but a standalone test. If any CUDA-like collective fails here, this is a dead end. If it works and concurrency occurs, record throughput.

## Phase 2: Full H3 forward (one process per GPU)

We now attempt to run H3 DiT in two-GPU mode *without ComfyUI first*, just to ensure the model logic can operate across GPUs.

1. **Separate processes:** Use `torch.multiprocessing.spawn` with 2 ranks, `init_process_group` on RCCL (likely fails, but we have to try). If it fails, skip to approach B.

   - If DDP fails, try a single-process multi-device approach: manually set device for encoder and model (though a single process cannot easily do two GPUs in one forward without duplicated model). Hard.

   - Alternative: Use ComfyUI’s custom node approach (but at PyTorch level, we can hack the PyTorch source with `FSDP` or manual splits).
  
   - Given the negativity of [14], I suspect we skip direct DDP. Instead, *simulate* the beneficial effect: put encoder on GPU1, model on GPU0 and measure one-run time vs baseline, but that’s just what we had.

Since actual parallel computation (like FSDP or Raylight) is hard without dist, Phase 2 might largely confirm we cannot do true parallel inference via PyTorch DDP. 

Instead, we might pivot: test the **Turbo LoRA** Sampler (Comfy has one) to ensure it runs at 4 steps without errors. Then integrate possibly Raylight or MultiGPU nodes. But Raylight needs NCCL too.

If we accept single process, maybe test an approach: use `torch.fx` or model parallel (pipeline) but as noted, pipeline might not gain time.

Given constraints, Phase 2 might be simply: prepare code integration points. If DDP with BWo fails, skip.

## Phase 3: ComfyUI / Turbo integration (fallback)

Since direct 2-GPU H3 seems blocked, we proceed with *CPU-vs-GPU layout experiments*: encode on 7900, DiT on 9700 (the current), and try adding disabled Dynamic VRAM etc. We have already done this ("current dual-component placement").

Then try to replicate any parallelism using available ComfyUI extensions:

- **ComfyUI-MultiGPU**: Pollock’s extension can place models on different GPUs or CPU per node. We can try using its GPU-select loaders: load the MiniMax DiT on cuda:0 (R9700) and encoder on cuda:1 (7900). If MultiGPU can handle Gemma4 encoder. It likely can, since Qwen is just a checkpoint loader. Then run the normal workflow and see if clocks show both GPUs busy (though only encoder on 7900, which is quick). If it does something like FSDP, measure memory difference. But FSDP on AMD with ROCm? Unlikely to work. However, use minimal block: just ensure both GPUs see activity.

- **Raylight (USP)**: It’s CUDA-only, skip unless someone ports to HIP (not feasible for us).

- **Test Nitro**: Is there any ROCm multi-GPU tool? SGLang is an option but it’s its own system. Out-of-scope (requires building SGLang and running server, heavy).

- **Experiment with VAE placement**: Use MultiGPU to put VAE on 7900. Benchmark R9700 VRAM usage and time. If VAEs on 7900, R9700 frees maybe ~1GB at decode, not big.

**End Phase 3** with establishing a "baseline dual-component config" (encode on 7900, rest on 9700) as the multi-GPU baseline (we have those 4–8% gains measured). Document all settings.

## Phase 4: Hybrid or creative hacks

If any steps above hint at a partial win, try one last idea: **Low-level stream splitting.** For instance, could we split inference by alternating steps on GPUs? (e.g. step0 on GPU1, step1 on GPU0, etc.) That would require copying entire model between devices each step - pointless. Or one GPU does odd steps, one even steps, but then all intermediate states must be exchanged - again heavy.

Likely skip. The objective is single-video latency. The results of Phase 3 likely conclude dual-GPU yields only small speedups now.

## Phase 5: Benchmarks

For whichever configurations are viable (Single R9700 vs Single+comp offload vs any partial parallel we make work), run the full battery:

- Cold run (no warm cache)
- Warm run 1
- Warm run 2
- Same prompt new seed
- New prompt

Record: total time, DiT time (via ComfyUI instrumentation if possible), encode time, decode/save time. Log VRAM usage on each GPU (via ROCm queries, e.g. `torch.cuda.max_memory_allocated`). GPU util% if possible, and any RCCL/NCCL debug logs.

Compare: ideally Single vs Dual exactly. Specifically check GPU timelines (via `rocprof`) to confirm if indeed both GPUs used concurrently. Use rocprof to see concurrency and memory copy patterns.

If debug logs show any HIP errors or warnings (in `stderr`), capture them. Also ensure output video/audio correct (hash + visual check). 

**Key pass/fail:**

- If at least some overlapped compute is seen (which is unlikely), measure speedup. If no concurrency, likely revert (so “don’t pursue further”).

- If any run triggers a GPU reset or OOM, abort.

- If numeric output differs unacceptably (frame mismatch, audio missing), fail.

# 12. Risk Register and Abort Conditions

We list known risks and clear aborts:

- **RCCL failure (critical):** If any NCCL or ROCm collective fails in testing, we abort *all* distributed designs. (Mitigation: only use manual HIP copy with single process, or scrap TP entirely.)

- **Memory overflow:** If any attempt (e.g. head-split) requires holding >32 GB on a card, abort that branch. (Given 64 GB needed, we already know it's impossible.) Skip those.

- **Compute idle GPU:** If tests show the “second” GPU stays mostly idle (e.g. 0% util) in all attempts, then adding it is meaningless. For latency, we need >0% util during DiT. If not, abort parallel efforts.

- **Communication too slow:** If P2P bandwidth <20 GB/s in initial test, two-GPU will certainly be slower. Possibly abort or label as too slow.

- **Stability:** Any GPU hang, global sync error, or library crash in PyTorch or ComfyUI during testing aborts that line of work. Continue only if clean.

- **Output divergence:** If multi-GPU runs (once we manage any) produce visibly worse or NaN outputs (likely from sync issues), abort.

- **Engineering time:** If after basic verify we see virtually no speedup (<5% H3 time saved), we deem it uneconomic (baseline plus the R&D is too costly). Then pivot focus to capacity rather than latency.

- **Licensing/distribution:** We rely on MiniMax Community License compliance (just note: no issues here as we’re using available weights).

# 13. Recommendation

**First Prototype:** *Implementing a single transformer block head-split in PyTorch (Phase 1) to verify concurrent compute.* This will test if any multi-GPU compute overlap is possible given ROCm’s current state. Why this first? Because if even one-layer head-splitting *or* sequence-splitting in pure code shows that both GPUs can do useful work concurrently (and complete in less wall-time than sequential), we have a shot. It also directly confronts the RCCL issue (does manual peer copy allow progress?). Early success or failure here will strongly guide the rest. 

If this block test cannot run or yields negligible concurrency, we know tensor/sequence parallel is dead and can stop. If it shows promise (both GPUs busy, correct output), then we scale to multiple layers carefully.

**Effort:** Phase 1 can be done with pure PyTorch code in a day or two. It requires moderate familiarity with torch.distributed or manual CUDA operations (maybe using `torch.cuda.nccl` if workable). The fallback is more coding.

**Safeguards:** We keep our original single-GPU workflow intact at all times. The prototype code can be isolated in a separate script. If anything fails, revert to baseline.

If Phase 1 fails (likely due to RCCL or memory), we label “not pursue for latency” and shift to capacity: e.g. "use multi-GPU purely for length". If it succeeds, we proceed through plan and aim for 10–15% improvement threshold. 

In summary: **Prototype first** is the plan. We do not deploy any half-baked multi-GPU ComfyUI nodes on the workstation yet. Instead, we try the simplest concurrency to gauge viability. The earliest falsification is: if two GPUs cannot compute any part of the same block in under-sequential time (phase 1), then stop. If success, continue.

# 14. Open Questions & Evidence Gaps

- **ROCm Mixed GPU fixes:** Will future ROCm releases fix dual-Radeon collectives? (There’s no timeline, and current known regressions are severe.) We must assume “not solved” for our timeframe. 

- **H3 memory tiling:** The actual AMD SGLang approach presumably uses some streaming or sparse attention. Does H3 have any “windowed” attention in code? Official docs say full attention, so likely not. Could a *subregion approach* be engineered? Unknown.

- **Efficiency of encodings:** Would using a lower quant (Q6 instead of Q8) meaningfully reduce H3 compute? Possibly minor, but if Q6 uses the same kernels (TI, AWQ) maybe not. Unsloth uses AWQ (W4A16) for GPU; presumably no speed difference between Q6 and Q8 on GPU beyond memory.

- **Dynamic VRAM / AIMDO:** Our plan disables DynamicVRAM for consistency. If we had it on, maybe it would auto-swap weights to CPU, which we explicitly want to avoid for latency (that’s block swap).

- **Partial layer streaming:** Could we manually split one block’s attention into sub-blocks and overlap? Possibly, but that’s advanced.

- **Alternate frameworks:** Could JAX/XLA or RadeonML or other frameworks do multi-GPU better? Not at scale for H3.

- **Other HPC modes:** The AMD AITER kernels reportedly work on MI300 (gfx942) but we have gfx1201/1100. No easy port.

Each of these could change feasibility if solved, but for now they remain open.

# 15. Annotated Sources

- AMD Tech Article (Aug 2026) – *Official MiniMax-H3 architecture specs and AMD multi-GPU demo.* Provides H3 hidden size=5376, heads=56, patch size, etc, and shows 8-way sequence parallel on AMD. Also the core ROCm test (8 GPUs) used by AMD. Informs on model internals and performance context.

- ROCm Issue #6074 (Mar 28, 2026) – *ROCm dev issue documenting dual-Radeon collectives failure.* Reproduces a 2-GPU torch.distributed barrier error on ROCm7.2.1. Essential evidence that basic NCCL/RCCL doesn’t work on 2×RDNA3 (and likely RDNA3+4) in this environment.

- Komikndr’s Raylight README (Oct 2023) – *ComfyUI multi-GPU parallel node (Raylight)*. Explicitly lists “Minimax H3” in supported models and explains the concept of Unified Sequence Parallelism (USP) + FSDP weight sharding. This shows that H3 can be split across GPUs (via USP) on CUDA/NCCL, and GPU usage explanation.

- Pollockjj’s ComfyUI-MultiGPU docs (Aug 2023) – *Custom nodes for weight offload.* Explains DisTorch (model offload to other GPUs/CPU) and lists MultiGPU loader nodes. Indicates ability to place model components on specific GPUs. Relevant for encoder placement and potential baseline.

- Hugging Face Unsloth H3-GGUF (Feb 2024) – *stable-diffusion.cpp H3 instructions.* Shows the use of `--backend te=cpu` to keep the 12 GB Qwen encoder on CPU, and `--offload-to-cpu` to fit smaller GPUs. Illustrates separate CPU text encoding vs GPU model compute. Confirms that stable-diffusion.cpp cannot split a model across GPUs, only components to CPU.

- ComfyUI Wiki on H3 Turbo LoRA v4 (Aug 8, 2026) – *Community guide to Turbo LoRA.* Confirms that Turbo LoRA is a LoRA patch (no architecture change) and details (v4 distills H3 to 2–3 steps). Relevant for understanding that Turbo’s impact is on step count, not model structure.

- Hugging Face MiniMaxAI/MiniMax-H3 Discussion (Sep 2026) – *User reports ComfyUI multi-GPU.* (Specifically [4] lines 115-118: “Implementing sequence parallelism is easy…” from user “komixenon”). This is anecdotal: someone effectively ran H3 with Raylight, stating H3 packing is simple. Not citable formally here, but indicates community interest. (We have Raylight doc instead.)

- Hugging Face Docs (“MiniMax H3 in ComfyUI”) – *Official ComfyUI doc.* Confirms ComfyUI native H3 support and usage, not parallel. (Not cited because it’s obvious info: we know ComfyUI runs single-GPU H3.)

- PyTorch Distributed (Accelerate) guide – *Data-parallel inference.* Not H3-specific; cited to contrast that typical multi-GPU inference frameworks target data-parallel (multiple prompts) rather than model splitting.

We focus citations where factual claims needed. The above cover architecture, existing multi-GPU attempts, and ROCm issues, which are the factual backbone of our analysis. Each is from late 2026 / mid-2026, satisfying freshness.