# Wan-derived optimization candidates for LingBot World 1.3B

This note mines the broader Wan inference ecosystem for ideas that may transfer to the persistent LingBot World v2 1.3B interactive path.

The product metric remains **keypress -> first genuinely new visible frame**.

Current accepted R9700/gfx1201 steady state:

- actual geometry: 464x832
- chunk_size=1
- local/sink window: 18+6
- DiT: ~1.56 s (5 forwards)
- FP16 VAE decode: ~1.25 s
- setup/presentation: ~1 ms
- total first-visible: ~2.81 s
- peak VRAM: ~13.50 GiB

The purpose of this file is not to import every Wan optimization. It is to keep a ranked backlog of proven ideas and reject ecosystem tricks that do not match this causal, four-step, persistent-KV workload.

## Priority A: low-risk DiT execution improvements

### 1. `torch.compile` / TorchInductor on transformer blocks

WanVideoWrapper exposes block-only `torch.compile` as a normal optimization path, including `default`, `reduce-overhead`, and `max-autotune` modes. AMD documents TorchInductor/Triton as a supported ROCm optimization route and recommends `max-autotune` as a performance-oriented starting point.

Why it may transfer:

- fixed model shapes at chunk_size=1 are favorable for compilation;
- persistent sessions amortize compile cost over many actions;
- potential gains include fusion, fewer intermediate VRAM writes, lower Python/dispatch overhead, and tuned generated kernels.

Risks:

- mutable KV/cross-attention caches may cause graph breaks;
- dynamic cache indices can trigger recompilation;
- whole-model compilation may be brittle.

First bounded test: compile transformer blocks only, leave cache management and outer causal loop eager. Compare steady-state full-window DiT and end-to-end action latency. If compile overhead/recompiles persist after warmup, reject.

Theoretical ceiling: 1.56 s (entire DiT).
Expected useful win: tens to a few hundred milliseconds, to be measured.

### 2. ROCm attention backend selection: CK vs current SDPA backend

The current compatibility path uses PyTorch SDPA. Recent AMD Wan2.2 measurements report CK-backed Flash/SDPA paths outperforming the default AOTriton path by roughly 20-50% in some Wan workloads, depending on hardware and shape.

PyTorch/ROCm can prefer CK for SDPA/Flash attention. This should be tested on the *actual LingBot incremental attention shapes*, not inferred from offline Wan2.2.

Why it may transfer:

- no model or algorithm change;
- keeps SDPA semantics;
- potentially a backend-only win;
- our rolling KV creates long K/V with short Q, which may favor a different backend than normal full-sequence Wan.

Required checks:

- asymmetric Q/K lengths;
- local rolling window;
- sink retention;
- causal alignment;
- exact output comparison against current SDPA path;
- attention-only and whole-DiT timing.

Do not assume CK is faster. Measure each actual steady-state shape.

### 3. Native PyTorch RMSNorm

WanVideoWrapper exposes PyTorch native RMSNorm because it can be faster than its original Wan RMSNorm implementation when not using `torch.compile`, with small numerical differences.

Only test if the DiT profile shows normalization has meaningful cumulative cost. This is a likely small, easy win rather than a primary lever.

## Priority B: algorithmic denoising reuse/caching

The Wan ecosystem contains TeaCache, MagCache, EasyCache, DBCache/CacheDiT, TaylorSeer, TaoCache, and newer adaptive caching schemes. These reduce video diffusion cost by reusing intermediate outputs or selectively skipping expensive model evaluations across denoising timesteps.

TeaCache's Wan2.1 implementation reports large speedups in ordinary multi-step Wan generation, and WanVideoWrapper directly supports TeaCache and MagCache. MagCache also includes Wan2.1 support.

However LingBot causal-fast is unusual:

- only 4 denoising forwards per action;
- plus 1 mandatory clean x0 / t=0 KV-writing forward;
- the clean fifth forward cannot reuse denoising K/V and must remain.

Therefore ordinary Wan cache speedup claims do **not** transfer directly.

The useful question is narrower:

> Can one or more of the four denoising forwards, or expensive blocks inside them, be safely reused/predicted while retaining action following and long-horizon world stability?

Approximate arithmetic opportunity:

- DiT total ~1.56 s / 5 forwards ~= 0.31 s average/forward;
- removing/replacing one denoising forward could save on the order of ~0.3 s;
- two could approach ~0.6 s;
- the clean fifth forward remains mandatory.

This is one of the largest remaining algorithmic levers, but it is higher risk than backend/compile work.

Validation must emphasize:

- camera/action following;
- world persistence after many actions;
- revisitation consistency;
- KV correctness;
- temporal seams;
- drift accumulation.

If a TeaCache/MagCache-style signal is tested, calibrate specifically on the four LingBot timesteps rather than importing Wan thresholds intended for 20-50 step generation.

Newer adaptive cache methods such as ACID are relevant only after a base cache method proves useful; they should not be the first implementation.

## Priority C: precision and GEMM paths

### 4. DiT FP16 versus BF16

The current DiT is BF16. The broader Wan ecosystem commonly supports both FP16 and BF16, and ComfyUI's Wan guidance has often preferred FP16 model files for output quality.

The VAE experiment also demonstrated an important local lesson: do not assume BF16 is automatically best on this workload.

A bounded DiT FP16 A/B may be worthwhile if profiling shows GEMMs dominate and the backend has a better FP16 path on gfx1201.

Measure:

- per-forward latency;
- numerical difference;
- 24+ action causal drift;
- output quality;
- peak VRAM.

Do not change precision merely for memory: current VRAM is already comfortable.

### 5. FP8 / weight quantization

DiffSynth-Studio and WanVideoWrapper support FP8/quantized Wan inference, primarily for memory reduction and sometimes speed.

This is not an immediate priority:

- the 1.3B model already fits comfortably;
- dequantization/unsupported-kernel fallbacks can erase speed gains;
- gfx1201 FP8 software maturity must be proven on the exact stack;
- quality and persistent-world drift matter more than one-shot similarity.

Promote only if DiT GEMM profiling shows a strong hardware-supported FP8 path with low conversion overhead.

## Priority D: attention sparsity

### 6. Sparse/video sparse attention

AMD has published substantial attention-kernel speedups from video sparse attention on long video sequences. LingBot's steady-state shape is unusual: one new latent frame (~1508 query tokens) attends into a rolling window of up to 27144 tokens.

This could make attention sparsity valuable if self-attention is a major fraction of the DiT.

But this is not automatically compatible with the current incremental KV layout, and earlier local VSA work showed sparse kernels can lose at small token counts.

Do not implement until the steady-state DiT profile establishes attention as a large enough target and the candidate supports asymmetric Q/K plus rolling cache semantics.

## Priority E: VAE structural work

### 7. Causal Conv3d -> temporal Conv2d decomposition

Still available. FP16 reduced VAE decode to ~1.25 s, so this is no longer the first target, but Conv3d remains the decoder's dominant operation class.

If the DiT side does not yield an easy >=0.3 s win, benchmark the three dominant VAE shapes using mathematically equivalent temporal Conv2d accumulation.

Reject immediately if the isolated kernels are not clearly faster.

### 8. Tiled VAE decode

Wan/ComfyUI widely uses tiled VAE decoding for memory-constrained generation. It is primarily a memory tool, not a latency optimization, and community reports include possible temporal/seam artifacts when tiling parameters are poor.

Our model fits with ~13.5 GiB peak and already has an incremental causal decoder. Do not pursue tiling for speed unless a specific profile proves otherwise.

## Priority F: memory/offload techniques that do NOT help this target

WanVideoWrapper contains block swapping, model offload, prefetch, and aggressive VRAM-management paths.

These are useful when large Wan models do not fit. They are not useful for this 1.3B interactive target because the model fits comfortably and host/device movement would add latency.

Status: reject unless future resolution/context changes create a memory constraint.

## SageAttention: do not chase by default on AMD

SageAttention is popular in the Wan community and can accelerate NVIDIA Wan workloads, but recent AMD measurements report it generally slower than PyTorch SDPA on Navi31 and Strix Halo, with gfx12xx requiring additional patching and having stability/compile issues in some configurations.

Therefore SageAttention is not a default candidate for R9700. Prefer testing native ROCm SDPA backend selection (especially CK) first.

## Recommended order from the Wan ecosystem

1. Profile the steady-state DiT at chunk_size=1 / window 18+6.
2. Test block-only `torch.compile` if graph structure permits stable reuse.
3. Test ROCm SDPA backend choice (current AOTriton versus CK) on actual attention shapes.
4. Take any easy measured RMSNorm/GEMM/layout wins revealed by the profile.
5. Then test one bounded denoising-cache/reuse strategy adapted to the four causal-fast timesteps.
6. Revisit VAE Conv2d decomposition if the above does not supply the needed ~0.3-0.5 s.
7. Consider FP16 DiT / FP8 only when operator profiling justifies the precision path.
8. Sparse attention only if attention is proven dominant and the implementation supports LingBot's incremental KV semantics.

## Standing rule

Wan ecosystem speedup numbers are candidate generators, not evidence for LingBot.

Every imported idea must be re-measured on:

- R9700/gfx1201;
- 464x832;
- chunk_size=1;
- full 18+6 rolling context;
- persistent KV/cross-attention/VAE state;
- the fixed long traversal regression.

Keep the idea only if it improves the working interactive model.