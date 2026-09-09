# LingBot-World-V2 1.3B causal-fast on R9700 / gfx1201 — 2026-09-09

**Status: bring-up PASSED.** Native BF16, single R9700, no quantization, no
generic CPU offload, no distributed inference, upstream causal-fast settings.

- AMD Radeon AI PRO R9700, `gfx1201`, 32 CU, 31.86 GiB reported
- Ubuntu 24.04.4 LTS, kernel `7.0.0-28-generic`
- ROCm 7.2.1, HIP `7.2.53211-e1a6bc5663`
- PyTorch `2.9.1+rocm7.2.1.lw.gitff65f5bc`, torchvision `0.24.0+rocm7.2.1.gitb919bd0c`
- Upstream `robbyant/lingbot-world-v2` @ `45fa40673607c9acba6cf96a1f9396c95bcef25f`
- Model `robbyant/lingbot-world-v2-1.3b-causal-fast` @ `7e36a5f919f86cb4255cc9bfc30adb44963fbde1`

## Summary

| Question | Answer |
|---|---|
| Did it work? | Yes |
| Native BF16? | Yes — all 1,709,502,016 DiT parameters `torch.bfloat16`, 3.18 GiB |
| Single R9700? | Yes — one gfx1201, no FSDP, no Ulysses, no NCCL |
| Resolution | `--size 480*832` requested; **464x832 actually produced** (see below) |
| Peak VRAM | 19.19 GiB allocated, 23.62 GiB reserved |
| Warm generation | 63.0 s for 29 frames = 0.460 effective fps |
| Main compatibility issue | FlashAttention is CUDA-only → SDPA path |
| Main bottleneck | The FP32 VAE — **89% of warm wall time**, even after a 6.75x fix |

The single most valuable finding is not the compatibility work. It is that
MIOpen 3.5.1 silently selects a naive convolution kernel for gfx1201 above a
workspace threshold, costing a **17x throughput collapse** on the VAE's
full-resolution 3D convolutions, and that an exact temporal restructuring
recovers it.

### On the resolution

`--size 480*832` is an **area** target, not a shape. `_generate_causal_fast`
derives the latent grid from the input image's aspect ratio at that area, so
`examples/03/image.jpg` (5145x2894, exactly 16:9) yields `lat_h=58, lat_w=104`
and a 464x832 output. This is upstream arithmetic and would produce the same
464x832 on any GPU; it is not an AMD artifact and not a fallback. The
requested operating point was accepted without reduction.

## Hardware

Three GPUs are present. `HIP_VISIBLE_DEVICES=1` selects the R9700; device 0 is
an RX 7900 XT (gfx1100) and device 2 an integrated gfx1036. Every measurement
here is single-device on gfx1201.

Raw `rocminfo` / `rocm-smi` in `experiments/lingbot-world-v2-1.3b/logs/phase0-system.txt`.

## Software

Isolated venv at `/ai/envs/lingbot-world-v2`, built from the local ROCm 7.2.1
wheelhouse. **No CUDA PyTorch, no NVIDIA packages, no `flash-attn`.** Existing
ComfyUI and ROCm installations were not touched. Full freeze in
`logs/phase0-pip-freeze.txt`.

`LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib` is required — `/opt/rocm` on this host
does not carry `libroctx64.so.4`, which this torch build links against, and
`import torch` fails without it.

## Model — required release reconstruction

The 1.3B Hugging Face repository is **not a runnable pipeline**. It contains
only the DiT: six safetensors shards plus an index, and nothing else. There is
no VAE, no text encoder, no tokenizer, and no `config.json`. Upstream's own
README still lists "Release the causal-fast model of the 1.3B model" as an
unchecked TODO while linking the weights.

The upstream code likewise has **no configuration for this model** —
`wan/configs/__init__.py` registers only `i2v-A14B`.

Reconstruction, all recorded in `wan/configs/wan_i2v_1_3B.py`:

- **Architecture derived from the safetensors headers**, not guessed:
  `patch_embedding.weight [1536,36,1,2,2]` → `dim=1536`, `in_dim=36`;
  `blocks.*.ffn.0.weight [8960,1536]` → `ffn_dim=8960`;
  `blocks.{0..29}` → `num_layers=30`;
  `text_embedding.0.weight [1536,4096]` → `text_dim=4096`.
- **`num_heads=12` is the one value not recoverable from tensor shapes.** It
  follows the Wan2.1-1.3B backbone (head_dim 128). Every shape-derivable
  parameter is consistent with it, and the model loads with no missing or
  unexpected keys, but it remains an inference rather than a published fact.
- **VAE, T5 and tokenizer taken from the 14B release** (`Wan2.1_VAE.pth`,
  `models_t5_umt5-xxl-enc-bf16.pth`, `google/umt5-xxl/`).
- The 1.3B shards sit at the repository root, where the 14B release uses a
  `transformers/` subfolder, so an empty `fast_checkpoint` now means "root".
- diffusers resolves sharded weights through
  `diffusion_pytorch_model.safetensors.index.json`; the published file is named
  `model.safetensors.index.json`, so the assembled directory carries both.

Assembled checkpoint: `/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled`
(symlinks, 18 GiB resolved).

**`sample_shift = 10.0` is inherited from the 14B causal-fast config and is
not verified for the 1.3B model.** No upstream 1.3B reference exists to check
it against. It materially affects the sampling schedule and should be treated
as an open variable, not a validated default.

## Upstream architecture (from source inspection)

`generate.py` → `WanI2VCausal._generate_causal_fast`. Per chunk:

1. Four denoising steps at `timesteps[[0, 250, 500, 750]]`. `run_causal` calls
   `generate()` without `timesteps_index`, so `generate()`'s own default wins
   and `_generate_causal_fast`'s `[0,179,358,679]` is shadowed — a live trap
   for anyone reading only the inner function.
2. A fifth forward at timestep 0 whose sole purpose is to write the accepted
   `x0` into the KV cache.

**Five DiT forwards per chunk**, confirmed empirically (`dit_forwards=10` over
2 chunks).

Locality is **not** implemented as an attention mask. `CausalWanSelfAttention`
calls `attention(roped_query, k_cache, v_cache)` with no `causal`, no
`window_size` and no lengths: it attends densely over whatever the KV cache
currently holds. The window is enforced by *rolling the cache* — evicting
oldest tokens while preserving the first `sink_size * frame_seqlen` tokens.
This matters for the AMD port, because it means the SDPA replacement needs no
mask at all on the self-attention path to be exactly equivalent.

Cross-attention passes `context_lens`, but `WanModelFast.forward` sets
`context_lens = None` unconditionally and zero-pads the text to
`text_len=512`, so it too attends densely.

### Verified settings (from the checked-out source, not assumed)

| Setting | Value | Provenance |
|---|---|---|
| `local_attn_size` | **18** | `run_fast.sh`. The argparse default is `-1` (global attention) |
| `sink_size` | **6** | `run_fast.sh`. The argparse default is `0` |
| `chunk_size` | **4** | `generate.py` argparse default; not overridden in `run_fast.sh` |
| diffusion steps/chunk | **4** | `generate()` default `timesteps_index=[0,250,500,750]` |
| model forwards/chunk | **5** | 4 sampling + 1 KV-cache update; confirmed by measurement |
| `sample_shift` | 10.0 | Inherited from 14B — **not verified for 1.3B** |
| `sample_fps` | 16 | `shared_config.py` |
| `vae_stride` | (4, 8, 8) | config |

Note that 18 and 6 are upstream's *recommended* values from its own launch
script, not the CLI defaults. Both were confirmed in source before use.

Frame counts are quantised twice: `lat_f = (F-1)//4 + 1`, then truncated to a
multiple of `chunk_size`, then `F = (lat_f-1)*4 + 1`. With `chunk_size=4` the
reachable counts are 13, 29, 45, 61… A request for 21 frames silently becomes
13. **29 frames was chosen as the nearest reachable count to the 21 target**
that also exercises more than one chunk.

## AMD compatibility findings

Most CUDA references are harmless. `torch.cuda.*`, `torch.amp.autocast('cuda')`
and `torch.device('cuda:N')` all map to HIP, and `q.device.type == 'cuda'` is
true for HIP tensors. `torch.cuda.amp.autocast(dtype=torch.float32)` also works
on this build (verified). None of these needed changing.

Two things did.

### 1. FlashAttention → SDPA

`flash_attention()` ends in `assert FLASH_ATTN_2_AVAILABLE`. FlashAttention 2/3
have no CUDA-free build, and upstream's install instructions call for
`pip install flash-attn`, which would be wrong here.

Upstream *has* an SDPA fallback in `attention()` — but it silently ignores
`window_size`, `q_scale`, `softmax_scale` and all length masking, and
`WanCrossAttention` bypasses `attention()` entirely by calling
`flash_attention()` directly, so it would still have asserted.

Added `_sdpa_attention`, a semantically-equivalent replacement handling
padding masks, sliding windows, causal masking, GQA/MQA head expansion,
`q_scale` and `softmax_scale`, and routed both entry points through it.

Validated against a dense reference implementation across causal, window
`(3,0)`, window `(3,2)`, `k_lens`, `q_scale` and custom-scale cases: **max
error 3e-07** in every case.

One real trap found and fixed: **SDPA's `is_causal` aligns the mask top-left
while FlashAttention aligns it bottom-right.** They agree only when Lq == Lk.
Taking SDPA's fast path unconditionally gave a max error of **3.46** — not a
rounding difference, a wrong answer. The fast path is now taken only when
Lq == Lk. This does not affect the 1.3B causal-fast path, which never passes
`causal=True`, but it would have silently corrupted any path that did.

### 2. Explicit T5 lifetime (memory correction, not offload)

umt5-xxl is ~10.9 GiB in BF16. Upstream releases it only under
`offload_model`, which also enables per-forward DiT offload and
`empty_cache()` churn. With offload off, **14.58 GiB was still allocated when
VAE encode asked for its ~12 GiB peak** → OOM with 3.10 GiB free.

The encoder is semantically finished the moment the prompt is encoded, and its
output is cached in `_t5_cache`, so a warm repeat never touches it. It is now
released unconditionally after encoding. Allocation at VAE encode drops
**14.58 → 4.00 GiB**.

**The DiT stays resident throughout.** This is sequential residency of a
finished component, not generic CPU offload.

## gfx1201 / MIOpen workaround

This is the substantive finding, and it is a **platform workaround, not a
model optimization**. It changes which kernel MIOpen selects; it does not
change the model, its precision, its resolution or its arithmetic intent.

### Mechanism

```
FP32 Wan VAE Conv3d at full resolution
  → GemmFwdRest (im2col+GEMM) workspace scales with output temporal extent
    (11.6 GiB at out_T=3; ~15.5 GiB at out_T=4)
  → at out_T>=4 the workspace exceeds what MIOpen will allocate,
    so GemmFwdRest is dropped from the candidate list entirely
  → ConvDirectNaiveConvFwd is selected, silently, with no warning
  → 0.16 TFLOP/s against 2.6 TFLOP/s: a 17x collapse

  temporal output split = 1
  → each sub-convolution has out_T=1, well under the threshold
  → GemmFwdRest remains eligible and is selected
  → same convolution, restructured exactly
```

MIOpen's own FindDb states it outright:

```
96-5-482-834-3x3x3-96-3-480-832-1-...-FP32-F
  = GemmFwdRest:219.499,12421693440,miopenConvolutionFwdAlgoGEMM;
    ConvDirectNaiveConvFwd:3539.7,0,miopenConvolutionFwdAlgoDirect
96-6-482-834-3x3x3-96-4-480-832-1-...-FP32-F
  = ConvDirectNaiveConvFwd:5072.33,0,miopenConvolutionFwdAlgoDirect
```

At `out_T=3` MIOpen has both and picks the 219 ms GEMM. At `out_T=4` the GEMM
entry is simply **absent**.

Contributing factor: **`/opt/rocm-7.2.1/share/miopen/db` ships no gfx1201
database at all** (gfx1030, 803, 900, 906, 908, 90a, 942, 950 only), so every
shape is resolved into the user FindDb at runtime with no vendor-tuned
fallback.

### Trigger

The trigger is the **output temporal extent**, and nothing else. Fresh
contiguous `[1,96,T,482,834]` x `[96,96,3,3,3]`:

| in T | out T | warm median | effective |
|---:|---:|---:|---:|
| 3 | 1 | 0.096 s | 2.06 TFLOP/s |
| 4 | 2 | 0.169 s | 2.35 TFLOP/s |
| 5 | 3 | 0.508 s | 1.17 TFLOP/s |
| 6 | 4 | **5.036 s** | **0.16 TFLOP/s** |
| 8 | 6 | 7.590 s | 0.16 TFLOP/s |
| 12 | 10 | 12.621 s | 0.16 TFLOP/s |

Sharp cliff at `out_T=4`, then linear at the naive kernel's throughput.

### Intervention

`CausalConv3d.forward` splits along the output temporal axis. Each output
frame depends only on its own receptive field, so this is an exact
restructuring. It handles stride and dilation generally rather than assuming
the 3-tap unit-stride case, because one VAE `CausalConv3d` uses
`stride=(2,1,1)`.

Split selection, full 480x832 FP32 decode:

| split | cold | warm | speedup | peak alloc | peak reserved | rel err |
|---|---:|---:|---:|---:|---:|---:|
| 0 (upstream) | 110.71 s | 110.82 s | 1.00x | 13.28 GiB | 19.44 GiB | reference |
| **1 (selected)** | **18.64 s** | **16.48 s** | **6.73x** | **11.33 GiB** | **16.49 GiB** | 3.64e-06 |
| 2 | 16.91 s | 16.55 s | 6.69x | 15.33 GiB | 23.76 GiB | 3.64e-06 |
| 3 | 66.19 s | 16.35 s | 6.78x | 19.19 GiB | 25.69 GiB | 3.64e-06 |

**split=1 selected** because warm latency is flat across 1–3 (16.35–16.55 s,
within run-to-run noise) so the only real axis is memory, and split=1 is both
the cheapest and the only setting that stays below the unsplit baseline's own
peak allocation. split=3 additionally pays a 66 s cold search penalty for the
extra shapes it introduces.

Results are **not bit-identical** to upstream: `GemmFwdRest` and
`ConvDirectNaiveConvFwd` accumulate in different orders, giving 3.64e-06
relative difference — FP32 round-off, and the GEMM path is the
better-conditioned of the two. The relative error is identical across all
split values because it comes from the solver change, not from split
granularity.

`WAN_VAE_CONV3D_TEMPORAL_SPLIT=0` restores exact upstream behaviour.

## Attention implementation

**PyTorch SDPA**, via `_sdpa_attention`. On this build SDPA reports the
AOTriton Efficient Attention backend.

Chosen for semantic equivalence, not speed, per the bring-up brief. It is not
currently a bottleneck: the entire DiT — all 30 layers, 10 forwards, both
attention paths and every GEMM — is 6.9 s of a 63.0 s warm run.

## Reproduction

```sh
git clone https://github.com/robbyant/lingbot-world-v2.git
cd lingbot-world-v2 && git checkout 45fa40673607c9acba6cf96a1f9396c95bcef25f
git apply .../experiments/lingbot-world-v2-1.3b/patches/lingbot-rocm-r9700.patch

python3 -m venv /ai/envs/lingbot-world-v2
/ai/envs/lingbot-world-v2/bin/pip install \
    /ai/cache/amd-wheels/rocm-7.2.1/{torch,torchvision,torchaudio,triton}-*.whl
/ai/envs/lingbot-world-v2/bin/pip install "numpy<2" opencv-python "diffusers>=0.31.0" \
    "transformers<=4.51.3" accelerate tqdm "imageio[ffmpeg]" easydict ftfy \
    imageio-ffmpeg scipy einops sentencepiece protobuf safetensors

hf download robbyant/lingbot-world-v2-1.3b-causal-fast \
    --local-dir /ai/models/lingbot-world-v2-1.3b-causal-fast
hf download robbyant/lingbot-world-v2-14b-causal-fast \
    Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
    google/umt5-xxl/{special_tokens_map.json,spiece.model,tokenizer.json,tokenizer_config.json} \
    --local-dir /ai/models/lingbot-world-v2-14b-assets
# then assemble both into one directory (see experiments/.../README.md)

experiments/lingbot-world-v2-1.3b/run_baseline.sh 29 2 baseline
```

`run_baseline.sh` pins `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib` and
`HIP_VISIBLE_DEVICES=1`.

## Baseline results

29 frames, 464x832, BF16 DiT, FP32 VAE, `local_attn_size=18`, `sink_size=6`,
`chunk_size=4`, seed 42, split=1, no offload.

| Measurement | Cold | Warm 1 | Warm 2 |
|---|---:|---:|---:|
| Total wall | 66.54 s | **62.98 s** | **63.08 s** |
| Effective fps | 0.436 | 0.460 | 0.460 |
| T5 encode | 0.76 s | cached | cached |
| VAE encode | 21.23 s | 21.34 s | 21.37 s |
| DiT total (10 forwards) | 6.88 s | 6.92 s | 6.92 s |
| VAE decode | 34.46 s | 34.54 s | 34.57 s |
| Time to first chunk | 3.114 s | 3.117 s | 3.119 s |
| Per-chunk DiT | 3.114 / 3.771 s | 3.117 / 3.803 s | 3.119 / 3.805 s |
| Mean per-forward DiT | 0.688 s | 0.692 s | 0.692 s |

Model load is 31.7 s, measured separately and excluded from generation wall
time.

Two warm runs agree to **0.16%**. Chunk 2 costs 22% more than chunk 1, which
is the KV cache growing — at 29 frames `lat_f=8`, below the 18-frame window,
so nothing is ever evicted and the cache only grows.

13-frame smoke, same configuration: load 31.6 s, cold 32.16 s, warm 28.46 s,
one chunk, 5 forwards, VAE encode 9.62 s, VAE decode 15.47 s.

**The cold/warm gap is small and shrinking for a reason worth recording.** The
first-ever 13-frame run took 76.77 s cold; the rerun took 32.16 s. The
difference is MIOpen's user FindDb, which persists across processes — a truly
cold machine pays a one-time per-shape search cost that no later run repeats.

## Memory results

| Phase | Allocated | Peak allocated in phase |
|---|---:|---:|
| After model load | 3.68 GiB | 3.68 GiB |
| During T5 encode (cold only) | 14.43 GiB | 14.74 GiB |
| After T5 release | 4.09 GiB | — |
| VAE encode | 4.09 GiB | 13.03 GiB |
| DiT generation | 8.75 GiB | 13.03 GiB |
| VAE decode | 8.87 GiB | **19.19 GiB** |

- **Peak allocated: 19.19 GiB.** Peak reserved: 23.62 GiB. System RSS: 12.63 GiB.
- DiT weights: 3.18 GiB (1,709,502,016 params, all BF16).
- VAE weights: 0.473 GiB (FP32).
- KV cache: `local_attn_size=18` x `frame_seqlen=1508` x 30 layers x 2 (K,V)
  x 12 heads x 128 dim x 2 bytes = **4.66 GiB** if fully populated. At 29
  frames only 8 of 18 window slots are used, which matches the 4.09 → 8.75 GiB
  step across generation.

**The phases do not overlap.** T5, VAE encode, DiT and VAE decode each peak
sequentially, which is precisely why 32 GiB is sufficient. Peak reserved
(23.62 GiB) overstates the physical requirement — it is allocator high-water
mark across all phases, not simultaneous residency.

## Correctness

- No NaN, no Inf, in any run.
- Output range fully spans [-1, 1]; mean 0.334, std 0.446.
- Per-frame std 0.3989–0.4634, rising smoothly — no black or collapsed frames.
- Frame-to-frame L1 0.0258–0.1662, mean 0.0705, **no zeros** — no frozen or
  repeated chunks, and no discontinuity at the chunk boundary (frame 12→13).
- **Bit-identical across cold, warm 1 and warm 2** — all three 29-frame
  outputs hash to `745f251ceaa5c1b0…`. The pipeline is deterministic at fixed
  seed on this stack.

No CUDA reference output was available for comparison. Internal consistency
and the 3e-07 attention validation are what support correctness here.

## Failures and negative findings

Preserved because they constrain the conclusion.

**Failed hypotheses about the VAE slowdown, all disproven:**

- *Layout / non-contiguity.* Dead. At the slow shape, `F.pad` output, a fresh
  allocation, `.contiguous()` and `.clone()` all give 5.05 s with identical
  ordinary contiguous strides `(231545088, 2411928, 401988, 834, 1)`.
  `channels_last_3d` was marginally worse (5.16 s).
- *Spatial extent / alignment.* Dead. Sweeping H over 480–496 and W over
  832–864 stays at 0.14–0.16 TFLOP/s throughout.
- *Repeated kernel search, JIT or cache misses.* Dead. The dominant conv costs
  the same on call 1 as on call 54 (5.068 s vs 5.061 s), and three consecutive
  decodes were 110.76 / 110.74 / 110.74 s.
- *A missing or non-persistent MIOpen cache.* Dead. The gfx1201 user FindDb
  exists, is written, persists across processes, and honours
  `MIOPEN_USER_DB_PATH`.
- *`MIOPEN_FIND_MODE=FAST`.* No effect whatsoever — identical timings to three
  decimals and an identical output hash. Consistent with there being no search
  to avoid.
- *Bypassing MIOpen (`torch.backends.cudnn.enabled = False`).* OOMed at
  15.43 GiB. Informative rather than useless: that is the same im2col buffer
  MIOpen refuses to allocate, which is what confirmed the workspace mechanism.

**Not required, and therefore not done:**

- Vulkan or any alternative convolution backend
- BF16 VAE (upstream FP32 preserved)
- Tiled VAE
- Generic model CPU offload (`offload_model` stayed off; DiT never leaves the GPU)
- Quantization or GGUF
- Reduced resolution
- FSDP, Ulysses, NCCL or any multi-GPU machinery
- A ROCm upgrade

**Process errors worth recording:**

- My first Conv3d microbenchmark used the shape my instrumentation logged,
  which was the **pre-padding** input — `CausalConv3d` sets `padding=(0,0,0)`
  and pads itself. That shape benchmarks at 0.146 s and briefly suggested the
  cost lay outside the convolution. It did not. Instrumenting a wrapper means
  recording what the wrapped call receives, not what the wrapper receives.
- I initially reported a "7.46 GiB workspace". That was a PyTorch allocator
  delta, not a MIOpen-reported workspace. The real figure (12.4 GB) came later
  from MIOpen's own FindDb.
- `bench.py` chdirs into the upstream repo, so a relative `--out_json` was
  written into the wrong tree and lost a complete run's JSON. Paths are now
  resolved before the chdir.
- Wrapping `text_encoder.__call__` on the *instance* measured nothing, because
  Python looks dunders up on the type. T5 timing was silently absent from the
  first JSON.

## Profiling

Warm 29-frame run, 63.0 s total:

| Component | Time | Share |
|---|---:|---:|
| **VAE decode** | 34.54 s | **54.8%** |
| **VAE encode** | 21.34 s | **33.9%** |
| DiT (30 layers x 10 forwards) | 6.92 s | 11.0% |
| T5 | 0 s (cached) | 0% |
| Orchestration / save | ~0.2 s | 0.3% |

**The FP32 VAE is 88.7% of warm wall time even after a 6.75x improvement.**
The transformer — the actual model — is 11%.

Within the unfixed VAE decode, one Conv3d shape was 82% of the total. Attention
never appeared as a cost at any point in this investigation.

## Output

- `/ai/outputs/lingbot-world-v2-1.3b/baseline_cold_480x832_29f_w18_s6.mp4`
- `/ai/outputs/lingbot-world-v2-1.3b/baseline_warm1_480x832_29f_w18_s6.mp4`
- `/ai/outputs/lingbot-world-v2-1.3b/baseline_warm2_480x832_29f_w18_s6.mp4`
- `/ai/outputs/lingbot-world-v2-1.3b/smoke_{cold,warm1}_480x832_13f_w18_s6.mp4`

All three baseline files are byte-identical.

## Conclusions

The R9700 runs LingBot-World-V2 1.3B causal-fast comfortably in native BF16 at
the upstream operating point, with 12.7 GiB of headroom at peak. The model is
not the constraint — the DiT is 3.18 GiB of weights and 0.69 s per forward.

What the card is currently limited by is **MIOpen's 3D convolution coverage
for gfx1201**, not by capacity, bandwidth or precision. A vendor-tuned database
for this architecture would likely remove most of the remaining VAE cost
without any code change at all.

The 29-frame baseline does not yet exercise the causal machinery that makes
this model interesting: with `lat_f=8` under an 18-frame window, the KV cache
never evicts and the sink tokens never matter. Window scaling needs
`frame_num >= 77` to cross that boundary.

## Follow-up leads

Evidence-ranked. These are leads, not a roadmap.

1. **VAE encode wastes ~21 s encoding zeros.** `_generate_causal_fast` builds
   `concat([image, zeros(3, F-1, h, w)])` and encodes the whole tensor, though
   only the first frame carries information — the mask channels tell the DiT
   the rest is padding. 33.9% of warm wall time. Needs verification that the
   causal VAE's temporal state makes the zero frames genuinely inert.
2. **Remaining VAE decode cost after the split fix** — 34.5 s, 54.8%. Now that
   `GemmFwdRest` is selected, the cost is im2col + GEMM on FP32 at full
   resolution. Worth profiling per-shape again post-fix; the split threshold
   may not be optimal per-shape.
3. **A gfx1201 MIOpen database.** The absence of any tuned db is the root
   enabler of finding 3. `MIOPEN_FIND_MODE=NORMAL` exhaustive tuning was never
   run to completion and might populate better solvers.
4. **Upstreaming the temporal split** as a guarded ROCm workaround, and
   reporting the silent naive-kernel fallback to MIOpen as a bug — a solver
   being dropped for workspace reasons should not be invisible.
5. **DiT is only 11%**, so attention, KV-cache layout and BF16 GEMM work have
   little headroom to buy at this size. Deprioritise until the VAE is resolved.
6. **`sample_shift=10.0` is unverified for 1.3B** and affects output quality
   rather than speed. A shift sweep against operator judgement would close it.
