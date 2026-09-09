# Phase A — implementation inspection (14B GGUF community path)

Source inspection only. No benchmarks in this document.

## What the community target actually is

The brief named two GitHub repositories; **both 404**:

- `RealRebelAI/ComfyUI_Rebels_LingBotWorld` — 404 (this is the link the HF
  model card itself uses, so the model card is stale)
- `RealRebelAI/LingBot_World_V2_ComfyUI` — 404

The real artefacts:

| Thing | Location | SHA / revision |
|---|---|---|
| Node pack | `github.com/RealRebelAI/Rebels_LingBot-World-V2_GGUF_ComfyUI` | `f310acc1109e7af447d5c6efb94f29fe746408a1` (2026-07-11) |
| GGUF weights | `hf.co/realrebelai/LingBot_World_V2_ComfyUI` | `79be5d5f30d63327ac6020ced7f653cdbc301f49` |
| Dequant kernels | `github.com/city96/ComfyUI-GGUF` | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` |

Note the *inner* package directory inside the node pack is still named
`ComfyUI_Rebels_LingBotWorld`, which is where the dead link came from.

### Available quantizations

| File | Size |
|---|---:|
| `LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf` | 11.226 GiB |
| `LingBot-World-14B-Q4_K_S.gguf` | 10.385 GiB |
| `LingBot-World-14B-Q5_K_M.gguf` | 12.784 GiB |
| `LingBot-World-14B-Q6_K.gguf` | 14.438 GiB |
| `LingBot-World-14B-Q8_0.gguf` | 18.567 GiB |
| `umt5-xxl-encoder-Q4_K_S.gguf` | 3.257 GiB |
| `Wan2.1_VAE.pth` | 0.473 GiB |

All declare `base_model: robbyant/lingbot-world-v2-14b-causal-fast`. Only the
Q4_K_M carries `causal-fast_merged` in its filename; **whether the other tiers
are the same merged causal-fast weights has not been verified from their GGUF
metadata yet** and must be checked before any Phase D quant comparison is
treated as like-for-like.

## The vendored `wan/` is byte-identical to upstream

`diff -rq` of the pack's `ComfyUI_Rebels_LingBotWorld/wan/` against pristine
upstream `45fa40673607c9acba6cf96a1f9396c95bcef25f` reports **no differing
files**. The pack vendors upstream verbatim.

This single fact answers most of Phase A, because it means the model, VAE,
attention, KV cache and causal semantics are exactly the ones already
characterised in the native 1.3B bring-up.

The pack does not subclass or monkey-patch the pipeline either. `LBWorldLoader`
builds the upstream object with `object.__new__(WanI2V)` and populates its
attributes by hand — deliberately skipping `__init__` so it never force-loads
the 11 GiB T5 `.pth` or the 28 GiB dense DiT — then `LBWorldSampler` calls
upstream `pipe.generate(...)` unchanged.

## GGUF streaming machinery

- **Dequant** is delegated to city96's `ComfyUI-GGUF/dequant.py`, loaded by
  path. That file is **pure PyTorch tensor arithmetic** with a numpy fallback —
  no CUDA extensions, no Triton, no compiled kernels. It is therefore
  ROCm-compatible by construction, not by porting. (Correctness on ROCm still
  has to be measured, not assumed.)
- **Only `nn.Linear` weights are streamed.** `GGUFLinearWrapper` holds the
  quantized bytes as a plain attribute — explicitly *not* `register_buffer`,
  with a comment saying that would make `model.to(device)` drag all 11 GiB onto
  the card. Each `forward` copies its own `qdata` H2D and dequantizes to the
  activation dtype, then frees it.
- **Everything else is dequantized once to BF16 and stays resident**: norms,
  convolutions, embeddings, and any tensor stored as F16/F32 in the GGUF.
- **mmap**: `torch.from_numpy(t.data)` over `GGUFReader`'s memmap shares the
  file mapping — no RAM copy. So mapped bytes are not resident bytes.
- **Pinning** is budgeted (`pin_gb`, default 2 GiB). Pinned tensors *are* real
  host copies; everything past the budget stays mmapped and pageable.
- **GPU-resident DiT weight at any instant is therefore roughly one layer's
  dequantized Linear set plus all the dense non-Linear parameters** — not the
  11.2 GiB file. Actual figures must come from measurement.

There is **no hardware-specific branching anywhere in the pack** — no `gfx`,
no Strix, no ROCm/CUDA fork. The same path is taken on any device
`torch.cuda.is_available()` accepts, which on this stack is HIP.

## Attention — the pack cannot run on AMD as published

Because `wan/modules/attention.py` is vendored verbatim, it carries upstream's
`assert FLASH_ATTN_2_AVAILABLE` inside `flash_attention()`, and
`WanCrossAttention` calls `flash_attention()` **directly**, bypassing the
`attention()` dispatcher that has the SDPA fallback.

FlashAttention 2/3 have no CUDA-free build. **The pack as published cannot
complete a single forward on ROCm.** This is not a performance issue, it is a
hard failure, and it is inherited from upstream rather than introduced by the
pack.

Fix applied for this experiment: the same `_sdpa_attention` path validated to
3e-07 in the native 1.3B bring-up, copied onto the vendored `attention.py`.

## VAE — the community path does NOT avoid the gfx1201 MIOpen pathology

Answering the brief's Q3 directly:

| Question | Answer |
|---|---|
| Different VAE implementation? | **No** — byte-identical upstream `Wan2_1_VAE` |
| Different precision? | **No** — FP32 (`dtype=torch.float` default) |
| Tiled decoding? | **No** — no tiling anywhere in the pack or in `wan/` |
| Temporal tiling/splitting? | **No** — only upstream's internal 4-frame `encode` loop |
| Bypasses MIOpen? | **No** — plain `F.conv3d` through `CausalConv3d` |
| Is our temporal split unnecessary? | **No — it is still required** |

The pack's only VAE-specific logic is `_move_vae`, which shuttles the module
between CPU and GPU around sampling. It does not change a single convolution.

So the mechanism characterised in the native bring-up applies unchanged:
`out_T >= 4` → `GemmFwdRest` workspace ineligible → `ConvDirectNaiveConvFwd` →
collapse. What the community path *does* change is the **magnitude**: their
default preset is 256x448, roughly 3.5x fewer pixels than 480x832, so the
absolute cost of the naive kernel is proportionally smaller and easier to
mistake for "slow but tolerable" on the NVIDIA cards it was developed against.
The temporal extent — which is what actually triggers the cliff — is unchanged.

This will be measured in isolation rather than asserted.

## The zero-future-frame encode is present

Yes. `wan/image2video.py` is vendored verbatim, so `_generate_causal_fast`
still builds `concat([image, zeros(3, F-1, h, w)])` and encodes the whole
tensor. Same shared-Wan optimization lead identified in the native bring-up;
not to be acted on during this bring-up.

## KV cache / causal semantics

Identical to upstream: `local_attn_size` and `sink_size` are passed into
`WanModelFast`, locality is enforced by *rolling the cache* rather than by any
attention mask, and sink frames are retained at the head of the window.

Cache dtype is `pipe_dtype` = BF16, shape
`[batch, frame_seqlen * local_attn_size, num_heads, head_dim]` per layer.

### A real bug in the pack's KV pre-flight

`LBWorldSampler` refuses to start if the estimated cache exceeds budget:

```python
kv_gb = 2 * 40 * (local_ + sink_) * tokens * 5120 * 2 / 1e9
```

But upstream allocates (`image2video.py`):

```python
kv_size = frame_seqlen * self.local_attn_size
```

**`sink_size` does not add to the cache.** Sink frames are the first
`sink_size` frames *inside* the local window, preserved during eviction — not
extra capacity. The pre-flight therefore overestimates by
`(local + sink) / local`, which at 18+6 is **1.33x**.

The pack's own README states "upstream 18+6 @ 480x832 ~ 21GB". The true
allocation is `2 x 40 x 18 x 1508 x 5120 x 2 = 22.2 GB (20.7 GiB)` — so the
README's figure is right and its *code* is the thing that is wrong, computing
29.7 GB for the same configuration.

Consequence on a 32 GiB R9700: the pre-flight will refuse some configurations
that would actually fit. Predicted true KV at 480x832, 1508 tokens/frame,
1.1506 GiB per window frame:

| window | their pre-flight | true allocation |
|---|---:|---:|
| 6+2 | 9.21 GiB | **6.90 GiB** |
| 12+4 | 18.41 GiB | **13.81 GiB** |
| 18+6 | 27.62 GiB | **20.71 GiB** |

To be confirmed against measured allocation.

## T5 lifecycle

The pack does **not** use upstream's T5 at all. `_T5Shim` is a stub whose
`__call__` returns embeddings that ComfyUI's own `CLIPLoaderGGUF` computed
upstream in the graph, keyed by `sha256(prompt)`; `LBWorldSampler` calls
`mm.unload_all_models()` before sampling so the encoder is gone from VRAM
before the DiT starts. That is a cleaner lifecycle than upstream's, and
equivalent in effect to the explicit T5 release added in the native bring-up.

Because this harness runs headless with no ComfyUI graph, it encodes the
prompt once with the real `T5EncoderModel` on CPU and feeds the shim — same
result, no GPU residency at any point.

## VAE lifecycle

Built on **CPU** (a comment records that loading straight to CUDA died on
Windows commit exhaustion), moved to GPU around sampling by `_move_vae`, moved
back to CPU in a `finally`. `offload_model=True` is passed to `generate()`, so
upstream additionally moves the DiT to CPU before the final VAE decode and
calls `empty_cache()` between chunks.

## Convention differences to respect

- **`chunk_size` default is 3**, not upstream's CLI default of 4.
- **`shift` default is 3.0**, not the config's 10.0 — matching upstream's
  README note that 3.0 suits 480p.
- Resolution is a **preset enum**, `256x448` by default; `480x832` is offered
  and labelled "needs tiny window".
- `frame_num` default 81, forced to 4n+1.
- Frame quantisation is upstream's, so reachable counts still depend on
  `chunk_size`: with `chunk_size=3`, `lat_f` truncates to a multiple of 3.

## Consequences for the plan

1. Q3 is answered: **the temporal split is still needed.** Not transplanted
   blindly — it will be measured on the community path both ways.
2. The pack needs the SDPA fix to run at all on ROCm.
3. The pre-flight bug may block Phase C lanes that would otherwise fit; the
   true allocation is what will be reported.
4. Phase D's premise needs checking first: only the Q4_K_M file is named
   `causal-fast_merged`.
