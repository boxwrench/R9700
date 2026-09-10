# Phase B — 14B GGUF short compatibility baseline on R9700 / gfx1201

Q4_K_M, window 6+2, seed 42, `examples/03` image/actions, shift 3.0.
Community pack defaults except where noted.

## Baseline nomenclature

| | |
|---|---|
| Requested size preset | **480x832** |
| Actual executed dimensions | **464x832** |
| Requested frame count | **29** |
| Community `chunk_size` | **3** (upstream CLI default is 4) |
| Actual frame count | **21** |
| Chunks | 2 |
| DiT forwards per chunk | 5 (4 sampling + 1 KV write) |

Upstream truncates `lat_f` to a multiple of `chunk_size`: `(29-1)//4+1 = 8`,
`8 - 8%3 = 6`, so `F = (6-1)*4+1 = 21`. **21 frames is the accepted
community-pack baseline, not a failure.** Getting 29 frames on this pack
requires `chunk_size=4`.

`464x832` is upstream's aspect-ratio arithmetic at the 480x832 *area* target
for a 16:9 input, identical to the native 1.3B bring-up. Not a fallback.

## Headline: the community VAE path does NOT avoid the gfx1201 cliff

Measured, not inferred. Identical settings, only `WAN_VAE_CONV3D_TEMPORAL_SPLIT`
differs:

| | split=0 (community as published) | split=1 (our fix) | ratio |
|---|---:|---:|---:|
| **Warm wall** | **328.15 s** | **86.48 s** | **3.79x** |
| Cold wall | 330.09 s | 86.76 s | 3.80x |
| VAE encode | 107.25 s | 15.58 s | 6.88x |
| VAE decode | 177.03 s | 25.15 s | 7.04x |
| DiT total (10 forwards) | 43.51 s | 45.39 s* | 0.96x |
| Peak allocated | 20.41 GiB | **18.48 GiB** | lower |
| Peak reserved | 23.53 GiB | 23.53 GiB | same |
| Host RSS | 16.01 GiB | 16.01 GiB | same |
| Output mean / std | 0.3649 / 0.4221 | 0.3649 / 0.4224 | equivalent |
| NaN / Inf | none | none | |

\* the split=1 DiT figure carries `--profile_dequant` instrumentation
(4910 extra synchronisations); the uninstrumented DiT is 43.5 s, identical to
split=0. The VAE split does not touch the DiT.

**The temporal split is required on the community path too.** It is not
redundant with anything the pack does, and it also *lowers* peak allocation.

### Exact dominant Conv3d, measured on this geometry

Instrumented VAE decode at 14B baseline geometry (lat 58x104, 6 latent
frames), split disabled — 17 unique Conv3d shapes, one dominates:

| Conv3d input (pre-pad) | weight | calls | first | repeat | total |
|---|---|---:|---:|---:|---:|
| `[1, 96, 4, 464, 832]` | `[96,96,3,3,3]` | 30 | 4.905 s | 4.893 s | **146.79 s** |
| `[1, 96, 4, 464, 832]` | `[3,96,3,3,3]` | 5 | 2.979 s | 2.984 s | 14.92 s |
| `[1, 192, 4, 232, 416]` | `[192,192,3,3,3]` | 30 | 0.277 s | 0.258 s | 7.75 s |

One shape is **83% of a 177 s decode**, and first call equals repeat call —
the same signature as the native bring-up: not search, not caching, a
silently-selected naive kernel. `CausalConv3d` pre-pads, so the convolution
actually sees `[1,96,6,466,834]` with `out_T=4`, right past the cliff.

## Reconciled component breakdown (accepted split=1 warm run)

Warm wall **86.48 s**:

| Component | Time | Share | Notes |
|---|---:|---:|---|
| T5 encode | 2.39 s | — | CPU, once at setup, outside the timed loop |
| VAE encode | 15.58 s | 18.0% | |
| DiT total | 45.39 s | 52.5% | 10 forwards over 2 chunks |
| — GGUF H2D | 3.76 s | 4.3% | 4910 transfers, 0.77 ms mean |
| — GGUF dequant | 7.60 s | 8.8% | 4910 calls, 1.55 ms mean |
| — GGUF matmul | 13.45 s | 15.6% | 4910 calls, 2.74 ms mean |
| — rest of DiT | 20.58 s | 23.8% | attention, norms, rope, conv, elementwise |
| VAE decode | 25.15 s | 29.1% | |
| Save / serialize | 1.27 s | 1.5% | |
| **Sum** | **87.39 s** | | vs 86.48 s wall; -0.91 s residual is timer overhead |

The GGUF H2D/dequant/matmul rows are **nested inside** `dit_forward`, not
additional to it.

**GGUF streaming is 24.81 s — 54.7% of DiT time, 28.7% of the whole run.**
Dequant alone (7.60 s) costs more than the H2D transfer (3.76 s), so this is
compute-bound on unpacking, not PCIe-bound.

### What "~22 s per chunk" means

**The entire causal chunk.** Per-chunk DiT totals are 21.96 s and 23.44 s, each
being 5 forwards at ~4.5 s. So:

- one DiT forward = **4.54 s**
- one diffusion step = one forward = 4.54 s
- one causal chunk = 5 forwards = **~22 s**

Chunk 2 is 6.7% slower than chunk 1 as the KV cache fills.

## Memory accounting

Model file size is **not** VRAM residency.

| Quantity | Value |
|---|---:|
| GGUF file (mapped) | 11.23 GiB |
| Quantized bytes held host-side | 10.77 GiB |
| Pinned host memory | 2.01 GiB (`pin_gb=2` budget) |
| Process RSS at load | 7.69 GiB |
| Process RSS during DiT | 16.01 GiB |
| Process VmSize | 31.8 GiB (mmap, not resident) |
| GPU alloc immediately after load | **0.00 GiB** |
| Resident dense (non-streamed) GPU params | 0.44 GiB |
| DiT + VAE on GPU before generation | 0.94 GiB |
| KV cache growth across generation | ~6.8 GiB |
| VAE encode peak | 12.30 GiB |
| VAE decode peak (split=1) | 18.48 GiB |
| **Total peak GPU allocation** | **18.48 GiB** |
| Peak reserved | 23.53 GiB |

Loader takes **0.65 s** because `torch.from_numpy` over `GGUFReader`'s memmap
shares the mapping — no copy. 563 Linear modules are wrapped; their quantized
bytes never leave the host, and each forward streams its own.

**Caveat on free VRAM:** two production ComfyUI processes were resident on this
GPU throughout (PIDs 267209 and 578406, ~2.1 GB combined). They were left
untouched. Per-process torch allocator figures above are unaffected, but free
VRAM for these runs was ~29.7 GiB rather than the full 31.86 GiB.

## The three AMD findings

### GGUF — clean for bring-up

- Dequant is city96's `dequant.py`: **pure PyTorch tensor arithmetic**, numpy
  fallback, no CUDA extension, no Triton. ROCm-compatible by construction.
- 1421 tensors attached, **0 unmatched, 0 still-meta** — the bundled 14B
  config matches the GGUF exactly.
- Quantized data stays host-resident (`FIRST LINEAR: x.device=cuda:0
  qdata.device=cpu`), confirming streaming actually occurs rather than a
  silent full upload.
- Output is clean: no NaN, no Inf, sensible statistics, and the split=0 and
  split=1 lanes agree to 3 decimal places on mean and std.

### Attention — not ROCm-compatible as published

`wan/modules/attention.py` is vendored verbatim from upstream, so
`flash_attention()` still ends in `assert FLASH_ATTN_2_AVAILABLE`, and
`WanCrossAttention` calls it **directly**, bypassing the `attention()`
dispatcher that carries the SDPA fallback. FlashAttention 2/3 have no
CUDA-free build.

The pack therefore cannot complete a single forward on ROCm as published.
Substituted the `_sdpa_attention` path already validated in the native 1.3B
bring-up to **max error 3e-07** against a dense reference across causal,
sliding-window, padding, `q_scale` and custom-scale cases.

### Memory — file size is not residency

See the table above. An 11.23 GiB Q4_K_M file produces **0.00 GiB** of GPU
allocation at load and 0.44 GiB of resident dense parameters. The 18.48 GiB
peak is dominated by the KV cache (~6.8 GiB) and the VAE decode spike, not by
weights.

## Corrected KV pre-flight

Confirmed from source — `wan/image2video.py`:

```python
if self.local_attn_size > -1:
    kv_size = frame_seqlen * self.local_attn_size
```

**KV temporal extent = `local_attn_size`.** Sink frames are the first
`sink_size` frames *inside* that window, preserved during eviction; they do not
extend it.

The pack's pre-flight used `(local_ + sink_)`:

| | 6+2 at 464x832 |
|---|---:|
| Pack's estimate | 9.88 GB |
| True upstream allocation | 7.41 GB |
| **Measured growth during generation** | **~6.8 GiB (7.3 GB)** |

Measurement matches the corrected formula, not the pack's. Patched in
`patches/lbworld_nodes-kv-preflight.patch`. Note the pack's own `need_window`
*suggestion* already used the correct per-frame cost — only the check was
inconsistent with it.

Corrected feasibility at 464x832, 1508 tokens/frame, 1.1506 GiB per window
frame:

| window | pack's estimate | true allocation |
|---|---:|---:|
| 6+2 | 9.21 GiB | **6.90 GiB** (measured ~6.8) |
| 12+4 | 18.41 GiB | **13.81 GiB** |
| 18+6 | 27.62 GiB | **20.71 GiB** |

## Process corrections made during this phase

- **My first "split=1" lane did not actually have the split enabled.** I had
  copied only `attention.py` into the lab pack, not `vae2_1.py`, so the env var
  had nothing to act on and both lanes ran the unmodified community VAE. The
  numbers stood (328 s), but the label was wrong. Both lanes were re-run after
  patching the vendored VAE. The originally-reported 327.99 s is now correctly
  identified as the **community control**, and it reproduces at 328.15 s.
- `_T5Shim.set` takes the `[L, C]` tensor `_cond_to_embed` would produce, not a
  list; the harness passed a list and failed on first use.
- An over-broad `pkill -f` pattern killed my own foreground command.
- A stray duplicate lane was left holding 29.9 GB of VRAM and had to be
  stopped; checking `--showpids` first was what prevented me from killing the
  user's two production ComfyUI processes on the same GPU.

## Comparison with the native 1.3B baseline (context, not benchmark)

| | native 1.3B BF16 | 14B GGUF Q4_K_M |
|---|---:|---:|
| Frames | 29 | 21 |
| Window | 18+6 | 6+2 |
| Warm wall | 63.0 s | 86.5 s |
| DiT total | 6.92 s | 45.39 s |
| Per DiT forward | 0.692 s | 4.54 s |
| VAE encode | 21.34 s | 15.58 s |
| VAE decode | 34.54 s | 25.15 s |
| Peak allocated | 19.19 GiB | 18.48 GiB |

Different frame counts and windows — not like-for-like. The informative
contrast is the **DiT forward: 0.69 s vs 4.54 s, a 6.6x gap** for a 8.2x
larger model that is also streamed and dequantized per layer. VAE times are
lower here only because there are fewer frames.

The role reversal is the point: in the native 1.3B run the VAE was 88.7% of
wall time and the DiT 11%. Here, with the split applied, the DiT is 52.5% and
the VAE 47.1%. At 14B the transformer finally dominates.
