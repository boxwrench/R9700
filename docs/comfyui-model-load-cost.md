# Slow ComfyUI model loading on the R9700 is mmap, not quantization

## Outcome

On an AMD Radeon AI PRO R9700 (`gfx1201`, ROCm 7.2.1), ComfyUI model loads that
took **fifteen minutes** complete in **1.2 seconds** with one launcher flag:

```
--disable-mmap
```

| Measurement | Default | `--disable-mmap` | Change |
|---|---:|---:|---:|
| MiniMax H3 Qwen3-VL-32B encoder (24.71 GiB, 1,836 tensors) | 931 s | **1.2 s** | **776x** |
| MiniMax H3 fl2va transformer (19.52 GiB) | 419 s | **~20 s** | **21x** |
| Full H3 warm-up (encoder + DiT + VAE + 1-step sample) | 1,362 s | **51.07 s** | **27x** |

**Quantization format is not the variable.** `bf16`, `fp8`, `int8_convrot` and
`nvfp4` all sit on the same path.

## Root cause

`weight.to(device)` on **mmap-backed safetensors costs roughly 0.5 seconds per
tensor, independent of tensor size.** It is per-call overhead, not bandwidth.

Micro-benchmark on this card, moving tensors out of a real checkpoint:

| Path | 6 large tensors (1,924.7 MiB) | 400 tensors (median 7.5 KiB) |
|---|---:|---:|
| mmap → GPU | 1.370 s — 1,404 MiB/s | **199.934 s — 25.3 MiB/s (499.84 ms/tensor)** |
| normal RAM → GPU | 0.073 s — 26,428 MiB/s | 0.243 s — 20,836 MiB/s (0.61 ms/tensor) |
| pinned RAM → GPU | 0.071 s — 27,039 MiB/s | — |
| **ratio** | **19x** | **822x** |

With a handful of large tensors the penalty is a survivable 19x. With hundreds
of small ones it is 822x, because the cost is paid per call.

That yields a predictive model:

    load_time ≈ tensor_count × 0.5 s

| Model | Tensors | Predicted | Measured |
|---|---:|---:|---:|
| LTX-2.5 Gemma-4-12B encoder `int8_convrot` | 1,342 | 11 m 10 s | **11 m 11 s** |
| MiniMax H3 Qwen3-VL-32B encoder `fp8` | 1,836 | 15 m 17 s | **15 m 31 s** |
| LTX-2.5 22B transformer `int8_convrot` | 7,229 | 60 m 13 s | >35 m, aborted |

Two independent predictions land within 1% and 1.5%.

## Ruled out by measurement

Each of these was tested and **failed** to explain the behaviour. They are
recorded so the work is not repeated:

- **Quantization format.** A 9.57 GiB `bf16` encoder and a 14.32 GiB
  `int8_convrot` encoder both load slowly; a 24.46 GiB `bf16` encoder was
  aborted at 8 m 06 s. Apparent format correlation was a coincidence of tensor
  counts.
- **`--disable-dynamic-vram`.** 931 s with the flag, 938 s without — a 7-second
  difference on a 15-minute load.
- **`--disable-smart-memory`, `--disable-async-offload`, `--disable-pinned-memory`.**
  No measurable effect on load time.
- **ComfyUI version.** The fast 2026-08-12 baseline (`c2bcbecd`) and the slow
  runs (`7cee3ceb`, v0.33.2) are separated by 9 commits, none of which touch
  loading, quantization, or memory management.
- **Changed model files.** Byte-identical, unmodified since 2026-08-12.
- **Upstream [#15665](https://github.com/Comfy-Org/ComfyUI/issues/15665)**
  (H3 ~4x slower since v0.32.0). That is a *sampling* regression; this
  workstation does not reproduce its signature — sampling here runs at 100% GPU
  and 299 W against a 300 W cap.

## Diagnostic signature

While a load is stuck on this path:

- one CPU core pegged at ~100%, the rest idle
- GPU at **42 MHz / ~20 W**, utilisation ~3%
- VRAM filling gradually rather than in a burst
- disk reads well below device capability (most bytes come from page cache)
- logs show `Using MixedPrecisionOps for text encoder` and
  `Found quantization metadata version 1`

That matches upstream
[#15001](https://github.com/Comfy-Org/ComfyUI/issues/15001) —
*"[ROCm][gfx1201] General model loading became extremely slow on R9700"* — which
is open, on the same GPU, and whose reporter independently found `--disable-mmap`
and profiled the same mmap→GPU transfer cost.

## Unrelated open item: comfy-kitchen Triton backend is disabled

Separate from loading, quantized *compute* on this card falls back to the
`eager` backend. `comfy/quant_ops.py` gates Triton like this:

```python
elif args.enable_triton_backend: # or (torch.version.hip is not None and _rocm_kitchen_arch_supported()):
```

The ROCm auto-enable clause is **commented out**, although `cli_args.py` still
documents `--disable-triton-backend` as overriding "the automatic ROCm/AMD
default". `gfx1201` satisfies the architecture gate (`gfx12*`), so the only
other blocker is the Triton version: this stack ships **Triton 3.5.1**, below
the 3.7 the code requires — and forcing it on older Triton is documented to
hard-crash the INT8 path via a missing `libdevice.rint`.

Backends actually active here: `hip` and `eager`. `triton` and `cuda` are
disabled. The `hip` backend lacks `rotate_int8_convrot_weight`,
`dequantize_int8_convrot_weight`, `dequantize_int8_embedding` and
`dequantize_int8_simple`, so those fall through to `eager`.

This affects compute throughput, **not** load time, and is not the subject of
this page.

## Hardware and software

- AMD Radeon AI PRO R9700, 31.86 GiB VRAM, `gfx1201`
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads; 190,817 MB host RAM
- Ubuntu 24.04.4 LTS; ROCm 7.2.1 (HIP 7.2.53211)
- Python 3.12.3; `torch` 2.9.1+rocm7.2.1; `triton` 3.5.1+rocm7.2.1;
  `safetensors` 0.8.0; `comfy-aimdo` 0.4.13; `comfy-kitchen` 0.2.31
- ComfyUI v0.33.2 (`7cee3ceb`)
- Launcher: `--disable-dynamic-vram --disable-smart-memory --disable-mmap --bf16-vae --reserve-vram 2`

Loads are process-cold/model-cold but **not** disk-cold: the Linux page cache
and persistent compiled-kernel caches are preserved, per the benchmark
standard's cold-state definition.

## LTX 2.5 / R9700 — SELECTED CONFIGURATION (2026-08-18)

Closing this investigation. Final production config and the reasoning behind
each choice, plus the hypotheses that were tested and rejected along the way.

| Setting | Selected | Why |
|---|---|---|
| `--disable-mmap` | **on** | The catastrophic loader fix. mmap -> GPU transfer cost scales with tensor *count*, not bytes (~0.5 s/tensor, ~19x on large tensors vs ~822x on many small ones). See above. |
| Encoder | **INT8-ConvRot** | Same conditioning time as BF16 (22.56 vs 22.80 s, statistically a tie), faster load (1.97 vs 4.47 s), **~7.2 GiB less peak VRAM**, ~10 GiB less disk. |
| DiT | **INT8-ConvRot** | Profiler confirms it runs on `comfy_kitchen`'s native HIP/WMMA path, not an eager fallback. No broken quantized op found here. |
| Gemma token floor | **256** (was hardcoded 1024) | 1024->256: conditioning 22.51 -> 5.35 s, wall 43.78 -> 25.69 s (**-41%**). Sampling unchanged (10.65 vs 10.61 s), confirming the DiT receives equivalent conditioning. Two spot checks (111-token detailed prompt, 96-token motion prompt) both PASS on adherence and motion, no artifacts. 256 chosen over 128: 1024->256 saves 17.2 s, 256->128 saves only a further 0.75 s — not worth the risk of truncating longer prompts, since the floor is a *minimum*. |
| Negative-conditioning reuse | **automatic** | ComfyUI's own node cache already skips a `CLIPTextEncode` node when its inputs are unchanged; no workflow edit needed. Verified: 3 changed-positive runs with a fixed negative executed the negative once, not three times. |
| `tile_size` (VAEDecodeTiled) | **1280** (was 768) | Fixes the reported "snow" artifact — a tiling seam at exactly x=768 on 1024px-wide output, confirmed by column-band noise measurement. Validated snow-free on all subsequent renders including both spot checks. |

### Sampler profile

One profiled `SamplerCustomAdvanced` call (torch.profiler, CPU+CUDA activities).
**Do not add the percentages below** — `comfy_kitchen::int8_linear`'s 49.1%
Self CUDA time already contains the `gemm_wmma_kernel` entries beneath it in
the table; they are nested, not additional. The defensible summary:

> Quantized linear/GEMM execution is the dominant DiT cost —
> `comfy_kitchen::int8_linear` alone accounts for 49.1% of Self CUDA time, with
> native HIP WMMA kernels underneath it. Attention is a distant second at ~12%
> (`_flash_attention_forward` / `attn_fwd`). Normalization and RoPE together are
> under 10%.

No broken fallback was found demanding immediate repair. Triton 3.7 therefore
becomes a "does a better GEMM kernel beat an already-native WMMA one" research
question, not a fix for something broken — parked in the backlog behind the
broader H3/MiniMax Music survey.

### Rejected hypotheses (kept so they are not re-tested)

- **Dynamic VRAM as the load-time cause.** `--disable-dynamic-vram` alone: 931 s
  vs 938 s without — a 7-second difference on a 15-minute load. Not it.
- **BF16 encoder as the fix.** Chased before mmap was diagnosed; a 24.46 GiB
  BF16 encoder was still loading when aborted at 8m06s under the *mmap-enabled*
  config. Once mmap was fixed, BF16 and INT8-ConvRot conditioning came back
  statistically tied — format was never the variable, mmap was.
- **"GPU is idle during conditioning."** An early telemetry read sampled
  `rocm-smi --showuse` across the whole run rather than node-correlated, and
  wrongly concluded the encoder wasn't using the GPU. A per-node-correlated
  probe showed the opposite: 299-300 W sustained, 1% reported "GPU use" — see
  the telemetry warning below. Conditioning is real, power-capped GPU compute.
- **The 11-second warm-conditioning result meant "no staging cost."** It was
  actually one cached `CLIPTextEncode` node (the unchanged negative prompt)
  costing 0 s, not a genuinely fast 11 s pass. A 3-run cache-bust test (change
  positive only, then change both) isolated the real per-pass cost at ~11 s
  each, ~22 s for a fully novel pair — before the token-floor fix.

### Telemetry warning for this card

`rocm-smi --showuse` (GPU-use%) reported ~0-1% while the R9700 sustained
~300 W against its 300 W cap during real, node-confirmed compute (both LTX
conditioning and sampling reproduce this). **Do not infer idleness from
GPU-use% alone on gfx1201.** Correlate power, VRAM, and node-level timing
instead; the utilization counter is not trustworthy here. It read correctly
(100%) during H3 sampling on the same card, so the fault is situational, which
makes it more dangerous, not less.

## Method

Load timings are taken from ComfyUI's journal, between `Requested to load
<Model>` and the following `loaded completely` line. The transfer
micro-benchmark reads tensors from a real checkpoint with `safetensors.safe_open`
and times `.to(device)` for mmap-backed, cloned-to-RAM, and pinned-RAM copies of
the same tensors, synchronising the device around each pass. CPU occupancy was
sampled from `/proc/<pid>/stat` deltas; GPU state from
`rocm-smi --showpower --showuse --showgpuclocks`.

Raw rows: [`data/experimental/comfyui-load-cost.tsv`](../data/experimental/comfyui-load-cost.tsv)
