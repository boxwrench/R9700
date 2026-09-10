# gfx1201 Conv3d investigation — running findings

Chronological. Superseded conclusions are kept, marked, and not deleted.

## F1. No MIOpen system database ships for gfx1201 (Experiment A)

`/opt/rocm-7.2.1/share/miopen/db` carries gfx1030, gfx803, gfx900, gfx906,
gfx908, gfx90a, gfx942 and gfx950. There is **no gfx1201 entry**, so every
convolution shape must be resolved into the *user* FindDb.

That user DB does exist and does persist across processes:
`~/.config/miopen/gfx1201_32.HIP.3_5_1_dabb6df2b9.ufdb.txt` (191 entries)
plus `~/.cache/miopen/3.5.1.dabb6df2b9/gfx1201_32.ukdb`. `MIOPEN_USER_DB_PATH`
is honoured. So caching is *not* broken.

## F2. The VAE decode cost is not search, JIT or cache misses (Experiment B)

Three decodes in one process: 110.76 / 110.74 / 110.74 s. Flat.

17 unique Conv3d shapes. The dominant one costs **the same on call 1 as on
call 54** (5.068 s vs 5.061 s). If search or compilation were responsible the
first call would be an outlier. It is not.

One shape accounts for ~82% of the decode:

| Conv3d input (pre-pad) | weight | calls/3 decodes | first | repeat | total |
|---|---|---:|---:|---:|---:|
| `[1,96,4,480,832]` | `[96,96,3,3,3]` | 54 | 5.068 s | 5.061 s | 273.3 s |
| `[1,96,4,480,832]` | `[3,96,3,3,3]` | 9 | 3.086 s | 3.096 s | 27.9 s |
| `[1,192,4,240,416]` | `[192,192,3,3,3]` | 54 | 0.307 s | 0.286 s | 15.4 s |
| `[1,384,2,120,208]` | `[384,384,3,3,3]` | 45 | 0.143 s | 0.136 s | 6.1 s |

## F3. `MIOPEN_FIND_MODE=FAST` changes nothing (Experiment C)

110.65 / 110.68 / 110.74 s, byte-identical output
(`sha256[:32] = 9a45019d4b5646636ed72d67377ac013` in both lanes), and the
per-shape table matches the default lane to three decimals. Consistent with
F2: there is no search to avoid.

## F4. Bypassing MIOpen runs out of memory (Experiment F)

`torch.backends.cudnn.enabled = False` on a clean process OOMs inside the
decoder: *"Tried to allocate 15.43 GiB"*. The native ATen fallback appears to
materialise an im2col buffer. Not usable as-is at 480x832, and therefore not
yet a valid negative control — it needs re-running at a reduced shape.

## F5. Terminology correction

The `transient_gib` column in `logs/expB-default.json` is the **peak PyTorch
allocation delta across the call**, not a demonstrated MIOpen workspace. No
MIOpen-reported workspace figure has been obtained yet. The earlier session
note of a "7.46 GiB workspace" was an allocator figure, not a solver figure.

## F6. Correction: the first microbenchmark used the pre-padding shape

`CausalConv3d` sets `self.padding = (0,0,0)` and applies `F.pad` itself, so
the shape my instrumentation logged (`[1,96,4,480,832]`) is what
`CausalConv3d.forward` receives, **not** what `F.conv3d` receives. The real
convolution sees the padded tensor.

Benchmarked at the logged (pre-pad) shape, the convolution is fast — 0.146 s,
2.70 TFLOP/s — which briefly suggested the cost lay outside the convolution.
That was wrong. At the true padded shape it reproduces exactly:

| input to `F.conv3d` | weight | output | warm median | effective |
|---|---|---|---:|---:|
| `[1,96,4,480,832]` | `[96,96,3,3,3]` | `[1,96,2,478,830]` | **0.146 s** | 2.70 TFLOP/s |
| `[1,96,6,482,834]` | `[96,96,3,3,3]` | `[1,96,4,480,832]` | **5.045 s** | 0.157 TFLOP/s |
| `[1,96,6,482,834]` | `[3,96,3,3,3]` | `[1,3,4,480,832]` | **3.168 s** | 0.005 TFLOP/s |
| `[1,96,4,480,832]` `padding=1` | `[96,96,3,3,3]` | `[1,96,4,480,832]` | **5.549 s** | 0.143 TFLOP/s |

The slow case does **2x** the output work of the fast case and takes **34x**
the time — a ~17x collapse in efficiency. Passing `padding=1` to `F.conv3d`
rather than pre-padding reproduces it too, so it is not specific to `F.pad`.

Neither autocast (fp32 or bf16), `no_grad`, real weights, nor going through
the real `CausalConv3d` module changes the fast-case number (all 0.146-0.148 s),
so pipeline context is ruled out.

**Reproducer** — 5 lines, ~5 s, no VAE, no model:

```python
import torch, torch.nn.functional as F
w = torch.randn(96, 96, 3, 3, 3, device='cuda')
x = torch.randn(1, 96, 6, 482, 834, device='cuda')
F.conv3d(x, w); torch.cuda.synchronize()   # ~5.0 s on gfx1201 / ROCm 7.2.1
```

## Open question

Whether the trigger is the padded *shape* (odd spatial extents 482x834, or
the temporal extent), the tensor *layout*, or the padding content. Sweep in
progress.

## F7. Root cause: MIOpen silently falls back to a naive kernel above a workspace threshold

MIOpen logging (`MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6`) plus the gfx1201
user FindDb give the whole story. The two problems differ only in input depth:

```
96-5-482-834-3x3x3-96-3-480-832-1-0x0x0-1x1x1-1x1x1-0-NCDHW-FP32-F
  = GemmFwdRest:219.499,12421693440,miopenConvolutionFwdAlgoGEMM;
    ConvDirectNaiveConvFwd:3539.7,0,miopenConvolutionFwdAlgoDirect

96-6-482-834-3x3x3-96-4-480-832-1-0x0x0-1x1x1-1x1x1-0-NCDHW-FP32-F
  = ConvDirectNaiveConvFwd:5072.33,0,miopenConvolutionFwdAlgoDirect
```

At out_T=3 MIOpen has two candidates and picks `GemmFwdRest` (im2col + GEMM,
219 ms) over `ConvDirectNaiveConvFwd` (3540 ms). At out_T=4 the GEMM entry is
**absent entirely** — its workspace would be ~15.5 GiB (it is 11.6 GiB at
out_T=3, and scales with output depth), which exceeds what MIOpen will
allocate, so the solver is dropped from the candidate list. Only the naive
direct kernel remains, and it is chosen with no warning, no fallback message
and no error.

That 15.5 GiB also explains F4: with `cudnn.enabled=False`, ATen's own
im2col fallback asked for 15.43 GiB and OOMed. Same buffer, different owner.

## F8. The trigger is the output temporal extent, and nothing else

Fresh contiguous input, `[1,96,T,482,834]` x `[96,96,3,3,3]`:

| in T | out T | warm median | effective |
|---:|---:|---:|---:|
| 3 | 1 | 0.096 s | 2.06 TFLOP/s |
| 4 | 2 | 0.169 s | 2.35 TFLOP/s |
| 5 | 3 | 0.508 s | 1.17 TFLOP/s |
| 6 | 4 | **5.036 s** | **0.16 TFLOP/s** |
| 8 | 6 | 7.590 s | 0.16 TFLOP/s |
| 12 | 10 | 12.621 s | 0.16 TFLOP/s |

Sharp cliff at out_T=4, then linear at the naive kernel's throughput.

Layout is **not** involved — at the slow shape, every provenance gives the
same time and the same ordinary contiguous strides
`(231545088, 2411928, 401988, 834, 1)`:

| case | contiguous | warm |
|---|---|---:|
| `F.pad` result | True | 5.052 s |
| fresh contiguous | True | 5.056 s |
| `padded.contiguous()` | True | 5.053 s |
| `padded.clone()` | True | 5.054 s |
| `channels_last_3d` | False | 5.156 s |

This is stop condition **B — shape trigger**. The layout hypothesis is dead.

Spatial extent is irrelevant too: sweeping H over 480-496 and W over 832-864
at in_T=6 stays at 0.14-0.16 TFLOP/s throughout.

## F9. Intervention: split the convolution along the output temporal axis

Each output frame depends only on its own receptive field, so slicing the
output temporal axis and concatenating is an exact restructuring of the same
convolution. It keeps every sub-convolution below the cliff, so MIOpen picks
`GemmFwdRest` for each.

Isolated, on the dominant shape (5.066 s unsplit):

| split | warm | speedup | max abs err | relative |
|---|---:|---:|---:|---:|
| `(2,2)` | 0.312 s | 16.2x | 7.55e-04 | 2.36e-06 |
| `(1,1,1,1)` | 0.316 s | 16.0x | 7.55e-04 | 2.36e-06 |
| `(3,1)` | 0.311 s | 16.3x | 7.55e-04 | 2.36e-06 |

Not bit-identical, because `GemmFwdRest` and `ConvDirectNaiveConvFwd`
accumulate in different orders. 2.4e-06 relative is FP32 round-off, and the
GEMM path is the better-conditioned of the two.

Implemented in `CausalConv3d.forward` behind
`WAN_VAE_CONV3D_TEMPORAL_SPLIT` (default 2, `0` disables). Handles stride and
dilation generally rather than assuming the 3-tap unit-stride case, because
one VAE `CausalConv3d` uses `stride=(2,1,1)`.

Whole VAE decode, 480x832, FP32, upstream chunking, same latent:

| lane | cold | warm | peak alloc | peak reserved | output |
|---|---:|---:|---:|---:|---|
| unsplit (upstream) | 111.61 s | **110.64 s** | 13.22 GiB | 19.44 GiB | mean -0.269880 std 0.069726 |
| split=2 | 49.38 s | **16.39 s** | 15.22 GiB | 24.09 GiB | mean -0.269880 std 0.069726 |

**6.75x faster warm.** Peak allocation rises because `GemmFwdRest` now
actually runs and needs its im2col workspace — the naive kernel needed none.
Smaller split values should trade some of that back; not yet measured.

---

# TAEW2.1 streaming presentation decoder (gfx1201, 2026-09-10)

## Upstream pin (verified, not assumed)

- repo `https://github.com/madebyollin/taehv`, commit
  `011dfc2112197741c540e0bdd5b7b67bcc930771` (fresh clone HEAD equals pin)
- weights `taew2_1.pth`, SHA-256
  `d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e`
  (recomputed locally; enforced at load, staged at `/ai/models/taehv-011dfc2/`)
- architecture from pinned source: name-selected `patch_size=1`,
  `latent_channels=16`; decoder temporal upscale 4 (TGrow pattern
  False,True,True); `frames_to_trim = 3`, consumed internally by
  `StreamingTAEHV.decode`; NTCHW in/out; identity latent mean/std
  (decoder input is `tanh(x/3)*3`, not a user-side normalization);
  output `clamp_(0,1)` RGB. Decode-only world-model streaming is the
  documented use case (`decode(latent)` returns the first frame
  immediately, `decode()` drains the rest, one frame per call minimum work).
- Decoder ops are Conv2d/Upsample/PixelShuffle/ReLU only — no Conv3d, so
  the gfx1201 MIOpen Conv3d cliff class does not apply. `taehv.py` imports
  cleanly in `/ai/envs/lingbot-world-v2`.

## Latent contract (traced in current code)

- Sampler emits model-space normalized Wan VAE `x0`, `[16,1,48,84]` fp32
  at 384x672 (measured, not assumed).
- Canonical decode applies `z = x0/scale[1] + scale[0]` (fp32) then fp16.
- TAE input is the **raw accepted x0** with only NCTHW -> NTCHW layout
  conversion, fp16. No canonical scale/shift (verified against the pinned
  Wan 2.1 TAE convention; a scale/shift trial was never needed here —
  identity confirmed first try by sane output).
- Feeding TAE from a random late latent was never done; all tests primed
  from stream start or contiguous history.

## Isolated saved-latent A/B (24-latent 384 stream, gfx1201)

| | canonical FP16 | StreamingTAEHV fp16 |
|---|---|---|
| cold first latent | 0.434 s | 4.77 s (one-time MIOpen autotune; FindDb persists) |
| warm steady per latent | 0.8158 s P50 | first RGB 3.6 ms / all-4 5.9 ms P50 |
| peak VRAM (standalone) | 3.71 GiB | 1.82 GiB |
| frames per latent | 4 (1 from stream head) | 1 from head latent (trim), 4 thereafter |

Warm saving: **~0.810 s/latent**. Artifacts: `ab_report.json`,
`x0_*.pt`, `canon_*.pt`, `tae_*.pt` under `/ai/outputs/lingbot-exp/tae/`.

## Quality (identical latents, gfx1201)

- MAD 0.0269 mean; frame means 0.6798 vs 0.6797 (no color/contrast shift);
  intra-chunk deltas 0.0269 vs 0.0279; chunk-boundary deltas 0.0310 vs
  0.0306 (no TAE seam pathology).
- Turn-vs-stay divergence per frame, uint8 L1: TAE
  [6.04, 11.65, 25.23, 28.05] vs canonical [5.38, 12.17, 24.29, 27.46] —
  same profile, first materially action-conditioned frame is index 2
  for **both** decoders. TAE preserves action-response structure.
- Visual spot-checks (turn, late-session, divergence frames): same scene
  and geometry, TAE slightly softer in foliage/cloud texture, marginally
  more saturated blue. Coherent and pleasant for navigation.

## Generation independence (measured)

TAE reads x0 read-only plus its own disjoint streaming memory. A second
session with TAE presentation (same seed/script) produced **bit-identical
accepted x0** (max diff 0.00e+00) with correct KV positions. Decoder is
presentation-only. Canonical conditioning encode, DiT, clean-KV,
cross-attention unchanged.

## Live integration (`--presentation_decoder taew2_1`, decode-first)

27-action regression at 384 through 9 KV evictions, filled-window steady:

| metric | canonical decode-first | TAE decode-first | saving |
|---|---|---|---|
| first RGB | 1.5217 (max 1.5284) | 0.6993 (max 0.7051) | **-822 ms** |
| next-action-ready | 1.6962 (max 1.7028) | 0.8773 (max 0.8831) | **-819 ms** |
| DiT (denoise/clean) | 0.6986 / 0.1718 | 0.6949 / 0.1707 | identical work |
| decoder all-output | 0.8241 | 0.0061 | -818 ms |
| peak VRAM | 10.26 GiB | 7.12 GiB | -3.1 GiB |

105/105 frames, 0 frozen, seam ratio 1.0316, finite, KV 27216/18144
evicting, identical camera trajectories ([7.22,0,17.19] both modes).
Action-conditioned RGB (frame idx 2, +125 ms @16fps): ~1.647 s -> ~0.824 s.

Ordering used: denoise -> TAE first frame -> display -> clean-KV commit
-> TAE drain -> next action. TAE all-4 is ~6 ms, so completing the drain
after the commit costs nothing measurable; no playback gap introduced.
Decoder selected at fresh-session start; no mid-session switching (TAE
and canonical maintain different causal decoder state). Canonical decode
never runs in TAE mode. `WAN_VAE_CONV3D_TEMPORAL_SPLIT=1` intact.

Accepted: TAEW2.1 retained as opt-in presentation decoder on gfx1201.

---

# 3-denoise sampler Candidate A (gfx1201, 2026-09-10): ACCEPTED as opt-in

## Phase 0 — exact baseline sampler semantics (logged from live objects)

- Scheduler: `FlowUniPCMultistepScheduler`, 1000-step grid,
  `set_timesteps(1000, shift=10.0)`.
- `timesteps_index=[0,250,500,750]` -> model timesteps **[999, 967, 908, 768]**,
  sigmas [0.9999, 0.967617, 0.908925, 0.768994].
- Per evaluation: `model(xt, t)` -> `x0 = xt - sigma_t * pred` (float64,
  nearest on-grid sigma); if another timestep remains,
  `xt = add_noise(x0, fresh_noise, next_t)` with sigma from exact grid
  lookup (`index_for_timestep`), `xt = alpha*x0 + sigma*noise`.
- The causal-fast path never calls `scheduler.step`: no hidden multistep
  history. Removing an evaluation removes only its model call + renoise draw.
- Initial noise: randn [16,1,48,84] fp32 from world RNG; 3 renoise draws
  per action. Clean pass input is `t = timesteps[-1]*0.0`, unchanged.

## Candidate A: grid [0,500,750] -> t [999,908,768] (drop 967)

RNG-controlled matched-state A/B from identical full-window state
(KV/metadata/cross-attn/pose/cond/init-latent/RNG/TAE priming); the
candidate reuses the baseline's own noise tensors at retained timesteps:

| | 4-step baseline | 3-step A |
|---|---|---|
| per-forward | 188/174/173/172 ms | 188/171/173 ms |
| denoise total | 0.707 s | 0.533 s (**-0.174 s**, one forward, no hidden behavior) |
| clean-KV | 0.175 s | 0.173 s (unchanged) |
| x0 diff (same noises) | — | MAD 0.0038, max 0.099 (0.5% of signal) |
| x0 diff (natural RNG) | — | MAD 0.103 (noise-draw dominated, expected) |

Quality (identical-state decodes): injected-noise candidate vs baseline —
canonical MAD 0.0012, TAE MAD 0.0014, means equal. Natural-RNG candidate —
MAD ~0.025 both decoders (same order as the TAE approximation itself).
Visuals coherent, same geometry/camera. Turn-vs-stay TAE profile:
candidate [7.0, 13.1, 19.5, 22.3] vs baseline [6.0, 11.7, 25.2, 28.1] —
**first conditioned frame stays index 2**; turn magnitude marginally softer.

## Persistent validation (50 actions, 32 evicting, exact pose round-trip)

Independent worlds, same seed/script (out-and-back: action 35 reproduces
action 3's pose exactly, action 39 reproduces the origin — verified
bit-close in both worlds), turns/reversals/stay/abrupt change/continued
travel, canonical + TAE presentation each:

| world | frozen | seam ratio | finite | KV | late visual |
|---|---|---|---|---|---|
| base canon | 0 | 1.007 | yes | 50400/18144 | stable |
| base TAE | 0 | 1.034 | yes | same | stable |
| cand canon | 0 | 1.084 | yes | same | stable |
| cand TAE | 0 | 1.080 | yes | same | stable, no drift at action 49 |

Return-point frames (same pose): same scene class, diverged content as
expected for independent sampler trajectories — no collapse in either.
Candidate B never tested (A passed; per plan, no sweep).

## Accepted latency (filled-window steady, 384, TAE decode-first)

| | 4-step | 3-step A |
|---|---|---|
| first RGB | 0.699 s (max 0.705) | **0.544 s** (max ~0.55) |
| conditioned RGB idx2 | ~0.824 s | **~0.669 s** |
| next-action-ready | 0.877 s (max 0.883) | **~0.759 s** |
| DiT denoise / clean | 0.695 / 0.171 | 0.521 / 0.171 |

Integration: `--denoise_schedule {4-step,3-step-A}` (default 4-step
reference); 3-step-A maps to grid [0,500,750] with the exact transition
rule and mandatory clean pass. TAE/384/clean-KV/canonical-fallback/SDPA
all unchanged. Target answered: action-conditioned 0.824 s -> ~0.67 s.
