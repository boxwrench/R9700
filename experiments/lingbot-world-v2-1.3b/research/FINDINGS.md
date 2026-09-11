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

## 2026-09-10 — viewer session quit after 5 frames (FIFO sticky-EOF)

TASK: diagnose "went 5 frames and quit" (2 actions, then session exited).
HYPOTHESIS: viewer opened+closed the key FIFO per keypress; after the 2nd
key's writer closed, the runner's next input() saw EOF, and Python stdin
holds EOF sticky -> EOFError -> session quit path.
CONTROL: test_viewer.py delivery (single key) passed; code read confirmed
os.close(fd) in send_key finally-block + bare EOFError->break in runner.
CHANGE: view_session.py holds ONE persistent FIFO writer (retry-once on
EPIPE); interactive_world.py treats EOF on non-tty stdin as transient
(sleep+continue), keeps Ctrl-D-quit on real ttys. Added
test_key_burst_no_eof (3 sequential keys, reader held open).
RESULT: test_viewer.py 8/8 OK; py_compile OK. fix untested live (GPU left
free for user relaunch; user's next session is the e2e proof).
DECISION: SELECTED (pending live confirm). Relaunch with the same command;
frames 1-5 from the quit session are intact in OUT.
NEXT: user relaunches, presses 3+ keys; if it survives past 5 frames, mark
CONFIRMED.

## 2026-09-10 — play.sh press-and-play launcher

TASK: single command to start viewer + session (user: retyping is dumb).
CHANGE: new play.sh — boots viewer :8734 in background (reuses if up),
then execs the forge session (TAE/decode-first/3-step-A/384x672) in the
foreground with the forge-awakening image+prompt baked in as defaults.
Uses <> for the key fifo so boot prints progress immediately instead of
blocking for the first keypress. Env overrides: IMAGE/PROMPT/OUT/PORT/
KEYFIFO. run_product.sh untouched.
RESULT: live-tested to world-load start (viewer 200, checkpoint load
began, boot logs immediate). Fixed IMAGE typo (awakening not wakening).
DECISION: SELECTED. User runs ./play.sh, opens the page, presses keys.

## 2026-09-10 — window-12 candidate (local_attn_size 18→12, sink 6)

QUESTION: can LingBot keep a useful persistent world with 12 full-res
historical frame-equivalents instead of 18, moving action-conditioned
response ~0.67 s toward ~0.6 s?
SEMANTICS (verified code + live): sink is INSIDE local_attn_size. Actual
geometry is 368x672 / latent 46x84 / 966 tokens-per-frame (brief assumed
1008; corrected live). Baseline 18 = 6 sink + 11 rolling + 1 current =
17388 KV tokens; candidate 12 = 6 sink + 5 rolling + 1 current = 11592.
Eviction flips exactly at chunk 18 (base) / 12 (cand). Live Q=966,
attended K == allocated valid K (17388 / 11592). BF16, 30 layers x 12
heads x 128. Backend: no flash_attn on gfx1201 -> torch SDPA fallback,
observed only, untouched.
LATENCY (matched 30x stay, same image/prompt/seed/schedule/TAE; P50/P95
saturated): denoise 519.6/521.5 -> 421.4/424.7 ms (-98); clean-KV
168.8/171.2 -> 139.2/140.8 (-30); first RGB 524.0/525.8 -> 425.7/428.9
(-98); conditioned RGB 699.0/700.8 -> 570.7/574.4 (-128); next-ready
700.1/702.2 -> 571.8/575.8 (-128). Peak VRAM 6.93 -> 5.94 GiB; KV cache
2.99 -> 1.99 GiB. Equal-K controls identical (K=2898: 303 vs 307 ms;
K=8694: 373 vs 375 ms): latency scales with actual K, not capacity.
Self-attention/forward saturated: 82 -> 51.5 ms (ratio 0.628 vs K ratio
0.667); 4 forwards save ~122 of the 128 ms. No fix attempted per brief.
ROLLOUT (90 actions, matched script, adv. legs + returns at gaps <6 /
6-12 / >12): poses identical; all 180 finite. Revisit L1 base vs cand:
<6: 15.66 vs 14.39; 6-12: 17.98 vs 16.28; >12: 17.50 vs 16.94 (cand
equal-or-better everywhere). Sharpness end 576.9 vs 773.0; move energy >
stay energy both runs. Canonical sequential decode of 7 saved x0:
bit-identical through x0_011, then bounded divergence (L1 9-16, within
own revisit variance). Cold-cache single-latent canonical decode
collapses (dark) — documented non-method, superseded by sequential.
DECISION: RETAIN 12-frame mode as supported option (--local_attn_size 12).
Default unchanged (18); no production change without update gate + approval.
Artifacts: /ai/outputs/lingbot-window12/ (runs, probes, latents, canonical,
harness/attn_probe.py). No repo source files modified for this experiment.

## 2026-09-10 — exact-transformer reassessment (window-12 config)

Groundwork for next-experiment choice. Shape-exact microbench + live hooks:
fp32 time MLP is ~12.5 ms/forward in-path (embed ~2.2 + projection ~10.3;
t fixed at [999,908,768,0] every action) = ~50 ms/action, largest exactly
removable cost. fp32 GEMM path is ~50x slower than bf16 on identical shapes
(m3 fp32 20.2 ms cold vs bf16 ~0.4 ms). Everything else measured dead for
exact purposes: RoPE 0.06 ms, evict clone 0.04-0.09 ms, text-embed 0.14 ms,
TunableOp/GEMM headroom ~1-2 ms (FFN already ~70 TFLOPS effective in bf16).
torch.compile smoke works on gfx1201 (2.6 s first, single-GEMM warm nil).
Artifacts: harness/op_breakdown.py, harness/timeprobe.py, timeprobe/ run.

## 2026-09-10 — timecond_cache: RETAIN (opt-in --timecond_cache)

Dims reconciled live: dim=1536, freq_dim=256, heads=12, layers=30,
time_embedding 256->1536->1536, time_projection 1536->9216, e0 [1,966,9216]
FP32. Earlier 2048-dim microbench estimates discarded; live hooks rule.
Remeasured time-conditioning cost in-path: embed ~2.2 + projection ~10.3 ms
per forward; t fixed at [999,908,768,0] every action.
Implementation: precompute (e, e0) once with existing modules under identical
autocast; serve via per-forward module swap keyed (timestep, seq_len); miss
raises. No upstream edits, no dtype change, no rewritten math.
Bit-identity (20-action off vs on, window 12): e/e0 cache-vs-live recompute
equal all 4; x0 hashes equal all 20; KV equal all 30 layers k+v; 77/77 PNGs
byte-identical; actions.jsonl differs only in timing fields.
Warmed saving (30 stays, saturated P50): next-ready 575.9 -> 530.5 ms
(-45.4); conditioned RGB -45.3; first RGB -33.6; clean -12.2. Peak VRAM
5.94 -> 6.05 GiB (+cache). P95 tight. Retain threshold >=20 ms: PASS.
RETAIN as opt-in flag; default unchanged. Next per roadmap: clean-pass
KV-write-only structural. Artifacts: outputs/lingbot-window12/tc_*,
harness/tc_validate.py + timeprobe.py.

## 2026-09-10 — camcond hoist: REJECTED, code reverted

Phase 0 (live, 966-token, window 12): model cam prep ~0.21 ms/forward;
per-block injector->shift ~8.1 ms/forward (event-timed, 30 blocks); same
plucker object+values across all 4 forwards; per-block cam_scale/shift
bitwise equal across t=999/908/768/0 (56 forwards) — dependency proof held.
Predicted ceiling ~25 ms/action.
Implementation (--camcond_cache): per-action precompute with original
modules under matching autocast + const-serve swaps. Bit-identity FULLY
held: served-vs-live recompute 20/20, x0 20/20, KV 30 layers, 77/77 PNGs,
trajectory identical.
A/B/A sandwich (24 stays, saturated P50 next-ready): off1 541.4 / on 548.1
/ off2 538.1 ms => ON +8 ms (P95 tight, off anchors reproducible). The
+365 MB persistent const footprint perturbs execution beyond the removed
GEMM work on this box. Verdict REJECT (<20 ms bar, negative saving).
Implementation fully reverted; code path unchanged. Profile + artifacts:
outputs/lingbot-window12/cam_{probe,probe_proof,time,off,on,aba_*},
harness/cam_{probe,time,validate}.py. Next: clean-pass KV-write-only.

## 2026-09-10 — regional torch.compile Phase 0: STOP before implementing

Fresh kineto trace (2 saturated actions, window12+timecond): 9,281 kernel
launches/action (3.2-4.4 us API each), 478 .item()/.tolist() syncs/action
(2-4 per block-forward: rope tolist, frame_seqlen item, eviction items,
cross is_init), avg drain 625 us = GPU stays fed (lockstep eager:
CPU submits ~1 ms, sync drains; almost no CPU/GPU overlap as a result).
Per-forward composition (live SDPA + shape-exact bench, dim1536/ffn8960):
SDPA 51.5, FFN 17.3, QKV/O 7.5, cam 7.5, cross 8.9, norms/elem 5.6 ms.
Dynamo would fragment at every data-dependent sync (~100+ graphs/forward,
tiny fragments, ~0 fusion benefit, 30-60 min startup, K-shape recompiles
during fill). Only clean single-graph region is FFN (shape-stable always):
ceiling ~4-5 ms/action < 15 ms bar. Clean-pass final tail (head+unpatchify)
~0.03 ms, not implementable usefully. VERDICT: close compile path.
Structural note (not a proposal): the ~478 syncs/action enforce lockstep
(~100 ms non-overlap); removing them is upstream cache-logic surgery,
out of scope. Artifacts: harness/compile_phase0.py, compile_phase0.log.

## 2026-09-10 — static host-sync removal: REJECTED, code reverted

Phase 0 exact census (instrumented wrappers on current local code, window-12 saturated):
rolled action = 704 torch syncs (fill 644), fully reconciled: per block-forward 2x rope
tolist + KV-index items per branch (fwd0: 141+142 in all 30 blocks, evict iff full; fwd1-3:
141-short-circuit or else-branch), plus 4 timecond-key items, 4 unpatchify tolists,
2 scheduler items, 4 post-sync bookkeeping items per action. Profiler's 478 ≈ GPU-side
items only (460) + explicit syncs/presentation; CPU tolists (244) are kineto-invisible.

Key correction to the experiment premise: rope `grid_sizes.tolist()` (240/action) is on a
CPU tensor (~1 us, never a drain). `frame_seqlen` and cross `is_init` are already zero on
the hot path. Removable static exposure totaled ~0.35 ms/action (~0.1 ms real GPU drain).
The ~2.6 ms x120 serialized drains are the dynamic KV-index reads (stage 2, not attempted).

Candidate (rope_pygrid + unpatchify-pygrid + CPU timecond keys): 704 -> 456 syncs, hot path
left with dynamic KV items only. Bit-identity FULLY held (x0 20/20, KV 30+30 layers,
cross 30+30+init, indices/trajectory/evict-12, 121/121 PNGs byte-identical across 3 runs).
A/B/A sandwich rolled P50 next-ready: off1 527.4 / on 529.3 / off2 527.6 ms (P95 flat).
Verdict REJECT (<5 ms bar): drains serialize critical-path GPU work; removing non-draining
syncs changes nothing. Both trees reverted and verified byte-identical to pre-experiment
state. Scratch harnesses kept in /tmp only (sync_census.py, identity.py).

## 2026-09-10 — Python KV cursor (dynamic sync removal): REJECTED, code reverted

Phase 0A (metric boundaries, saturated evicting action): setup 0.5 ms, denoise 386.2 ms
(3 forwards), TAE-first 3.4 ms, first RGB on host at 390.3 ms (BEFORE clean), clean-KV
126.6 ms, TAE-drain 2.1 ms, all-host 520.9 ms, next-ready 524.6 ms. Earlier "~526 ms
first RGB" was latency_seconds (post-clean); true first-visible is ~390 ms. Decode-first
does present before clean; clean is what makes next-ready ~525 ms.

Phase 0B (layer cursor equivalence, 22 actions x120 block-forwards = 2640): all 30 layers
identical on (gb, lb, cap, cs, ce, new, ga, la) for every (action, forward). Zero
divergence. fwd0 advances/evicts; fwd1-3 + clean overwrite same slots, indices unchanged.
Shared model-level cursor authorized (not assumed).

Phase 0C (pure-Python shadow): 2640/2640 agreement on (local_end, new_global_end),
10/10 evict-branch predictions match (acts 12-21 fwd0, evicted=966, rolled=4830).

Candidate (--python_kv_cursor, since reverted): one plan/forward, all 450 dynamic KV
.items gone (rolled 704 -> 254; only CPU tolists, 4 timecond-key items, scheduler(2) and
post-sync bookkeeping(4) remain). Device scalars kept as fill_ mirrors, never read back;
per-action mirror asserts passed throughout. Bit-identity FULLY held over a 51-action
matched session crossing 39 evictions: x0 51/51, KV k/v 30/30, cross k/v/init 30/30,
indices/trajectory/RNG progression equal; 121/121 TAE PNGs byte-identical across the
A/B/A runs.

A/B/A sandwich (31 scripted actions, saturated rolled P50): denoise 388.3/389.0/388.8,
clean 127.9/127.6/128.2, first-frame 392.5/393.5/393.0, next-ready 525.6/528.8/527.6 ms;
P95 flat everywhere; VRAM 6.05 GiB all lanes. Delta ≈ 0, within variance.

Verdict REJECT (<10 ms bar): dynamic host syncs also serialize GPU work already on the
critical path. Both trees reverted and verified byte-identical to pre-experiment state.
Sync-removal work is CLOSED per the experiment gate: no further cursor rewrites. The
compile-revisit pointer (sync topology changed nothing measurable) is moot.

## 2026-09-10 — presentation-only 2x upscale: RETAIN async (commit 2dd8140)

Question: 368x672 TAE RGB -> ~720p present with no world-model loop cost.
Implementation (opt-in --upscale2 bicubic/lanczos, default off; --upscale2_sync
for inline mode): CPU PIL resampling of host RGB only; async mode upscales on
the archive thread, sync mode inline on the present path. Upscaled frames are
terminal output (frames_2x/), never fed back. Scoped commit 2dd8140
(interactive_world.py only, 142+/2-).

Timing, 31-action matched runs, saturated rolled P50 (P95):

- OFF: denoise 386.5, clean 126.6, first-frame 390.7, next-ready 524.8 (531.4)
- bicubic async: per-frame upscale 7.0/8.9 ms; next-ready 524.5 (529.1)
- lanczos async: per-frame upscale 9.0/10.7 ms; next-ready 524.3 (528.5)
- bicubic sync: per-frame 7.6/8.7 ms inline; next-ready 555.6 (+31 ms) REJECT

Async worker always caught up (max archive queue 1, all 121 frames); first
upscaled frame ~65 ms after native-ready, well before next action. VRAM
identical 6.05 GiB all lanes (CPU resize, zero GPU contention).

State: native PNG sets byte-identical across OFF/bicA/lanA/bicS (same hash as
all prior runs); KV indices, eviction, trajectory, latent-frame counts equal.
Quality (identical source frames, video + contact sheets inspected):
nearest-blocky -> smooth bicubic -> marginally crisper lanczos; no ringing or
shimmer observed; frame-to-frame deltas match native (0.021-0.022), i.e. no
added temporal instability. No "generated detail" claimed.

Verdict RETAIN async only (both methods; default off). Sync-inline mode
rejected (+31 ms critical path = 4x inline upscale, as predicted).
Artifacts: /tmp/sbs_2x.mp4, /tmp/sbs_contact.png, /tmp/crop_detail.png,
upscale2_summary.json per run dir (scratch, regenerable with one command).
