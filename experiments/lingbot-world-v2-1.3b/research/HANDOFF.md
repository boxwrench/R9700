# LingBot-World-V2 on R9700 — session handoff

Read this first on a cold start. Everything below is measured on this machine.

## Where things stand

Branch `experiment/lingbot-world-v2-1.3b`, head **`7ece228`** (pushed).

A persistent interactive world runner works and is the product. Two research
tracks behind it are closed and documented.

| doc | what it covers |
|---|---|
| `docs/lingbot-world-v2-1.3b-r9700-bringup-20260909.md` | native BF16 1.3B bring-up, ACCEPTED |
| `experiments/lingbot-world-v2-1.3b/INTERACTIVE.md` | the interactive runner + every optimization pass |
| `experiments/lingbot-world-v2-1.3b/FINDINGS.md` | the gfx1201 MIOpen Conv3d investigation |
| `experiments/lingbot-world-v2-14b-gguf/PHASE-A.md`, `PHASE-B.md` | 14B GGUF community path (closed) |
| `experiments/lingbot-world-v2-1.3b/LATENCY_LEDGER.md` | latency ledger — added from outside this session |
| `experiments/lingbot-world-v2-1.3b/WAN_OPTIMIZATION_CANDIDATES.md` | Wan-ecosystem candidate backlog — added from outside this session |

The last two arrived on the branch (`8bec022`, `a112838`) from another author
or session. I have not verified their claims against measurements here; treat
the tables in this file and in `INTERACTIVE.md` as the measured record and
those two as a planning overlay.

## Run it

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export HIP_VISIBLE_DEVICES=1
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1

/ai/envs/lingbot-world-v2/bin/python \
  /ai/github/R9700/experiments/lingbot-world-v2-1.3b/interactive_world.py \
  --image /ai/repos/lingbot-world-v2/examples/03/image.jpg \
  --prompt "A serene lakeside scene..." \
  --output /ai/outputs/lingbot-sessions/mysession
```

All three env vars are required. Add `--script "forward forward turn_right"`
to run non-interactively.

Controls: `w/s` forward/back, `a/d` strafe, `q/e` yaw, `r/f` up/down,
`t/g` pitch, `x` stay, `<action> <amount>`, `script ...`, `reset`, `stats`,
`quit`.

## Current performance (steady state, chunk 1, window 18+6, 464x832)

| | |
|---|---:|
| keypress → first visible frame | **2.81 s** |
| DiT (5 forwards) | 1.56 s (55%) |
| FP16 VAE decode | 1.25 s (44%) |
| setup + presentation | 1.2 ms (<1%) |
| frames per action | 4 |
| peak VRAM | 13.50 GiB |
| init, conditioning cached | ~35 s |
| init, first ever run for an image | ~296 s |

Journey: 18.4 s → 6.93 s (streaming + chunk 1) → 3.45 s (FP16 VAE) → 2.81 s
(presentation overhead). Milestone 2 is <2.5 s.

## Paths

| what | where |
|---|---|
| reporting repo | `/ai/github/R9700` (`/ai/repos` is a symlink to `/ai/github`) |
| upstream checkout | `/ai/github/lingbot-world-v2`, branch `rocm/r9700-1.3b-bringup`, head `7cf8109`, based on upstream `45fa4067` |
| venv | `/ai/envs/lingbot-world-v2` (ROCm 7.2.1 wheels, no CUDA torch, no flash-attn) |
| assembled checkpoint | `/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled` |
| sessions | `/ai/outputs/lingbot-sessions/` |
| conditioning cache | `~/.cache/lingbot-world-cond/` |
| 14B GGUF lab (closed) | `/ai/lab/lingbot-gguf-r9700`, models in `/ai/models/lingbot-gguf` |

Upstream patch is exported to
`experiments/lingbot-world-v2-1.3b/patches/lingbot-rocm-r9700.patch`.

## Things that will cost you hours if you rediscover them

**`LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib` is mandatory.** `/opt/rocm` on this
host lacks `libroctx64.so.4`; `import torch` fails without it.

**`HIP_VISIBLE_DEVICES=1` selects the R9700.** Device 0 is an RX 7900 XT
(gfx1100), device 2 an integrated gfx1036.

**Two production ComfyUI processes live on the R9700** (~2.1 GB). Check
`rocm-smi --showpids` before killing anything. Free VRAM for experiments is
~29.7 GiB, not 31.86.

**Never use a loose `pkill -f` pattern.** It has killed my own foreground
command and nearly killed the user's production ComfyUI. Kill by PID after
checking `ps`.

**MIOpen ships no gfx1201 database.** Every conv shape is searched into the
*user* FindDb at `~/.config/miopen/`. It persists across processes, so a truly
cold machine pays a one-time per-shape cost that later runs do not.

**The gfx1201 Conv3d cliff.** FP32 Wan VAE convs with output temporal extent
>= 4 lose `GemmFwdRest` (its workspace would exceed what MIOpen allocates) and
silently fall to `ConvDirectNaiveConvFwd` at 0.16 TFLOP/s vs 2.6. The fix is
`WAN_VAE_CONV3D_TEMPORAL_SPLIT` (default 1), which splits the output temporal
axis so every sub-conv stays eligible. `=0` restores exact upstream behaviour.
Whole-VAE effect: 110.6 s → 16.4 s.

**FP16 beats BF16 for this VAE**, which is counterintuitive since BF16 is used
everywhere else here. Decoder activations live in ~[-1,1], so FP16's narrower
exponent costs nothing and its four extra mantissa bits win: 3.77x vs 3.70x
speedup, and 3.3e-03 vs 2.6e-02 relative error.

**SDPA's `is_causal` aligns the mask top-left; FlashAttention aligns it
bottom-right.** They agree only when Lq == Lk. Getting this wrong gave max
error 3.46 — a wrong answer, not a rounding difference.

**Don't decode in a worker thread when the caller waits anyway.** Blocks freed
on another thread carry stream events that the main thread's next allocation
must reconcile — ~0.5 s per action, charged to whatever GPU call came first.

**Same-GPU DiT/VAE overlap is rejected on gfx1201.** Measured: DiT 4.17 s → 19.83 s
under contention, and a separate HIP stream is no better than a worker on the
default stream. The VAE saturates the device.

**Per-chunk MP4 encoding must not run during interaction** — it cost ~1.7 s of
the *next* action's DiT. Written at session end instead.

**Frame counts quantise twice.** `lat_f = (F-1)//4+1`, truncated to a multiple
of `chunk_size`, then `F = (lat_f-1)*4+1`. At `chunk_size=4` the reachable
counts are 13, 29, 45, 61 — asking for 21 silently gives 13.

**`--size 480*832` is an area target, not a shape.** With a 16:9 image it
executes at **464x832**. Upstream arithmetic, same on any GPU.

## Model facts worth not re-deriving

- The 1.3B HF repo ships **only the DiT** — no VAE, no T5, no tokenizer, no
  `config.json`. Those come from the 14B release. Upstream has no config for
  the 1.3B either; `wan/configs/wan_i2v_1_3B.py` was derived from safetensors
  headers. `num_heads=12` is the one value inferred rather than measured.
- `sample_shift = 10.0` is inherited from the 14B config and **is not verified
  for 1.3B**. Open variable.
- Actions are camera motion only. `control_dim = 6` (Plücker rays) and upstream
  sets `wasd_action = None`. **There is no attack/jump/interact input.**
- 5 DiT forwards per chunk: 4 denoising + 1 that re-encodes the accepted clean
  `x0` at t=0 into the KV cache. The fifth **cannot** be removed — forward 4's
  V differ from it by more than the reference magnitude (relative 1.131).
- Locality is enforced by *rolling the KV cache*, not by an attention mask.
  `local_attn_size=18`, `sink_size=6`. Sink frames live **inside** the window;
  they do not extend it. KV = `frame_seqlen * local_attn_size`.
- Window fills at 27144 tokens (18 x 1508) and then evicts. At `chunk_size=1`
  that takes 18 actions, so eviction regressions need >= 24.

## Regression

```sh
--script "forward forward turn_right forward turn_right turn_right forward \
          forward forward forward forward forward forward forward forward \
          forward forward forward forward forward forward forward forward forward"
```

Must hold: finite output, no NaN/Inf, no frozen frames (frame-to-frame L1 has
no zeros), boundary/interior L1 ratio ~1.0 (0.985 is the current value), KV
global advancing in `chunk * 1508` steps, eviction at exactly 27144, camera
pose continuous.

## What is explicitly closed — do not reopen without a reason

14B GGUF work; same-GPU DiT/VAE overlap; chunk-size sweeps; Vulkan;
quantization of the 1.3B; further VAE precision; broad parameter sweeps.

The fifth-forward removal is closed as **impossible**, not as unfinished.

## Open leads, in the order they currently look worth taking

1. **DiT is now the larger term (1.56 s, 55%).** Not yet profiled at
   `chunk_size=1`. Profile before touching anything.
2. **VAE decode 1.25 s.** Conv3d → causal Conv2d decomposition was queued and
   deferred when FP16 landed; gfx1201 may have better Conv2d solvers.
3. **VAE encode wastes ~21 s encoding zeros** in the batch pipeline
   (`concat([image, zeros(F-1)])`). Only matters for cold init now, since the
   conditioning horizon is cached.
4. **Second-GPU pipeline** — the RX 7900 XT is idle in this box. Same-GPU
   overlap failed for contention reasons that a second device would not have.
