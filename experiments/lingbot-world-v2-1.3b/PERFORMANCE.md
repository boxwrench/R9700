# Performance — LingBot World v2 1.3B on R9700

## Metrics

Three separate numbers, all per action at saturated window, seed 42:

| metric | definition (`actions.jsonl` field) | meaning |
|---|---|---|
| first RGB | `first_frame_latency` | keypress → first displayable frame (TAE: post-denoise, pre-clean-commit) |
| action-conditioned RGB | `latency_seconds` | keypress → full chunk presented after the clean commit |
| next-action-ready | `next_ready_latency` | keypress → session ready for the next action |

Report P50/P95 over saturated-window actions. Never compare across different
geometries or window occupancies without saying so.

## Stacks

Reference (runner defaults — conservative, upstream-faithful):

```text
--denoise_schedule 4-step   --presentation_decoder canonical
--schedule clean-first      --local_attn_size 18 --sink_size 6
```

Fast validated stack (all opt-in flags):

```sh
--size 384*672 --chunk_size 1             # actual 368x672, 966 tok/frame
--local_attn_size 12 --sink_size 6        # 6 sink + 5 recent + 1 current
--denoise_schedule 3-step-A               # 999 → 908 → 768 + clean t=0
--presentation_decoder taew2_1 --schedule decode-first
--timecond_cache                          # exact time-embedding memo
```

Measured fast budget (saturated P50): **first RGB 0.40 s, conditioned
0.53 s, next-ready 0.53 s**, peak VRAM ~6.0 GiB. Reference at the same
geometry: canonical first RGB ≈1.71 s / conditioned ≈1.84 s (INTERACTIVE.md
TAE section — older config, retained for scale, not a controlled A/B).

## Latency progression

One line per accepted milestone; metric and workload stated each time.

| era | first RGB | conditioned | next-ready | notes |
|---|---|---|---|---|
| persistent runner, chunk 3 | — | — | ~18.4 s full action | 464×832 actual; proof of continuous world (KV rolls, evicts on schedule) |
| chunk 1 + streaming present | 6.93 s | 6.93 s | — | chunk 1 valid; first==full (nothing left to stream) |
| FP16 canonical VAE | 3.45 s | — | — | 3.77× decode, rel err 3.3e-03, no drift accumulation |
| Phase A overhead removal | 2.81 s display-ready | — | — | allocator stall, 2nd D2H, PNG/MP4 off path; bit-identical output |
| reduced geometry + TAE + decode-first | 0.699 s | ~0.824 s | 0.877 s | 384×672 target, window 18, 4-step |
| 3-step-A sampler | 0.544 s | ~0.669 s | 0.759 s | drop t=967; 50-action rollout + pose round-trip accepted |
| window 18 → 12 | 0.426 s | 0.571 s | 0.572 s | 6 fewer history frames; 90-action adversarial rollout, revisit L1 equal-or-better |
| time-conditioning memo | **0.395 s** | **0.530 s** | **0.531 s** | exact; −45 ms whole-action; +0.1 GiB |

VRAM along the way: 18.9 → 13.5 (FP16) → 7.1 (TAE) → 5.9 (window 12) →
6.1 GiB (timecond cache).

## Exact versus quality-affecting

**Exact / state-preserving** (bit-identical where stated, never merely "validated"):

- decode-first scheduling (final state bit-identical; only presentation order moves)
- time-conditioning memoization (e/e0/x0/KV/PNGs all bit-identical, 20 actions)
- inline decode, single D2H copy, CPU PNG archive, session-end MP4 (all proven bit-identical)
- conditioning-horizon cache (exact by construction; tiling rejected as approximate)

**Interaction behavior** (changes what runs, not render math): latest-key-wins
input coalescing, key-repeat suppression.

**Presentation-only** (generation state untouched, displayed RGB approximate):
TAEW2.1 streaming decoder.

**Quality/state tradeoff** (different outputs by design, accepted on evidence):

- 3-step-A — different diffusion trajectory (x0 MAD 0.0038 same-noise);
  accepted after matched A/B + 50-action rollout with exact pose round-trips
- window 12 — 6 fewer full-resolution history frames (11→5 recent); accepted
  after 90-action adversarial rollout with <6 / 6–12 / >12-frame revisit
  horizons and sequential warm canonical-decode checks

## Rejected with evidence

| idea | result |
|---|---|
| camera-conditioning hoist | redundant ~33 ms/action, ceiling ~25 ms, but A/B/A next-ready 541.4 / **548.1** / 538.1 ms (**~8 ms slower**, +365 MB) — bit-identical and correct, rejected as a system-level regression |
| fifth-forward removal | impossible: clean K/V differ from denoise-4 K/V by rel 1.13 (V); the commit is load-bearing |
| partial fifth-forward prefix | only ~3–4% of one forward theoretically skippable; rejected as too small/risky |
| same-GPU DiT/VAE overlap | DiT 4.17 s → 19.83 s under contention; device saturated, not a stream-ordering problem |
| zero-latent tiling for conditioning | tail latents still drifting 8.3e-03 at frame 13; caching is exact, tiling is not |
| BF16 VAE | fast but 8× less accurate than FP16 here |
| cold single-latent canonical decode as evaluator | collapses to near-black without causal cache; invalid probe (use sequential warm decode) |
| chunk 2/3, resolution upscaling games, 14B GGUF path, Vulkan, 1.3B quantization, broad sweeps | closed; see research index |

Next candidate already queued by prior analysis: clean-pass KV-write-only
structural forward (skips everything downstream of the cache store on the
discarded-output clean pass). Not started.
