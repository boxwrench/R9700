# Milestone: R9700 1.3B product stack (2026-09-10)

Frozen snapshot of the accepted configuration. For the live progression see
PERFORMANCE.md; for the chronology see FINDINGS.md.

## Stack

```text
GPU: AMD Radeon AI PRO R9700 / gfx1201, ROCm 7.2.1, torch 2.9.1+rocm7.2.1
upstream: /ai/repos/lingbot-world-v2 @ 7cf8109 (+ local fast-path plumbing)
model: lingbot-world-v2-1.3b-causal-fast-assembled (6 DiT shards + T5/VAE)
TAE: taew2_1.pth, SHA d26151e7…c469c797e (enforced at load)

sim 368×672 / latent 46×84 / 966 tok/frame
window 12 (6 sink + 5 recent + 1 current), chunk 1
3-step-A 999 → 908 → 768 + exact clean t=0 KV commit
TAEW2.1 FP16 decode-first, timecond memoization
async CPU Lanczos 2× present → 736×1344 (terminal, never fed back)
```

Launch: `./run_product_2x.sh` (product) / `./run_product.sh` (reference,
same world state, native output only).

## Measured saturated behavior (seed 42, matched 31-action runs)

```text
denoise            ~386 ms
first native RGB   ~391 ms
first 2× RGB       ~456 ms
clean-KV           ~127 ms
next-ready (P50)   ~524 ms   (P95 ~529 ms)
VRAM               ~6.05 GiB
upscale            9.0/10.7 ms per frame P50/P95 (async worker, max queue 1)
```

## Classification of accepted changes

- exact/state-preserving: persistent runner, chunk-1 streaming path,
  decode-first scheduling, time-conditioning memoization, inline decode /
  single-D2H / CPU archive plumbing
- presentation-only: FP16 canonical VAE path, TAEW2.1 renderer, async 2×
  Lanczos (native outputs byte-identical with/without)
- quality/state tradeoff: 3-step-A sampler (accepted on 50-action rollout +
  pose round-trips), window 18 → 12 (accepted on 90-action adversarial
  rollout with revisit horizons)

## Commits pinning this milestone

- `2dd8140` feat: presentation-only 2x upscale (async bicubic/lanczos)
- docs/chore commits in this session: product milestone docs, research
  index, `run_product_2x.sh` preset (see git log)
