# LingBot World v2 1.3B on AMD ROCm

Persistent interactive world generation on one Radeon AI PRO R9700.

```text
368×672 simulation
736×1344 async presentation
~456 ms first 2× RGB
~524 ms closed-loop action cadence
~6.05 GiB peak VRAM
```

The world model simulates at 368×672 and asynchronously presents at 736×1344.
The 736×1344 frames are presentation-only resampling — never native model
generation, never fed back into world state.

## What this is

One process holds a persistent causal world: every keypress advances the same
DiT KV cache by one chunk (966 tokens/frame, 12-frame window with 6-frame
sink), commits the accepted latent with an exact clean forward, and presents
frames — first natively, then upscaled 2× on an async CPU worker that never
touches the model loop.

## Hardware / software tested

- AMD Radeon AI PRO R9700 (`gfx1201`, HIP device 1), ROCm 7.2.1,
  PyTorch `2.9.1+rocm7.2.1`, venv `/ai/envs/lingbot-world-v2`
- Upstream `/ai/repos/lingbot-world-v2` @ `7cf8109` (+ local fast-path
  plumbing, uncommitted — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md))
- Assembled 1.3B checkpoint + TAEW2.1 weights (SHA enforced at load);
  full provenance in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## Quickstart

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib HIP_VISIBLE_DEVICES=1
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1
./run_product_2x.sh        # validated product stack (2× Lanczos present)
# ./run_product.sh         # reference: same world state, native 368×672 only
# ./play.sh                # press-and-play with browser viewer (own defaults)
```

`IMAGE=`, `PROMPT=`, `OUT=` override the defaults; extra args append.
Details, controls, and troubleshooting: [QUICKSTART.md](QUICKSTART.md).

## Architecture overview

```text
keys → Plücker camera conditioning → persistent DiT (chunk = 1 latent frame)
  → accepted x0 → exact clean t=0 KV commit → TAEW2.1 → 368×672 RGB
  → async CPU Lanczos → 736×1344 presentation (terminal, never fed back)
```

Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Current measured latency (saturated, seed 42)

| metric | latency |
|---|---:|
| keypress → first native RGB | ~391 ms |
| keypress → first 2× RGB | ~456 ms |
| keypress → next-action-ready | ~524 ms |

Definitions and the full milestone progression: [PERFORMANCE.md](PERFORMANCE.md).
What each change preserved or traded:
[PERFORMANCE.md](PERFORMANCE.md#exact-versus-quality-affecting).

## Technical evidence

- [PERFORMANCE.md](PERFORMANCE.md) — stacks, milestone progression, rejected ideas
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — provenance, pins, benchmark recipe
- [research/FINDINGS.md](research/FINDINGS.md) — full experiment chronology
- [research/LATENCY_LEDGER.md](research/LATENCY_LEDGER.md) — interactive latency ledger
- [research/RESOLUTION_PATHS.md](research/RESOLUTION_PATHS.md) — why presentation
  upscale won over native higher-resolution DiT
- [research/rejected/](research/rejected/) — closed paths, so they stay closed
- [research/milestones/](research/milestones/) — frozen product snapshots
