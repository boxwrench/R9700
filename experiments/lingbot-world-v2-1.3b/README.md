# LingBot World v2 1.3B — interactive persistent world on R9700

Interactive, persistent, causal world generation with the 1.3B LingBot World v2
model on a single AMD Radeon AI PRO R9700 (`gfx1201`, ROCm). One process holds
the whole world: every keypress advances the same DiT KV cache by one chunk,
and frames render in a browser window as they are generated.

Current validated budget (fast stack, saturated window, seed 42):

| metric | latency |
|---|---:|
| keypress → first RGB | **~0.40 s** |
| keypress → action-conditioned RGB | **~0.53 s** |
| keypress → next-action-ready | **~0.53 s** |

Definitions of the three metrics: [PERFORMANCE.md](PERFORMANCE.md#metrics).

## Start here

- [QUICKSTART.md](QUICKSTART.md) — environment → models → launch → controls → what to expect
- Fastest validated launch: [`./play.sh`](play.sh) (starts viewer + session; open http://localhost:8734/)
- Reference launch (upstream-faithful defaults): [`./run_product.sh`](run_product.sh) without the fast flags — see [PERFORMANCE.md](PERFORMANCE.md#stacks) for both flag sets

## How it works (one paragraph)

Movement keys become camera motions, encoded as 6-channel Plücker ray
embeddings — the only conditioning this checkpoint accepts. Each action runs
3 denoising DiT forwards plus one mandatory exact clean forward that commits
the accepted latent into a rolling KV cache (6 sink + 5 recent + 1 current
frame at the fast setting), so the next action continues the same world.
Decoded RGB comes from the pinned TAEW2.1 presentation decoder or the
canonical FP16 VAE. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

Active experiment on branch `experiment/lingbot-world-v2-1.3b`, isolated from
the production ComfyUI work on `main`. Runner defaults are the conservative
reference (4-step sampler, canonical renderer, clean-first order, 18-frame
window); every faster behavior is an opt-in flag — see
[PERFORMANCE.md](PERFORMANCE.md#exact-versus-quality-affecting) for which
changes are bit-exact and which trade quality.

## Deeper records

- [QUICKSTART.md](QUICKSTART.md) — setup and controls
- [ARCHITECTURE.md](ARCHITECTURE.md) — world state, caches, renderers
- [PERFORMANCE.md](PERFORMANCE.md) — stacks, latency progression, rejected ideas
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — hardware, software, provenance, benchmark recipe
- [research/](research/) — full experiment history, raw logs, harnesses, negative results
