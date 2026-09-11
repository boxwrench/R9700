# Architecture — LingBot World v2 1.3B interactive session

```text
controls (keys / script)
  ↓  6-channel Plücker camera conditioning — the only input the weights take
persistent DiT world simulator (one process, causal chunk = 1 latent frame)
  ↓  accepted latent x0 + mandatory exact clean t=0 KV commit
TAE or canonical renderer (display only; never feeds generation)
```

## Controls → camera conditioning

The causal-fast checkpoint's only conditioning channel is a 6-dim Plücker ray
embedding (`control_dim = 6`); upstream sets `wasd_action = None`. The action
space is therefore camera motion, listed in [QUICKSTART.md](QUICKSTART.md#4-controls).
Each action builds framewise relative poses (translation unit-scaled per step —
upstream's whole-sequence normalization is impossible interactively, and unit
scaling is what it produces for constant-speed segments), embeds them once per
action, and passes the same tensor to all four DiT forwards.

## Persistent world state (all allocated once, mutated in place)

- **DiT self-attention KV cache**, rolling window with sink. `sink_size=6`
  frames live *inside* `local_attn_size`: reference 18 = 6 sink + 11 recent +
  1 current (17,388 tokens); fast option 12 = 6 + 5 + 1 (11,592 tokens).
  Eviction starts exactly when the window fills (chunk 18 / 12); locality
  comes from rolling the cache, not from an attention mask.
- **Cross-attention cache**: text conditioning encoded once, reused all session.
- **VAE causal feature cache**: carried across chunks by `IncrementalDecoder`
  (upstream `decode` clears it on entry/exit, which would seam every chunk).
- **Camera pose** `(R, t)` integrated deterministically from the action stream.
- **RNG**: one generator, seed 42 by default; identical action scripts
  reproduce identical worlds.

## Geometry (measured, not assumed)

`--size` is an area target, not a shape. `384*672` on a wide image executes at
**368×672** actual pixels → **46×84** latent → **966 tokens/frame** (read back
from `metadata.json`: `resolution`, `latent`, `frame_seqlen`). Do not use the
obsolete 384×672 / 1008-token figures found in older notes.

## The four forwards per action

Reference sampler: 4 denoising evaluations + 1 clean commit. Fast sampler
(`3-step-A`): grid `[0,500,750]` → t `[999,908,768]`, then the same mandatory
clean pass that overwrites the chunk's noisy K/V with the accepted `x0` at
t=0. The clean pass is load-bearing — denoising forwards leave measurably
wrong cache content (V off by more than reference magnitude), so it cannot be
removed or approximated. Research history: `research/INTERACTIVE.md` § Phase B.

## Renderers

- **Canonical** (default): FP16 incremental VAE decode. FP16 was selected over
  FP32 (3.77× faster) and BF16 (8× less accurate here); decoder activations
  live in ~[-1,1], so FP16's mantissa wins. Research history:
  `research/INTERACTIVE.md` §§ VAE pass.
- **TAEW2.1** (opt-in `--presentation_decoder taew2_1`): pinned Tiny AE
  (`/ai/models/taehv-011dfc2`, weights SHA enforced at load). Streaming,
  display-only, first RGB in milliseconds; generation state provably
  untouched. Approximate RGB by design.

## Resolution pipeline (presentation upscale)

```text
DiT world simulation, 368×672
  ↓ accepted x0
TAEW2.1 → 368×672 RGB (native frames/)
  ↓ async CPU Lanczos (--upscale2, default off)
736×1344 presentation (frames_2x/, terminal output only)
```

The upscale adds no generated scene information: it improves presentation
sampling/sharpness only and never feeds back into VAE, DiT, KV cache, camera
conditioning, or world state (native PNGs are byte-identical with and without
it). Measured: ~9.0/10.7 ms per frame P50/P95, first upscaled frame ~65 ms
after native-ready, next-action-ready unchanged. PNG archival I/O is not
upscale compute time. Sync-inline mode exists (`--upscale2_sync`) but was
measured +31 ms on the loop and is not retained as a mode.

`--schedule decode-first` presents the accepted latent before the clean
commit (same final state, earlier first RGB); `clean-first` is the reference
order. Scheduling changes are bit-identical where proven — see
[PERFORMANCE.md](PERFORMANCE.md#exact-versus-quality-affecting).

## Session artifacts

`metadata.json` (resolution, latent grid, KV capacity, init timings, cache
hit), `initial.png`, `chunk_*.mp4` (written at session end, off the critical
path), `session.mp4`, `frames/`, `frames_2x/` + `upscale2_summary.json` (only
with `--upscale2`), `actions.jsonl` — one record per action with timings,
`kv_global_end`/`kv_local_end`, capacity, eviction flag, seed, camera
position, and VRAM. The latency metrics are defined in
[PERFORMANCE.md](PERFORMANCE.md#metrics).
