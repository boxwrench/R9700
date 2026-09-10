# Interactive LingBot-World 1.3B session on R9700

A persistent single-process world runner: `interactive_world.py`.

## Launch

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export HIP_VISIBLE_DEVICES=1            # device 0 is an RX 7900 XT
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1  # the gfx1201 Conv3d workaround

/ai/envs/lingbot-world-v2/bin/python interactive_world.py \
  --image /ai/repos/lingbot-world-v2/examples/03/image.jpg \
  --prompt "A serene lakeside scene with a lone tree standing in calm water..." \
  --output /ai/outputs/lingbot-sessions/mysession
```

Add `--script "forward forward turn_right forward"` to run non-interactively.

## Controls

The causal-fast checkpoint's **only** conditioning is a 6-channel Plücker ray
embedding (`control_dim = 6` in `model_fast.py`), and upstream sets
`wasd_action = None` unconditionally. **The action space is camera motion. There
is no attack / jump / interact input in this model** — exposing one would mean
inventing an encoding the weights never saw.

| key | action | key | action |
|---|---|---|---|
| `w` | forward | `s` | back |
| `a` | strafe left | `d` | strafe right |
| `q` | turn left (yaw) | `e` | turn right (yaw) |
| `r` | up | `f` | down |
| `t` | tilt up (pitch) | `g` | tilt down |
| `x` | stay | | |

Also: `<action> <amount>` to override step size (metres, or degrees for turns),
`script w w e w`, `reset`, `stats`, `help`, `quit`.

Axes are OpenCV (+x right, +y **down**, +z forward), matching upstream's
`poses.npy`. A negative yaw about y swings the forward axis toward −x, which is
why `turn_left` is the negative rotation.

## Proof that world state persists

Each action advances one causal chunk of 3 latent frames = 4524 tokens
(`1508 tokens/frame`). From `actions.jsonl` of the test traversal:

| chunk | action | `kv_global_end` | `kv_local_end` | capacity | evicting |
|---:|---|---:|---:|---:|---|
| 0 | forward | 4524 | 4524 | 27144 | no |
| 1 | forward | 9048 | 9048 | 27144 | no |
| 2 | turn_right | 13572 | 13572 | 27144 | no |
| 3 | forward | 18096 | 18096 | 27144 | no |
| 4 | turn_right | 22620 | 22620 | 27144 | no |
| 5 | turn_right | 27144 | 27144 | 27144 | no |
| 6 | forward | **31668** | **27144** | 27144 | **yes** |

Three independent confirmations that this is one continuous world and not
seven separate image-to-video generations:

1. **`kv_global_end` advances monotonically** by exactly one chunk per action,
   and `current_start` is derived from it, so each chunk attends to every
   preceding chunk's keys and values.
2. **The window fills and then evicts exactly on schedule.** 18 latent frames ×
   1508 = 27144 tokens, reached at chunk 5. At chunk 6 `global_end` (31668)
   exceeds `local_end` (27144) — the cache is rolling, retaining the 6 sink
   frames. If state were rebuilt per action this could never happen.
3. **DiT time grows as the cache fills, then flattens**: 2.26 → 2.55 → 2.95 →
   3.35 → 3.75 → 4.17 → 4.18 s. Attention cost tracks occupancy up to capacity
   and is then constant. That curve is only possible with a shared cache.

The cross-attention cache is likewise computed on the first forward
(`cross_attn_first_call`) and reused for the rest of the session, and the VAE
decoder's causal feature cache is carried across chunks.

## Measurements

Test traversal `forward forward turn_right forward turn_right turn_right
forward`, 464×832, chunk 3, window 18+6, seed 42.

| | |
|---|---|
| Initialization (first ever run) | **295.9 s** — model 31.2 s, T5 3.5 s, conditioning encode 261.1 s |
| Initialization (conditioning cached) | **~35 s** |
| Action-to-chunk latency | **13.2 s** (first) then **16.7–18.4 s** |
| Frames per chunk | 9 (first chunk) then **12** |
| Effective generation fps | 0.65–0.72 |
| Peak VRAM | **18.93 GiB** |
| KV cache | 4.66 GiB (18 frames × 1508 tokens × 30 layers × 12 heads × 128 × 2 × 2 B) |

Per-action breakdown at steady state:

| Component | Time | Share |
|---|---:|---:|
| **VAE decode** | **14.18 s** | **77%** |
| DiT (5 forwards) | 4.18 s | 23% |
| Setup (poses, Plücker, noise) | ~0.01 s | <1% |

### Initialization: why the conditioning encode is cached

Upstream encodes `concat([image, zeros(F-1)])` for a known `F`. Interactively
the session length is open-ended, so the runner encodes a bounded horizon (90
latent frames by default) once. On gfx1201 that costs ~2.9 s per latent frame —
261 s — and it depends only on `(image, resolution, horizon)`, never on the
session. It is therefore cached to `~/.cache/lingbot-world-cond/`, making every
subsequent session on the same image start in ~35 s.

Tiling a converged tail latent instead was measured and **rejected**: the
zero-input latents decay but do not converge, still drifting 8.3e-03 at latent
frame 13. Caching is exact; tiling would not be.

## Latency bottleneck

**The FP32 Wan VAE decode, at 77% of action latency** — 14.18 s against the
DiT's 4.18 s. This is *after* the gfx1201 temporal-split workaround; without it
the same decode would be roughly 7× slower again.

Decode cost is ~1.18 s per output pixel frame and is flat across the session,
while DiT cost saturates at 4.18 s once the window fills. So the split is
structural, not a warm-up artefact:

- shrinking `--chunk_size` shortens both the chunk and the decode roughly
  proportionally, so it lowers latency per action but not per frame;
- anything that speeds the VAE decode is worth ~4× more, per second saved,
  than anything that speeds the transformer.

Two further observations, recorded but not acted on:

- The decoder runs frame-by-frame with a causal feature cache, so it is
  inherently serial in the temporal dimension — a natural target for
  pipelining decode against the next chunk's DiT rather than for raw kernel
  work.
- The DiT curve flattening at exactly the window boundary confirms attention
  over the KV cache is the growing term; at window 18 it costs ~1.9 s of the
  4.18 s.

## Session artifacts

```
session/
  metadata.json     resolution, latent grid, KV capacity, init timings, cache hit
  initial.png       the conditioning frame at working resolution
  chunk_000.mp4 ... one clip per action
  session.mp4       the whole traversal concatenated
  actions.jsonl     one record per action
```

Each `actions.jsonl` record carries timestamp, action, amount, chunk index,
total latency, DiT/VAE/setup split, frames, effective fps, latent frames done,
`kv_global_end`, `kv_local_end`, capacity, eviction flag, seed, camera
position, and allocated/peak/reserved VRAM.

## What was changed relative to the one-shot pipeline

Upstream's `_generate_causal_fast` allocates the caches, loops every chunk and
decodes once at the end. The per-chunk maths here is identical — same four
timesteps `[0, 250, 500, 750]`, same fifth KV-writing forward, same Plücker
conditioning, same scheduler. Three changes were required by interactivity, all
commented at their sites:

1. **Translation normalisation.** Upstream divides every translation by the
   largest in the sequence. Interactively that is impossible: "forward" would
   be scaled by moves the player has not made yet, and would change
   retroactively. Each step now emits a framewise relative whose translation is
   already unit-scaled — what upstream's normalisation produces for a
   constant-speed segment.
2. **Incremental VAE decode.** `Wan2_1_VAE.decode` calls `clear_cache()` on
   entry and exit. The decoder's 3D convolutions are causal and carry a
   per-conv feature cache across latent frames; dropping it between chunks
   would restart temporal context and put a seam at every chunk boundary.
   `IncrementalDecoder` keeps it.
3. **Bounded conditioning horizon**, encoded once and cached, as above.


---

# Responsiveness pass (from 18fdf92)

Objective: reduce keypress → first visible frame, preserving the persistent
process, KV/world state, cross-attention cache, causal VAE decoder cache and
camera continuity. All preserved; nothing about the model changed.

## Result

**Keypress → first visible frame: 18.4 s → 6.93 s (2.7x).**

Steady state, one action at a time, 464x832, window 18+6, seed 42:

| chunk | keypress→DiT done | **keypress→first frame** | keypress→full chunk | frames/action | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| **1 (default)** | **1.57 s** | **6.93 s** | **6.93 s** | 4 | 18.92 GiB |
| 2 | 2.45 s | 7.71 s | 12.52 s | 8 | 18.92 GiB |
| 3 | 4.17 s | 9.48 s | 19.12 s | 12 | 18.93 GiB |
| 3, before this pass | 4.18 s | *n/a — no frame until the end* | 18.4 s | 12 | 18.93 GiB |

Two independent changes contribute, and they are separable:

- **Streaming decode.** The decoder was already frame-serial — one pass per
  latent frame, yielding 1 pixel frame for the very first frame of a session
  and 4 thereafter. Turning `decode_chunk` into a generator and writing PNGs
  inside the loop makes the first frames visible without waiting for the rest.
  At chunk 3 this alone takes first-frame from 19.12 s to 9.48 s.
- **Smaller chunks.** chunk 1 is valid — KV advances 1508 tokens per action,
  the scheduler runs its usual four timesteps plus the KV-writing fifth
  forward, Plücker conditioning is built per chunk, and the VAE cache carries
  across. It is faster on *both* metrics, so there is no trade to make.

At chunk 1 first-frame and full-chunk coincide: one latent frame is one decode
group, so there is nothing left to stream within the action.

## Chunk size validity

All three sizes are valid. Verified per size: KV progression in exact
`chunk * 1508` steps, eviction at the right boundary, no NaN/Inf, finite range,
no frozen frames, and no chunk-boundary seam.

Eviction at chunk 1 needs 18 actions to reach the window, so the regression was
extended to 24. It fills at exactly 27144 tokens and then rolls, with DiT
flattening at 1.57 s and latency stable at 6.92-6.95 s.

## Decode/DiT overlap: implemented, measured, and turned off

Dependency analysis says overlap is safe — the DiT owns the KV and
cross-attention caches, the decoder owns the VAE feature cache, and the next
causal step never reads decoded RGB. So it was built: a decode worker on its
own HIP stream, handed the latent through a CUDA event with `record_stream` to
keep the allocator honest.

**It does not help on gfx1201. It hurts.** With the queue saturated:

| lane | DiT latency | full-chunk latency |
|---|---:|---:|
| one action at a time | 4.17 s | 19.12 s |
| worker thread, default stream (`--overlap 0 --queue_max 3`) | 20.28 s | 34.72 s |
| worker thread, separate stream (`--overlap 1 --queue_max 3`) | 19.83 s | 34.28 s |

The separate stream makes essentially no difference to the default-stream
worker, which is the tell: this is device contention, not a stream-ordering
problem. The FP32 VAE decode saturates the GPU, so the DiT does not run
*alongside* it, it runs *after* it while its own measured latency absorbs the
wait. Overlap is kept behind `--overlap 1` for future hardware but defaults to
off, and `--queue_max` defaults to 1 so each action is uncontended.

This also corrects a measurement trap worth recording: with a worker thread
running, `dit_seconds` stops being DiT compute and becomes DiT wall time
including queueing. The first action of each run was uncontended and always
showed the true cost (0.78 / 1.42 / 2.24 s for chunks 1/2/3).

## Input queueing

`--queue_max` bounds in-flight chunks; the default of 1 keeps each action
uncontended. Raising it to 2-4 lets actions be entered while frames are still
being presented, and input can never build up beyond that bound. Each
`actions.jsonl` record carries `t_entered`, `queue_wait`,
`dit_complete_latency`, `first_frame_latency` and `latency_seconds`, so the
full keypress → accepted → DiT done → first frame → chunk complete chain is
recoverable per action.

## Regression traversal

`forward forward turn_right forward turn_right turn_right forward`, re-run at
every chunk size:

- KV global/local positions correct at every step
- eviction begins at exactly 27144 tokens (18 latent frames x 1508)
- no NaN, no Inf, range [0, 1] fully spanned
- frame-to-frame L1 has no zeros — nothing frozen or repeated
- **no seam at chunk boundaries**: boundary/interior L1 ratio 0.985 at chunk 1
  (23 boundaries) and 1.091 at chunk 3 (7 boundaries), against 1.0 for a
  perfectly seamless join. Within normal frame-to-frame variation.
- camera pose continuous

Output is not bit-identical to 18fdf92 at chunk sizes other than 3, which is
expected: chunk size changes how many latent frames each scheduler pass covers.
At chunk 3 the DiT figures reproduce exactly (4.17 s vs 4.18 s).

## Remaining bottleneck

Still the FP32 VAE decode: **4.81 s of the 6.93 s action, 69%**, against the
DiT's 1.57 s. Decode is ~1.2 s per output pixel frame and flat across the
session; DiT saturates at 1.57 s once the window fills.

Because decode and DiT demonstrably do not overlap on this device, latency is
now simply their sum, and the VAE remains the term worth attacking. The
per-frame cost has not moved since the temporal-split workaround — what changed
here is when frames become visible, not how fast they are produced.
