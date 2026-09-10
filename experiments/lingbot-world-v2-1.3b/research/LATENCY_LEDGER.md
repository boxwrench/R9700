# LingBot World 1.3B — Interactive Latency Ledger

This document is the persistent record of what controls interactive latency, what has already been tested, what worked, what failed, and what remains.

The product metric is:

> **keypress → first genuinely new visible frame**

Everything else is secondary unless it improves that number without breaking world continuity.

Current accepted configuration:

- Radeon AI PRO R9700 / gfx1201
- LingBot World v2 1.3B causal-fast
- 464×832 actual output geometry
- chunk_size=1
- local_attn_size=18
- sink_size=6
- persistent KV cache
- persistent cross-attention cache
- persistent causal VAE decoder cache
- BF16 DiT
- FP16 VAE decode
- FP32 conditioning encode
- SDPA attention compatibility path
- `WAN_VAE_CONV3D_TEMPORAL_SPLIT=1`

Current steady-state latency:

| Component | Time | Share |
|---|---:|---:|
| DiT, 5 forwards | ~1.56 s | ~55% |
| FP16 VAE decode | ~1.25 s | ~44% |
| setup/presentation | ~1 ms | <1% |
| **keypress → first visible** | **~2.81 s** | |

Peak VRAM is ~13.50 GiB.

---

## 1. Resolution

Resolution is one of the largest global levers because image area determines how many spatial latent tokens exist per frame.

Increasing resolution increases work in both major stages:

- DiT has more tokens for attention, projections, MLPs, RoPE, KV traffic, and elementwise work.
- VAE has more pixels/features for convolution and upsampling.
- KV memory also grows because every retained latent frame contains more spatial tokens.

This is therefore a multiplicative lever rather than a single-kernel tweak.

Current accepted actual geometry is 464×832.

Potential future test: one carefully selected lower native resolution plus upscale, only if model/kernel work cannot reach the latency target first.

Status: **available but deferred**.

---

## 2. Causal chunk size

Chunk size controls how many latent frames are generated per player action.

Larger chunks can amortize some overhead, but they increase the time before the user sees the result of an action.

Measured:

| chunk_size | DiT done | first visible | full chunk | visible frames/action |
|---:|---:|---:|---:|---:|
| **1** | **1.57 s** | **6.93 s before later optimizations** | **6.93 s** | 4 |
| 2 | 2.45 s | 7.71 s | 12.52 s | 8 |
| 3 | 4.17 s | 9.48 s | 19.12 s | 12 |

All three sizes were valid. KV progression, scheduler semantics, Plücker conditioning, causal decoder state, and temporal continuity remained correct.

`chunk_size=1` is now the default because it is faster on both first-visible and full-action latency.

Status: **closed; chunk_size=1 selected**.

---

## 3. World-memory / local attention window

The rolling KV cache is the model's recent world memory.

The configured local window affects:

- steady-state self-attention cost,
- KV memory footprint,
- how much recent visual history the model can consult,
- world consistency during longer exploration and revisitation.

Measured behavior confirms the cost grows with occupancy. DiT time rises while the cache fills and then becomes flat once the 18-frame window is full.

At chunk_size=1, the global KV position advances by 1508 tokens/action. Capacity is:

`18 latent frames × 1508 tokens = 27144 tokens`

Eviction begins exactly at that boundary while sink frames remain retained.

Reducing the window may lower steady-state attention latency, but it risks worse world persistence. This is a real quality/latency trade rather than a free optimization.

Status: **important future lever; do not change without revisitation/continuity testing**.

---

## 4. Sink size

Sink frames are preserved early context inside the local attention window.

Important implementation detail: sink frames are **inside** `local_attn_size`; they do not add extra temporal KV extent.

The community 14B pack incorrectly estimated KV as `(local + sink)`. The actual upstream allocation is based on `frame_seqlen * local_attn_size`.

Sink size affects what survives eviction more than total steady-state KV compute.

Status: **correct semantics established; not currently a latency target**.

---

## 5. Number of DiT forwards per action

Each causal action currently performs:

- 4 denoising forwards at noisy timesteps,
- 1 additional forward on accepted clean `x0` at t=0 to write correct K/V into the persistent cache.

This is a major algorithmic multiplier: every extra forward is nearly another complete transformer execution.

The fifth forward was investigated directly.

Measured cache difference between the fourth denoising forward and the clean t=0 forward:

- K relative difference: ~0.170
- V relative difference: ~1.131

The fourth forward therefore cannot be reused. It represents noisy latent state, while future chunks need clean-world K/V.

A bounded-prefix replacement also does not help: layer L's K/V require hidden state `h_L`, which depends on all preceding layers. Only the tail after the final layer's K/V formation is avoidable, estimated at roughly 3–4% of one forward and not worth the correctness/maintenance risk.

Status: **fifth forward required; simple 5→4 reuse rejected**.

---

## 6. DiT self-attention

Self-attention cost depends on both current query tokens and retained KV context.

For this model, attention gets more expensive as the rolling context fills. That was observed directly in the action-time curve before saturation.

Potential levers:

- better AMD-compatible attention kernel,
- fewer unnecessary temporaries,
- improved memory layout,
- reduced synchronization,
- shorter context window if quality permits,
- lower-precision KV if validated,
- sparse/local implementations if mathematically identical to the required semantics.

Any replacement must preserve:

- causal alignment,
- asymmetric Q/K lengths,
- local rolling window,
- sink retention,
- correct cache writes.

Current attention compatibility uses PyTorch SDPA because the original FlashAttention path is not usable as-published on this ROCm setup.

Status: **profiling target once ranked against MLP/projections**.

---

## 7. Q/K/V projections and attention output projections

Each transformer layer performs dense projections around attention.

Even with an efficient attention kernel, these matrix multiplications can consume a large fraction of transformer time.

Potential levers:

- fused QKV projection,
- improved GEMM/kernel selection,
- fewer layout conversions,
- fused output projection/residual paths,
- reduced intermediate writes to VRAM.

Status: **not yet ranked in the steady-state DiT profile**.

---

## 8. MLP / FFN work

Feed-forward blocks often consume as much or more arithmetic than attention in transformer layers.

Potential levers:

- fused activation/gating/projection paths,
- improved GEMM kernels,
- reducing intermediate materialization,
- precision changes only if validated,
- shape-specific kernel selection where broadly safe.

Do not assume attention is the dominant transformer cost. The next DiT profile must establish the actual breakdown.

Status: **not yet ranked**.

---

## 9. Layer-to-layer movement and VRAM traffic

On a single GPU, "communication between layers" is mostly not network communication. It is:

- reading previous activations from VRAM,
- writing new activations,
- temporary tensor materialization,
- cache reads/writes,
- transposes/layout changes,
- synchronization between kernels.

A mathematically cheap operation can still be slow if it repeatedly moves large tensors through memory.

Potential levers:

- fusion,
- in-place or reuse-safe operations,
- persistent/reused buffers,
- avoiding unnecessary contiguous copies,
- avoiding repeated casting/layout transforms,
- keeping producer/consumer operations close enough to improve cache locality.

Status: **inspect after DiT operator profile**.

---

## 10. Kernel launch and dispatch overhead

The model executes many GPU kernels per forward.

At multi-second latency, launch overhead is unlikely to be the largest term, but it becomes increasingly important as heavy kernels are optimized.

Potential levers:

- kernel fusion,
- graph capture/compiled execution where compatible with persistent mutable caches,
- reducing Python-level dispatch,
- eliminating unnecessary synchronization between small kernels.

Status: **lower priority until profiles show many small kernels matter**.

---

## 11. Precision

Precision is one of the largest proven levers in this project.

### VAE decoder

Original decoder: FP32.

Measured isolated decoder:

| dtype | decode wall | speedup vs FP32 | relative error |
|---|---:|---:|---:|
| FP32 | 4.681 s | 1.00× | reference |
| BF16 | 1.264 s | 3.70× | 2.57e-02 |
| **FP16** | **1.242 s** | **3.77×** | **3.26e-03** |

FP16 was both faster and roughly 8× more accurate than BF16 for this decoder.

Reason: decoder activations remain roughly within [-1,1], so BF16's large exponent range provides no benefit while FP16's additional mantissa precision matters.

End-to-end FP16 VAE decode changed:

- VAE: 4.81 s → ~1.25–1.33 s
- first-visible: 6.93 s → 3.45 s before later overhead removal
- peak VRAM: 18.92 GiB → ~13.5 GiB

Across 93 frames, FP16 vs FP32 mean absolute pixel difference was ~0.00020 and remained flat across the session, showing no accumulating causal drift.

FP32 conditioning encode remains unchanged.

Status: **major accepted optimization; FP16 decode selected**.

### DiT

DiT already runs BF16.

Any additional precision changes should be selective and justified by profile data and correctness testing.

---

## 12. VAE decode

The VAE converts latent video into visible RGB frames.

This was originally the largest steady-state bottleneck.

Original FP32 decode at chunk_size=1: ~4.81 s/action.

Profile showed CausalConv3d was 94.4% of decoder time. The three largest shape families were approximately:

- 96 channels at 464×832,
- 192 channels at 232×416,
- 384 channels at 116×208.

The earlier gfx1201 temporal split fixed a pathological MIOpen solver cliff. After that fix, remaining FP32 Conv3d operations were already on the healthy `GemmFwdRest` path at roughly 2.6 TFLOP/s; the remaining problem was simply FP32 arithmetic.

FP16 decode reduced this to ~1.25 s.

Potential remaining lever: causal Conv3d → sum of temporal Conv2d contributions, if measured to be faster. This is now deferred because VAE is no longer the dominant term and the likely end-to-end gain is smaller.

Status: **optimized substantially; still ~44% of current latency**.

---

## 13. Causal Conv3d solver behavior

A major gfx1201-specific issue was discovered earlier.

For certain temporal extents, MIOpen stopped offering the efficient GEMM/im2col solver because workspace requirements exceeded practical allocation limits and silently fell to a naive direct Conv3d path.

The dominant bad shape produced roughly 5-second calls.

The accepted fix is exact temporal output decomposition with:

`WAN_VAE_CONV3D_TEMPORAL_SPLIT=1`

This keeps each temporal subproblem on a healthy solver path and preserves causal semantics.

This fix is required both for native 1.3B and the community 14B VAE path on gfx1201.

Status: **accepted structural workaround**.

---

## 14. VAE encode / initialization

VAE encode is primarily an initialization concern, not steady-state action latency.

The interactive implementation needs a conditioning horizon derived from the starting image. The first ever horizon encode was ~261 s and dominated initial startup.

The result depends only on:

- initial image,
- working resolution,
- configured horizon.

It is therefore cached exactly. Subsequent matching sessions start in roughly ~35 s instead of ~296 s.

A proposal to tile a supposedly converged tail latent was tested and rejected because the latent tail continued to drift; caching preserves exact behavior.

Status: **steady-state issue solved by exact caching; startup still has room for future product work**.

---

## 15. Prefill / startup work

"Prefill" here refers to one-time or initialization work rather than the LLM-specific token prefill concept.

Relevant startup work includes:

- model load,
- prompt/T5 encode,
- cross-attention conditioning setup,
- starting-image conditioning encode,
- initial cache construction.

These should never be confused with per-action latency.

T5 and conditioning work are already cached/released appropriately so they do not dominate steady-state interaction.

Status: **separate startup metric; not current playability bottleneck**.

---

## 16. Persistent KV cache

Persistent KV is what makes this a continuous world rather than repeated independent image-to-video generation.

Evidence of persistence:

- global position advances monotonically,
- local cache fills and then rolls,
- sink frames remain retained,
- DiT time tracks occupancy and flattens at capacity.

Potential latency levers:

- context length,
- cache dtype,
- cache layout,
- cache read/write efficiency,
- attention implementation.

Any optimization must preserve world-memory behavior.

Status: **core product state; optimize carefully**.

---

## 17. Cross-attention cache

Cross-attention conditioning is computed once and reused across actions.

Recomputing it every action would be unnecessary work.

Current implementation keeps the cache persistent.

Status: **already optimized structurally**.

---

## 18. Action conditioning / Plücker construction

The checkpoint's supported control space is camera motion encoded by 6-channel Plücker rays.

Supported controls include forward/back, strafe, yaw, vertical movement, pitch, and stay.

Plücker and pose construction itself is sub-millisecond and is not a meaningful compute target.

An apparent ~500 ms setup cost was traced elsewhere to allocator synchronization, not Plücker math.

Status: **not a latency target**.

---

## 19. GPU synchronization

Synchronization can turn asynchronous GPU execution into large hidden wall-time stalls.

A major example was found after introducing a presentation worker. The first GPU operation of every new action appeared to cost roughly 477–576 ms, even when that operation was a 64-byte H2D copy or `torch.zeros(1)`.

Investigation ruled out:

- device wake-up,
- Plücker computation,
- unfinished GPU work.

The cause was cross-thread caching-allocator reconciliation: memory freed by the decode worker carried stream events that the main thread's next allocation had to process.

Running decode inline for the uncontended `queue_max=1` case removed the stall.

Result:

- overhead ~560 ms → ~1.2 ms
- first-visible 3.45 s → ~2.81 s

Status: **major hidden synchronization issue fixed**.

---

## 20. Allocator behavior and buffer ownership

Repeated allocation/free cycles and cross-thread/stream ownership can create synchronization even when GPU kernels themselves are fast.

Known lesson from this project:

> if an inexplicably tiny first GPU operation takes hundreds of milliseconds, test whether it is absorbing allocator/event reconciliation from previous asynchronous work.

Potential future levers:

- persistent buffers,
- fewer allocations,
- single-owner hot-path memory,
- careful stream ownership,
- avoiding needless producer/consumer thread boundaries.

Status: **known class of latency bug; monitor continuously**.

---

## 21. Same-GPU overlap

Because next-chunk DiT does not require decoded RGB, DiT generation and VAE presentation are logically independent enough to pipeline.

This was implemented using a worker, separate HIP stream, event handoff, and `record_stream`.

It was slower.

Measured at chunk_size=3:

| lane | DiT latency | full chunk |
|---|---:|---:|
| serial | 4.17 s | 19.12 s |
| worker/default stream | 20.28 s | 34.72 s |
| worker/separate stream | 19.83 s | 34.28 s |

Separate-stream behavior was essentially identical to default-stream contention, showing device saturation rather than stream-ordering failure.

Status: **rejected on single R9700; do not retry unless workloads/hardware change materially**.

---

## 22. Multi-GPU stage separation

A second GPU changes the overlap problem because DiT and VAE would no longer compete for the same execution resources.

Potential architecture:

- GPU A: DiT + persistent KV/cross-attention state
- GPU B: persistent VAE decoder + presentation
- transfer only latent output between devices

Potential benefit: true pipeline overlap of DiT chunk N+1 with VAE decode N.

Costs to measure first:

- latent transfer size,
- PCIe topology,
- peer-to-peer availability,
- host-staged transfer latency if needed,
- synchronization,
- decoder residency and feature-cache ownership.

Status: **future architectural lever**.

---

## 23. Device↔host copies

These were suspected to be significant but measured as negligible in the current path.

Before cleanup:

- redundant second D2H copy: ~1.3 ms
- first D2H + conversion: sub-millisecond class

The duplicate was removed anyway because every unnecessary copy should stay removed.

Status: **cleaned up; not currently material**.

---

## 24. PNG / MP4 / filesystem work

Presentation and archival work must not be confused with model latency.

Measured PNG encode/write for four frames: ~92 ms.

PNG archival was moved to a CPU thread.

Per-chunk MP4 encoding during interaction was much worse: ffmpeg feeding caused roughly ~1.7 s of extra latency in the next DiT action due to CPU/resource contention.

Per-chunk MP4s are now produced at session end, off the interaction path.

Status: **critical-path archival overhead removed**.

---

## 25. Compiler/backend/solver maturity

Backend selection can outweigh model-level arithmetic changes.

The gfx1201 Conv3d episode is the primary example: mathematically identical work changed by many multiples depending on which MIOpen solver was selected.

For every dominant operator, distinguish:

- inherent arithmetic cost,
- memory bandwidth limit,
- poor solver/kernel selection,
- workspace constraint,
- synchronization artifact.

Do not optimize model math until backend pathology is ruled out.

Status: **standing rule for all future profiling**.

---

## 26. Thermals, clocks, and GPU power state

Sustained clocks and thermals can alter measurements.

Device wake-up was explicitly tested during the allocator investigation and was not responsible for the ~500 ms stall; first GPU work after 0.02–2.0 s idle remained under ~1 ms in that test.

Still, long-run measurements should record enough repetition to detect throttling or clock variance before attributing small changes to code.

Status: **measurement hygiene**.

---

# Optimization history

This table is the cumulative record of major wins and rejected paths.

| Change | Before | After | Result |
|---|---:|---:|---|
| gfx1201 temporal Conv3d split | ~110 s-class VAE paths in affected workloads | ~16 s-class one-shot VAE / usable incremental decode | **accepted structural fix** |
| persistent interactive runner | independent clip workflow | continuous KV/world session | **product milestone** |
| chunk_size 3 → 1 | ~18.4 s full action | ~6.93 s first-visible | **accepted** |
| streaming decode | full-chunk wait | earlier first frame at larger chunks | **accepted** |
| same-GPU DiT/VAE overlap | serial | much slower under contention | **rejected** |
| VAE FP32 → FP16 | 4.81 s decode | ~1.25–1.33 s | **accepted, ~3.6×** |
| VAE BF16 | FP32 baseline | fast but much less accurate than FP16 | **rejected in favor of FP16** |
| remove cross-thread allocator stall | ~560 ms overhead | ~1 ms | **accepted** |
| remove duplicate D2H | duplicate copy | one copy | **accepted cleanup** |
| async PNG archive | PNG on critical path | CPU archival | **accepted** |
| per-action MP4 encode | caused ~1.7 s next-action penalty | session-end encode | **accepted** |
| reuse 4th denoise K/V as clean cache | proposed 5→4 forwards | K/V materially wrong | **rejected** |
| partial fifth-forward prefix | proposed | only ~3–4% of one forward theoretically avoidable | **rejected as too small/risky** |

Current progression of first-visible latency:

`~18.4 s → 6.93 s → 3.45 s → 2.81 s`

The system is now approximately:

`1.56 s DiT + 1.25 s VAE + ~0 s overhead`

---

# What remains most important

The hot path is now balanced rather than dominated by one component.

## Immediate

1. Profile the steady-state DiT at chunk_size=1 and full 18-frame context.
2. Rank attention, projections, MLP, normalization, RoPE, KV operations, copies, and synchronization.
3. In parallel, keep one bounded VAE structural candidate available: causal Conv3d → temporal Conv2d decomposition.
4. Choose the easiest measured intervention capable of saving at least ~0.3 s/action.

## Next threshold

Milestone 2:

> **<2.5 s first-visible**

That requires only about 0.3 s from the current path.

## Playability target

A useful product target is approximately:

- 2–3 s: interactive demo / command-like
- 1.5–2 s: clearly interactive
- ~1 s: plausibly playable
- ~0.5–0.7 s: starts to feel game-like
- <0.3 s: genuinely responsive controls

The engineering goal should therefore continue beyond 2.5 s toward roughly 1 s/action.

A plausible long-term budget would be approximately:

- DiT: 0.5–0.7 s
- VAE: 0.3–0.5 s
- other: <0.05 s
- total: ~0.9–1.2 s

---

# Standing optimization rules

1. **Measure the end-to-end interactive path.** A microbenchmark win is not sufficient.
2. **Optimize the largest remaining term first unless a smaller term has a much easier near-free win.**
3. **Remove work before making work faster.**
4. **Cache invariant work exactly whenever possible.**
5. **Keep archival, logging, and presentation encoding off the critical path.**
6. **Treat unexplained tiny-operation latency as possible synchronization/allocator debt.**
7. **Do not assume the backend selected a good kernel. Verify dominant shapes.**
8. **Prefer simple structural wins over elaborate custom kernels.**
9. **Validate persistent causal state, not just individual-frame similarity.**
10. **Keep every measurable fraction of a second if it is clean and maintainable. Small wins accumulate.**
11. **Do not spend time on 14B, broad sweeps, or unrelated optimization while the 1.3B interactive path is the product target.**
12. **Every accepted optimization must preserve world continuity, KV behavior, and usable output quality.**

---

# Standard regression

Use the existing deterministic traversal after meaningful changes, extended long enough at chunk_size=1 to fill and roll the 18-frame KV window.

Validate:

- finite output / no NaN or Inf,
- no frozen frames,
- correct frame count and dimensions,
- global KV position advances correctly,
- local KV capacity remains correct,
- eviction starts at 27144 tokens,
- sink frames remain retained,
- cross-attention cache remains persistent,
- VAE causal feature cache remains persistent,
- camera pose remains continuous,
- no obvious boundary seams,
- no accumulating causal drift.

The final test is always the working interactive session: **did keypress → genuinely new visible world state get faster?**
