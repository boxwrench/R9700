# Qwen3.8-27B on one R9700 — running experiment log

A chronological index of the local-inference experiments run against
Qwen3.8-27B on a single AMD Radeon AI PRO R9700. Each entry states what was
asked, what was measured, and what survived scrutiny. Detailed records are
linked where they exist; entries without a separate document carry their
numbers here.

Retractions are kept in place rather than deleted. Two conclusions in this
campaign were withdrawn after further controls, and the reasons are recorded
because they are the most instructive part of the sequence.

## Standing configuration

Unless an entry says otherwise: Qwen3.8-27B UD-Q4_K_XL, llama.cpp Vulkan/RADV
backend at commit `ad1de39`, device `Vulkan1` (R9700, gfx1201) with the RX 7900
XT excluded, full offload, `--parallel 1`, six threads, flash attention on, f16
K and V, unified KV, `ctx-size 163840`, `ubatch-size 512`, MTP speculative
decoding with `n-max 2` and `p-min 0.3`.

Device isolation is mandatory on this host. Both Vulkan and HIP enumerate the
7900 XT first, and omitting the visibility filter sends work to the wrong GPU.

## Index

| # | Experiment | Result | Record |
|---|---|---|---|
| 1 | Q4 vs Q6 quantization | Prefill identical, decode −23.8% at Q6 | [quant comparison](qwen3-8-27b-quant-comparison.md) |
| 2 | Router VRAM overcommit | Production bug found and fixed | [quant comparison](qwen3-8-27b-quant-comparison.md) |
| 3 | `reasoning_effort` serving mode | `low` is faster in tokens/s, not in answers | [quant comparison](qwen3-8-27b-quant-comparison.md) |
| 4 | MTP head quantization | Structurally identical heads, no acceptance gain | [quant comparison](qwen3-8-27b-quant-comparison.md) |
| 5 | Vulkan vs ROCm/HIP | Vulkan +41.6% prefill, −11.6% decode | [parameter sweep](qwen3-8-27b-parameter-sweep.md) |
| 6 | 29-configuration parameter sweep | No configuration beat production | [parameter sweep](qwen3-8-27b-parameter-sweep.md) |
| 7 | KV-cache precision | Closed as a throughput lever; large memory lever | this document |
| 8 | Speculative-vs-serial equivalence | MTP changes greedy output; n-gram does not | this document |
| 9 | Divergence localization | Traced to `n_rs_seq` / GDN snapshot path | this document |
| 10 | `n_rs_seq` sufficiency | Snapshot configuration alone is sufficient | this document |
| 11 | K=1 vs K=3 verifier agreement | Neutral; branch closed for performance | this document |
| 12 | Qwen3.6 vs Qwen3.8 native MTP | No proposer gap; hypothesis rejected | this document |
| 13 | MTP round-cost decomposition | Verification is 89% of round cost | this document |
| 14 | 1-token vs 3-token kernel profile | FFN matmul multi-column path is the lever | this document |
| 15 | IQ4_XS tiny-column path A/B | No existing alternate wins; dequant mechanism proposed, later refuted by 16 | this document |
| 16 | IQ4_XS tiny-N dequant-reuse and occupancy | Both variants fail the gate; kernel is DRAM-bound | this document |
| 17 | Tiny-N MUL_MAT + ADD fusion | Fusion restored and correct, but no throughput gain | this document |

## 1–6. Earlier entries

Covered in full by the two linked documents. The load-bearing results:

- Q4 and Q6 prefill identically (compute-bound); Q6 decodes 23.8% slower
  (bandwidth-bound). Bandwidth ratio alone predicts 0.691 against 0.762
  measured; the gap is attributed to MTP amortization but was not isolated.
- The router evicts by model **count**, not VRAM. At `models-max 4` two 27B
  models stayed resident, weights spilled to host RAM, and decode collapsed to
  3.85 tokens/s. Fixed with `--models-max 1`. This was a production fault, not
  a benchmark artifact.
- Both quantizations ship structurally identical MTP heads (424,699,392
  parameters) quantized differently. Q6's higher-precision head bought no
  measurable acceptance; the 1.2-point gap is within binomial standard error.
- The parameter sweep is a negative result and that is its value: production
  sits on a measured local optimum. Only `spec-draft-n-max` showed a real
  effect. Acceptance moved opposite to throughput in both stages that moved it.

## 7. KV-cache precision

Question: is target KV-cache precision limiting decode on the R9700?

Three configurations, everything else frozen, greedy generation with
`ignore_eos` so every arm emitted exactly 256 tokens, five measured repetitions
after a discarded warmup.

| KV | decode (MTP) | Δ | prefill | Δ | VRAM | Δ | accept | mean acc len | p0 | p1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f16 | **53.82** ±0.27 | — | 721.3 | — | 27.73 GiB | — | 0.750 | 2.493 | 0.857 | 0.637 |
| q8_0 | 52.59 ±0.32 | −2.3% | 700.9 | −2.8% | 23.04 GiB | −4.69 | 0.749 | 2.480 | 0.858 | 0.622 |
| q4_0 | 52.44 ±0.22 | −2.6% | 689.2 | −4.4% | 20.54 GiB | −7.19 | 0.749 | 2.487 | 0.861 | 0.627 |

MTP-off reference decode: 29.41 / 29.24 / 29.21. MTP is worth **1.83×**, and
that multiplier is unaffected by KV precision.

**KV precision is closed as a throughput lever.** Neither quantized cache
improves decode. The reason is structural: at these prompt lengths the KV cache
is a small share of per-token memory traffic next to 17.9 GB of weights, so
shrinking it saves little bandwidth while adding dequantization work on every
attention access. The penalty is about four times larger with MTP on (−2.3%)
than off (−0.6%), consistent with MTP attending over the cache for n+1
positions per committed token.

**Memory is the real result.** q4_0 frees 7.19 GiB — 21% of the card — for 2.6%
decode. q8_0 and q4_0 are statistically tied on speed while q4_0 saves 2.50 GiB
more, so q4_0 is the better trade of the two if headroom is ever wanted.
Context is already at the model maximum and two 27B copies do not fit
regardless, so the realistic use is a small second model resident alongside
this one, plus headroom for the vision path.

Acceptance, mean accepted length, and positional survival are flat to three
digits across all three. KV precision does not touch the proposer.

## 8. Speculative-vs-serial equivalence

A correctness gate in the KV experiment showed greedy output differing between
MTP-on and MTP-off. Investigating it consumed the rest of the campaign.

The acceptance rule is not at fault. `common_sampler_sample_and_accept_n`
(`common/sampling.cpp:678`) commits the target model's own sampled token at
every position and uses the draft token only to decide whether to continue.
Greedy equivalence cannot be violated there.

**Retraction.** An initial three-arm control appeared to show MTP and n-gram
speculation producing identical output that differed from serial decode,
implying a generic speculative-batching cause. That run warmed up on the *same*
prompt as the measurement, so the prompt cache restored recurrent state and
altered every arm — including the reference. Under a correct unrelated-prompt
warmup the result reverses.

| warmup | serial | n-gram depth 2 | MTP n-max 2 |
|---|---|---|---|
| same-prompt (cache reuse) | `efae1ff4` | `6f067801` | `6f067801` |
| unrelated (fresh) | `553a7d8a` | `553a7d8a` ✓ identical | `6f067801` ✗ diverges @ token 7 |

All arms are individually reproducible across independent server launches, so
the comparisons are sound. **n-gram speculation is lossless; MTP is not.**

A separate unexplained fact is recorded here rather than folded in: prompt-cache
reuse changes greedy output in non-speculative decode by itself. Not chased.

## 9. Divergence localization

Instrumented worktree at the same commit, log-only probes on the GDN dispatch
and on the sampler's pre-chain candidate array. The production tree was never
modified.

`build_delta_net` dispatch counts: serial **1344**, n-gram **2832**, MTP **0**.
MTP never enters that function at all, which rules out the autoregressive-vs-
chunked branch as the axis — n-gram drives identical 3-token chunked
verification batches and stays bit-identical to serial decode.

The mechanism is a second GDN entry point (`src/models/delta-net-base.cpp:582`):

```cpp
const bool keep = cparams.n_rs_seq > 0;
if (!keep) { build_delta_net(...); }          // K = 1
const int64_t K = cparams.n_rs_seq + 1;
ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);   // K snapshot slots
```

and its trigger (`common/common.h:386`): `need_n_rs_seq()` returns `draft.n_max`
for model-based draft types (MTP, EAGLE3, DFLASH, DSPARK) and `0` for all
n-gram types. MTP at `n-max 2` therefore sets `n_rs_seq = 2`, `K = 3`, and
routes every token — prefill included — through the snapshot kernel.

At the first divergence, with argmax verified identical for all preceding
tokens, the target logits already differ:

| | serial | MTP |
|---|---:|---:|
| top-1 | 20438 · 22.986334 | **3766 · 23.088205** |
| top-2 | 3766 · 22.876259 | 20438 · 22.866669 |
| margin | 0.110075 | 0.221537 |

Mean \|Δlogit\| 0.196, max 0.673 over 29 shared top-32 tokens — two to three
orders of magnitude above floating-point reduction noise. The perturbation is
substantial; the decision it flipped was marginal (each arm's top-1 is rank 1
under the other, a paraphrase choice: ` passage` against ` provided`).

## 10. `n_rs_seq` sufficiency

Does the snapshot configuration alone reproduce the shift, without any
drafting? A forced-`n_rs_seq` control answers yes.

| arm | config | K | drafting | hash | vs serial |
|---|---|---:|---|---|---|
| A | no spec, `n_rs_seq=0` | 1 | none | `553a7d8a` | — |
| B | no spec, `n_rs_seq=2` forced | 3 | **none** | `efae1ff4` | diverges @ 7 |
| C | draft-mtp n-max 2 | 3 | 213 drafted | `6f067801` | diverges @ 7 |

Arm B ran with zero draft-head sampler calls, all `spec_decode_*` counters at
zero, and no multi-token verification batches — its batch-shape profile matches
arm A exactly. Recurrent layout under `n_rs_seq=2`: `ssm_states_all=[786432,3]`,
widened to `1+n_rs_seq` snapshot groups.

At token 7, B and C agree to about 1×10⁻⁴ (top-1 logits 23.088337 against
23.088205; mean \|Δ\| versus serial 0.195573 against 0.195586). Both flip the
same near-tie the same way.

**The snapshot configuration is sufficient.** Speculative batch shape and
rollback traffic are ruled out — arm B performs neither and reproduces the
divergence. Actual speculation contributes only a second-order residual: B and
C hold identical argmax for **223** tokens before parting, differing by mean
0.0017 across that region against the 0.196 shift the configuration alone
produces.

Speculation is therefore the *trigger*, not the cause: selecting a model-based
draft type enables recurrent-state snapshots, and that configuration changes
the GDN computation for every token.

Two limits on the claim. `n_rs_seq > 0` also changes ubatch splitting
(`src/llama-memory-hybrid.cpp:89`), so the control isolates the configuration
rather than the `K` value alone. And neither arm is ground truth — this shows
the paths differ, not which is more accurate.

## 11. K=1 vs K=3 verifier agreement

Entries 9 and 10 established that the snapshot path shifts target logits by
about 0.2 at a divergent prefix. This entry asks the question that matters for
performance: does that shift explain the MTP acceptance ceiling?

Method: production MTP ran once over four prompts, and every speculative round
was captured with its exact committed prefix, both proposals, and the
accept/reject outcome — 414 rounds. The proposals were then **frozen** and
re-scored under two target contexts, V1 (`n_rs_seq=0`, K=1) and V3 (forced
`n_rs_seq=2`, K=3, no drafter). Neither verifier was permitted to generate a
continuation. Scoring reads the target distribution at an exact token prefix
with `cache_prompt` disabled, since prompt-cache restoration is known on this
model to alter recurrent state. Position 1 was scored at `prefix + d0`
regardless of whether d0 would have been rejected, separating proposal-position
quality from joint survival.

The offline V3 scorer reproduced production's decisions on **249/250** rounds
(99.6%) at position 0 and 215/216 at position 1. The residual is expected
rather than a fault: production evaluates a position inside a 3-token
verification batch while offline scoring evaluates it as a single-token
continuation. Both verifiers are scored identically, so the comparison holds
batch shape constant and isolates `n_rs_seq`.

| metric | K=1 | K=3 |
|---|---:|---:|
| P0 | 0.8800 | 0.8600 |
| P1-counterfactual | 0.7440 | 0.7440 |
| Joint-2 | 0.6680 | 0.6520 |
| Conditional-P1 | 0.7591 | 0.7581 |
| expected accepted drafts per round | 1.5480 | 1.5120 |
| mean target margin, position 0 | 4.3288 | 4.3436 |
| mean target margin, position 1 | 4.2904 | 4.3440 |

Only 7 of 250 rounds disagree at position 0 — K=3 breaks six K=1 acceptances
and rescues one; joint survival splits 5 against 1 the same way. Every
disagreement sits at a near-tie: K=1 margin averages 0.060 and K=3 margin 0.169
on those rounds, against an overall mean margin of 4.33. The ~0.12 median
\|Δlogit\| on the proposed token only flips outcomes in the thin tail where the
margin is below roughly 0.2.

**Neutral. The branch is closed for the performance project.** Per-token
acceptance is 0.7740 under K=1 against 0.7560 under K=3 — 1.8 points, 2.4%
relative. Measured production acceptance is 0.75–0.77 against a ceiling of 1.0,
so the snapshot path accounts for about 2 of roughly 23 missing points, under a
tenth of the gap. The K=1/K=3 divergence remains a real correctness and
numerical-equivalence question, but it is not the performance lever.

The study is deliberately underpowered and was not extended. McNemar exact on
the discordant pairs gives p = 0.125 at position 0 and p = 0.219 for joint
survival, so a small real effect is bounded rather than excluded; the direction
is consistently against K=3. Resolving 2 points would need roughly 1,500–2,000
rounds against the 250 scored here. That was judged not worth the GPU time,
because the conclusion does not depend on it — even taking the point estimate
at face value, the effect is under a tenth of the gap.

The acceptance ceiling is therefore a property of the proposer, consistent with
the rest of the campaign: acceptance was flat to three digits across every KV
precision, and `n-max 2` was decisively optimal in the parameter sweep.

## 12. Qwen3.6 vs Qwen3.8 native MTP control

Entry 11 left intrinsic head quality as the leading explanation for the ~75%
acceptance ceiling. This entry tests it directly on the same machine: is
Qwen3.8's learned proposer worse than the preceding release's?

Control model: `unsloth/Qwen3.6-27B-MTP-GGUF`, `Qwen3.6-27B-UD-Q4_K_XL.gguf`,
17,909,097,600 bytes, SHA-256
`4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095`.

A local `Qwen3.6-27B-Q6_K` was rejected rather than substituted: it has 64
blocks, no `nextn_predict_layers`, and no MTP tensors, so it is a non-MTP build.

The two releases are comparable at the layout level. Same architecture
(`qwen35`), 65 blocks, `nextn_predict_layers = 1`, context length 262144, 24
heads / 4 KV, key and value length 256, `full_attention_interval` 4. The MTP
blocks are structurally identical — the same 15 `blk.64.*` tensors with the same
shapes and the same **424,699,392** parameters — and the weight files differ by
under 0.1%, so decode bandwidth is matched. Native MTP sets `n_rs_seq = n_max`
for both, so both run the K=3 snapshot path and the comparison is about the
proposer, not the runtime.

Same harness as the Qwen3.8 f16 reference arm: greedy with `ignore_eos` at 256
tokens, five repetitions after a discarded unrelated warmup, fresh server per
arm, cache-busted prompts with `cache_prompt` disabled, ctx 163840, ubatch 512,
n-max 2, p-min 0.3.

| | Qwen3.6 | Qwen3.8 |
|---|---:|---:|
| base decode tok/s | 29.89 | 29.41 |
| MTP decode tok/s | 50.95 | **53.82** |
| MTP multiplier | 1.705 | **1.830** |
| aggregate acceptance | 0.7435 | 0.7502 |
| p0 | 0.8196 | **0.8566** |
| joint-p1 | **0.6471** | 0.6365 |
| conditional-p1 | **0.7895** | 0.7431 |
| expected accepted drafts per round | 1.4667 | **1.4931** |
| mean accepted length | 2.4667 | 2.4931 |
| prefill tok/s | 713.1 | 721.3 |
| VRAM | 27.66 GiB | 27.73 GiB |
| verification rounds | 510 | 509 |
| drafted / accepted | 1006 / 748 | 1013 / 760 |

**No proposer gap. The hypothesis is rejected.** On the primary metric Qwen3.8
is slightly ahead — 1.4931 expected accepted drafts per round against 1.4667 —
and it also leads on p0 (+0.0370), aggregate acceptance, and the MTP multiplier.
Nothing approaches a 7–10 point advantage for Qwen3.6, and the sign is reversed
in any case. Neither positional difference is significant at n ≈ 510
(two-proportion z = +1.60 for p0, −0.35 for joint-p1), which only reinforces the
conclusion: no deficit was found to recover.

The positional-depth characterization at n-max 1–4 was gated on Qwen3.6 clearly
winning and was therefore not run.

The one axis where Qwen3.6 leads is conditional second-position survival, 0.7895
against 0.7431: given a correct first proposal its second is more often correct
too. But it proposes a correct first token less often and the net favours
Qwen3.8. The two heads trade first-position accuracy against depth robustness,
consistent with `n-max 2` being the parameter sweep's decisive optimum.

Recorded and not corrected: the MTP blocks are quantized differently — Qwen3.6
carries `eh_proj` at Q8_0 with attention and FFN mostly Q4_K, Qwen3.8 `eh_proj`
at Q6_K with attention and FFN mostly Q5_K and IQ4_XS. Entry 4 found no
acceptance gain from substantially higher MTP precision, and since the measured
difference is small and favours Qwen3.8 there is no Qwen3.6 advantage for the
quantization mix to explain.

**Consequence for the campaign.** A 74–75% acceptance ceiling appears to be the
normal operating point for this architecture's native MTP at depth 2, not a
Qwen3.8 defect. Head retraining is dropped as a direction. The remaining lever
is not better proposer weights but the cost and structure of each speculative
round — reducing per-round overhead, or changing the speculative mechanism
itself.

## 13. MTP round-cost decomposition

With both the runtime-path and proposer-weight explanations closed, this entry
asks where the time in a speculative round actually goes.

`draft-mtp` reports cumulative `dur(b,g,a)` = `t_begin_us`, `t_draft_us`,
`t_accept_us` (`common/speculative.cpp:2764`, counts at `:2786`). The timers
themselves are plain RAII wall clocks with no synchronization
(`common/common.h:40`), but `dur(g)` is still reliable: the region it brackets
runs `llama_decode(ctx_dft, ...)` at `common/speculative.cpp:1596` and then
`common_sampler_sample` at `:1614`, whose first statement is
`llama_synchronize` (`common/sampling.cpp:595`). Draft GPU work must therefore
complete inside the timed region. Target verification is **not** inside any of
these timers -- it runs in the server loop at `server-context.cpp:3819` -- so it
is obtained by subtraction. Trace output requires `-lv 4`.

Four arms, same 256-token harness, five repetitions after a discarded unrelated
warmup, fresh server, `cache_prompt` disabled:

| | T0 serial K=1 | T1 serial K=3 | T2 n-gram K=3 | T3 MTP |
|---|---:|---:|---:|---:|
| decode tok/s | 29.56 | 29.02 | 30.62 | **53.24** |
| ms per output token | 33.702 | 34.327 | 32.533 | **18.713** |
| verification rounds | 0 | 0 | 123 | 509 |
| output tokens per round | — | — | 1.7886 | 2.4931 |

Round cost is solved from `total = n_serial x t_serial + n_rounds x t_round`
rather than `total / rounds`, because not every output token comes from a round.
For T3 this barely matters -- 11 of 1280 tokens are serial -- so the estimate is
robust; for T2 it is far more sensitive and is used only as a cross-check.

| component | ms per round | share |
|---|---:|---:|
| target verification, 3-position batch | **41.259** | 89.1% |
| MTP draft head (`dur(g)`) | **5.057** | 10.9% |
| accept and begin bookkeeping | ~0.000 | ~0% |
| **total round** | **46.317** | 100% |

Against a single serial K=3 step of 34.327 ms, the excess is **+11.990 ms**,
split 6.932 ms (57.8%) extra verification for two more positions and 5.057 ms
(42.2%) draft head. **The K=3 snapshot overhead is separable and small: +0.625
ms per serial step, 1.9%**, measured directly as T1 - T0 with no drafting
involved, and it sits inside the baseline rather than in the excess.

Two independent arms agree on the verification cost: T2 solves to 42.728 ms per
round with a proposer costing 0.001 ms per call, against T3's 41.259 -- within
3.5%.

Accounting ceilings, not achievable targets:

| eliminate | ms per output token | tok/s | gain |
|---|---:|---:|---:|
| nothing (current) | 18.713 | 53.24 | — |
| draft head entirely | 16.549 | 60.43 | +13.5% |
| extra verification cost | 15.797 | 63.30 | +18.9% |
| all | 13.518 | 73.98 | +38.9% |

**Verification is the lever, not the proposer.** It is 89% of round cost and 58%
of the excess. Verifying 3 positions costs 41.26 ms against 34.33 for 1 -- +20%
for three times the positions, already sublinear because decode is
bandwidth-bound and the weights stream once regardless of batch size. Whether
that marginal ~6.9 ms is recoverable is a kernel question this entry does not
answer; it establishes only where the time is.

The sublinearity has a structural implication: deeper drafting spreads a fixed
~34 ms weight-streaming cost across more candidate tokens. That is in tension
with `n-max 2` being the sweep's decisive optimum, where the binding limit is
acceptance decay rather than verification cost. A cheaper proposer at greater
depth is the shape of any remaining win.

## 14. Kernel profile of 1-token vs 3-token verification

Entry 13 located the +6.9 ms marginal cost of evaluating three positions instead
of one but did not explain it. This entry profiles it.

**Microbench.** A standalone tool (`examples/vbench` in the instrumented
worktree) removes the proposer entirely and issues single-sequence decode
batches of N tokens at a fixed KV depth, synchronizing after each call. Both
arms advance the same number of sequence positions from the same depth, so
per-call times compare at matched depth. It reproduces production closely:

| | V1, 1 token | V3, 3 tokens | marginal |
|---|---:|---:|---:|
| microbench at 1.2K | 34.287 ms | 41.256 ms | **+6.969 ms** |
| entry 13 decomposition | 34.327 ms | 41.259 ms | +6.932 ms |

**Profiling method.** The Vulkan perf logger accumulates across graph computes,
so each arm was run at two iteration counts and differenced, cancelling prefill
and warmup exactly. Profiling adds about 2.1 ms to both arms but preserves the
delta (+6.83 against +6.97 unprofiled), so proportions are valid while absolute
profiled times are not comparable to production.

| family | V1 ms | V3 ms | delta ms | share of delta |
|---|---:|---:|---:|---:|
| **MUL_MAT** | 30.224 | 35.179 | **+4.955** | **72.6%** |
| COPY/TRANSFORM | 1.837 | 2.267 | +0.430 | 6.3% |
| ELEMENTWISE | 0.648 | 1.056 | +0.408 | 6.0% |
| NORM | 1.260 | 1.626 | +0.365 | 5.4% |
| GATED_DELTA_NET | 0.300 | 0.542 | +0.242 | 3.5% |
| FLASH_ATTN_EXT | 0.353 | 0.432 | +0.079 | 1.2% |
| other, activation, rope | 1.125 | 1.173 | +0.049 | 0.7% |
| **GPU kernel total** | 35.746 | 42.276 | **+6.530** | **95.6%** |
| unexplained | 0.677 | 0.975 | +0.299 | 4.4% |

The three largest contributors are the per-layer FFN matmuls, 64 dispatches each:

| operation | per dispatch, n=1 to n=3 | delta | share |
|---|---:|---:|---:|
| `MUL_MAT_VEC iq4_xs m=17408 k=5120` (ffn_gate) | 92.45 to 120.11 us, **x1.299** | +1.770 | 25.9% |
| `MUL_MAT_VEC q5_K m=5120 k=17408` (ffn_down) | 103.4 to 119.9 us, x1.160 | +1.057 | 15.5% |
| `MUL_MAT_VEC q5_K m=17408 k=5120` (ffn_up) | 109.76 to 119.63 us, x1.090 | +0.632 | 9.3% |

All three converge to about 120 us at n=3 despite different quantizations, so
`iq4_xs` loses its n=1 advantage entirely.

**One implementation change at N=3, and it is a hard cliff.** The MUL_MAT+ADD
fusion is disabled. V1 runs two fused ops totalling 7.242 ms; V3 runs none, and
its standalone ADD dispatches rise from 96 to 176. The gate is at
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:16438`:

```cpp
// mat-vec only
if (ggml_nrows(mul) != 1) { return false; }
```

`ggml_nrows` is ne[1]*ne[2]*ne[3], so n=3 fails outright. This is **not** a
MMVQ-to-GEMM transition: both arms stay on `MUL_MAT_VEC`, whose shader handles
up to `mul_mat_vec_max_cols = 8`. Only the fusion is lost.

**Context sensitivity.** Repeating V1/V3 at two depths separates the two
components:

| depth | V1 | V3 | marginal |
|---|---:|---:|---:|
| ~1.2K | 34.138 ms | 40.788 ms | **+6.649 ms** |
| ~25K | 36.761 ms | 44.735 ms | **+7.974 ms** |

A 20x increase in KV depth raises the marginal cost only 19.9%. In the profile,
the MUL_MAT delta does not grow with depth (+4.955 to +4.380, drifting down
within noise) while FLASH_ATTN_EXT grows about tenfold, from +0.079 (1.2%) to
+0.773 ms (9.9%), accounting for roughly 72% of the increase. **The verification
tax is predominantly context-independent model matmul**, with an attention
component that is negligible at production context lengths.

The 25K profile is less trustworthy in absolute terms: GPU totals explain 84.8%
of the wall delta against 95.6% at 1.2K, and V1's residual is negative
(-0.413 ms), meaning summed kernel time exceeds measured wall time. That is a
profiler limitation, not a real cost. The attention-growth conclusion survives
because it rests on an order-of-magnitude change in the FLASH_ATTN_EXT delta
between depths, far larger than the accounting error.

**Next target: the `mul_mat_vec` multi-column path, specifically `iq4_xs`.** An
ideal bandwidth-bound kernel would cost about 1.0x going from one column to
three, since the weights stream once either way; these cost 1.09x to 1.30x.
`iq4_xs` is both the worst amortizer and the single largest contributor, and its
cost is context-independent, so any gain applies at every context length. The
fusion cliff is the more tractable fix but is worth only about 0.4 ms, 5.6% of
the delta. Attention is explicitly not the first target at 1.2% of the marginal
cost.

## 15. IQ4_XS tiny-column path A/B

Entry 14 named `iq4_xs m=17408 k=5120` (ffn_gate) the largest single contributor
to the verification tax, at 25.9%. This entry asks whether N=3 selects the wrong
Vulkan path and whether an already-implemented alternate is faster. It changes
no production behaviour.

**Path selection is identical at both widths.** Both arms reach
`ggml_vk_mul_mat_vec_q_f16` and differ only in a specialization constant:

| tensor / shape | N=1 | N=3 |
|---|---|---|
| ffn_gate `iq4_xs m=17408 k=5120` | `mul_mat_vec_iq4_xs_f32_f32` NUM_COLS=1 | same shader, NUM_COLS=3 |
| ffn_up `q5_K m=17408 k=5120` | `mul_mat_vec_q5_k_f32_f32` NUM_COLS=1 | same, NUM_COLS=3 |
| ffn_down `q5_K m=5120 k=17408` | NUM_COLS=1, fused MUL_MAT+ADD | NUM_COLS=3, fusion lost |

The branch is `ggml-vulkan.cpp:10042` (`dst->ne[1] <= mul_mat_vec_max_cols`,
which is 8); the pipeline is indexed `[dmmv_wg][a_type][num_cols-1]` at
`:7839`. Workgroup is `DMMV_WG_SIZE_SUBGROUP`, 64 invocations — the
larger-workgroup heuristic at `:7818` is gated to NVIDIA and Intel, so AMD never
takes it. `IQ4_XS` is absent from the Q8_1 type list at `:7761`, but that is not
operative here: the R9700 under RADV reports `int dot: 0`, so
`integer_dot_product` is false and no type uses MMVQ on this device.

**No cliff at N=2–5.** Isolated `test-backend-ops perf` on Vulkan1, each case
emitted twice with the second pass read (the first absorbs GPU clock ramp, worth
40% on the leading case and under 2% everywhere else):

| N | iq4_xs us | ratio | per column | q5_K up us | ratio | per column |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 76.49 | 1.000 | 1.000 | 61.64 | 1.000 | 1.000 |
| 2 | 94.31 | 1.233 | 0.617 | 85.44 | 1.386 | 0.693 |
| 3 | 106.92 | 1.398 | 0.466 | 107.42 | 1.743 | 0.581 |
| 4 | 123.59 | 1.616 | 0.404 | 137.62 | 2.233 | 0.558 |
| 5 | 138.44 | 1.810 | 0.362 | 168.15 | 2.728 | 0.546 |

Scaling is smooth and `iq4_xs` amortizes *better* than `q5_K`, inverting entry
14's in-graph ordering. **The isolated benchmark is not a valid proxy for this
question.** Isolated `q5_K` at N=1 reaches 994 GB/s, 154% of the R9700's
644.6 GB/s DRAM peak: both weight matrices (47.35 and 61.28 MB) fit Navi 48's
64 MB Infinity Cache, so the isolated bench is cache-resident while the
production graph streams 64 layers of cold weights from DRAM. Entry 14's
in-graph delta profile remains the only valid measurement of the tax, and any
future fix must be validated in-graph rather than here.

**The one existing alternate loses decisively.** With MMVQ unavailable and
`KHR_coopmat` present, `mul_mm` was the only real candidate. An env-gated
override (`GGML_VK_FORCE_MUL_MM`, worktree only) routes the same shapes through
dequant+GEMM. Correctness passed 10/10 against the CPU reference at the
NMSE 5e-4 gate for both types at N=1–5.

| path, iq4_xs N=3 | us | vs incumbent |
|---|---:|---:|
| `mul_mat_vec` (incumbent) | 107.03 | — |
| `mul_mm` (dequant + GEMM, coopmat) | 277.14 | 2.59x slower |

Per the validation chain, no V3 or MTP run followed: there was no isolated
improvement to propagate, so no throughput projection is claimed.

**Redundant dequantization is the mechanism.** RADV pipeline statistics, no
spills and SGPR 128 throughout:

> **Retraction (see entry 16).** The resource statistics below are correct and
> stand. The inference drawn from them — that redundant dequantization was the
> factor *limiting performance*, and that removing it was worth 1.77–2.98 ms per
> round — was wrong. Entry 16 built the hoisted kernel: it cut shader code from
> 81.2 KB to 48.9 KB and removed two thirds of the dequantization work, and the
> in-graph kernel moved under 2.5%, failing the 0.2 ms wall-time gate. In full
> graph execution the matmul is DRAM-bandwidth and latency bound, and the
> redundant dequant was already hidden behind memory latency. Read the paragraphs
> below as a measured description of the two shaders' code structure, not as a
> diagnosis of the tax.

| NUM_COLS | iq4_xs VGPR / code / occupancy | q5_K VGPR / code / occupancy |
|---|---|---|
| 1 | 48 / 38192 B / 32 | 60 / 5440 B / 24 |
| 3 | 72 / 81228 B / 20 | 108 / 9492 B / 14 |
| 5 | 84 / 124292 B / 18 | 192 / 13412 B / 8 |

`iq4_xs` code grows about 21.5 KB per column against `q5_K`'s 2.0 KB. In
`vulkan-shaders/mul_mat_vec.comp:47-78` the column loop is the *outer* loop and
`dequantize4()` sits inside it, so a full dequant — including the
`kvalues_iq4nl[]` LDS gathers — is emitted per column even though `ib` and `iqs`
are column-invariant. The specialized `mul_mat_vec_q5_k.comp:14-78` inverts the
nesting, decoding once per row before looping columns.

The two shaders sit at opposite ends of one tradeoff and land together at N=3:
`iq4_xs` pays redundant dequant but holds occupancy (32 to 20), `q5_K` hoists the
dequant into registers and collapses occupancy (24 to 14). Neither is near
optimal.

**Headroom, measured in-graph where it is real.** `iq4_xs` at N=3 runs at
394 GB/s against 512 GB/s at N=1. Bytes read are identical at both widths, so a
kernel that amortized properly would cost the same at N=3 as at N=1 — worth
1.77 ms per round, with a DRAM-bound floor of 2.98 ms. Both clear the >1.0 ms
high-value threshold.

The change recommended at the time was a loop interchange in the generic
`mul_mat_vec.comp`, hoisting IQ-type dequantization out of the `NUM_COLS` loop and
staging it so it does not inherit the VGPR blowup that costs `q5_K` its occupancy,
A/B'd in-graph with `vbench` and scoped behind a specialization constant. Entry 16
carried that out. It was built and measured, and it did not pay.

## 16. IQ4_XS tiny-N dequant-reuse and occupancy evaluation

Entry 15 identified redundant dequantization in generic `mul_mat_vec.comp` as the
primary source of code bloat across columns and hypothesized that hoisting dequant
above the `NUM_COLS` loop, combined with workgroup accumulator sizing (`ROWS=2`),
could recover N=3 amortization.

This entry evaluates two experimental variants implemented in the scratch probe:
1. **Tiny-N dequant-reuse (`TINYN_REUSE=1`, default rows=4):** hoists decoded weight
   fragments (`dequantize4`) out of the column loop so each fragment is decoded once
   and consumed across all active columns.
2. **Tiny-N dequant-reuse with reduced row tile (`GGML_VK_IQ4XS_ROWS=2`):** halves
   the row accumulator array per workgroup to reduce VGPR pressure and increase
   occupancy from 20 to 24 subgroups/SIMD.

### Correctness and shader resource statistics

All experimental variants passed the CPU-reference correctness gate (`16/16 OK` on
`test-backend-ops test -b Vulkan1 -o MUL_MAT`).

RADV pipeline statistics on GFX1201 (N=3, `m=17408, k=5120`):

| Variant | SGPR | VGPR | Code size | Occupancy (subgroups/SIMD) | Notes |
|---|---:|---:|---:|---:|---|
| Incumbent (`TINYN=0`) | 128 | 72 | 81,228 B | 20 | Dequant inside column loop |
| Default Tiny-N Reuse (`TINYN=1, ROWS=4`) | 128 | 72 | 48,868 B | 20 | Code size -39.8%, duplicate dequant eliminated |
| Occupancy Variant (`TINYN=1, ROWS=2`) | 128 | 60 | 27,704 B | 24 | Code size -65.9%, VGPR -12, occupancy +20% |

### Authoritative in-graph V3 wall-time measurement

Measured using `llama-vbench` on Qwen3.8-27B-UD-Q4_K_XL at matched conditions
(`dev=1 (R9700)`, `n_prefill=1200`, `n_iter=200`, `n_rs_seq=2`, `batch_n=3`, 5 clean repetitions):

| Arm / Config | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Rep 4 (ms) | Rep 5 (ms) | Mean (ms) | Stdev (ms) | Median (ms) | p10 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Incumbent (`TINYN=0`)** | 41.2308 | 41.2311 | 41.2274 | 41.2412 | 41.2420 | **41.2345** | 0.0066 | 41.2311 | 41.2274 | 41.2420 |
| **Default Tiny-N Reuse** | 41.0076 | 41.0471 | 41.0475 | 41.0697 | 41.0817 | **41.0507** | 0.0283 | 41.0475 | 41.0076 | 41.0817 |
| **Occupancy Variant (`ROWS=2`)** | 40.9908 | 41.1096 | 41.2567 | 41.4036 | 41.5970 | **41.2715** | 0.2391 | 41.2567 | 40.9908 | 41.5970 |

- Default Tiny-N reuse reduced V3 wall time by **0.1838 ms** (below the 0.2 ms minimum gate).
- The `ROWS=2` occupancy variant was **0.0370 ms slower** than incumbent and **0.2208 ms slower** than default Tiny-N reuse.

An independent run of the same A/B, three repetitions per arm on the same
binaries, measured incumbent V3 41.080 ms against tiny-N 41.003 ms — a saving of
**0.077 ms**, and an IQ4_XS kernel delta of 114.476 to 112.511 us/dispatch
(-1.7%, -0.126 ms over 64 layers). The magnitudes differ from the table above by
more than either run's internal spread, so treat the absolute saving as bounded
somewhere under 0.2 ms rather than pinned at 0.18 ms. Both runs agree on the
decision and on the mechanism, and in both the V1 arm — whose path is unchanged —
drifted by 0.04 ms, which sets the floor on what this harness can resolve.

### Isolated kernel profiling (differenced 100 vs 200 iterations)

| Arm / Config | IQ4_XS `ffn_gate` (us/dsp) | 64-dsp total (us) | Total MUL_MAT (ms) | Wall time (ms) |
|---|---:|---:|---:|---:|
| Incumbent (`TINYN=0`) | 118.454 | 7,581.040 | 34.767 | 42.702 |
| Default Tiny-N Reuse | 115.661 | 7,402.283 | 34.813 | 42.702 |
| Occupancy Variant (`ROWS=2`) | 122.716 | 7,853.852 | 35.391 | 43.401 |

- Default Tiny-N reuse shaved only **2.793 us/dispatch** (-2.4%, ~0.179 ms total over 64 layers).
- `ROWS=2` degraded kernel dispatch time to **122.716 us/dispatch** (+3.6% slower than incumbent, +6.1% slower than default Tiny-N reuse).

### Statistical and architectural interpretation

1. **Dequant arithmetic is not the bottleneck:** Removing redundant dequantization in SPIR-V
   collapsed code size from 81.2 KB to 48.9 KB but yielded only a 1.7–2.4% kernel-level delta.
   In full graph execution streaming 64 layers of cold weights from DRAM, the kernel is
   predominantly DRAM bandwidth and latency bound.
2. **Why `ROWS=2` failed despite higher occupancy:** Halving rows-per-workgroup reduced VGPR
   and unlocked 24 subgroups/SIMD, but doubled the total number of workgroups dispatched
   ($M=17408$ requires 8,704 workgroups instead of 4,352). On R9700/RADV, the increased command
   processor / workgroup scheduling overhead and reduced per-workgroup arithmetic amortization
   more than wiped out the occupancy gain.

### Gate decision

Both variants fail the $\ge 0.2\text{ ms}$ wall-time threshold ($0.18\text{ ms}$ and $-0.04\text{ ms}$).

**Decision:**
```text
ABANDON IQ4_XS AS PRIMARY TARGET → START MUL_MAT+ADD FUSION
```

No further workgroup/row tuning will be pursued on `mul_mat_vec_iq4_xs`. The project moves directly to the `ffn_down` `MUL_MAT + ADD` fusion cliff (`ggml-vulkan.cpp:16505`), where $N=3$ currently drops fusion because `ggml_nrows(mul) != 1`.

## 17. Tiny-N MUL_MAT + ADD fusion evaluation

Entry 16 pivoted to the `ffn_down` ($Q5\_K, M=5120, K=17408$) `MUL_MAT + ADD` fusion cliff,
where $N=3$ verification dropped fusion due to the legacy `ggml_nrows(mul) == 1` predicate
in `ggml-vulkan.cpp:16505`.

### Source/path audit

1. **Root cause:** The restriction was host-side predicate-only. When multi-column batching
   ($N=2..8$) was originally added to `mul_mat_vec`, the underlying SPIR-V shaders (`mul_mat_vec_base.glsl`)
   and C++ push constant dispatch (`ggml_vk_mul_mat_vec_q_f16`) were already parameterized with
   `NUM_COLS` and `batch_stride_d` to index `data_fuse0[j * p.batch_stride_d + d_offset + first_row + n]`.
   The fusion validator `mm_add_ok()` had retained the older 1D constraint `ggml_nrows(mul) == 1`.
2. **Implementation:** Generalized `mm_add_ok()` under `GGML_VK_MM_ADD_TINYN=1` to allow
   multi-column shapes satisfying `mul_mat_vec` eligibility ($N \le 8, \text{ne2} \times \text{ne3} == 1$).

### Correctness verification

Passed CPU-reference correctness suite across all matrix widths $N=1..8$ (`21/21 OK` on
`test-backend-ops test -b Vulkan1 -o MUL_MAT -p ".*type_a=q5_K.*"`).

### Direct cliff & dispatch measurement ($N=1..5$)

Profiling the full graph across sequence widths under incumbent vs tiny-N fusion:

| $N$ | Fused (Incumbent) | Fused (Tiny-N) | Incumbent Total ADDs | Tiny-N Total ADDs | ADD Dispatches Eliminated |
|---|---|---|---|---|---|
| 1 | True | True | 96 dsp / 0.348 ms | 96 dsp / 0.346 ms | 0 (already fused) |
| 2 | False | **True** | 176 dsp / 0.641 ms | **96 dsp / 0.370 ms** | **-80 dispatches** |
| 3 | False | **True** | 176 dsp / 0.703 ms | **96 dsp / 0.440 ms** | **-80 dispatches** |
| 4 | False | **True** | 176 dsp / 0.748 ms | **96 dsp / 0.457 ms** | **-80 dispatches** |
| 5 | False | **True** | 176 dsp / 0.735 ms | **96 dsp / 0.458 ms** | **-80 dispatches** |

At $N=3$, enabling fusion eliminates 80 separate ADD dispatches (64 `ffn_down` residual ADDs + 16 other layers)
and reduces raw elementwise ADD GPU time from 0.703 ms to 0.440 ms (-0.263 ms).

### Authoritative in-graph $V_3$ A/B (5 Repetitions)

Measured using `llama-vbench` on Qwen3.8-27B-UD-Q4_K_XL (`batch_n=3`, `n_prefill=1200`, `n_iter=200`, `n_rs_seq=2`):

| Arm / Config | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Rep 4 (ms) | Rep 5 (ms) | Mean (ms) | Stdev (ms) | Median (ms) | p10 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Incumbent (`TINYN=0`)** | 40.7516 | 40.9065 | 41.0050 | 41.0985 | 41.2168 | **40.9957** | 0.1782 | 41.0050 | 40.7516 | 41.2168 |
| **Tiny-N Fused (`TINYN=1`)** | 41.0515 | 41.0711 | 41.1229 | 41.1186 | 41.1381 | **41.1004** | 0.0371 | 41.1186 | 41.0515 | 41.1381 |

- Mean $\Delta = \mathbf{-0.1048\text{ ms}}$ (Fused is within noise / ~0.10 ms slower than incumbent).
- Median $\Delta = \mathbf{-0.1136\text{ ms}}$.

### Kernel family delta analysis ($N=3$)

Differenced 100 vs 200 iterations under `GGML_VK_PERF_LOGGER=1`:

| Kernel Family | Incumbent | Fused | Delta |
|---|---|---|---|
| **ELEMENTWISE** | 288 dsp / 1035.52 us | 208 dsp / 758.37 us | **-277.15 us** (-80 dispatches) |
| **MUL_MAT** | 497 dsp / 35030.73 us | 497 dsp / 35249.40 us | **+218.67 us** (+3.4 us/layer bias fetch) |
| NORM / OTHER | 465 dsp / 2001.25 us | 465 dsp / 1879.90 us | -121.35 us |
| ALL OTHERS | 256 dsp / 4015.52 us | 256 dsp / 4043.21 us | +27.69 us |
| **TOTAL GPU TIME** | 1506 dsp / 42083.02 us | 1426 dsp / 41930.88 us | **-152.14 us (-0.152 ms)** |

### Architectural interpretation

While fusion eliminates 80 dispatches and saves 277 us of standalone elementwise execution,
the fused `mul_mat_vec` kernel must now perform uncoalesced/streamed reads of the bias tensor
`data_fuse0` ($M=5120, N=3$) from memory inside `reduce_result`, adding +218 us across the 64 layers.
The net GPU saving is only ~0.15 ms, which translates to a flat/negligible graph wall-time difference (<0.1 ms).

### Gate decision & recommendation

The measured wall-time saving is $<0.1\text{ ms}$, failing the $\ge 0.2\text{ ms}$ threshold for MTP deployment.

**Recommendation:**
```text
CLOSE FUSION BRANCH
```

Tiny-N `MUL_MAT + ADD` fusion is technically correct and successfully removes 80 dispatches, but does not provide
meaningful throughput acceleration due to the offsetting bias fetch overhead in `mul_mat_vec`.

## Where this leaves the verification tax

Entries 14–17 close the kernel-level attack on the +6.9 ms verification tax.
Entry 14 attributed 72.6% of it to MUL_MAT and named two candidate levers;
entries 15–17 tested both to exhaustion and neither paid:

| lever | built and measured | wall effect | gate |
|---|---|---|---|
| alternate path (`mul_mm`) | entry 15 | 2.59x slower isolated | rejected |
| IQ4_XS dequant reuse | entry 16 | under 0.2 ms | rejected |
| tiny-N reuse + `ROWS=2` occupancy | entry 16 | slower than incumbent | rejected |
| `MUL_MAT + ADD` fusion restore | entry 17 | under 0.1 ms | rejected |

The common cause is the same in every case: at N=3 the FFN matmuls stream the
same weight bytes as at N=1 from DRAM, and they are bandwidth and latency bound
rather than limited by arithmetic, dispatch count, or shader code size. Work that
removes ALU or dispatches therefore returns almost nothing. Any future gain has to
reduce **bytes moved** — a smaller quantization for `ffn_gate`, weight residency
across the three verified positions, or a different speculative mechanism — not
reduce work per byte.

## 18. MTP Draft-Vocabulary Trimming Speculative Holdout Validation

**Question**: Does reducing the MTP draft LM-head row count (from full $248,320$ to $65,536$ or $32,768$ rows via `d2t` remapping) yield real end-to-end multi-token speculative decode gains on the R9700?

**Hypothesis**: Isolated microbenchmarks showed the Q6_K LM-head matvec drops from $1,652\ \mu\text{s} \to 452\ \mu\text{s}$ (64K) and $234\ \mu\text{s}$ (32K), saving $1.20\text{--}1.42\text{ ms}$ GPU kernel time per dispatch. If top-1 token coverage remains high ($>95\%$), multi-token speculative throughput should improve.

**Experimental Setup**:
* **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB VRAM), Vulkan/RADV
* **Speculative Policy**: Genuine multi-token speculative decoding ($n_{\text{max}}=2, p_{\text{min}}=0.3, c=8192, ub=512, fa=1, ctk=f16, ctv=f16, temp=0.0, top\_k=1$)
* **Arms**:
  * `FULL`: Full vocabulary ($M=248320$)
  * `64K`: Trimmed vocabulary ($M=65536$, map hash `3aa00816...`)
  * `32K`: Trimmed vocabulary ($M=32768$, map hash `f6d89a06...`)
* **Holdout Matrix**: 16 diverse, previously unseen prompts across 6 domains (Code, Structured JSON, Technical Architecture, Mathematics, Chat/Prose, Long-Context Continuation) evaluated in interleaved/rotated order (FULL-64K-32K, 64K-32K-FULL, 32K-FULL-64K) totaling $>1,500$ verification rounds per arm.

### Measured Holdout Results (16 Unseen Prompts)

| Metric | FULL (248K) | 64K TRIMMED | 32K TRIMMED | Classification |
|---|---:|---:|---:|---|
| **Direct Decode Throughput (tok/s)** | **16.82 ± 0.89** | **8.32 ± 0.04** | **8.69 ± 0.05** | **MEASURED** |
| **Throughput Delta vs FULL** | — | **−50.5%** | **−48.3%** | **CALCULATED** |
| **Verification Rounds ($N_{\text{rounds}}$)** | 1,253 | 3,088 | 3,088 | **MEASURED** |
| **Draft Tokens Generated** | 2,437 | 0 | 0 | **MEASURED** |
| **Draft Tokens Accepted** | 1,862 | 0 | 0 | **MEASURED** |
| **Accepted Drafts / Round** | **1.486** | **0.000** | **0.000** | **CALCULATED** |
| **Committed Tokens / Round** | **2.486** | **1.000** | **1.000** | **CALCULATED** |
| **Step 0 Acceptance ($p_0$)** | **0.838** | **0.000** | **0.000** | **MEASURED** |
| **Joint Step 1 Acceptance ($p_1$)** | **0.641** | **0.000** | **0.000** | **MEASURED** |
| **Proposer Latency (`dur_g` / round)** | **5.58 ms** | **33.19 ms** | **27.91 ms** | **MEASURED** |
| **Greedy Output Equivalence** | 16 / 16 match | 0 / 16 match (collapsed) | 0 / 16 match (collapsed) | **MEASURED** |

### Findings & Root Cause Analysis

1. **Speculative Collapse**: Across all 16 unseen holdout prompts, the trimmed vocabulary arms generated exactly zero accepted draft tokens (`#gen drafts = 0, #acc drafts = 0, p0 = 0.0`). The speculative engine completely collapsed to serial 1-token-at-a-time decoding.
2. **Proposer Penalty**: Scattering trimmed logits into a 248,320-element `-INFINITY` buffer via `ggml_set_rows` distorted the candidate softmax probability distribution during `common_sampler_sample`, causing top candidate probabilities to fail the $p_{\text{min}} \ge 0.3$ threshold at step 0.
3. **Severe End-to-End Regression**: In addition to zero speculative draft generation, evaluating the failed proposer graph on every round incurred an unamortized $28\text{--}33\text{ ms}$ latency penalty per token, cutting decode speed in half ($16.82 \to 8.69\text{ tok/s}$).
4. **Proposer Duration Accounting (`dur_g`)**: Inspection of `common/speculative.cpp` confirms that `dur(g)` (`t_draft_us`) measures synchronized host wall time from the entry of `impl->draft()` to its return. It includes CPU batch construction, GPU graph execution, synchronous GPU-to-host readback (`llama_get_logits_ith`), and CPU softmax evaluation. It is not an isolated GPU kernel timer.

### Pre-Registered Decision: `FAILED IMPLEMENTATION / CONCEPT REMAINS OPEN`

* **Outcome**: **`NO WIN (≤0%)`** for the *reconstruction* implementation as built.
* **Action**: Do not deploy this implementation. Do **not** deploy 32K or 64K
  trimming via the reconstruction path to production.

> [!IMPORTANT]
> **What this entry disproved, precisely.** The measured failure is located in
> the FILL(`-INFINITY`) + `SET_ROWS` reconstruction and backend-sampling path,
> **not** in the reduced-vocabulary concept. Specifically:
>
> * The reduced-head logits were subsequently proven **bit-exact for copied
>   rows** — the head computes the right numbers.
> * The reduced-head **mechanism timing remains valid** (1652 → 452 / 234 µs
>   per dispatch, isolated).
> * The failure occurred **downstream**, in reconstructing a 248,320-entry
>   logit tensor and handing that view to the backend sampler. Scattering into
>   a `-INFINITY` buffer distorted the candidate softmax so nothing cleared
>   $p_{\text{min}}$.
> * Severe **graph-split latency** made that architecture unsuitable
>   regardless of the sampling distortion.
>
> The concept therefore **remains open**. **Entry 19** tests direct
> reduced-vocabulary sampling — sampling in the reduced space and mapping the
> local index through `d2t`, with no reconstruction at all.

**The raw holdout results above are preserved unchanged.** They are evidence
that the end-to-end validation gate worked: component timings and acceptance
rates looked favourable, and only the wall-clock holdout exposed the collapse.

## 18B. Root-Cause Diagnosis of the Reconstruction Failure

**Status**: `ESTABLISHED`

Diagnostic work following Entry 18, isolating *where* the reconstruction path
broke. Raw diagnostic output:
[`qwen38_diag_raw_logs.txt`](../data/experimental/qwen38_diag_raw_logs.txt).

Established:

1. The reduced 64K / 32K heads are **genuinely materialized** — not a fallback to
   the full head.
2. **Copied rows produce matching logits.** The slicing from `output.weight` is
   faithful.
3. `d2t` is a **valid I64 tensor and maps correctly** to target vocabulary IDs.
4. **`FILL` + `SET_ROWS` works in isolation.** The primitives are not themselves
   defective.
5. The reconstruction *architecture* nonetheless **interacts badly with the real
   backend sampling and scheduler path**, and should not be revived.

Points 1 and 3 were independently re-confirmed during Entry 19 (head widths
reported as 65536 / 32768 / 248320; local index `39882` → target `39138`).

The practical consequence: the failure was never in the reduced vocabulary, the
head, the map, or the individual ops. It was in composing them into a
full-vocabulary reconstruction that the backend sampler then had to consume.
**Do not attempt to repair the `SET_ROWS` reconstruction path.**

## 19. Direct Reduced-Vocabulary MTP Sampling

**Status**: `BLOCKED AT EARLY GATE — IMPLEMENTATION INCOMPLETE`

**Hypothesis**: the reduced 64K/32K MTP heads can yield a real end-to-end gain if
sampled *directly* — sample in the reduced space, map the local index through
`d2t`, and never allocate, fill, or scatter into a 248,320-logit destination.

### Implementation state

The direct-sampling path already existed, uncommitted and **never compiled**, in
the experimental worktree (`/ai/scratch/llamacpp-probe`, detached at `ad1de39`).
It was built and exercised for the first time here. Captured as
[`qwen38_entry19_direct_sampling.patch`](../data/experimental/qwen38_entry19_direct_sampling.patch).

Confirmed working:

* The reduced heads are genuinely materialized — the MTP proposer reports
  `nextn.shared_head_head rows=65536` and `rows=32768`, versus `model.output
  rows=248320` for FULL. They are built by slicing rows out of `output.weight`
  through `d2t`, which is why copied rows are bit-exact.
* `d2t` mapping produces plausible target IDs (e.g. local `39882` → target
  `39138`).
* No reconstruction, no FILL, no `SET_ROWS` remains in the trimmed path.

### Blocking defect found in the pre-existing baseline

> [!WARNING]
> An uncommitted modification to `src/llama-context.cpp` in the worktree had
> **silently corrupted the FULL arm**. It dropped the `needs_raw_logits()` guard
> for all contexts and replaced the per-row logits copy with a single clamped
> flat copy. Effects, both measured here:
>
> * **Target output became two interleaved token streams** — e.g. `"WeThe need
>   user answer wants in complete English production.-ready Need Python produce
>   async code task."` — draft tokens committed alongside target tokens instead
>   of being verified against them.
> * **Throughput read a false 42.32 tok/s.** With the change reverted and a
>   correct patch applied, FULL returns coherent output at **52.96–53.08 tok/s**,
>   matching the historical Track A native-MTP reference of ~53.2–53.8.
>
> Any measurement taken from this worktree before this fix is invalid.

The corrected patch keeps the guard for the target path, widens it only for a
narrower-than-`n_vocab` head, and copies row-by-row because `logits.data` is
strided by `n_vocab` while `t_logits` is only `n_vocab_res` wide.

### Early gate result: FAIL

| Arm | decode tok/s | draft acceptance | drafts |
|---|---:|---|---|
| FULL 248K | **52.96** | 0.747 (118 acc / 158 gen), mean len 2.48 | yes |
| 64K DIRECT | 27.58 | *no acceptance reported* | **none** |
| 32K DIRECT | 27.79 | *no acceptance reported* | **none** |

Both trimmed arms collapse to serial decoding. **64K and 32K are within 0.2
tok/s of each other despite a 2× vocabulary difference**, which rules out
vocabulary-dependent cost (the host-side sort) as the explanation and points at a
fixed per-step defect.

**Root cause, traced**: the reduced logits reach the sampler on only *alternate*
draft steps. On the others the extraction block is skipped (`n_outputs = 0`) and
the sampler reads an **all-zero** buffer. Ten equal zeros give
$p_{\text{top}} = 1/\text{top\_k} = 0.1000$, which fails $p_{\text{min}} = 0.3$,
so drafting stops on that step every time. Trace excerpt:

```
[E19] step=1 p_top=0.5810 max_l=17.1758 local0=39882 tgt=39138   <- valid
[E19] step=2 p_top=0.1000 max_l=0.0000  local0=8     tgt=18      <- all-zero
[E19] step=3 p_top=0.1687 max_l=3.9074  local0=3457  tgt=18865   <- valid
[E19] step=4 p_top=0.1000 max_l=0.0000  local0=8     tgt=18      <- all-zero
```

Making the reduced-head copy synchronous fixed the *first* head's decode but not
the second, consistent with `n_max=2` alternating between the two MTP heads.

### Decision

**STOP at the Phase 9 early gate**, as pre-registered. No holdout was run: the
gate exists precisely to prevent another 16-prompt holdout on an implementation
that cannot generate drafts. Phases 4–6 (top-1 coverage, retained probability
mass, $p_{\text{min}}$ agreement) were **not** measured, because they require a
proposer that produces valid logits on every step.

**Nothing here disproves the direct-sampling concept.** The one step that did
receive valid logits behaved exactly as intended — a confident draft
($p_{\text{top}} = 0.58$) correctly mapped through `d2t`. The defect is in logits
extraction for the second MTP head, not in the sampling architecture.

## Open threads

- Reducing bytes moved during verification, the only lever entries 15–17 did not
  close. Nothing has been designed here yet.
- Whether a cheaper proposer at greater depth beats the current n-max 2 point,
  given that verification is sublinear in batch size.
- Which of the two K paths is closer to unquantized reference output. Requires
  a reference the campaign does not currently have.
- Whether the ~0.2 logit perturbation is quantization-sensitive. A Q6 repeat of
  the localized prompt would test it.
- Why prompt-cache reuse alters greedy output in serial decode.
- llama.cpp reports `logprob = 0.0` for MTP-committed tokens, which would
  mislead anyone scoring outputs from that path.

## Reproduction

Harnesses are under [`experiments/qwen3-8-27b/`](../experiments/qwen3-8-27b/):
`bench.py` (bucketed benchmark), `sweep.py` (parameter sweep), `kvsweep.py`
(entry 7), `equiv_step1.py` and `equiv_localize.py` (entries 8–9),
`rs_seq_test.py` (entry 10), `k1_vs_k3.py` (entry 11), `qwen36_control.py`
(entry 12), `round_cost.py` (entry 13), `kernel_profile.py` with the
`examples/vbench` microbench (entries 14 and 16–17). Data files are under
[`data/experimental/`](../data/experimental/): `qwen3-8-27b-kv-cache.tsv`
(entry 7), `qwen3-8-27b-iq4xs-path.tsv` and
`qwen3-8-27b-iq4xs-pipeline-stats.tsv` (entries 15–16),
`qwen3-8-27b-mtp-proposer.tsv`.

Entries 9–11 and 14–17 require an instrumented llama.cpp build. Most probes are
log-only; these three deliberately change behaviour and default to off:
`LLAMA_FORCE_N_RS_SEQ` (changes `cparams`), `GGML_VK_FORCE_MUL_MM` (changes
Vulkan path selection), and `GGML_VK_IQ4XS_TINYN` with `GGML_VK_IQ4XS_ROWS`
(selects the experimental tiny-N pipelines). None exist in the production tree.

**Line numbers.** Citations in entries 1–15 are against the production tree at
`ad1de39`. Entries 16–17 cite the instrumented worktree, whose additions shift
`ggml-vulkan.cpp` by roughly 70 lines: the fusion predicate cited there as
`:16505` is `:16439` upstream, and entry 17 removes it outright, so it will not
be found in the worktree at all. Resolve any citation against `ad1de39` first.

