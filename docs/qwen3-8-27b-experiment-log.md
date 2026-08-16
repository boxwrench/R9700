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

## Open threads

- Whether the marginal ~6.9 ms cost of verifying three positions instead of one
  is recoverable at the kernel level, or is irreducible compute. Entry 13
  locates it but does not profile it.
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
(entry 12), `round_cost.py` (entry 13). Entries 9-11 require an instrumented
llama.cpp
build; the probes they depend on are described inline in each script and are
log-only except `LLAMA_FORCE_N_RS_SEQ`, which deliberately changes `cparams`.
