# Qwen3.8-27B runtime parameter sweep on one R9700

## Outcome

Twenty-nine configurations of Qwen3.8-27B UD-Q4_K_XL were measured on a single
AMD Radeon AI PRO R9700 under llama.cpp's Vulkan backend. **No configuration
beat the production baseline.**

| | Value |
|---|---:|
| Best single measurement | 51.35 tokens/s |
| Production configuration, five repeats | **50.91 tokens/s**, sd 0.22, range 50.74–51.35 |
| Improvement over the prior published 51.30 | **none** |
| Aggregate draft acceptance | 0.7068 |
| Mean accepted length | 2.4008 |

The result is negative and that is its value. The production setting sits on a
measured local optimum rather than merely appearing fast, and the parameter
space immediately around it is now mapped, so it does not need retesting.

| Parameter | Optimum | Character of the result |
|---|---|---|
| `ctx-size` | 163840 | Flat; free for decode |
| `ubatch-size` | 512 | Flat for decode, matters for prefill |
| `spec-draft-n-max` | 2 | **Decisive real optimum** |
| `spec-draft-p-min` | 0.3 | Flat 0.0–0.3, declines above |

## Method

Each configuration ran in a fresh `llama-server` process so the Prometheus
counters at `/metrics` were scoped to it. Ten generations of up to 256 tokens
on a fixed ~1,175-token prompt, temperature 0.6, fixed seeds, thinking
disabled, each prefixed uniquely to defeat the prompt cache. A first
generation was discarded and the counters were read before and after the
measured block.

Everything not being swept was frozen at the production values: Vulkan device
`Vulkan1`, full offload, `--parallel 1`, six threads, flash attention on, f16
K and V, unified KV, `--load-mode none`, `--split-mode none`, `--n-cpu-moe 0`,
`--spec-type draft-mtp`, and the F16 vision projector loaded.

Stages ran in sequence, each starting from the winner of the previous one.
Where a stage was statistically flat the baseline value was retained rather
than the nominal maximum, so that a noise-selected value would not propagate
into later stages.

Per-configuration standard error on decode is roughly 0.6 tokens/s, so
differences below about 1.8 tokens/s are not resolvable at ten samples. Most
stages here returned differences below that threshold.

### Acceptance metric definitions

From `tools/server/server-task.cpp` and `server-context.cpp`:

- `spec_decode_num_draft_tokens_total` — draft tokens generated
- `spec_decode_num_accepted_tokens_total` — draft tokens accepted
- `spec_decode_num_drafts_total` — verification steps
- `spec_decode_num_accepted_tokens_per_pos_total{position=i}` — a **survival**
  count, incremented for every verification step in which at least `i+1` draft
  tokens were accepted

Mean accepted length is `1 + accepted / verification_steps`. Positional
acceptance is `n_accepted_per_pos[i] / verification_steps`.

## Stage 1: micro-batch size

| ubatch | Decode tokens/s | Prefill tokens/s | Acceptance | Mean accepted length |
|---:|---:|---:|---:|---:|
| 128 | 50.06 ± 0.57 | 639.2 | 0.684 | 2.353 |
| 192 | 50.56 ± 0.45 | 521.9 | 0.696 | 2.380 |
| 256 | 51.07 ± 0.54 | 628.9 | 0.707 | 2.402 |
| 288 | 51.29 ± 0.62 | 574.6 | 0.714 | 2.414 |
| 384 | 51.15 ± 0.41 | 706.2 | 0.710 | 2.408 |
| 512 | 50.87 ± 0.63 | 704.5 | 0.707 | 2.401 |

Decode across 256–512 spans 0.42 tokens/s against roughly 0.85 standard error
on the difference. Not resolvable. **Micro-batch tuning cannot improve
generation throughput at `--parallel 1`**, which is consistent with there being
no concurrent slots to batch across.

Prefill is a different matter and is not monotonic in `ubatch`. 384 and 512
deliver about 705 tokens/s while every smaller value falls between 522 and 639.
Selecting the nominal decode maximum of 288 would have cost roughly 130
tokens/s of prefill for no decode gain. `ubatch` is a prefill parameter here,
not a decode parameter.

## Stage 2: MTP draft depth

This is the only stage with a significant effect.

| n-max | Decode tokens/s | Acceptance | Mean accepted length |
|---:|---:|---:|---:|
| 1 | 44.46 ± 0.31 | 0.811 | 1.811 |
| **2** | **50.80 ± 0.64** | 0.707 | 2.401 |
| 3 | 48.39 ± 0.85 | 0.598 | 2.758 |
| 4 | 45.64 ± 0.59 | 0.529 | 3.033 |
| 5 | 40.06 ± 0.54 | 0.442 | 3.090 |
| 6 | 37.36 ± 0.63 | 0.394 | 3.217 |
| 7 | 33.22 ± 0.68 | 0.346 | 3.186 |
| 8 | 28.41 ± 0.57 | 0.295 | 3.035 |

n-max 2 beats its neighbours by 2.4 and 6.3 tokens/s against standard errors
near 0.7. Throughput falls away steeply past it, reaching 28.41 tokens/s at
depth 8, a 44% regression.

The three columns diverge. Acceptance falls monotonically with depth. Mean
accepted length rises, saturates near 3.2 around depth 6, then falls again.
Throughput peaks well before either turns over, because every rejected draft
token still costs target-model verification.

`nextn_predict_layers = 1` — the head natively predicts one token ahead — so
large draft depths have no architectural basis in this model.

## Stage 3: confidence gating

| p-min | Decode tokens/s | Acceptance | Mean accepted length |
|---:|---:|---:|---:|
| 0.0 | 50.84 ± 0.64 | 0.691 | 2.379 |
| 0.2 | 50.73 ± 0.63 | 0.692 | 2.380 |
| 0.3 | 50.74 ± 0.63 | 0.707 | 2.401 |
| 0.4 | 50.29 ± 0.57 | 0.724 | 2.406 |
| 0.5 | 49.63 ± 0.64 | 0.757 | 2.420 |
| 0.6 | 48.31 ± 0.53 | 0.798 | 2.436 |
| 0.7 | 46.70 ± 0.59 | 0.841 | 2.447 |
| 0.8 | 44.73 ± 0.61 | 0.886 | 2.463 |

0.0, 0.2, and 0.3 fall within 0.11 tokens/s of one another. Above 0.3 decode
declines monotonically, losing 6.1 tokens/s by 0.8. **Confidence gating does
not help.** Any value in the flat region is equivalent; 0.3 is retained.

This supersedes an earlier claim in this campaign that p-min 0.3 outperformed
0.0 at 46.0 against 45.8 tokens/s. That measurement came from unseeded
story-generation prompts and was run-to-run noise. It was retracted at the
time and is now closed out with ten seeded samples at each of eight settings.

## Stage 4: context allocation

| ctx-size | Decode tokens/s | Prefill tokens/s | Acceptance | Mean accepted length |
|---:|---:|---:|---:|---:|
| 32768 | 51.02 ± 0.64 | 707.5 | 0.7068 | 2.4008 |
| 49152 | 50.82 ± 0.64 | 704.8 | 0.7068 | 2.4008 |
| 65536 | 50.88 ± 0.64 | 704.2 | 0.7068 | 2.4008 |
| 98304 | 50.89 ± 0.63 | 704.3 | 0.7068 | 2.4008 |
| 131072 | 50.87 ± 0.64 | 702.3 | 0.7068 | 2.4008 |
| 163840 | 50.80 ± 0.63 | 703.9 | 0.7068 | 2.4008 |

Decode spans 0.22 tokens/s across a fivefold change in allocation. Acceptance
and mean accepted length are identical to four decimals, because KV allocation
size does not change what is computed for a given prompt.

**Context allocation is effectively free for decode.** There is no throughput
argument for reducing it. The only cost is VRAM, which is what constrains
Q6_K_XL to 65,536 tokens in the
[quantization comparison](qwen3-8-27b-quant-comparison.md).

## Positional acceptance

Survival curves, P(at least `k+1` draft tokens accepted at a verification
step), measured across the draft-depth sweep:

| n-max | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.811 | | | | | | | |
| 2 | 0.810 | 0.591 | | | | | | |
| 3 | 0.792 | 0.577 | 0.389 | | | | | |
| 4 | 0.786 | 0.584 | 0.394 | 0.270 | | | | |
| 5 | 0.765 | 0.544 | 0.359 | 0.238 | 0.183 | | | |
| 6 | 0.773 | 0.525 | 0.349 | 0.241 | 0.180 | 0.149 | | |
| 7 | 0.754 | 0.516 | 0.327 | 0.225 | 0.160 | 0.116 | 0.087 | |
| 8 | 0.740 | 0.467 | 0.280 | 0.195 | 0.127 | 0.097 | 0.071 | 0.057 |

At the production point, n-max 2, position 0 accepts 0.810 and position 1
accepts 0.591.

Within any single row, survival roughly halves per position: 0.81, 0.59, 0.39,
0.27, and under 0.20 by position 4. That decay explains why depth 2 wins.
Position 2 would land under 40% of the time while costing full verification at
every step.

**Rows are not directly comparable to one another.** Changing `n-max` changes
where verification cycles fall in the generated sequence, so the population of
position-0 proposals differs from row to row. The apparent decline in the
position-0 column across rows is therefore reported as observed and is **not**
evidence that deeper drafting degrades the first proposal. Establishing that
would require holding the verification-cycle positions fixed.

## What acceptance rate is good for

Both stages that moved acceptance moved it opposite to throughput.

- n-max 1 has the highest acceptance in the study, 0.811, and the
  second-slowest decode, 44.46 tokens/s.
- p-min 0.8 reaches 0.886 acceptance and is 12% slower than the flat region.

**Acceptance rate is not a proxy for speed and is an actively misleading
tuning target.** An operator optimizing for acceptance alone would select
either endpoint and land on a materially slower configuration.

The aggregate 0.7068 at the production point should be read as a diagnostic of
the proposer, not as a number that ordinary llama.cpp knobs can improve.
Raising it further would require a better draft head, and
[no usable alternative head exists for this model on AMD/Vulkan/GGUF](qwen3-8-27b-quant-comparison.md).

## Backend selection

The sweep was run on Vulkan because a ROCm/HIP comparison at the same commit
selected it for this workload. Both backends were built from `ad1de39` and
measured with the same harness on identical weights at ctx 163840.

| | Vulkan/RADV | ROCm/HIP | Delta |
|---|---:|---:|---:|
| Prefill, ten-sample mean | 705.99 tokens/s | **999.67 tokens/s** | +41.6% |
| Decode, ten-sample mean | **51.30 tokens/s** | 45.36 tokens/s | −11.6% |
| Draft acceptance | 0.716 | 0.719 | — |
| R9700 allocation | 29.784 GB | 31.359 GB | +1.58 GB |

ROCm processes prompts about 1.4× faster and generates about 12% slower, and
the decode deficit widens with prompt length — at 25,867 tokens it is 39.39
against 46.63. Acceptance is unchanged, so speculation is not the cause.

The better backend depends on workload shape. Long prompt and short answer
favours ROCm; short prompt and long answer favours Vulkan. This host serves an
agent gateway, which is predominantly the latter, so Vulkan is retained.

This is one ten-sample run per backend rather than repeated cold loads. The HIP
build also sets `GGML_HIP_MMQ_MFMA=ON`, and MFMA is a CDNA instruction while
`gfx1201` is RDNA4, so that flag may be inert here; build flags beyond the
shared source commit were not otherwise equalized.

### HIP requires explicit device isolation

The HIP build segfaulted on load, including at `--n-gpu-layers 0`:

```
resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
(chunked) is assigned to device ROCm0
Thread 1 "llama-server" received signal SIGSEGV
#5 ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul,1>> ... libggml-hip.so
#0-4 libamdhip64.so.7
```

Work was dispatched to `ROCm0`, the RX 7900 XT, rather than the R9700.
`HIP_VISIBLE_DEVICES=1` — after which the R9700 becomes `ROCm0` — loads
cleanly.

This is the same failure class recorded for Vulkan in the
[DeepSeek V4 record](deepseek-v4-flash-inference.md), where selecting the R9700
while both GPUs remained visible caused a drafter assertion and
`GGML_VK_VISIBLE_DEVICES=1` was required. **On this host, explicit device
isolation is mandatory for both backends.** Selecting a device is not
sufficient.

## Limitations

- One prompt shape, roughly 1,175 tokens, with 256-token generations. Optima
  may differ for other shapes; `ubatch` in particular is prefill-sensitive and
  a prefill-dominated workload would want its own sweep.
- Ten samples per configuration, standard error near 0.6 tokens/s. Effects
  below roughly 1.8 tokens/s are not detectable.
- Staged search, not a full grid. Interactions between parameters were not
  measured, and a joint optimum away from the staged path would be missed.
- Speed only. No quality measurement of any configuration.
- Positional survival rows are not comparable across `n-max` values, as noted
  above.
- Single host, single card, single build. Not portable without remeasurement.
