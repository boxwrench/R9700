# Qwen3.8-27B on one R9700: Q4_K_XL versus Q6_K_XL

## Outcome

Two Unsloth dynamic quantizations of Qwen3.8-27B were measured on a single AMD
Radeon AI PRO R9700 under llama.cpp's Vulkan backend, with the model's built-in
MTP head driving speculative decoding. Both quantizations were fully offloaded.

| | UD-Q4_K_XL | UD-Q6_K_XL |
|---|---:|---:|
| Weights on disk | 17.923 GB | 25.924 GB |
| Context size | 163,840 | 65,536 |
| R9700 allocation | 29.784 GB | 30.409 GB |
| Decode, ten-sample mean | **51.30 tokens/s** | **39.08 tokens/s** |
| Decode standard deviation | 1.48 | 1.18 |
| Prefill, ten-sample mean | **705.99 tokens/s** | **700.05 tokens/s** |
| MTP draft acceptance | 71.6% | 70.4% |

Q4_K_XL is the selected profile for routine local serving on this workstation.
Q6_K_XL costs 23.8% of decode throughput and forces a context reduction, in
exchange for weight precision that this benchmark does not attempt to score.

**This document measures speed, not answer quality.** No perplexity, no
benchmark suite, and no side-by-side output grading were run. A reader deciding
between these quantizations on quality grounds will not find that evidence here.

## Hardware and software

- AMD Radeon AI PRO R9700, 32 GB VRAM, `gfx1201`, Vulkan device index 1
- AMD Radeon RX 7900 XT and a `gfx1036` iGPU present but not selected
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads; 188 GiB host RAM
- Ubuntu 24.04.4 LTS; ROCm 7.2.1
- llama.cpp Vulkan/RADV backend, build `b10448`, commit `ad1de39`
  (`GGML_VULKAN=ON`, `GGML_HIP=OFF`; every measurement here is Vulkan, not ROCm/HIP)
- [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), `UD-Q4_K_XL` and `UD-Q6_K_XL`
- Vision projector: `mmproj-F16.gguf`, 927,607,488 bytes, shared by both

RADV reports itself as a non-conformant Vulkan implementation. That warning is
present on every run here and is not treated as an error.

## Model architecture

GGUF metadata reports `general.architecture = qwen35` with 65 blocks and a
`full_attention_interval` of 4. Only every fourth block is full attention; the
remainder are Gated DeltaNet layers with SSM-style state. `head_count_kv` is 4
with key and value length 256. `nextn_predict_layers = 1` is the built-in MTP
head used for speculation — there is no separate drafter file.

This shapes the KV budget. With roughly 16 full-attention layers holding f16 K
and V, the KV cache costs on the order of 64 KiB per token, an order of
magnitude below a conventional 65-block transformer. That is why 163,840 tokens
fit alongside Q4 weights at all. The figure is derived from metadata, not
measured directly; the VRAM column above is the measured quantity, and it runs
somewhat below a naive weights-plus-KV-plus-projector sum.

llama.cpp confirms the fused kernels are active at full offload:

```
Flash Attention enabled
fused Gated Delta Net (autoregressive) enabled
fused Gated Delta Net (chunked) enabled
Lightning Indexer enabled
DeepSeek V4 HC pre/comb/post enabled
```

The chunked Gated DeltaNet path is the one that matters for prefill. An earlier
reading that it was unsupported came from a diagnostic run pinned to
`--n-gpu-layers 1`, which placed block 0 on the CPU. It is enabled under the
production configuration.

## Selected configuration

The host runs `llama-server` in router mode over `/ai/models`, with per-model
overrides in an INI preset. The full preset is at
[`experiments/qwen3-8-27b/llama-models-preset.ini`](../experiments/qwen3-8-27b/llama-models-preset.ini).

```
[Qwen3.8-27B-UD-Q4_K_XL]
ctx-size = 163840
threads = 6
threads-batch = 6
flash-attn = on
cache-type-k = f16
cache-type-v = f16
kv-unified = 1
load-mode = none
split-mode = none
n-cpu-moe = 0
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-p-min = 0.3
```

The Q6 section is identical apart from `ctx-size = 65536`.

Router invocation:

```bash
llama-server \
  --models-dir /ai/models \
  --models-preset llama-models-preset.ini \
  --models-max 1 \
  --device Vulkan1 \
  --n-gpu-layers 999 \
  --parallel 1 \
  --jinja \
  --reasoning-preserve \
  --host 0.0.0.0 --port 8080
```

Two precedence rules govern this setup, both from
`tools/server/server-models.cpp`. Presets merge cached, then models-dir, then
custom; the router's own command-line arguments are then overlaid on top of
every per-model preset. A flag on the router command line therefore **wins over
the same key in any preset**. `--ctx-size` was deliberately removed from the
router invocation so that each model could carry its own context size.

## Results

Prompt lengths are the server's reported `prompt_n`. Each bucket is one
generation capped at 256 tokens, temperature 0.6, thinking disabled.

| Bucket | Prompt tokens | Q4 prefill | Q4 decode | Q4 accept | Q6 prefill | Q6 decode | Q6 accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 153 | 374.6 | 48.09 | 62.2% | 372.3 | 38.74 | 68.4% |
| medium | 1,175 | 702.5 | 49.39 | 66.1% | 699.5 | 39.34 | 70.1% |
| long | 6,236 | 806.7 | 49.24 | 68.9% | 797.8 | 38.21 | 68.9% |
| very long | 25,868 | 742.0 | 46.63 | 70.3% | 735.1 | 39.36 | 77.5% |
| vision | 84 | — | 53.53 | — | — | 37.66 | — |

Ten cache-busted generations on the medium bucket:

| | Q4_K_XL | Q6_K_XL run 1 | Q6_K_XL run 2 |
|---|---:|---:|---:|
| Prefill mean | 705.99 | 699.62 | 700.05 |
| Prefill stdev | 1.98 | 1.90 | 1.21 |
| Decode mean | 51.30 | 39.07 | 39.08 |
| Decode stdev | 1.48 | 1.19 | 1.18 |
| Draft acceptance | 0.716 | 0.704 | 0.704 |

The two Q6 runs were separate cold loads with a full model eviction between
them. Decode means agree to 0.01 tokens/s and the acceptance counts are
identical at 1473 of 2093 drafted tokens, the latter because seeds were fixed.

### Prefill is unchanged; decode is not

Prefill differs by 0.84% between quantizations, inside the run-to-run spread.
Prompt processing is compute-bound here, so weight precision does not move it.

Decode falls 23.8%. Decode is memory-bandwidth-bound, and Q6 weights are 1.446×
larger. Pure bandwidth scaling predicts a ratio of 0.691; the measured ratio is
0.762. The gap is consistent with speculative decoding amortizing some weight
reads across accepted draft tokens, but that mechanism was not isolated.

### Acceptance does not explain the difference

MTP acceptance is 71.6% at Q4 and 70.4% at Q6 — effectively unchanged. The draft
head does not degrade measurably at the lower weight precision, so the decode
cost of Q6 buys weight precision only; it returns nothing through speculation.

Acceptance rises with prompt length in both quantizations, from roughly 62–68%
at 153 tokens to 70–78% at 25,868 tokens. Longer context appears to make the
MTP head's continuations more predictable.

### The MTP head is quantized differently between the two files

Both files declare `nextn_predict_layers = 1` and carry one extra block,
`blk.64`, with identical tensor shapes. Unsloth quantized that block
differently in each. Dumped with `gguf-py/gguf/scripts/gguf_dump.py`:

| Tensor | Parameters | UD-Q4_K_XL | UD-Q6_K_XL |
|---|---:|---|---|
| `blk.64.attn_q.weight` | 62,914,560 | Q5_K | Q8_0 |
| `blk.64.attn_k.weight` | 5,242,880 | Q5_K | Q8_0 |
| `blk.64.attn_v.weight` | 5,242,880 | Q6_K | Q8_0 |
| `blk.64.attn_output.weight` | 31,457,280 | Q5_K | Q8_0 |
| `blk.64.ffn_down.weight` | 89,128,960 | Q5_K | Q8_0 |
| `blk.64.ffn_gate.weight` | 89,128,960 | IQ4_XS | Q6_K |
| `blk.64.ffn_up.weight` | 89,128,960 | Q5_K | Q6_K |
| `blk.64.nextn.eh_proj.weight` | 52,428,800 | Q6_K | Q8_0 |
| norms (5 tensors) | 20,736 | F32 | F32 |
| **total** | **424,699,392** | mixed 4–6 bit | uniform 6–8 bit |

The draft head is a 425M-parameter attention-plus-FFN block with a 52.4M
`eh_proj`, sharing the base model's embedding and output head. It is a
substantial drafter in its own right, which is why MTP performs here without a
separate checkpoint.

Q6's head is uniformly higher precision — every tensor Q8_0 or Q6_K, against
Q4's Q5_K/Q6_K mix with `ffn_gate` down at IQ4_XS. Measured acceptance was
71.6% for Q4 and 70.4% for Q6.

With 2,076 and 2,093 drafted tokens, the binomial standard error on each rate is
about one point, so a 1.2-point gap is **not distinguishable from zero**. The
supportable claim is no detectable difference, not that Q4's head is better.
Even so, that is a useful negative: quantizing this MTP head down to 4–5 bits
does not measurably degrade draft acceptance, and the extra precision in the Q6
file buys nothing on this workload. Sourcing a higher-precision MTP head is
therefore not an available tuning lever.

## Reasoning effort: the production serving mode

The quantization comparison above was run with `enable_thinking` disabled, which
holds conditions constant across both quantizations but is **not** how this host
serves the model. The hermes gateway sends `reasoning_effort: low`, so the model
emits thinking tokens before its answer. Q4_K_XL was re-measured in that mode.

| Q4_K_XL, ten-sample mean | Thinking disabled | `reasoning_effort: low` |
|---|---:|---:|
| Decode | 51.30 tokens/s | **52.93 tokens/s** |
| Decode stdev | 1.48 | 1.17 |
| Prefill | 705.99 tokens/s | 686.44 tokens/s |
| MTP draft acceptance | 71.6% | **74.8%** |

Per bucket, with reasoning low:

| Bucket | Prompt tokens | Prefill | Decode | Accept |
|---|---:|---:|---:|---:|
| short | 181 | 321.5 | 54.92 | 75.7% |
| medium | 1,173 | 699.8 | 54.96 | 77.0% |
| long | 6,234 | 822.1 | 52.74 | 74.1% |
| very long | 25,866 | 755.4 | 51.28 | 80.0% |
| vision | 82 | — | 47.84 | — |

Enabling reasoning **raises** decode throughput by 3.2% and lifts MTP acceptance
by 3.2 points. The most plausible reading is that chain-of-thought text is more
formulaic than final prose, so the MTP head predicts it better and more drafted
tokens survive. That mechanism was not isolated and this is a single ten-sample
run, so treat it as an observation rather than a characterized effect.

**Higher decode throughput here does not mean faster answers.** Tokens per
second counts thinking tokens, and reasoning mode generates additional tokens
before the response begins. Time to a finished answer is longer even though the
token rate is higher. This benchmark caps generation at 256 tokens and does not
measure time-to-first-content or total tokens per completed answer, so it cannot
quantify that tradeoff.

Vision decode moves the other way, 53.53 to 47.84 tokens/s. With only one image
prompt at 82 tokens this is a single observation, not a characterized result.

## Speculative decoding tuning

`--spec-draft-p-min` was swept at 0.0, 0.2, 0.3, and 0.75 on Q4.

An initial sweep on unseeded story-generation prompts suggested 0.3 was a win at
46.0 tokens/s against 45.8. Repeating with `seed` and `temperature` fixed showed
0.0, 0.2, and 0.3 all landing near 45 tokens/s. **That apparent gain was
run-to-run noise.** The 0.3 setting is retained as a reasonable midpoint, not as
a demonstrated optimum.

The 0.75 setting is a genuine and instructive result: acceptance rose to 80.7%
while throughput fell to 38.6 tokens/s. A high acceptance rate is not a proxy
for speed. Raising `p-min` makes the drafter propose only where it is confident,
which accepts more of what it drafts but drafts less often.

`--spec-draft-n-max` is set to 2. `nextn_predict_layers = 1` means the head
natively predicts one token ahead, so large draft depths have no basis here.

## Negative and null results

These are recorded because they cost time and would otherwise be repeated.

**Router VRAM overcommit.** `--models-max` defaults to 4 and evicts by model
*count*, never by VRAM. With four 17–26 GB models in `/ai/models`, the router
loaded Q6 and Q4 concurrently on a 32 GB card, reaching 34.004 GB of 34.209 GB.
Q4 weights spilled to host memory over PCIe and decode collapsed to **3.85
tokens/s**, roughly a thirteenfold regression, with nothing in the logs
identifying the cause. `--models-max 1` restores LRU eviction. Any single-card
multi-model router is exposed to this; the row is retained in the TSV as
`invalid`.

**Prompt-cache contamination in benchmarking.** Varying only the sampler `seed`
between samples leaves the prompt identical, so llama.cpp serves it from cache
and reports prefill over the one or two genuinely new tokens. This produced a
101 tokens/s prefill mean with a standard deviation of 200 across ten samples.
Prefixing each sample with a unique string drops the standard deviation to
roughly 2. Decode figures were unaffected. Any prefill number in this repository
predating that fix should be discarded.

**Batch and micro-batch sizing.** `batch-size 8192` with `ubatch-size 4096`,
carried over from a Windows configuration, cost roughly 3 GB of VRAM and
produced no measurable throughput change against the defaults of 2048 and 512.
At `--parallel 1` there are no concurrent slots to batch across. The VRAM was
reclaimed as context.

**Host scheduling flags.** `--poll 100` with `--prio 2` measured 45.2 and 45.4
tokens/s against 45.82 and 45.7 for the defaults. Null result; reverted.

**GPU clock and power state.** Not measurable. Every relevant sysfs entry on
this card — `power_dpm_force_performance_level`, `pp_dpm_sclk`, `pp_dpm_mclk`,
and the hwmon frequency and power-cap nodes — returns `Device or resource busy`,
and `rocm-smi --showperflevel` reports `unknown`. ROCm 7.2.1 does not expose
power management for this `gfx1201` part. Whether the card sustains peak clocks
under these workloads is **unverified**, and clock-locking could not be tested
as a tuning lever.

**Alternative draft heads.** No llama.cpp-usable EAGLE3 or dflash checkpoint
exists for this model. `Ex0bit/Qwen3.6-27B-PRISM-EAGLE3` is SGLang-only with no
GGUF conversion. Four Hugging Face repositories advertising MTP optimization
(`Youssofal/MTPLX`, `esatapedico/NVFP4-MTP-GGUF`, `lued/INT8-W8A16-MTP`,
`sakamakismile/MTP-NVFP4`) are re-quantizations that leave the stock MTP head
untouched, and all target MLX, CUDA Blackwell, or vLLM. None is usable on
AMD/Vulkan/GGUF. Pairing a conventional small draft model has been reported to
regress on the closely related Qwen3.6-27B and was not pursued.

## Method

The harness is [`experiments/qwen3-8-27b/bench.py`](../experiments/qwen3-8-27b/bench.py).

```bash
# Quantization comparison: thinking disabled, conditions held constant.
NONCE=run1 MODEL=Qwen3.8-27B-UD-Q4_K_XL python3 bench.py

# Production serving mode, matching the hermes gateway.
NONCE=low1 REASONING=low MODEL=Qwen3.8-27B-UD-Q4_K_XL python3 bench.py
```

It drives the OpenAI-compatible `/v1/chat/completions` endpoint and reads
`prompt_per_second`, `predicted_per_second`, `draft_n`, and `draft_n_accepted`
from the server's `timings` object. Prefill and decode rates are therefore the
server's own measurements, not wall-clock timings taken by the harness.

Filler text is synthetic repeated English prose. Each bucket prompt and each of
the ten samples carries a unique leading string so no request is served from
prompt cache. Generation is capped at 256 tokens with temperature 0.6, fixed
seeds, and `enable_thinking` disabled.

Both quantizations were driven through the same router process with
`--models-max 1`, so exactly one model was resident during any measurement.

## Limitations

- Speed only. No quality, perplexity, or accuracy measurement was attempted.
- The Q4-versus-Q6 comparison holds `enable_thinking` disabled on both sides.
  Q6 was **not** measured under `reasoning_effort: low`, so the reasoning
  results are a Q4-only observation and should not be assumed to transfer.
- Token rate under reasoning counts thinking tokens. Time to a finished answer,
  time to first content, and tokens per completed answer were not measured.
- The two quantizations run at different context sizes because Q6 weights leave
  too little VRAM for 163,840 tokens. Decode is nearly flat across prompt length
  in both, so the comparison is not obviously distorted, but it is not a
  context-matched test.
- Ten samples on one prompt bucket. The per-bucket rows are single generations
  and should be read as indicative.
- The vision rows use one 256×256 synthetic PNG at 84 prompt tokens. They
  establish that the projector loads and generates; they are not an image
  workload benchmark, and their prefill is not comparable to the text buckets.
- Decode standard deviations near 1.2–1.5 tokens/s mean differences below
  roughly 3 tokens/s are not resolvable by this harness.
- Single host, single card, single llama.cpp build. Nothing here is portable to
  another driver stack without remeasurement.
