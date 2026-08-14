# DeepSeek V4 Flash: 32K local inference on one R9700

## Outcome

DeepSeek V4 Flash UD-Q4_K_XL plus the Q8_0 DSpark drafter ran locally with a
32,768-token context using one AMD Radeon AI PRO R9700 and system DDR5. The
selected profile passed exact retrieval from a 24,603-token prompt, scored 5/5
on a deterministic sanity check, and averaged **8.14 decode tokens/s** over two
256-token generations.

The RX 7900 XT was excluded from inference. Restricting Vulkan visibility—not
merely selecting a target device—was necessary for target and drafter to share
one device namespace.

## Hardware and software

- AMD Radeon AI PRO R9700, 32 GB VRAM, `gfx1201`
- AMD Radeon RX 7900 XT, excluded from inference
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads
- 192 GB installed DDR5; 188 GiB reported by Linux
- llama.cpp Vulkan/RADV backend, commit `7b13a84`
- Target: [`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), `UD-Q4_K_XL`
- Drafter: Unsloth DeepSeek V4 Flash DSpark `Q8_0`

The downloaded target consisted of five GGUF shards, approximately 155 GB in
decimal units. The drafter was approximately 10.9 GB.

## Selected configuration

```bash
GGML_VK_VISIBLE_DEVICES=1 llama-server \
  --model DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
  --model-draft dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  --device Vulkan0 \
  --gpu-layers all \
  --n-cpu-moe 41 \
  --spec-draft-ngl all \
  --spec-draft-n-cpu-moe 0 \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  --ctx-size 32768 \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 \
  --cache-type-v-draft q8_0 \
  --kv-offload \
  --flash-attn on \
  --fit off \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --threads 16
```

On this host, physical Vulkan index 1 is the R9700. After
`GGML_VK_VISIBLE_DEVICES=1`, it becomes the only visible device and is named
`Vulkan0` inside llama.cpp.

Placement was:

- The first 41 target MoE expert layers in system DDR5
- Remaining target experts and dense/shared tensors on the R9700
- Q8_0 DSpark drafter on the R9700
- Target and draft Q8_0 KV caches on the R9700

The selected full run used 32.156 GB of 34.209 GB reported VRAM, leaving about
2.05 GB. This is suitable for one slot but not additional parallel slots or
another target expert layer without a new fit test.

## Methodology

The short screening pass used one fixed code prompt, temperature 0, seed 12345,
and 128 generated tokens per profile. It compared context allocation, KV type,
KV location, and target expert placement while keeping the target quant and
DSpark drafter constant.

The selected profile then received three gates:

1. A deterministic five-answer JSON sanity check
2. Exact secret retrieval from a 24,603-token prompt, approximately 75% of the
   configured 32K context
3. Two 256-token generations with llama.cpp timing and draft-acceptance logging

The long-context prompt placed `R9700-CONTEXT-32768-PASS` near the beginning and
requested that exact marker at the end. The model returned it exactly. Prefill
took roughly 7.5 minutes. Early 2K-token batches exceeded 200 prompt tokens/s;
aggregate throughput declined with sequence length as attention work increased.

## Results

| Profile | Decode t/s | Acceptance | Quality | VRAM | Outcome |
|---|---:|---:|---:|---:|---|
| 16K Q8 GPU KV, CPU MoE 41 | 7.62 | 90.2% | 5/5 | 30.89 GB | Passed |
| 32K Q8 GPU KV, CPU MoE 41 | **8.32** | 90.2% | 5/5 | 29.78 GB | Promoted |
| 32K Q8 GPU KV, CPU MoE 42 | 7.37 | 90.2% | 5/5 | 26.35 GB | More margin, slower |
| 32K Q4 GPU KV, CPU MoE 41 | 7.99 | 86.7% | 5/5 | 29.69 GB | Slower, lower acceptance |
| 32K Q8 system-RAM KV, CPU MoE 41 | 5.86 | 84.3% | 5/5 | 29.52 GB | Capacity fallback |
| 32K mixed Q8-K/Q4-V | n/a | n/a | n/a | n/a | Unsupported by model |

The promoted profile's full two-run mean was **8.1372 decode tokens/s**. Its
individual samples were 8.1709 and 8.1035 tokens/s. Draft acceptance was 78.855%
for that code prompt. Acceptance is workload-dependent; the shorter screening
prompt accepted 90.196%.

Q8/Q8 GPU KV was selected because it passed long-context retrieval, retained the
highest tested cache precision, and outperformed both Q4 GPU KV and Q8 system-RAM
KV. Moving one more target expert layer to DDR5 created more VRAM margin but cost
about 11% decode speed in the screening run.

## DSpark and Vulkan device isolation

Two initial DSpark loads failed with this assertion while both physical Vulkan
GPUs remained visible, even though the target specified the R9700:

```text
pre-allocated tensor (output.weight) in a buffer (Vulkan1) that cannot run the operation (NONE)
```

DSpark omits its own embeddings/output head and borrows those tensors from the
target. Hiding the RX 7900 XT with `GGML_VK_VISIBLE_DEVICES=1` gave both contexts
the same single-device namespace and resolved the assertion. No separate
`--spec-draft-device` was used, consistent with the
[Unsloth DSpark guide](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark/README.md).

Mixed Q8-K/Q4-V cache initialization also failed. llama.cpp reported that this
DeepSeek V4 model does not support different K and V cache types, so only
matching Q8/Q8 and Q4/Q4 pairs are valid for this build.

## Recovery and resumability

Each profile had a dedicated directory containing status, server log, benchmark
TSV, normalized metrics, quality result, context result, VRAM snapshots, and
system-memory snapshots. A profile was marked complete only after its requested
probes were saved; rerunning skipped completed profiles.

After every profile, the runner validated that the saved PID belonged to the
managed DeepSeek llama-server, sent TERM, waited up to 15 seconds, and used KILL
only if required. It then verified R9700 VRAM release with ROCm SMI. A scoped
`rocm-smi -d 1 --gpureset` path was available but disabled by default and guarded
against any process holding the R9700 render node. No GPU reset was needed in
this campaign.

The normalized measurements are in
[`data/experimental/deepseek-v4-flash.tsv`](../data/experimental/deepseek-v4-flash.tsv),
and the compact reproduction matrix is in
[`experiments/deepseek-v4-flash/`](../experiments/deepseek-v4-flash/README.md).

## Limitations

- Screening measurements are single runs; the selected profile received two
  decode samples, not a publication-grade variance study.
- The 5/5 probe is a regression alarm, not a comprehensive model-quality eval.
- DSpark acceptance and speedup depend strongly on prompt type.
- Current llama.cpp DSpark with quantized targets has a reported greedy-output
  mismatch, so bit-for-bit equality was not treated as the quality criterion.
- The profile depends on current Vulkan device ordering. Reconfirm that host
  index 1 is the R9700 before reuse on a changed system.

## Sources

- [Unsloth DeepSeek V4 serving guide](https://unsloth.ai/docs/models/deepseek-v4)
- [Unsloth DSpark llama.cpp guide](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark/README.md)
- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
