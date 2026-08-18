# Agent instructions for this repository

If you are an AI agent helping a user configure, restore, benchmark, or update this R9700 system, treat the selected production state as intentional engineering output.

## Start here

Read, in order:

1. `docs/selected-production-configs.md`
2. `TROUBLESHOOTING.md`
3. `docs/comfyui-walltime-campaign-20260818.md`

If the user asks for the current fastest/selected setup, do not reconstruct it from upstream defaults. Use the selected files and settings recorded here.

## Do not silently upgrade production

A newer ComfyUI commit, model revision, quant, workflow, ROCm release, Torch release, or Triton release is a **candidate**, not an automatic replacement.

Before changing production:

- identify the currently selected ComfyUI commit
- verify the exact model filenames
- preserve local selected behavior
- benchmark the candidate separately where practical
- compare against the smallest relevant canary
- keep measured facts separate from mechanism hypotheses

If the candidate fails, leave production unchanged.

## Required selected behavior

### Shared

- `--disable-mmap` is selected. Do not remove it casually; the mmap-backed safetensors -> ROCm path produced catastrophic per-tensor load latency on this system.

### LTX 2.5

- INT8-ConvRot Gemma encoder
- INT8-ConvRot DiT
- `LTX_GEMMA_MIN_LENGTH=256`
- `tile_size=1280`
- reuse unchanged negative conditioning when graph caching allows

The Gemma floor behavior is locally modified/overridden and is vulnerable to ComfyUI updates.

### MiniMax H3

- single R9700 is the current selected default
- explicitly offload Qwen3-VL after conditioning and before sampling
- selected offload uses `comfy.model_management.unload_model_and_clones(clip.patcher)`

Do not infer from the historical dual-GPU experiment that single-GPU `--disable-smart-memory` should be removed.

### MiniMax Music 3

The production path is still the baseline path. Optimization is active.

Do not:

- force the existing ComfyUI FixedKV graph path on ROCm; installed flash decode is NVIDIA-only
- use naive `torch.compile` on the Qwen one-token backbone; the dynamic Python KV-cache index caused continual recompilation in the tested stack
- spend time on Music DiT quants for marginal gains while conditioning dominates wall time

## Symptom-first debugging

If the user reports a performance problem, consult `TROUBLESHOOTING.md` before changing the environment.

Especially:

- minutes-long model load -> check mmap path first
- H3 ~25-30 s sampling after changed prompt -> check Qwen residency before sampler
- LTX ~22 s short-prompt conditioning -> check Gemma minimum floor
- near-zero `rocm-smi` utilization at high power -> do not diagnose CPU fallback from that counter alone
- Music ~20 s conditioning -> inspect AR loop, not the ~3.6 s DiT

## Benchmark discipline

Use small discriminators. One or two clean A/B runs are usually enough for an engineering decision.

Record:

```text
TASK:
HYPOTHESIS:
CONTROL:
CHANGE:
RESULT:
DECISION:
NEXT:
```

Use statuses:

- CONFIRMED
- SELECTED
- EXPERIMENTAL
- REJECTED
- UNTESTED
- BLOCKED

Do not rewrite failed hypotheses out of the record; they prevent repeat work.

## Update safety

Until the automated production manifest/preflight system exists:

- do not update the production ComfyUI checkout in place without preserving the selected state
- preserve local source changes/patches before pulling upstream
- do not overwrite golden workflows simply because an upstream workflow changed
- compare candidate model hashes/filenames and workflow semantics before promotion

If a requested update would overwrite a known-good local modification, tell the user exactly what would be lost before proceeding.
