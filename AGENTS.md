# Agent instructions for this repository

If you are an AI agent helping a user configure, restore, benchmark, or update this R9700 system, treat the selected production state as intentional engineering output.

## Start here

Execute and read, in order:

1. Run the automated preflight:
   ```bash
   python3 scripts/production-preflight.py
   ```
2. Read `production/manifest.json` for exact machine-checked commit, model hashes, and workflow fingerprints.
3. Consult `TROUBLESHOOTING.md` before diagnosing any symptom.
4. Read `docs/selected-production-configs.md` and `docs/update-gate.md`.

If the user asks to **"restore my optimized R9700 setup"**, **"set up H3"**, or **"run the known-good LTX workflow"**, do not reconstruct from upstream defaults. Use the selected files, golden workflows (`production/workflows/`), and settings recorded here.

---

## Do not silently upgrade production

A newer ComfyUI commit, model revision, quant, workflow, ROCm release, Torch release, or Triton release is a **candidate**, not an automatic replacement.

Before changing production:
- Follow the update gate in [`docs/update-gate.md`](docs/update-gate.md).
- Never update production in place; use a candidate workspace.
- Identify the currently selected ComfyUI commit from `production/manifest.json`.
- Preserve local production modifications in [`production/patches/`](production/patches/).
- Benchmark candidates against the canaries in [`production/canaries/`](production/canaries/).
- Verify with `scripts/production-preflight.py` and `scripts/workflow-fingerprint.py`.

If the candidate fails or regresses, leave production unchanged.

---

## Required selected behavior

### Shared
- `--disable-mmap` is selected. Do not remove it; the mmap-backed safetensors -> ROCm path produced catastrophic per-tensor load latency on this system.

### LTX 2.5
- INT8-ConvRot Gemma encoder
- INT8-ConvRot DiT
- `LTX_GEMMA_MIN_LENGTH=256` (overrides default 1024; preserved in [`production/patches/ltx-gemma-floor-256.patch`](production/patches/ltx-gemma-floor-256.patch))
- `tile_size=1280`
- reuse unchanged negative conditioning when graph caching allows

### MiniMax H3
- single R9700 is the selected default
- explicitly offload Qwen3-VL after conditioning and before sampling (preserved in [`production/patches/h3-qwen-presampler-offload.patch`](production/patches/h3-qwen-presampler-offload.patch))
- selected offload uses `comfy.model_management.unload_model_and_clones(clip.patcher)`

### MiniMax Music 3
- Characterization and optimization are **COMPLETE FOR THIS CAMPAIGN PASS**.
- Do not force the FixedKV graph path (NVIDIA-only `flash_attention_decode` dependency).
- Do not use naive `torch.compile` on the Qwen backbone (dynamic integer KV-cache index causes continuous Dynamo recompilation).
- Do not expect major speedups from `torch.compile` on RVQ (~1.03x speedup, ~0.94% wall-time win; sequential discrete passes are host-dispatch bound).

---

## Symptom-first debugging

If the user reports a performance problem, consult `TROUBLESHOOTING.md` before changing the environment.

Especially:
- minutes-long model load -> check mmap path first
- H3 ~25-30 s sampling after changed prompt -> check Qwen residency before sampler
- LTX ~22 s short-prompt conditioning -> check Gemma minimum floor
- near-zero `rocm-smi` utilization at high power -> do not diagnose CPU fallback from that counter alone
- Music ~20 s conditioning -> conditioning is AR loop bound; optimization is complete for this pass

---

## Benchmark discipline

Use small discriminators. One or two clean canary runs are usually enough for an engineering decision.

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
