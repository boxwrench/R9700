# R9700 safe update gate & candidate promotion procedure

This document establishes the mandatory protocol for evaluating, testing, and promoting software, models, workflows, or runtime updates on the AMD Radeon AI PRO R9700 production system.

## Core rules

1. **Never update production in place first.** The active production checkout and environment must remain untouched until a candidate is fully verified.
2. **Newer is not automatically better.** A new upstream ComfyUI commit, ROCm version, Torch release, model revision, or quant is a **candidate**, not an automatic replacement.
3. **Preserve selected production modifications.** Any update that touches files with local production behavior (such as `LTX_GEMMA_MIN_LENGTH` in `comfy/text_encoders/lt.py` or pre-sampler Qwen offload in `comfy_extras/nodes_minimax_h3.py`) must be verified to retain that behavior.
4. **Compare against small canaries.** Updates must reproduce the reference canaries within acceptable bounds before promotion.
5. **If the candidate fails, leave production unchanged.** A rejected candidate is discarded or iterated in isolation; production continues operating without downtime or regressions.

---

## Update lifecycle

```
PRODUCTION (/ai/comfyui, active models, active env)
    ↓
[LEAVE UNTOUCHED]
    ↓
CANDIDATE CHECKOUT / WORKSPACE (isolated directory/branch)
    ↓
Apply candidate changes (ComfyUI commit, new model file, etc.)
    ↓
Re-apply / verify local production patches (production/patches/)
    ↓
Run static preflight (scripts/production-preflight.py)
    ↓
Run semantic workflow comparison (scripts/workflow-fingerprint.py)
    ↓
Execute isolated canary runs (production/canaries/)
    ↓
Evaluate results against reference metrics & structural criteria
    ↓
Decision Gate:
    ├── PASS (correct output, no regressions) ──> PROMOTE TO PRODUCTION
    └── FAIL / REGRESSION / CORRUPTION     ──> REJECT (Production remains intact)
```

---

## Step-by-step update procedure

### Step 1: Candidate isolation
Create a separate candidate workspace or environment. Do not perform `git pull`, `git checkout`, or `pip install` inside the active `/ai/comfyui` or `/ai/environments/comfyui-h3` paths.

```bash
# Example candidate setup (isolated)
git clone /ai/comfyui /ai/candidates/comfyui-candidate
cd /ai/candidates/comfyui-candidate
git checkout <candidate-commit-or-branch>
```

### Step 2: Check & re-apply local production patches
Check whether the candidate modifies any files governed by production patches:
- [`production/patches/ltx-gemma-floor-256.patch`](../production/patches/ltx-gemma-floor-256.patch)
- [`production/patches/h3-qwen-presampler-offload.patch`](../production/patches/h3-qwen-presampler-offload.patch)

If modified upstream, re-apply the patch or verify that equivalent functionality exists in the candidate.

### Step 3: Run static preflight
Execute the production preflight script against the candidate configuration:

```bash
python3 /ai/github/R9700/scripts/production-preflight.py
```

Verify that all markers, environment flags, and model hashes pass.

### Step 4: Semantic workflow validation
If the update includes modified ComfyUI workflow JSON files, compare their semantic fingerprints against the golden workflows in [`production/workflows/`](../production/workflows/):

```bash
python3 /ai/github/R9700/scripts/workflow-fingerprint.py <candidate-workflow.json> <expected-semantic-sha256>
```

Ensure no unintended parameter drifts (e.g. accidental sampler/scheduler changes, missing VAE connections, or unintended resolution drops) occurred.

### Step 5: Execute canary benchmarks
Run the exact canary inputs defined in [`production/canaries/`](../production/canaries/):

- `ltx25-canary.json`
- `h3-t2v-canary.json`
- `h3-i2v-canary.json`
- `h3-r2v-canary.json`
- `music3-canary.json`

Verify:
1. **Strict structural criteria:**
   - Valid, non-corrupted video/audio output produced
   - Correct frame count / duration and resolution
   - Zero NaN/Inf tensors or ROCm memory access faults
2. **Performance bounds:**
   - Generation latency within the defined warning threshold (no >15% regression against reference baseline)

### Step 6: Promotion checklist
Before promoting any candidate to production:

- [ ] Static preflight passes 100% (`PRODUCTION STATE: VERIFIED`)
- [ ] Local production modifications verified and active
- [ ] Model SHA-256 signatures match or new models are documented with exact hashes
- [ ] All target canaries execute successfully without errors
- [ ] Wall-time performance matches or improves upon production baseline
- [ ] `production/manifest.json` updated with new commit hash / model hashes
- [ ] Git commit created on R9700 tracking repository documenting the promotion

---

## What to do if an update fails

If any canary fails or performance regresses:
1. Immediately abort the candidate promotion.
2. Record the failure mechanism in `TROUBLESHOOTING.md` or the relevant experiment record.
3. Clean up candidate workspaces.
4. Production remains active and undisturbed.
