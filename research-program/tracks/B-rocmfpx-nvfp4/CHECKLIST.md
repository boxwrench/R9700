# Track B — Stage Checklist

Companion to [`PLAN.md`](PLAN.md). Tick a box only when the artifact exists in
the repository, not when the work "feels done".

---

## Preparation (pre-B0)

- [x] Upstream audited read-only at a recorded SHA — [`upstream-audit/2026-08-17-upstream-audit.md`](upstream-audit/2026-08-17-upstream-audit.md)
- [x] Staged protocol written — [`PLAN.md`](PLAN.md)
- [x] Shared benchmark contract, metrics, and run-record schema in place
- [x] Reproducibility scripts written — [`scripts/`](scripts/)
- [x] Clean ROCmFPX checkout obtained (`/ai/scratch/ROCmFPX-audit`)
- [ ] **Repository identity resolved** — `charlie12345/ROCmFPX` vs `ciru-ai/ROCmFPX` HEADs differ; needs a user decision
- [ ] Reproduction model identified and available locally

---

## B0 — Immutable snapshot

- [ ] `snapshot.md` contains **no** `PENDING` field
- [ ] ROCmFPX remote, branch, SHA recorded; working tree verified clean
- [ ] ROCm, HIP, Vulkan loader, RADV/Mesa, kernel, CPU, RAM recorded
- [ ] R9700 identity, `gfx` arch, VRAM, and **backend enumeration index** recorded
- [ ] Device isolation confirmed — the R9700 is not index 0 on this host
- [ ] Build directory, exact configure line, exact build command, compiler recorded
- [ ] **Emitted code-object arch verified as `gfx1201`** (gfx1200/gfx1201 silent-failure hazard)
- [ ] Model path, byte size, SHA256, GGUF arch, ftype, tensor count recorded
- [ ] Presence/absence of `output.scale` / `output.input_scale` recorded
- [ ] FFN bias presence recorded, and absence of LoRA confirmed (`build_ffn` asserts)
- [ ] Launch command and every behaviour-changing environment variable recorded
- [ ] Harness script path and repo SHA recorded
- [ ] **No performance conclusion drawn in this stage**

## B1 — Upstream reproduction

- [ ] Stock upstream at the pinned SHA; zero Track A patches applied
- [ ] Model loads without error
- [ ] lm_head scale confirmed **loaded and applied** (silent-degradation guard)
- [ ] Generation is coherent; greedy output deterministic across repeats
- [ ] PP throughput measured
- [ ] Serial / no-MTP decode measured
- [ ] Native MTP decode measured
- [ ] VRAM high-water mark recorded
- [ ] ≥5 repetitions; distribution reported, not a single figure
- [ ] Raw logs preserved under `raw/`
- [ ] Baseline written up under `reproduction/`
- [ ] *(if failed)* minimal failure documented as an upstream finding, and the track stopped rather than patched around

## B2 — Matched comparison

- [ ] Same physical R9700 for both arms
- [ ] Prompts, token counts, seeds, context, flash-attention, parallelism held constant
- [ ] Serial and MTP comparisons kept strictly separate
- [ ] PP / serial decode / MTP decode / MTP multiplier / VRAM reported per arm
- [ ] Acceptance metrics reported: $p_0$, joint-$p_1$, conditional-$p_1$, accepted drafts/round, committed tokens/round
- [ ] Confounds stated explicitly (format, bpw, backend, checkpoint provenance)
- [ ] No conclusion drawn from acceptance rate alone

## B3 — Cost decomposition

- [ ] Serial target cost measured
- [ ] MTP round cost measured
- [ ] Verification cost isolated
- [ ] Proposer cost isolated, head cost within it
- [ ] Attention cost isolated
- [ ] Major matmul families broken out by shape and type
- [ ] Residual reported and **left labelled UNEXPLAINED**
- [ ] Redundant-dequant structure investigated for NVFP4 specifically
- [ ] Scale-multiply (`GGML_OP_MUL`) dispatch count and cost measured
- [ ] Bottleneck named, with the evidence that locates it
- [ ] Track A bottlenecks treated as hypotheses, not inherited results

## B4 — Native tuning

- [ ] Backend A/B run natively (Vulkan vs HIP) — not inherited from Track A
- [ ] ROCmFPX quant routing / ROCmFP4 targets evaluated
- [ ] NVFP4 kernel selection and upstream toggles evaluated
- [ ] MTP `n-max` / `p-min` re-swept natively
- [ ] One variable per experiment
- [ ] **No Track A patches imported in this stage**
- [ ] Negative results recorded

## B5 — Integration candidates

- [ ] Stable, characterized Track B baseline exists first
- [ ] Draft-vocabulary trimming re-tested as a **fresh** experiment with its own holdout
- [ ] 32K vs 64K decided on Track B evidence, not inherited
- [ ] Methodology ports (acceptance metrics, round-cost accounting, harness, matched control) applied
- [ ] Backend-specific Track A patches **not** ported
- [ ] Each candidate carries its own A/B and its own decision

## B6 — Upstream contribution

- [ ] Finding written from [`../../upstream-rocmfpx/findings/FINDING-TEMPLATE.md`](../../upstream-rocmfpx/findings/FINDING-TEMPLATE.md)
- [ ] Minimal reproducer written from [`../../upstream-rocmfpx/reproducers/REPRODUCER-TEMPLATE.md`](../../upstream-rocmfpx/reproducers/REPRODUCER-TEMPLATE.md)
- [ ] Exact upstream SHA, hardware, toolchain, model hash, command captured
- [ ] Raw before/after preserved
- [ ] Correctness invariance demonstrated
- [ ] Component **and** end-to-end results both present
- [ ] Patch is minimal and self-contained — not a fork dump
