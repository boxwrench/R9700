# Research Program Status

*Last updated: 2026-08-17*

---

### TRACK A — Vulkan / GGUF Q4 / Native MTP
* **Status**: `PAUSED / PRESERVED`
* **Evidence Base**: Mature (Entries 1–17 logged)
* **Authoritative Log**: [`docs/qwen3-8-27b-experiment-log.md`](../docs/qwen3-8-27b-experiment-log.md)
* **Latest Accepted Entry**: **Entry 17** (`Tiny-N MUL_MAT + ADD fusion evaluation`)
* **Experimental / Unaccepted**: draft-vocabulary trimming ($64\text{K} / 32\text{K}$) — `PROMISING / UNVALIDATED`; no true unseen $n_{\text{max}}=2$ speculative holdout exists. See [Track A README](tracks/A-vulkan-q4-mtp/README.md#experimental-work-at-pause--draft-vocabulary-trimming).
* **Files**: unchanged. No experiment log, dataset, or script was moved, renamed, rewritten, or deleted while establishing Track B.

---

### TRACK B — ROCmFPX / Native NVFP4
* **Status**: `PREPARATION COMPLETE` — B0 not started
* **Upstream snapshot**: `f4b2c5a3edfd183274641094d0db0fcc8092c0ad` (`charlie12345/ROCmFPX`, branch `main`, fetched 2026-08-17T03:53:08Z)
* **Upstream audit**: [complete](tracks/B-rocmfpx-nvfp4/upstream-audit/2026-08-17-upstream-audit.md)
* **Staged protocol**: [`PLAN.md`](tracks/B-rocmfpx-nvfp4/PLAN.md) (B0–B6), [`CHECKLIST.md`](tracks/B-rocmfpx-nvfp4/CHECKLIST.md)
* **Checkout**: `/ai/scratch/ROCmFPX-audit` — clean, unmodified, full history
* **Build — Vulkan**: `PASS` (2 m 16 s, 0 errors, 114 binaries)
* **Build — HIP `gfx1201`**: `PASS` (0 errors; upstream's code-object verification confirmed `gfx1201`)
* **Non-model tests**: `PASS` — NVFP4 `MUL_MAT` 26/26 (Vulkan) and 41/41 (HIP), `MUL_MAT_ID` 73/73 on both, `GET_ROWS` 4/4 (Vulkan). [Details](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-build-and-nonmodel-tests.md)
* **Model reproduction (B1)**: `NOT STARTED` — **blocked**, no native NVFP4 Qwen3.8-27B model is present locally. [Inventory](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-model-inventory.md)
* **Performance measurement**: `NONE`. No Track B throughput, latency, or acceptance number exists.

**Headline preparation finding.** Upstream's own build docs state that
*"published benchmark numbers and regression guards assume Strix Halo /
gfx1151"*, and gfx1151 (RDNA3.5) takes a **different HIP code path** from
gfx1201 (RDNA4). NVFP4 on the R9700 is therefore effectively untested upstream.
Preparation established that the NVFP4 kernels are **numerically correct** on
gfx1201 on both backends — which is new information, and says nothing about speed.

---

### INTEGRATION
* **Status**: `BLOCKED` (awaiting a stable Track B baseline)
* **Rule**: no Track A optimization is imported or assumed portable without independent Track B A/B validation
* **First candidate when unblocked**: draft-vocabulary trimming — as a **fresh experiment**, carrying its Track A caveats, with 32K-vs-64K undecided

---

### UPSTREAM ROCmFPX LANE
* **Status**: `READY FOR FINDINGS`
* **Templates**: [finding](upstream-rocmfpx/findings/FINDING-TEMPLATE.md), [reproducer](upstream-rocmfpx/reproducers/REPRODUCER-TEMPLATE.md)
* **Findings filed**: none. One operational issue (HIP segfault on a mixed-arch host without `HIP_VISIBLE_DEVICES`) was diagnosed and recorded as a **local configuration matter**, not an upstream defect.

---

## Open questions requiring a user decision

1. **Which ROCmFPX is canonical?** The work package named `charlie12345/ROCmFPX`
   (HEAD `f4b2c5a`); the pre-existing local checkout at `/ai/github/ROCmFPX`
   points at `ciru-ai/ROCmFPX` (HEAD `0d313da`) and is shallow, detached, and
   dirty. Both are live. This audit used `charlie12345`, as specified.
2. **Which model does B1 reproduce against?** No native NVFP4 Qwen3.8-27B exists
   locally. Options: authorize the `RadixArk/Qwen3.8-27B-NVFP4` download, supply
   the provenance of the reported ~72 tok/s R9700 result, or accept a
   locally-converted NVFP4 checkpoint as an explicitly-labelled substitute.
3. **The ~72 tok/s R9700 figure is unsubstantiated** by anything in the audited
   tree — no gfx1201 benchmark, no model identification. It is not currently a
   reproduction target.
