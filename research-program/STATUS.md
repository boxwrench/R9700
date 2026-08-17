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
* **Status**: `B1 REPRODUCED — UNIFORM PRIMARY`. Stopped at the B1 boundary per the work package.
* **Upstream snapshot**: `f4b2c5a3edfd183274641094d0db0fcc8092c0ad` (`charlie12345/ROCmFPX`, branch `main`, fetched 2026-08-17T03:53:08Z)
* **Upstream audit**: [complete](tracks/B-rocmfpx-nvfp4/upstream-audit/2026-08-17-upstream-audit.md)
* **Staged protocol**: [`PLAN.md`](tracks/B-rocmfpx-nvfp4/PLAN.md) (B0–B6), [`CHECKLIST.md`](tracks/B-rocmfpx-nvfp4/CHECKLIST.md)
* **Checkout**: `/ai/scratch/ROCmFPX-audit` — clean, unmodified, full history
* **Build — Vulkan**: `PASS` (2 m 16 s, 0 errors, 114 binaries)
* **Build — HIP `gfx1201`**: `PASS` (0 errors; upstream's code-object verification confirmed `gfx1201`)
* **Non-model tests**: `PASS` — NVFP4 `MUL_MAT` 26/26 (Vulkan) and 41/41 (HIP), `MUL_MAT_ID` 73/73 on both, `GET_ROWS` 4/4 (Vulkan). [Details](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-build-and-nonmodel-tests.md)
* **Model**: `RadixArk/Qwen3.8-27B-NVFP4` acquired, converted, and quantized. Mixed 28.2 GB / 1252 tensors / 193 NVFP4; uniform 15.5 GB / 4.55 BPW. [Inventory](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-model-inventory.md)
* **Gate 1 — lm_head scale**: `SOURCE VERIFIED` + `RUNTIME VERIFIED`
* **Gate 2 — model identity**: `PASSED`; all 193 original NVFP4 tensors bit-exact through `llama-quantize`. [Gates](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-b1-gates.md)
* **B1 measurement**: 4 arms × (1 warmup + 7 reps) = 32 runs, all `rc=0`. [Results](tracks/B-rocmfpx-nvfp4/reproduction/2026-08-17-b1-results.md)

| decode tok/s | MIXED | UNIFORM |
|---|---|---|
| serial | 20.32 ± 0.03 | **27.33 ± 0.02** |
| MTP (n_max=4) | 30.71 ± 0.04 | **37.26 ± 0.09** |

**Headline finding.** Upstream's own build docs state that *"published benchmark
numbers and regression guards assume Strix Halo / gfx1151"*, and gfx1151
(RDNA3.5) takes a **different HIP code path** from gfx1201 (RDNA4). NVFP4 on the
R9700 was therefore effectively untested upstream; B1 is an original measurement,
not a confirmation. **Uniform NVFP4 conversion is worth +34.5% serial and +21.3%
MTP decode on gfx1201, and saves 10.5 GB** — both deltas exceed the spread of
either arm by ≥72×.

**Not target-equivalent.** Greedy MTP output diverges from serial decoding, and
`--spec-mtp-strict-qwen` does not close the gap on Vulkan. The cause is
`UNRESOLVED`. The MTP numbers must not be presented as "same output, faster".

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
2. ~~**Which model does B1 reproduce against?**~~ **RESOLVED** — the user
   authorized the download. `RadixArk/Qwen3.8-27B-NVFP4` is confirmed to be the
   same checkpoint upstream tested (193 NVFP4 tensors, matching `5290625`).
3. **The ~72 tok/s R9700 figure is still unsubstantiated.** B1 did not reproduce
   it; the fastest measured arm is 37.26 tok/s. Nothing measured explains a 2×
   gap. If that figure is real, its configuration is still unknown.
4. **Should the Vulkan MTP/serial greedy divergence be filed upstream?** It is
   measured and reproducible but its cause is not isolated, so it would be an
   observation rather than a bug report. A draft is staged at
   [`2026-08-17-gfx1201-result-table.md`](tracks/B-rocmfpx-nvfp4/upstream-audit/2026-08-17-gfx1201-result-table.md).
   **Nothing has been submitted or pushed.**
