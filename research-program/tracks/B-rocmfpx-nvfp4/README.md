# Track B — ROCmFPX / Native NVFP4

## Status: PREPARATION COMPLETE — BASELINE NOT YET ESTABLISHED

**Start here:**

| Document | What it is |
|---|---|
| [`upstream-audit/2026-08-17-upstream-audit.md`](upstream-audit/2026-08-17-upstream-audit.md) | Read-only source audit of upstream at `f4b2c5a`, answering ten structural questions, with every claim tagged SOURCE FACT / COMMIT CLAIM / INFERRED / UNKNOWN. |
| [`PLAN.md`](PLAN.md) | The staged protocol, B0 through B6, with entry and exit conditions. |
| [`CHECKLIST.md`](CHECKLIST.md) | Per-stage checkboxes. |
| [`snapshot.md`](snapshot.md) | Environment freeze — partial; model fields still `PENDING`. |
| [`scripts/`](scripts/README.md) | Reproducibility harness. |

**Two things to know before going further:**

1. Upstream's build docs state that *"published benchmark numbers and regression
   guards assume Strix Halo / `gfx1151`"*, and gfx1151 (RDNA3.5) takes a
   different HIP code path from gfx1201 (RDNA4). **NVFP4 on the R9700 is
   effectively untested upstream**, so B1 is original measurement rather than
   confirmation of a known result.
2. Preparation established that NVFP4 matmul is **numerically correct** on
   gfx1201 on both Vulkan and HIP. **No performance number exists for this
   track**, and B1 is blocked pending a reproduction model.

## Mission

Establish a clean, reproducible, and fully characterized AMD Radeon AI PRO R9700 (`gfx1201`) native-NVFP4 baseline using Charlie's upstream ROCmFPX implementation before evaluating any local optimization mechanisms.

---

## Non-Contamination Principle

Track B begins with **ZERO imported Track A optimizations**.

Specifically, the initial Track B baseline **must NOT include**:
* No Track A `IQ4_XS` shader or dequant-reuse modifications
* No Track A `MUL_MAT + ADD` fusion patches
* No recurrent-state probe patches
* No 64K / 32K draft-vocabulary trimming
* No altered `n-max` or `p-min` speculative policies imported from Track A

All optimization mechanisms from Track A remain quarantined until the pure stock ROCmFPX baseline is measured and locked. Transfer of mechanisms will only occur during the subsequent Integration stage under formal A/B control.

---

## Upstream Starting Context

* **Upstream Repository**: `charlie12345/ROCmFPX`
* **Key Upstream Capabilities**:
  * Native NVFP4 loading and tensor allocation
  * Native NVFP4 compute kernel execution on AMD architectures
  * Dedicated NVFP4 `lm_head` scale tensor support (`LLM_TENSOR_OUTPUT` scale/input_scale)
  * Native MTP draft block compatibility
* **Starting Commit**: To be queried and recorded dynamically at experiment launch in [`snapshot.md`](snapshot.md).

---

## Workspace Structure

* [`snapshot.md`](snapshot.md): Hardware, driver, toolchain, commit, and model hash freeze.
* [`upstream-audit/`](upstream-audit/): Read-only audits of upstream source and history, dated and SHA-pinned.
* [`scripts/`](scripts/README.md): Reproducibility harness — environment snapshot, model hashing, command recording, repeated runs, metric extraction with provenance, and run comparison.
* `reproduction/`: Initial stock verification logs, build logs, and basic correctness tests.
* `matched-benchmark/`: Matched serial and native-MTP decode benchmarks against Track A hardware conditions.
* `decomposition/`: Profiled forward pass latency decomposition (verifier vs proposer, kernel breakdown).
* `raw/`: Unedited raw logs and profiler outputs.
