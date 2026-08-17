# Track B — ROCmFPX / Native NVFP4

## Status: ARCHIVED / REOPENABLE

*For full reproduction, benchmark results, semantic findings, and archival decision details, see the [Track B Closeout Document](CLOSEOUT.md).*

---

## Mission & Retrospective

> [!IMPORTANT]
> **Read [`CLOSEOUT.md`](CLOSEOUT.md) first.** B1 succeeded: native NVFP4 runs on
> gfx1201, and the uniform derivative is the preferred artifact *within this
> track*. Track B was nonetheless **not adopted** as the program foundation;
> primary effort returned to Track A. Reopen conditions are in the closeout.
>
> Passages below describing B1 as pending or blocked are **historical**.

**Start here:**

| Document | What it is |
|---|---|
| [`CLOSEOUT.md`](CLOSEOUT.md) | **Current.** Result, established findings, decision, reopen conditions. |
| [`reproduction/2026-08-17-b1-results.md`](reproduction/2026-08-17-b1-results.md) | The B1 four-arm baseline measurement. |
| [`reproduction/2026-08-17-b1-gates.md`](reproduction/2026-08-17-b1-gates.md) | Gate 1 (lm_head scale) and Gate 2 (model identity) evidence. |
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
   gfx1201 on both Vulkan and HIP. ~~No performance number exists for this
   track, and B1 is blocked pending a reproduction model.~~ **Superseded:** B1
   measured four arms; see [`CLOSEOUT.md`](CLOSEOUT.md).

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
