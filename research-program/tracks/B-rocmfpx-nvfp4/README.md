# Track B — ROCmFPX / Native NVFP4

## Status: NEW / BASELINE NOT YET ESTABLISHED

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
* `reproduction/`: Initial stock verification logs, build logs, and basic correctness tests.
* `matched-benchmark/`: Matched serial and native-MTP decode benchmarks against Track A hardware conditions.
* `decomposition/`: Profiled forward pass latency decomposition (verifier vs proposer, kernel breakdown).
* `raw/`: Unedited raw logs and profiler outputs.
