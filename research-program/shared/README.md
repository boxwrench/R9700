# Shared Scientific Methodology

This directory defines the common scientific contract, measurement standards, and metric terminology shared across Track A (Vulkan/Q4) and Track B (ROCmFPX/NVFP4).

---

## Documents

* [`benchmark-contract.md`](benchmark-contract.md): Strict protocol for all performance, microbenchmark, and regression testing.
* [`metrics.md`](metrics.md): Unambiguous definitions and formulas for all reported latency, throughput, and speculative acceptance metrics.
* [`run-record-schema.md`](run-record-schema.md): The per-run JSONL record — required fields, provenance and classification tagging, and the rule that an absent value beats a guessed one.

Tooling implementing this contract lives in [`../tracks/B-rocmfpx-nvfp4/scripts/`](../tracks/B-rocmfpx-nvfp4/scripts/README.md); it is track-agnostic despite its location.
