# Research Program Status

### TRACK A — Vulkan / GGUF Q4 / Native MTP
* **Status**: `PAUSED` (Preserved authoritative state)
* **Evidence Base**: Mature (Entries 1–17 logged)
* **Authoritative Log**: [`docs/qwen3-8-27b-experiment-log.md`](../../docs/qwen3-8-27b-experiment-log.md)
* **Latest Accepted Entry**: Entry 17 (`Tiny-N MUL_MAT + ADD fusion evaluation`)
* **Experimental / Unaccepted**: Preliminary draft-vocabulary trimming ($64\text{K} / 32\text{K}$) holdout validation paused

---

### TRACK B — ROCmFPX / Native NVFP4
* **Status**: `NOT STARTED`
* **Baseline**: Pending reproduction
* **Next Milestone**: Upstream stock reproduction and environment snapshot on R9700 (`gfx1201`)

---

### INTEGRATION
* **Status**: `BLOCKED` (Awaiting stable Track B baseline)
* **Rule**: No Track A optimization is imported or assumed portable without independent Track B A/B validation

---

### UPSTREAM ROCmFPX LANE
* **Status**: `READY`
* **First Contribution Candidate**: Pending Track B characterization findings
