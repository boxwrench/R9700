# Reproducer 0001 — Greedy Serial vs Native-MTP Output Divergence on Qwen3.8-27B-NVFP4

* **Status**: PREPARED / NOT SUBMITTED
* **Target Project**: `charlie12345/ROCmFPX`
* **Base Upstream Commit**: `a71e6c8a63ab947399a315095e08c8d8ad043dda`
* **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, Vulkan1)

---

## 1. Summary

Under deterministic greedy decoding (`temp = 0.0, top_k = 1`), native MTP generation diverges from serial target model generation on `Qwen3.8-27B-NVFP4`.

Furthermore, passing the mitigation flag `--spec-mtp-strict-qwen` does **not** prevent the first observed token divergence in the generation sequence; trace logs demonstrate that the strict mitigation mechanism becomes active only *after* the initial divergence has already occurred.

---

## 2. Minimal Reproduction Steps

### A. Serial Reference Generation (No Speculation):
```bash
./build-rocmfpx/bin/llama-cli \
  -m /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-uniform.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --temp 0.0 \
  --top-k 1 \
  --seed 42 \
  -n 128 \
  -p "Explain why RAM is considered volatile memory in computer systems."
```

### B. Native MTP Generation (Default):
```bash
./build-rocmfpx/bin/llama-speculative-simple \
  -m /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-uniform.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.3 \
  --temp 0.0 \
  --top-k 1 \
  --seed 42 \
  -n 128 \
  -p "Explain why RAM is considered volatile memory in computer systems."
```

### C. Native MTP Generation (With `--spec-mtp-strict-qwen`):
```bash
./build-rocmfpx/bin/llama-speculative-simple \
  -m /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-uniform.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.3 \
  --spec-mtp-strict-qwen \
  --temp 0.0 \
  --top-k 1 \
  --seed 42 \
  -n 128 \
  -p "Explain why RAM is considered volatile memory in computer systems."
```

---

## 3. Observed Divergence Point

* **Prompt**: `"Explain why RAM is considered volatile memory in computer systems."`
* **First Differing Output Position**:
  * **Serial Output Text**: `"...because it requires continuous electrical power to maintain its stored state..."`
  * **Native MTP Output Text**: `"...because it relies on continuous electric current to retain stored data..."`
  * **First Differing Token**: Token Index 34.
* **Strict Mitigation Timing**:
  * In trace logs with `--spec-mtp-strict-qwen`, strict fallback triggers at Token Index 48, which is 14 tokens *after* the initial divergence occurred.

---

## 4. Root Cause Status

* **Status**: `UNRESOLVED`.
* No specific root cause or speculative patch is proposed. This reproducer isolates the exact command and test vector for upstream investigation.
