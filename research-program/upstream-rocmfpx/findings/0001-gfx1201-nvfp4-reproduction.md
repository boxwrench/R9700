# Finding 0001 — AMD Radeon AI PRO R9700 (gfx1201) Native-NVFP4 Reproduction & Baseline Characterization

* **Status**: PREPARED / NOT SUBMITTED
* **Target Project**: `charlie12345/ROCmFPX`
* **Base Upstream Commit**: `a71e6c8a63ab947399a315095e08c8d8ad043dda`
* **Hardware Target**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB VRAM)
* **Backend**: Vulkan / RADV (Mesa 25.1.0-devel / LLVM 20.0.0git)

---

## 1. Environment & Artifact Provenance

* **Source Checkpoint**: `RadixArk/Qwen3.8-27B-NVFP4`
* **Original Mixed GGUF**:
  * File Size: `28,230,539,776 bytes` (28.23 GB)
  * Tensor Layout: 193 NVFP4 tensors, remainder in higher precision (FP16/Q8_0)
* **Uniform NVFP4 Derivative**:
  * File Size: `15,547,030,016 bytes` (15.55 GB)
  * Quantization Density: $\approx 4.55\text{ BPW}$
  * Tensor Layout: 505 NVFP4 tensors, 1 Q5_K tensor
  * Invariance: All 193 originally-NVFP4 tensors remained 100% bit-exact through uniformization.

---

## 2. Invocations

### Quantization / Uniformization Command:
```bash
./build-rocmfpx/bin/llama-quantize \
  /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-mixed.gguf \
  /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-uniform.gguf \
  NVFP4
```

### Inference Command:
```bash
./build-rocmfpx/bin/llama-speculative-simple \
  -m /ai/models/Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-uniform.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.3 \
  -c 163840 \
  -ub 512 \
  -fa 1 \
  -ctk f16 \
  -ctv f16 \
  -n 256 \
  -p "Explain the physics of semiconductor doping."
```

---

## 3. Measured Results (R9700 / gfx1201)

| Metric | Mixed NVFP4 (28.2 GB) | Uniform NVFP4 (15.5 GB) | Classification |
|---|---:|---:|---|
| **Serial Decode Throughput** | 20.32 tok/s | **27.33 tok/s** | **MEASURED** |
| **Native MTP Decode Throughput** | 30.71 tok/s | **37.26 tok/s** | **MEASURED** |
| **MTP Acceleration Multiplier** | 1.511× | **1.363×** | **CALCULATED** |
| **Serial VRAM Footprint** | 25.22 GB | **14.72 GB** | **MEASURED** |
| **MTP VRAM Footprint** | 26.30 GB | **15.80 GB** | **MEASURED** |

---

## 4. Key Technical Observations

1. **`lm_head` Scale Application**: Runtime scaling verification confirms that the NVFP4 `lm_head` scale tensor (`LLM_TENSOR_OUTPUT` scale/input_scale) is correctly bound and applied during logit generation.
2. **Runtime Execution Semantics (W4A16 vs W4A4)**:
   * While source model checkpoints describe the format as `W4A4`, runtime graph inspection reveals that input activation scale tensors are not consumed during forward evaluation.
   * Execution on `gfx1201` operates as **native NVFP4 weight execution with higher-precision activations (W4A16)**.
3. **Upstream Claim Distinction**: Upstream screenshots citing $\approx 72.4\text{ tok/s}$ were designated as `ROCmFP4 FAST`, a distinct proprietary/lossy representation, whereas standard native NVFP4 achieves $37.26\text{ tok/s}$ native MTP on the same hardware.
