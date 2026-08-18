# Selected R9700 production configurations

This page records the **known-good files and settings actually used in the 2026-08-18 campaign**. Newer is not automatically better. Treat a new ComfyUI commit, workflow, model revision, or quant as a candidate until it reproduces the canaries.

## Shared ComfyUI runtime

Current wall-time campaign:

```text
ComfyUI: v0.33.2
commit: 7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa
Python: 3.12.3
PyTorch: 2.9.1+rocm7.2.1.gitff65f5bc
HIP: 7.2.53211-e1a6bc5663
Triton: 3.5.1+rocm7.2.1.gita272dfa8
Ubuntu: 24.04.4 LTS
GPU: AMD Radeon AI PRO R9700 / gfx1201 / 32 GB
```

Selected launcher behavior:

```bash
/ai/environments/comfyui-h3/bin/python /ai/comfyui/main.py \
  --output-directory /ai/artifacts/runs/minimax-h3 \
  --disable-dynamic-vram --disable-smart-memory --disable-mmap \
  --bf16-vae --reserve-vram 2 \
  --listen 127.0.0.1 --port 8190 --disable-auto-launch
```

Environment used in the campaign:

```text
HIP_VISIBLE_DEVICES=1
MIOPEN_FIND_MODE=FAST
MIOPEN_USER_DB_PATH=/ai/cache/miopen
TORCH_HOME=/ai/cache/torch
TORCHINDUCTOR_CACHE_DIR=/ai/cache/torchinductor
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
TRITON_CACHE_DIR=/ai/cache/triton
```

Important scope note: the single-R9700 golden launcher intentionally retains `--disable-smart-memory`. The historical dual-GPU H3 lane required different memory-policy behavior and should not be used to generalize this flag.

---

## LTX 2.5 — selected

### Files

```text
Transformer:
ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors

Text encoder:
gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors

E2B:
gemma4_e2b_it_bf16.safetensors

Video VAE:
ltx-2.5-video-vae-bf16.safetensors

Audio VAE:
ltx-2.5-audio-vae-bf16.safetensors
```

### Selected behavior

```text
--disable-mmap
INT8-ConvRot Gemma encoder
INT8-ConvRot DiT
LTX_GEMMA_MIN_LENGTH=256
tile_size=1280
reuse unchanged negative conditioning when ComfyUI cache permits
```

The minimum-length change is a local ComfyUI modification/override preserved in [`production/patches/ltx-gemma-floor-256.patch`](../production/patches/ltx-gemma-floor-256.patch). The 256 value is a minimum sequence floor, not a 256-token truncation.

Reference optimization workload:

```text
768x448
41 frames
8 steps
```

Reference result:

```text
1024 floor: conditioning 22.51 s, wall 43.78 s
256 floor:  conditioning 5.35 s, wall 25.69 s
```

---

## MiniMax H3 — selected for this pass

### Files

```text
FL2VA transformer:
minimax_h3_fl2va_pruned_fp8_scaled.safetensors

Ref2VA transformer:
minimax_h3_ref2va_pruned_fp8_scaled.safetensors

Text / vision encoder:
qwen3vl_32b_minimax_h3_fp8.safetensors

Video VAE:
minimax_h3_video_vae_fp16.safetensors

Audio VAE:
minimax_h3_audio_vae_fp32.safetensors
```

### Selected behavior

```text
--disable-mmap
single R9700
FP8 scaled H3 transformer
BF16 VAE execution
explicitly unload Qwen3-VL after conditioning and before sampler
```

The selected pre-sampler action uses the local production patch in [`production/patches/h3-qwen-presampler-offload.patch`](../production/patches/h3-qwen-presampler-offload.patch):

```python
comfy.model_management.unload_model_and_clones(clip.patcher)
```

Reference T2V workload:

```text
608x352
39 frames
24 fps
20 steps
res_multistep / simple
```

Reference paired A/B:

```text
Qwen resident:
VRAM before sampler 26.18 GiB
sampling 29.51 s / 1.476 s per step

Qwen offloaded:
offload 4.03 s
VRAM before sampler 7.19 GiB
sampling 22.15 s / 1.108 s per step
```

Production sanity reproduced 22.42 s sampling.

---

## MiniMax Music 3 — complete for this pass

### Files

```text
DiT:
minimax_music3_dit_int8_convrot.safetensors

Text / lyrics encoder:
minimax_music3_text_encoder_pruned_int8_convrot.safetensors

DAV:
minimax_music3_dav.safetensors
```

### Benchmark configuration

```text
--disable-mmap
15.0 s output (375 frames)
res_multistep / simple
20 steps
CFG scale 1.5
top_k 50
```

Warm baseline:

```text
conditioning ~19.87-20.03 s
generation ~3.64 s
audio decode ~0.40 s
wall ~24.42 s
peak VRAM ~10.5 GiB
```

Conditioning decomposition:

```text
MiniMaxMusic3AR.generate ~19.80 s / 375 frames
Qwen3-8B one-token backbone ~12.04 s
RVQ depth decoder ~6.82 s
```

Closed paths summary:
- **ComfyUI FixedKV / graph path:** Blocked on ROCm because fixed-KV decode calls NVIDIA-only `comfy_kitchen.flash_attention_decode`.
- **Qwen backbone `torch.compile`:** Blocked due to changing Python KV-cache index causing continuous Dynamo guard failures.
- **RVQDepthDecoder `torch.compile`:** Evaluated and rejected (~1.03x speedup, ~0.94% wall improvement) due to sequential dependency and host-launch bounds.

---

## Production verification & update gate

To verify that the local system matches these known-good configurations:

```bash
python3 /ai/github/R9700/scripts/production-preflight.py
```

To safely evaluate candidate software/model updates without breaking production, follow [`docs/update-gate.md`](update-gate.md).
