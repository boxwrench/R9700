# R9700 troubleshooting: start with the symptom

This page is for the situation that started much of this repository: **the workflow technically runs, but something is obviously wrong**.

Do not begin by changing quantization, reinstalling ROCm, swapping GPUs, or tuning kernels. First match the behavior below and run the smallest discriminator.

The numbers here are engineering reference points from this specific Radeon AI PRO R9700 / ROCm / ComfyUI system. They are not universal performance guarantees.

## Fast triage

| If you see this | Check this first | Known result on this system |
|---|---|---|
| Model load takes many minutes; one CPU core busy; GPU mostly idle; VRAM rises slowly | **mmap-backed safetensors -> GPU transfer** | Add `--disable-mmap` before investigating storage, PCIe, power, or quantization |
| H3 is ~25–30 s sampling after a changed prompt but ~22 s when conditioning is cached | **Qwen3-VL still resident when sampler starts** | Explicit pre-sampler Qwen offload restored ~22.1–22.5 s sampling |
| H3 sampler starts with ~26 GiB allocated | **Text encoder residency** | Selected path enters sampler with roughly 7 GiB before DiT staging, then peaks around 21 GiB |
| LTX short prompts spend ~22 s in conditioning | **Gemma minimum sequence floor** | `LTX_GEMMA_MIN_LENGTH=256` reduced conditioning from 22.51 s to 5.35 s on the test workload |
| LTX output shows tile/seam/snow-like corruption at the tested small geometry | **VAE tile setting** | `tile_size=1280` is the retained workaround for the current workflow |
| `rocm-smi --showuse` says ~0–1% GPU while the card is drawing ~280–300 W | **Do not trust that utilization counter alone** | gfx1201 telemetry can under-report utilization during real GPU work; correlate power, clocks, VRAM, wall time and profiler data |
| Dual-GPU H3 keeps unloading/reloading Qwen instead of leaving it on the second GPU | **`--disable-smart-memory` in that dual-GPU lane** | In the historical dual-GPU test it defeated intended residency; see archived dual-GPU record |
| H3 changed-prompt sampling is slower and profiler shows large HtoD/copy time | **Model residency/offload before kernel tuning** | Removing Qwen from VRAM produced a much larger win than attacking H3 kernels first |
| MiniMax Music takes ~20 s before its very fast ~3.6 s DiT generation | **MiniMaxMusic3AR conditioning loop** | ~98.8% of conditioning was the iterative AR loop, not text tokenization or the Music DiT |
| Music conditioning draws ~23 W average and uses roughly one CPU core | **Serial host dispatch / tiny decode steps** | 375-frame conditioning scales linearly at about 51.6 ms/frame |
| Trying ComfyUI FixedKV/graph decode for Music fails on AMD | **NVIDIA-only flash decode dependency** | Installed `comfy_kitchen.flash_attention_decode` path requires the CUDA extension / SM80+; do not force it on ROCm |
| `torch.compile` on Music Qwen appears to compile but becomes unusable | **Dynamic Python KV-cache index guards** | The index changes every token and caused continual TorchInductor recompilation |
| Enabling a newer Triton backend looks tempting for INT8 | **Check stack compatibility first** | The stable production stack intentionally does not force the newer Triton path; do not replace Triton alone in the working environment |
| A model/router is inexplicably spilling to host RAM and decode collapses | **How many large models are being retained** | In the llama.cpp router study, model-count eviction allowed multiple large models to coexist and performance collapsed |
| H3 I2V/R2V node raises `execute() got an unexpected keyword argument 'image'` / `'ref_image'` / `'ref_image_0'` | **Node schema drift vs. saved workflow JSON** | `MiniMaxH3ImageToVideo` wants `first_frame`, not `image`. `MiniMaxH3ReferenceToVideo.ref_images` is a V3 `io.Autogrow` field — the prompt JSON key must be the dotted path `ref_images.ref_image_0`, not `ref_image` or a bare `ref_image_0`. See §7. |
| LTX or H3 R2V raises `RuntimeError: The size of tensor a (3072) must match the size of tensor b (128)` in `VAEDecodeAudio` | **Known-bad audio-VAE decode path for these shapes** | Deterministic, not transient. Workaround: drop the audio branch (remove `VAEDecodeAudio`, omit `audio` from `CreateVideo`) and ship video-only. See §7. |

## 7. V3 node schema drift (MiniMax H3 I2V / R2V) — 2026-08-18

### Symptom

A saved production workflow JSON that used to validate now fails at **execution** (not validation) with `TypeError: <Node>.execute() got an unexpected keyword argument '<name>'`.

### Root cause

Two independent, unrelated schema changes landed in the pinned ComfyUI node definitions (`comfy_extras/nodes_minimax_h3.py`) after the workflow JSON files in `production/workflows/` were last captured:

1. `MiniMaxH3ImageToVideo` takes `first_frame` (and optional `last_frame`), not `image`.
2. `MiniMaxH3ReferenceToVideo.ref_images` is declared as `io.Autogrow.Input(..., prefix="ref_image_", min=0, max=9)`. ComfyUI's V3 dynamic-input machinery (`comfy_api/latest/_io.py`: `parse_class_inputs` → `Autogrow._expand_schema_for_dynamic` → `build_nested_inputs`) only recognizes a filled autogrow slot when the **prompt JSON's flat input key is the dotted path** `<field_id>.<slot_name>` — e.g. `ref_images.ref_image_0` — wired directly as a node input (a link `["<image_node_id>", 0]`). A bare `ref_image_0` or the legacy singular `ref_image` is accepted by `/prompt` **validation** (no `node_errors`) but rejected at **execution**, because validation only checks the class's static `INPUT_TYPES()`, while the dynamic-path expansion that lets `execute()` actually receive `ref_images={"ref_image_0": <tensor>}` only fires for the dotted key. The node's own frontend widget already does this correctly (`custom_nodes/ComfyUI-ALLinONE-MinimaxH3/web/one_node_minimax_h3.js`: `wf["6"].inputs[\`ref_images.ref_image_${idx}\`]=[id,0]`); the drift was only in hand-built/legacy prompt JSON. `audio_vae` (a second `VAELoader`) and `ref_image_size` (`"match"` or `"max"`) are also now required/optional inputs that older saved graphs may be missing entirely.

### Fix

Do not edit the pinned ComfyUI node source. Fix the **workflow JSON**:

- I2V: rename the image input key from `image` to `first_frame`.
- R2V: replace `"ref_image": [<id>, 0]` with `"ref_images.ref_image_0": [<id>, 0]`, and add `"audio_vae": [<audio_vae_loader_id>, 0]` and `"ref_image_size": "match"` (or `"max"`).

### Known dead ends

- `ref_image_0` or `ref_image_1` (flat, undotted) — validates, fails at execution.
- Legacy singular `ref_image` — validates, fails at execution (the key does not exist in the live schema at all once `ref_images` autogrow is present).
- Nesting the value under an actual JSON object `"ref_images": {"ref_image_0": [...]}` — the prompt-graph link format does not support nested dict values; the flat dotted-string key is what `build_nested_inputs` expects to find in the flat prompt input dict.

## 8. LTX / H3 `VAEDecodeAudio` deterministic tensor-shape mismatch — 2026-08-18

### Symptom

`RuntimeError: The size of tensor a (3072) must match the size of tensor b (128) at non-singleton dimension 2` inside `comfy_extras/nodes_audio.py: vae_decode_audio`, for specific LTX and H3 latent shapes. Reproduces identically on retry with unchanged inputs — this is deterministic, not a transient/flaky GPU fault.

### Workaround used in this campaign

Drop the audio branch: remove the `VAEDecodeAudio` node and the `audio` input on `CreateVideo`, and save video-only. This is the same pattern already present in the known-good LTX return production run. Do not attempt to patch the audio VAE decode kernel itself — that is out of scope for showcase/acceptance work and was not investigated further this pass.

## 1. Catastrophically slow model loading

### Symptom

A safetensors model takes minutes to move into VRAM. One CPU core is near 100%, GPU power/utilization are low, VRAM climbs slowly, and the SSD does not look saturated.

### Do this first

Run ComfyUI with:

```bash
--disable-mmap
```

### Why this is first

On this machine the mmap-backed safetensors -> ROCm GPU path showed a roughly fixed per-tensor transfer penalty of about 0.5 s/tensor.

Measured examples:

- H3 Qwen3-VL-32B, 1,836 tensors: **931 s -> 1.2 s** with mmap disabled.
- H3 warm-up path: **1,362 s -> 51.07 s**.
- LTX Gemma4 encoder, 1,342 tensors: roughly **11m11s** on the pathological mmap path.

The microbenchmark showed hundreds of tiny mmap-backed tensors taking about 500 ms each to move to GPU while normal RAM-backed tensors took about 0.6 ms each.

### Do not start with

- replacing the SSD
- PCIe tuning
- GPU clock tuning
- changing BF16/FP8/INT8 formats
- `--disable-dynamic-vram`
- `--disable-smart-memory`

Those were not the root cause of this loader pathology.

## 2. H3 gets slower after a real prompt change

### Symptom

Same-prompt/new-seed H3 sampling is around 22 s, but changed-prompt sampling is around 25–30 s at 608x352, 39 frames, 20 steps.

### Check

Measure VRAM immediately before the sampler and determine whether Qwen3-VL remains resident.

Paired A/B on this system:

| State | VRAM before sampler | Sampling |
|---|---:|---:|
| Qwen resident | 26.18 GiB | 29.51 s |
| Qwen explicitly offloaded | 7.19 GiB | 22.15 s |

The explicit offload cost about 4.03 s but still improved paired end-to-end wall time by 8.1%. A production sanity run reproduced **22.42 s** sampling.

### Selected behavior

After H3 conditioning, explicitly unload the Qwen model through ComfyUI model management before sampling. See `docs/selected-production-configs.md`.

The measured residency effect is confirmed. Explanations involving cache thrash, paging, or memory-channel contention remain hypotheses unless separately profiled.

## 3. LTX short-prompt conditioning is strangely expensive

### Symptom

A short prompt takes roughly 22 s just to condition even though sampling is only around 10–11 s.

### Check

Inspect the LTX Gemma minimum sequence length.

The pinned ComfyUI code forced a minimum length of 1,024 positions. A 24-token prompt was therefore processed as 1,024 positions.

Measured full-run control:

| Gemma minimum | Conditioning | Sampling | Wall |
|---|---:|---:|---:|
| 1024 | 22.51 s | 10.65 s | 43.78 s |
| 256 | 5.35 s | ~10.61 s | 25.69 s |

### Selected behavior

```bash
LTX_GEMMA_MIN_LENGTH=256
```

This is a minimum, not a truncation to 256 tokens. Prompts longer than the floor continue at their own length.

The conditioning outputs were numerically very close in the spot checks, but diffusion amplified the small differences; changing the floor should be treated as changing the generation trajectory rather than preserving exact pixel reproduction.

## 4. GPU utilization telemetry looks impossible

### Symptom

`rocm-smi --showuse` reports almost no utilization while the GPU is near its power cap.

### Check

Correlate:

- board power
- clocks
- VRAM residency
- wall time
- profiler traces

Do not diagnose CPU fallback from the utilization percentage alone.

During one LTX encoder pass, the utilization counter reported roughly 0–1% while the board sustained approximately 278–300 W. The work was GPU-resident.

## 5. MiniMax Music spends almost all its time before diffusion

### Symptom

A 15 s Music generation takes about 24 s warm, but the Music DiT itself takes only about 3.6 s.

### Check

Instrument `MiniMaxMusic3TextEncode` / `MiniMaxMusic3AR.generate` before trying new DiT quants.

Measured 15 s / 375-frame decomposition:

- total conditioning: **20.03 s**
- AR loop: **19.80 s**
- Qwen3-8B one-token backbone: **~12.04 s**
- seven-pass RVQ depth expansion: **~6.82 s**
- initial prompt prefill/tokenization: **~0.225 s combined**

Conditioning scaled almost linearly with generated audio frames:

```text
conditioning ~= 0.25 s + 0.0516 s * frames
```

### Known dead ends

**Existing ComfyUI FixedKV / graph path:** blocked on ROCm because it calls the installed NVIDIA-only `comfy_kitchen.flash_attention_decode` kernel.

**Naive `torch.compile` on the Qwen one-token backbone:** tracing succeeded, but the Python integer KV-cache index changed each token and triggered continual recompilation.

**`torch.compile` on the RVQ depth decoder:** compiled cleanly with 0 graph breaks, but micro-sequence lengths ($S \in [2..8]$) across sequential discrete passes are host-dispatch bound (~1.03x speedup, ~0.94% wall-time win; rejected).

Do not rediscover these before checking the current Music optimization record.

## 6. Before changing anything major

Run through this order:

1. Compare your exact files and configuration with `docs/selected-production-configs.md`.
2. Check the symptom table above.
3. Confirm whether the problem is **cold**, **warm changed-input**, or **warm cached-input** behavior.
4. Record absolute wall times by stage.
5. Change one thing.
6. Keep measured results separate from mechanism hypotheses.

If a new ComfyUI/model/workflow version behaves differently, treat it as a **candidate** until it reproduces the known-good canaries. The planned production-lock/update-gate work will formalize that process.

## Related records

- `docs/comfyui-walltime-campaign-20260818.md`
- `docs/ltx25-r9700-optimization-20260818.md`
- `docs/minimax-h3-r9700-optimization-20260818.md`
- `docs/minimax-music3-r9700-baseline-20260818.md`
- `docs/archive/dual-gpu-residency-20260812.md`
