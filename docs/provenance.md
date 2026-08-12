# Model and software provenance

Weights are intentionally absent from this repository. The table records the
exact filenames and immutable revisions where the local source supports them.
`unknown` is deliberate: it means the local record did not provide a supported
immutable revision or file hash, and no value was inferred from a friendly
filename.

## MiniMax H3 lane

| Role | Repository | Revision | Filename | SHA-256 |
|---|---|---|---|---|
| H3 diffusion, FP8 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `014cd40f7e177756c6b2473c0d93b1c89a790dd2` | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | `12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993` |
| H3 text encoder | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `014cd40f7e177756c6b2473c0d93b1c89a790dd2` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| H3 video VAE | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `014cd40f7e177756c6b2473c0d93b1c89a790dd2` | `minimax_h3_video_vae_fp16.safetensors` | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| H3 audio VAE | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `014cd40f7e177756c6b2473c0d93b1c89a790dd2` | `minimax_h3_audio_vae_fp32.safetensors` | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| Turbo v4 adapter | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | `43a74557ac3f6539db8e0f2a959d03feb7a81480` | `minimax_h3_turbo_v4_step600_ema.safetensors` | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |

## LTX-2.5 lane

The local report and exact workflow identify [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
and the following five files. The source exposes `main` download URLs but no
immutable revision or file SHA-256 for this repository build, so those fields
remain `unknown` in `data/models.tsv`.

| Role | Filename | Revision | SHA-256 |
|---|---|---|---|
| Distilled transformer | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | unknown | unknown |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | unknown | unknown |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | unknown | unknown |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | unknown | unknown |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | unknown | unknown |

The workflow contains the official file URLs and is kept byte-for-byte under
[`workflows/ltx-2.5/`](../workflows/ltx-2.5/).

## Software provenance

- ComfyUI: [pinned upstream commit](https://github.com/Comfy-Org/ComfyUI/tree/c2bcbecd82ec5ae66594340b395c24ef0217b238)
- LTX-2.5 native support: [ComfyUI PR 15499](https://github.com/Comfy-Org/ComfyUI/pull/15499)
- H3 Turbo custom node: [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/55fee864dd7b2976b1c4ce3c3d5f7968f181409f)
