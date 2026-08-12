# LTX-2.5 distilled INT8 brass-robot comparison — 2026-08-12

> Superseded by `/ai/benchmarks/cold-video-comparison-20260812.md`. The historical and comparison-crop artifacts were removed; use `/ai/artifacts/runs/ltx-2.5/cold-brass-robot-LTX25-native-896x512.mp4` as the canonical LTX-2.5 benchmark.

## Outcome

- Status: successful single comparison render; queue empty afterward.
- Native model output: `/ai/artifacts/runs/ltx-2.5/brass-robot-LTX25-native-896x512.mp4`
- Standardized H3 comparison crop: `/ai/artifacts/runs/ltx-2.5/brass-robot-LTX25-vs-H3-864x480.mp4`
- Installed Studio workflow: `Workflows -> LTX-2.5-Brass-Robot-Comparison`

## Stack and model path

- Ubuntu 24.04; ROCm 7.2.1; Radeon AI PRO R9700 selected with `HIP_VISIBLE_DEVICES=1`.
- ComfyUI 0.32.0, pinned at `c2bcbecd82ec5ae66594340b395c24ef0217b238`.
- Official ComfyUI two-stage LTX-2.5 text-to-video graph.
- Transformer: `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`.
- Text encoder: `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`.
- Video decoder: `ltx-2.5-video-vae-bf16.safetensors`.
- Audio VAE: `ltx-2.5-audio-vae-bf16.safetensors`.
- Spatial upscaler: `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`.
- All five files passed SHA-256 verification against their Hugging Face download metadata.

## Matched workload

- Prompt: `A tiny brass robot carefully waters a glowing mushroom garden at night. Gentle rain, soft mechanical movements, warm lantern light. Audio: quiet rain, small servo sounds, and a soft bell chime. No text or logos.`
- Prompt enhancement disabled to preserve the exact H3 prompt.
- Seed: 8112026 for stage 1; stage-2 seed 42 as pinned by the official template.
- 121 frames, 24 fps, 5.041667 seconds. LTX requires `frames % 8 == 1`; H3 used 124 frames / 5.166667 seconds.
- Distilled schedule: eight first-stage Euler ancestral steps and three refinement steps, CFG 1 for video and audio.
- Native pipeline canvas/output: 896x512. The two-stage half-resolution latent must align to 32 pixels, making the native final grid effectively 64-pixel aligned.
- Standardized comparison: Lanczos scale to 864 pixels wide, center crop to 864x480, audio stream copied unchanged.

## Validation

- Corrected render prompt ID: `cef5fb30-d4c9-4767-90bf-49ba87981b55`.
- Corrected warm execution: 33.578 seconds. Model and text-conditioning nodes were cached from the immediately preceding load/validation attempt, so this is not a cold benchmark.
- First full load/generate/decode attempt reached SaveVideo in 60.84 seconds; only saving failed because a cross-directory symlink was rejected by ComfyUI's output safety check. No model or hardware fault occurred.
- Native media: H.264 896x512, exactly 121 frames at 24 fps; stereo AAC 48 kHz; 5.041667 seconds.
- Standardized media: H.264 864x480, exactly 121 frames at 24 fps; stereo AAC 48 kHz; 5.041667 seconds.
- Full video and audio decode passed without errors. Audio is non-silent (`mean -44.8 dB`, `max -27.3 dB`).
- Native SHA-256: `b2b61be8e7bd56f185ae8be10c12c15e5387eb06555268d168a66a693bced806`.
- Standardized SHA-256: `bc9071a6faaba8c3b6feb9060bc80b201c27f20073ee79e3c3e43fc62dfb3ef4`.
- Observed first-run telemetry: up to 22,352 MiB VRAM, 248 W, 59 C edge / 79 C hotspot / 65 C memory in 20-second samples.
- Kernel scan found no GPU reset, VM fault, page fault, or OOM. Two benign `svm_range_deferred_list_work` CPU-hog warnings appeared during model loading.

## H3 timing context

- Prior H3 native FP8 20-step validation: 260.91 seconds at 864x480, 124 frames, 24 fps.
- LTX's 33.578-second corrected run was warm, and the 60.84-second first pass used the smaller rounded 832x448 canvas, so neither is a controlled cold timing mate. They indicate LTX-2.5 distilled is materially faster here, but should not be presented as a rigorous speed ratio.

## PR and issue audit

- ComfyUI PR 15499 (native LTX-2.5 support): merged.
- workflow_templates PR 1113 (official LTX-2.5 templates): merged.
- Lightricks/LTX-2 PR 272 (public LTX-2.5 sync): merged.
- ComfyUI PR 11787 (AMD-safe LTX audio conversion): merged; stereo audio saved successfully on this system.
- ComfyUI issue 15540 concerns an incompatible third-party GGUF loader/checkpoint path. This setup uses the official native INT8 safetensors path and is unaffected.
- ComfyUI issue 13730 tracks AMD/ROCm LTX load stalls from host pinned-memory pressure. Proposed PR 14525 is still open, and its pin-budget guard is absent from this pinned ComfyUI. The current `--disable-smart-memory`, `--disable-dynamic-vram`, 2 GiB reserve, and 188 GiB host RAM completed cleanly, so no launch-flag change was made.
- ComfyUI PR 14916 (broader backend-aware inference handling) remains open; nothing in this successful run requires it.
- No open LTX-2.5 resolution/crop issue was found in ComfyUI, Lightricks/LTX-2, or workflow_templates at the time of this audit.

## Sources

- https://huggingface.co/Lightricks/LTX-2.5
- https://docs.ltx.io/open-source-model/integration-tools/comfy-ui
- https://github.com/Comfy-Org/ComfyUI/pull/15499
- https://github.com/Comfy-Org/workflow_templates/pull/1113
- https://github.com/Lightricks/LTX-2/pull/272
- https://github.com/Comfy-Org/ComfyUI/issues/13730
- https://github.com/Comfy-Org/ComfyUI/pull/14525
- https://github.com/Comfy-Org/ComfyUI/issues/15540
