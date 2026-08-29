# Methodology

## Controlled workload

- Output: 960x544, 124 frames, 24 fps
- Reference: recovered likely original asset `Gemini_Generated_Image_cajce7cajce7cajc.jpeg`, 2380x1792, SHA prefix `afc021`
- Transformer: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Encoder: `qwen3vl_32b_minimax_h3_fp8.safetensors`
- Turbo: `minimax_h3_turbo_v4_step600_ema.safetensors`, five-step Turbo sampler/schedule
- One fixed prompt and reference within the campaign; distinct seed/output prefix per repetition
- Reliability criterion: one cold plus at least three warm successes

The original API prompt graph was not stored in Comfy history. The journal identified the failing model, LoRA, schedule, activation shape, and failure location; file timestamps identified the likely reference. Accordingly, S1 is a controlled reconstruction rather than a bit-for-bit replay of the unavailable original submission.

## Lane isolation

Production `comfyui-h3.service` was stopped but never edited. The single experiment used the current `/ai/comfyui` checkout, current custom nodes, production flags/environment, a separate user/output directory, and a passive probe supplied through an extra custom-node path. Production was restarted and health-checked after the campaign.

The matched dual lane used the same current checkout, models, flags, `H3_FLEX_ATTENTION=auto`, and disabled smart memory. Logical device order was gated as `cuda:0=R9700`, `cuda:1=RX 7900 XT`. The only graph difference was `CLIPLoaderMultiGPU` targeting `cuda:1`; DiT and sampling remained on `cuda:0`. The required MultiGPU component came from the pinned historical dual runtime and is recorded in the launch configuration. D0 occupied historical port 8191, so the isolated matched lane used port 8192 during measurement; port choice has no inference effect.

## Phase order and stopping rules

1. S1: change reference sizing to `match`, retain active Turbo.
2. S2: remove only the Turbo LoRA compute branch.
3. Reduction ladder only if S1 fails.
4. D0: historical topology reproduction only.
5. D1: matched-dual sanity at 864x480/124f.
6. D2: matched-dual target at 960x544/124f.

S1 passed 4/4, so no spatial/frame reduction was needed. D0 exceeded the user-defined 20%-over-single cold-wall limit before sampler entry and was abandoned. D1/D2 used the current matched lane and completed because they remained useful causal tests and stayed within the wall-clock limit.

## Valid Turbo-off test

Turbo strength zero is not a memory bypass in commit `4274783`: the nested `F.linear` operations execute before scale is applied. S2 therefore removes the LoRA node/branch before model sampling, while retaining the same five-step Turbo sampler and scheduler. It isolates LoRA compute/memory but is not claimed as a quality-equivalent Native preset.

## Memory instrumentation

The passive server-side probe wraps `comfy.samplers.CFGGuider.inner_sample`. It records process-local PyTorch allocation immediately after model loading and before actual sampler execution, resets peak statistics there, then records peak/current values on success or exception. Probe SHA-256: `1c51477416f0b30e410b2c6330abc706d3788e57a734997441c474e1950fa8d0`.

Measurements are kept separate:

- process-local `memory_allocated`, `memory_reserved`, max allocated, and max reserved from inside ComfyUI
- device-wide used/free VRAM sampled externally with ROCm SMI

The harness gates probe capture with a unique run ID and joins `/tmp/h3-mem-<run_id>.json` into each raw run log. The primary encoder-offload metric is single R9700 sampler-entry allocated minus matched-dual R9700 sampler-entry allocated.

## Timing and output validation

Wall time is client submission through Comfy history completion. Sampler duration comes from the in-process probe. Cold and warm timings are never averaged together. Warm comparison uses the median of three runs.

Every successful output was inspected with `ffprobe` for resolution, frame rate, frame count, and audio stream shape, then fully decoded with `ffmpeg -v error` without errors. Raw output paths and validation data are stored in each JSON log. Failures and invalid pilots remain preserved and explicitly labeled.

## Interpretation limits

- The experiment isolates memory fit and runtime behavior, not perceptual quality between `match`, `max`, or LoRA-off.
- D0 is not compared causally because it uses older revisions and different memory/attention settings.
- Workflow SHA values vary per repetition because the seed and output prefix are intentionally unique.
- The 7900 XT’s model file residency during encoding must not be confused with residency at sampler entry.
- For publication, absolute Desktop staging paths in the two launch scripts/YAML files were replaced with the repository path and configurable state/output roots. This path-only cleanup occurred after measurement and does not alter inference settings.
