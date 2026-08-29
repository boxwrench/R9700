# MiniMax H3 R2V on Radeon AI PRO R9700

This directory preserves a controlled 2026-08-29 investigation of MiniMax H3 Reference-to-Video on a 32 GiB-class Radeon AI PRO R9700, with an RX 7900 XT tested as a Qwen3-VL/CLIP host.

## Finding

The full 960x544, 124-frame Turbo workload is reliable on one R9700 when `ref_image_size=match`: one cold plus three warm runs all passed. The original `max` run failed in the Turbo LoRA nested-linear path while requesting 3.30 GiB with only 2.97 GiB free.

The 7900 XT is not useful as a dedicated encoder GPU for this workload. In the matched experiment it freed only 12 MiB on the R9700 at sampler entry and changed warm wall time from 120.099 s to 120.101 s. The publication recommendation is therefore:

- single R9700
- `ref_image_size=match`
- Turbo retained
- 960x544 / 124 frames requires no workload reduction

See [RESULTS.md](RESULTS.md) for the complete answers and [tables/results.csv](tables/results.csv) for exact per-run data.

## Why reference size mattered

`max` does not upscale a small image. It retains a large input image, capped by the reference pipeline’s 2048 px short-edge policy. `match` only downsizes it toward the output video’s pixel area. Those reference tokens persist through sampling, enlarging the tensors handled by the Turbo LoRA branch. In the controlled reconstruction, changing to `match` reduced the relevant activation element count by 52.6% and made the unchanged output workload stable.

This is not a general claim that images always cost more than video. It says the oversized reference-image representation—not the 960x544/124f output latent by itself—was the variable that pushed this exact graph over the boundary.

## Community cross-check

The result agrees with current ComfyUI documentation, which defaults `MiniMaxH3ReferenceToVideo` to `match` and notes that reference tokens participate in every sampling step. Independent community workflows also recommend `match`; one reports `max` taking 24% longer in its own comparison.

- [ComfyUI MiniMaxH3ReferenceToVideo documentation](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ReferenceToVideo/en.md)
- [Community measured R2V workflow](https://github.com/jlucasmcrell/MiniMax-H3-HardMode-Workflow)
- [ComfyUI H3 memory-budget issue](https://github.com/Comfy-Org/ComfyUI/issues/15663)
- [vLLM-Omni one/two RTX 5090 memory-first recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md)

Layerwise offload, GGUF, smaller text encoders, and explicit encoder eviction are valid low-memory strategies, but they add transfer/loading tradeoffs. The sampler probe showed that this Comfy configuration already evicts nearly all Qwen residency before warm sampling, so those alternatives are not expected to beat the working single-GPU path under the 20% wall-clock rule.

## Reproducibility

- `tables/results.csv`: exact measurements for all 16 passing runs
- `tables/run-matrix.csv`: original pre-run plan retained as an archival record; its D labels/ports were superseded by the executed matrix in `results.csv`
- `logs/*.json`: raw joined Comfy history, in-process allocator probe, and device-wide ROCm telemetry
- `logs/BASE-known-oom.txt`: preserved original failure evidence
- `logs/D0-abandoned.json`: historical dual attempt and abandonment rationale
- `notes/methodology.md`: controls, instrumentation, caveats, and lane definitions
- `notes/preflight.json`: frozen hardware/software state
- `harness.py`: API graph construction, cold/warm repetitions, telemetry, and result capture
- `instrumentation/h3-sampler-mem-probe.py`: passive in-process sampler allocator probe
- `runtime/launch-single-probed.sh`: isolated single-GPU launch
- `runtime/launch-matched-dual.sh`: configuration-matched dual launch
- `workflows/r2v-r9700-baseline.json`: frozen source, SHA-256 `818172c67a07e1fbc355acb9714bb3ac7a6ee9e8d7306841ae56633ff0263587`

The frozen workflow is preserved as provenance; it was not a directly executable record of the original API submission. The harness constructs the controlled executable graph and records its SHA for each run. Seeds and output prefixes differ across repetitions to prevent cache reuse; all substantive workload settings remain fixed.

The published launch scripts differ from the executed copies only in path handling: Desktop staging paths were replaced with this repository path plus configurable `H3_R2V_STATE_ROOT` and `H3_R2V_OUTPUT_DIR` locations. GPU mapping, ComfyUI checkout, model paths, environment, flags, and ports are unchanged.

## Safety and production state

The production `comfyui-h3.service` was never edited. It was temporarily stopped while isolated experimental processes used the GPU, then restored and verified healthy on `127.0.0.1:8190`. The passive probe was loaded only through isolated extra custom-node paths because installing into the read-only `/ai` tree was not authorized.
