# R9700 Local Video Benchmark v1

## Headline metric

Always report these together and in this order:

1. Prompt-to-saved-artifact wall time.
2. Exact delivered clip duration.
3. Wall-time cost per output second: `wall time / clip duration`.

Preferred headline form:

`MODEL: X seconds wall time → Y seconds delivered video (Z wall seconds/output second)`

Do not headline throughput percentages. If percentages are useful, phrase them as `less wall time` and place them after the absolute numbers.

## Timing boundary

- Start: ComfyUI accepts the prompt and emits `execution_start`.
- Stop: ComfyUI emits `execution_success` after the MP4 has been saved.
- Source: millisecond timestamps in ComfyUI history.
- Exclude ComfyUI service startup from the headline number.
- Record service startup and restart-to-artifact separately.

## Cold-state definition

A v1 process-cold/model-cold lane requires:

- restart the ComfyUI service before the lane;
- wait for the HTTP API to become ready;
- verify a new service PID;
- submit exactly one workflow;
- require zero cached graph nodes in the execution record;
- preserve Linux filesystem cache and persistent compiled-kernel caches;
- state that this is process-cold/model-cold, not disk-cold.

One successful cold run is the v1 social baseline. For publication-grade variance, run three fresh-process trials and headline the median while preserving every individual result.

## Shared workload

- Positive prompt: use `PROMPT.txt` verbatim.
- Prompt enhancement: disabled.
- Numeric seed: 8112026.
- Target: approximately five seconds, 24 fps, synchronized generated audio.
- Use the nearest valid model-native geometry to the 864×480 target.
- Preserve official/recommended model-native sampler, scheduler, quantization, negative conditioning, and refinement stages; disclose all differences.

The same numeric seed is reproducible within each model lane but is not the same latent across different model families.

## Required validation

- ffprobe exact width, height, frame rate, frame count, duration, video codec, audio codec, sample rate, and channel count;
- full video decode-to-null with no errors;
- confirm non-silent audio;
- visual contact-sheet review for black/grey/corrupt frames;
- SHA-256 of every artifact;
- queue idle after completion;
- scan the run window for GPU reset, VM fault, ring timeout, and OOM.

## Specs shown after the headline

Report:

- GPU and VRAM;
- OS and kernel;
- ROCm version and HIP backend;
- PyTorch and ComfyUI versions;
- model/checkpoint and quantization;
- resolution, frames, fps, clip duration, and audio;
- sampler/scheduler and step count;
- launch flags that materially affect memory behavior.

## Benchmark v1 baseline

| Lane | Wall time | Clip | Wall/output second | Native output |
|---|---:|---:|---:|---|
| LTX-2.5 distilled INT8 | 67.035 s | 5.042 s | 13.30 | 896×512, 121f, 24fps |
| H3 Turbo v4 FP8 | 80.927 s | 5.167 s | 15.66 | 864×480, 124f, 24fps |
| H3 Standard FP8 | 261.038 s | 5.167 s | 50.52 | 864×480, 124f, 24fps |

Canonical detailed report: `/ai/benchmarks/cold-video-comparison-20260812.md`

