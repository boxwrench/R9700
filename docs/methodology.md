# Methodology

## Headline metric

Every result reports, in this order:

`MODEL: prompt-to-saved-artifact wall time → delivered video duration (wall seconds/output second)`

The start boundary is ComfyUI accepting the prompt and emitting
`execution_start`. The stop boundary is `execution_success` after the MP4 is
saved. The headline excludes service startup; startup and restart-to-artifact
are separate columns in `data/results.tsv`.

## Process-cold/model-cold procedure

For each v1 baseline lane, the service was restarted, the HTTP API was allowed
to become ready, a new service PID was verified, and exactly one workflow was
submitted. The execution record reported zero cached graph nodes. Linux
filesystem caches and persistent compiled-kernel caches were preserved. This is
process-cold/model-cold, not disk-cold.

One successful cold run per lane is a social baseline, not a variance study.
For publication-grade variance, run three fresh-process trials and publish the
median while preserving every individual result.

## Shared workload

- Positive prompt: `prompts/neutral-brass-robot.txt`, verbatim.
- Prompt enhancement: disabled.
- Numeric seed: `8112026` where accepted by the workflow.
- Target: approximately five seconds at 24 fps with synchronized generated audio.
- Model-native geometry, sampler, scheduler, quantization, and refinement stages
  are preserved and disclosed.

The same numeric seed is reproducible within a lane. It is not the same latent
across model families. LTX-2.5 natively produced 896×512 because its two-stage
latent grid has alignment constraints; H3 natively produced 864×480.

## Validation boundary

The source run report records exact stream geometry, frame count, frame rate,
duration, codecs, audio rate/channel count, full decode, non-silent audio,
queue-idle state, visual review, artifact SHA-256, and a run-window scan for GPU
reset, VM fault, ring timeout, and OOM. See
[`docs/validation.md`](validation.md) and the dated reports in
[`data/runs/2026-08-12/`](../data/runs/2026-08-12/).
