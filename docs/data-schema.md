# Data schema

`data/results.tsv` is the original normalized comparison table. Fields are tab-separated and use seconds, pixels, frames, hertz, and SHA-256 strings explicitly.

## Direct measurements

`run_id`, `date`, `model`, `lane`, `prompt_id`, `seed`, `startup_seconds`, `prompt_to_artifact_seconds`, `restart_to_artifact_seconds`, `width`, `height`, `frames`, `fps`, `duration_seconds`, `sampler`, `scheduler`, `steps`, `native_audio`, `workflow_path`, `workflow_sha256`, `artifact_source_path`, `artifact_sha256`, `validation_status`, and `measurement_notes` are copied from the canonical run records or their named source artifacts.

Model repository, revision, filename, quantization, encoder, VAE, and adapter columns are provenance fields. `unknown` means the source did not support a value. `cold_state` records the procedure rather than pretending it is a numeric measurement.

## Derived fields

`wall_seconds_per_output_second` is calculated as:

`prompt_to_artifact_seconds / duration_seconds`

The verifier recalculates it and allows only rounding-level disagreement. `restart_to_artifact_seconds` should equal startup plus prompt-to-artifact time within the source's rounding precision.

## Unsupported or intentionally absent data

The original canonical table does not fabricate component timing, peak VRAM, peak RAM, GPU power, quality scores, or variance statistics. Model file hashes are `unknown` for LTX-2.5 because the local source did not provide them.

That rule applies to the canonical table. Later experimental tables may include component timing, power, or VRAM only when those values were directly measured by the named experiment.

## 2026-08-18 ComfyUI wall-time campaign tables

These tables are engineering experiment records rather than replacements for `data/results.tsv`:

- [`h3-walltime-20260818.tsv`](../data/experimental/h3-walltime-20260818.tsv) — H3 cache/residency states, mode matrix, explicit Qwen offload A/B, and production sanity run.
- [`ltx25-walltime-20260818.tsv`](../data/experimental/ltx25-walltime-20260818.tsv) — LTX encoder-format, cache-state, and token-floor wall-time measurements.
- [`ltx25-token-floor-20260818.tsv`](../data/experimental/ltx25-token-floor-20260818.tsv) — direct Gemma sequence-floor timing and cropped conditioning-tensor comparison.
- [`workflow-transitions-20260818.tsv`](../data/experimental/workflow-transitions-20260818.tsv) — one representative transition per LTX/H3 lane.
- [`minimax-music3-baseline-20260818.tsv`](../data/experimental/minimax-music3-baseline-20260818.tsv) — MiniMax Music 3 cold/warm baseline decomposition.

Common interpretation rules:

- blank means not measured or not available from that run; it is not zero
- seconds are wall/component durations from the named harness unless the column explicitly says profiler/self GPU time
- peak VRAM and power values appear only where directly sampled
- profiler rows may be nested; nested percentages are not additive unless explicitly normalized into non-overlapping buckets
- these are mostly one- or two-run engineering discriminators, not publication-grade variance studies
- workloads with different geometry, frame count, step count, cache state, or model format must not be presented as direct before/after headline comparisons

## Experimental dual-GPU table

[The experimental nine-row table](../data/experimental/dual-gpu-residency.tsv) records three single-R9700 controls, three initial dual-GPU runs with aggressive offload, and three corrected dual-GPU residency runs. It uses the same timing boundary and derived ratio as the canonical table, while also recording topology, state, model placement, and source artifact hash. These single-pass measurements support an engineering decision; they are not publication-grade variance statistics.

The dual-GPU experiment is now a historical H3 record. Current H3 operational guidance is in [`minimax-h3-r9700-optimization-20260818.md`](minimax-h3-r9700-optimization-20260818.md).
