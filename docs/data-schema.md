# Data schema

`data/results.tsv` is the normalized comparison table. Fields are tab-separated
and use seconds, pixels, frames, hertz, and SHA-256 strings explicitly.

## Direct measurements

`run_id`, `date`, `model`, `lane`, `prompt_id`, `seed`, `startup_seconds`,
`prompt_to_artifact_seconds`, `restart_to_artifact_seconds`, `width`, `height`,
`frames`, `fps`, `duration_seconds`, `sampler`, `scheduler`, `steps`,
`native_audio`, `workflow_path`, `workflow_sha256`, `artifact_source_path`,
`artifact_sha256`, `validation_status`, and `measurement_notes` are copied from
the canonical run records or their named source artifacts.

Model repository, revision, filename, quantization, encoder, VAE, and adapter
columns are provenance fields. `unknown` means the source did not support a
value. `cold_state` records the procedure rather than pretending it is a
numeric measurement.

## Derived fields

`wall_seconds_per_output_second` is calculated as:

`prompt_to_artifact_seconds / duration_seconds`

The verifier recalculates it and allows only rounding-level disagreement.
`restart_to_artifact_seconds` should equal startup plus prompt-to-artifact time
within the source's rounding precision.

## Unsupported or intentionally absent data

The table does not fabricate component timing, peak VRAM, peak RAM, GPU power,
quality scores, or variance statistics. Model file hashes are `unknown` for
LTX-2.5 because the local source did not provide them.

## Experimental dual-GPU table

[The experimental nine-row table](../data/experimental/dual-gpu-residency.tsv)
records three single-R9700 controls, three initial dual-GPU runs with
aggressive offload, and three corrected dual-GPU residency runs. It uses the
same timing boundary and derived ratio as the canonical table, while also
recording topology, state, model placement, and source artifact hash. These
single-pass measurements support an engineering decision; they are not
publication-grade variance statistics.
