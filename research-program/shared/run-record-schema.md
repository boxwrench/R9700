# Run Record Schema

One record per benchmark run, across both tracks. Raw records are written as
**JSONL** — one JSON object per line, append-only, never rewritten in place.

A record describes **one execution**. Aggregation (mean, stdev, paired deltas)
is derived downstream by
[`../tracks/B-rocmfpx-nvfp4/scripts/06_compare_runs.py`](../tracks/B-rocmfpx-nvfp4/scripts/06_compare_runs.py),
never stored as if it were a measurement.

---

## Design rules

1. **Absent beats guessed.** A field that could not be determined is `null` or
   omitted. It is never filled with a default, an estimate, or a value carried
   over from another run.
2. **Every performance number carries provenance.** See `provenance` below.
   A number with no identifiable source line does not enter the record.
3. **Records are immutable.** Corrections are appended as a new record with a
   `supersedes` field, so the original stays auditable.
4. **One record, one variable.** If two things changed between runs, the record
   must say so in `notes`; it does not make the pair a valid A/B.

---

## Fields

### Identity

| Field | Type | Notes |
|---|---|---|
| `timestamp` | string | ISO-8601 UTC, run **start** |
| `campaign` | string | e.g. `qwen3-8-27b-r9700` |
| `track` | string | `A` or `B` |
| `experiment_id` | string | stable ID, e.g. `B1-serial-baseline-03` |
| `run_index` | int | repetition number within the experiment |
| `supersedes` | string\|null | `experiment_id` + `run_index` this corrects |

### Source

| Field | Type | Notes |
|---|---|---|
| `git_repo` | string | remote URL — **not** just a local path; the `charlie12345`/`ciru-ai` ambiguity makes this load-bearing |
| `git_sha` | string | full 40-char SHA |
| `git_branch` | string | |
| `git_dirty` | bool | true if `git status --porcelain` is non-empty |
| `git_dirty_files` | array\|null | required when `git_dirty` is true |
| `harness_sha` | string | research-repo SHA of the scripts used |

### Model

| Field | Type | Notes |
|---|---|---|
| `model_path` | string | |
| `model_sha256` | string | |
| `model_bytes` | int | |
| `model_arch` | string | GGUF `general.architecture` |
| `quant` | string | ftype, e.g. `NVFP4`, `Q4_K_XL` |
| `model_provenance` | string | `upstream-published` \| `locally-converted` \| `unknown` — a local conversion is **not** interchangeable with an upstream checkpoint |

### Hardware and toolchain

| Field | Type | Notes |
|---|---|---|
| `gpu` | string | e.g. `AMD Radeon AI PRO R9700` |
| `gfx_arch` | string | e.g. `gfx1201` |
| `gpu_index` | int | **enumeration index under the backend actually used** — the R9700 is not index 0 on this host |
| `code_object_arch` | string\|null | arch **verified emitted** by the build; guards the silent gfx1200/gfx1201 hazard |
| `driver` | string | e.g. `radv Mesa 25.2.8` |
| `rocm` | string\|null | |
| `mesa` | string\|null | |
| `kernel` | string | |
| `cpu` | string | |
| `ram_gib` | number | |

### Configuration

| Field | Type | Notes |
|---|---|---|
| `backend` | string | `vulkan` \| `hip` \| `cpu` |
| `build_dir` | string | |
| `command` | array | argv, **not** a shell string |
| `env` | object | only variables that alter behaviour, each recorded even when set to a default |
| `ctx` | int | |
| `batch` | int | |
| `ubatch` | int | |
| `parallel` | int | |
| `flash_attention` | bool | |
| `kv_type` | string | e.g. `f16` |
| `mtp_enabled` | bool | |
| `n_max` | int\|null | null when `mtp_enabled` is false |
| `p_min` | number\|null | null when `mtp_enabled` is false |

### Workload

| Field | Type | Notes |
|---|---|---|
| `prompt_id` | string | stable ID into the prompt set |
| `seed` | int | |
| `prompt_tokens` | int | |
| `generated_tokens` | int | |

### Results

| Field | Type | Notes |
|---|---|---|
| `pp_tok_s` | number\|null | |
| `decode_tok_s` | number\|null | serial **or** MTP per `mtp_enabled` — never a blend |
| `wall_seconds` | number\|null | |
| `vram_gib` | number\|null | high-water mark |

### Speculative counters

Raw counts first; derived rates are computed downstream, not stored as measurements.

| Field | Type | Notes |
|---|---|---|
| `verification_rounds` | int\|null | |
| `draft_generated` | int\|null | |
| `draft_accepted` | int\|null | |
| `p0` | number\|null | see [`metrics.md`](metrics.md) |
| `joint_p1` | number\|null | |
| `conditional_p1` | number\|null | |

### Provenance and classification

| Field | Type | Notes |
|---|---|---|
| `provenance` | object | maps a result field name → `{source_line, source_file}` for the log line the value was parsed from |
| `classification` | object | maps a result field name → `MEASURED` \| `CALCULATED` \| `ESTIMATED` \| `INFERRED` |
| `status` | string | `ok` \| `failed` \| `discarded` — failed runs are recorded, not deleted |
| `notes` | string | free text; the place to say what is unusual, confounded, or unexplained |

---

## Example

```json
{
  "timestamp": "2026-08-17T04:12:03Z",
  "campaign": "qwen3-8-27b-r9700",
  "track": "B",
  "experiment_id": "B1-serial-baseline",
  "run_index": 3,
  "supersedes": null,
  "git_repo": "https://github.com/charlie12345/ROCmFPX.git",
  "git_sha": "f4b2c5a3edfd183274641094d0db0fcc8092c0ad",
  "git_branch": "main",
  "git_dirty": false,
  "git_dirty_files": null,
  "harness_sha": "cf2454b69cc1146af789f2dbf1eeefad0a1fe935",
  "model_path": "/ai/models/PENDING/PENDING.gguf",
  "model_sha256": "PENDING",
  "model_bytes": null,
  "model_arch": "qwen3_5",
  "quant": "NVFP4",
  "model_provenance": "unknown",
  "gpu": "AMD Radeon AI PRO R9700",
  "gfx_arch": "gfx1201",
  "gpu_index": 1,
  "code_object_arch": null,
  "driver": "radv Mesa 25.2.8-0ubuntu0.24.04.2",
  "rocm": "7.2.1",
  "mesa": "25.2.8",
  "kernel": "7.0.0-28-generic",
  "cpu": "AMD Ryzen 7 9800X3D",
  "ram_gib": 186,
  "backend": "vulkan",
  "build_dir": "/ai/scratch/ROCmFPX-audit/build-vulkan",
  "command": ["llama-bench", "-m", "PENDING.gguf", "--device", "Vulkan1"],
  "env": {"GGML_VK_VISIBLE_DEVICES": "1"},
  "ctx": 8192,
  "batch": 2048,
  "ubatch": 512,
  "parallel": 1,
  "flash_attention": true,
  "kv_type": "f16",
  "mtp_enabled": false,
  "n_max": null,
  "p_min": null,
  "prompt_id": "std-256",
  "seed": 1234,
  "prompt_tokens": 256,
  "generated_tokens": 512,
  "pp_tok_s": null,
  "decode_tok_s": null,
  "wall_seconds": null,
  "vram_gib": null,
  "verification_rounds": null,
  "draft_generated": null,
  "draft_accepted": null,
  "p0": null,
  "joint_p1": null,
  "conditional_p1": null,
  "provenance": {},
  "classification": {},
  "status": "ok",
  "notes": "Schema example. Nulls are literal: no run has been performed."
}
```

**This example is a template, not a result.** Every performance field is `null`
because no Track B inference run has been executed.
