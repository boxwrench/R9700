# FastH3 + FastVideo VSA on gfx1201 — experiment bundle

Reproduction material for
[`docs/fasth3-vsa-r9700-bringup-20260902.md`](../../docs/fasth3-vsa-r9700-bringup-20260902.md).

**Status: EXPERIMENTAL.** Production H3 is unchanged and remains Turbo v4 FP8
with pre-sampler Qwen offload. Everything here ran in an isolated ComfyUI copy
and venv on port 8191; `/ai/comfyui` and `/ai/environments/comfyui-h3` were not
modified.

## Contents

| File | What it is |
|---|---|
| `probe_vsa_gfx1201.py` | Standalone VSA correctness/execution probe. No ComfyUI, no H3 weights. Run this first on any new stack. |
| `vsa-probe-gfx1201.log` | Probe output from the recorded run. |
| `run_workflow.py` | Submits a workflow, reports the v1 timing boundary, and **fails the run** if `[H3-VSA] ACTIVE` is missing or a fallback line appears. |
| `run_sweep.sh` | topk_ratio sweep driver. |
| `validate_outputs.sh` | ffprobe geometry/codec/stream checks, decode-to-null, non-silent audio, sha256. |
| `contact_sheet.py` | Matched contact sheets: one row per timestamp, one column per configuration. |
| `launch.sh` | Isolated ComfyUI launcher (port 8191, golden flags incl. `--disable-mmap`). |
| `workflows/` | API-format graphs for all lanes and the sweep. |
| `sweep-results.tsv` | Sampling / s-per-forward / wall / peak VRAM per topk ratio. |
| `quality-vs-dense.tsv`, `quality-vs-vsa100.tsv` | SSIM/PSNR diagnostics. |
| `temporal.tsv` | Frame-to-frame luma delta. |
| `asset-sha256.txt`, `stack.txt` | Provenance. |

Generated media: `/ai/artifacts/runs/fasth3-vsa/`.

## Setup notes that are easy to get wrong

1. **`vsa` must be installed as Python sources only.** `pip install vsa` builds a
   Hopper `-arch=sm_90a` extension and fails on this card. Take the `vsa==0.0.3`
   sdist, drop its `vsa/` directory into site-packages, and install `pytest`
   (imported at module load). `vsa/__init__.py` then selects the Triton path
   because `torch.cuda.get_device_capability(0)` is `(12, 0)`, not `(9, 0)`.
2. **Assert VSA is active.** A run can produce a perfectly valid video with
   attention silently dense. Always require the `[H3-VSA] ACTIVE — tokens=N
   topk=k/n gate=on` line.
3. **`min_tokens` matters.** Below roughly 2.7k tokens VSA is about 2x *slower*
   than dense on this card. The default 4096 guard is directionally right but
   was not tuned for gfx1201.
4. **VSA @1.00 is not the dense lane.** Keep both; they are different code paths
   with a 47 s vs 34 s sampling gap.
5. **ComfyUI caches whole graphs.** An identical prompt and seed returns the
   cached result in ~0.001 s. Change the seed or the graph for a real warm run.
6. **`torch.cuda.*` on ROCm is not evidence of a CUDA-only path.** PyTorch ROCm
   exposes that namespace; check for actual CUDA/PTX/CuTe kernels or compiled
   `_C` extensions instead.

## Reproduce

```bash
# 1. kernel-level check, no models needed
python probe_vsa_gfx1201.py

# 2. isolated server
./launch.sh &

# 3. one lane, with the anti-fallback assertion
python run_workflow.py workflows/fasth3-vsa-864x480-124f.json --port 8191 \
       --server-log logs/server.log

# 4. sweep + validation + sheets
./run_sweep.sh
./validate_outputs.sh
python contact_sheet.py -o sheet.png -t 1.5,2.75,4.0 \
       "dense=.../dense_00001_.mp4" "0.10=.../vsa-topk0.10_00001_.mp4"
```
