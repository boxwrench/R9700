# Dual topology — executed result

## Historical D0

- Port 8191, pinned ComfyUI `c2bcbecd82ec`
- `cuda:0 = R9700`, `cuda:1 = RX 7900 XT`
- Qwen/CLIP placement on `cuda:1` verified
- `H3_FLEX_ATTENTION=off`; smart memory enabled
- Abandoned before sampler entry after its cold loading exceeded the 20%-over-single wall-clock threshold
- Not used for causal comparison

## Matched D1/D2

- Port 8192 during the experiment to avoid collision with D0
- Current ComfyUI/custom-node revisions and the same flags/environment as the isolated single lane
- `H3_FLEX_ATTENTION=auto`; `--disable-smart-memory`
- `cuda:0 = R9700`, `cuda:1 = RX 7900 XT`, hard-gated before submission
- Qwen/CLIP alone targeted to `cuda:1`; ref2va/DiT/sampler remained on `cuda:0`
- Added component: historical `ComfyUI-MultiGPU` commit `b51c99a`

D2 passed 4/4 at 960x544/124f with `match + Turbo`, but freed only 12 MiB of R9700 sampler-entry allocation and had effectively identical warm timing. It is therefore a valid optional topology, not the recommended default.

The reproducible launch is `runtime/launch-matched-dual.sh`; exact per-run device maps are embedded in `tables/results.csv` and `logs/D*.json`.
