# Results — MiniMax H3 R2V on R9700

Experiment date: 2026-08-29. A pass requires one cold and three warm completions.

## Outcome

`ref_image_size=max -> match` was sufficient by itself. The target 960x544, 124-frame, five-step Turbo R2V job completed 4/4 times on one R9700. No spatial or frame reduction was required.

Moving Qwen3-VL/CLIP to the RX 7900 XT did not materially increase sampler headroom. At sampler entry it reduced R9700 process allocation by exactly 12,582,912 bytes (12 MiB, 0.0117 GiB) in both the cold and warm comparisons. Warm wall time was effectively identical. The recommended lane is therefore the single R9700 with `ref_image_size=match`.

## Summary

| Case | Topology | Workload | Result | Cold wall | Warm wall median | Warm sampler median | R9700 entry allocated | R9700 peak allocated |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BASE | single | 960x544/124f, `max`, Turbo | OOM | — | — | — | 25.62 GiB reported at failure | 29,131 MiB reported peak |
| S1 | single | 960x544/124f, `match`, Turbo | 4/4 pass | 168.124 s | 120.099 s | 95.753 s | 20.21 GiB warm | 23.12 GiB warm |
| S2 | single | 960x544/124f, `match`, LoRA branch removed | 4/4 pass | 164.122 s | 116.098 s | 91.800 s | 19.78 GiB warm | 22.07 GiB warm |
| D1 | matched dual | 864x480/124f, `match`, Turbo | 4/4 pass | 112.101 s | 90.094 s | 66.307 s | 20.19 GiB warm | 22.52 GiB warm |
| D2 | matched dual | 960x544/124f, `match`, Turbo | 4/4 pass | 144.959 s | 120.101 s | 95.870 s | 20.20 GiB warm | 23.11 GiB warm |

GiB values above are bytes / 2^30. Exact per-run values are in `tables/results.csv`.

## Answers to the primary questions

1. **Does `max -> match` alone fit?** Yes: S1 passed one cold and three warm runs at the full target workload.
2. **Is Turbo-off required?** No. The valid LoRA-off case also passed and confirms the LoRA branch costs memory, but S1 already works with Turbo.
3. **Smallest reduction required?** None. The reduction ladder was intentionally skipped after the unchanged 960x544/124f target proved stable.
4. **How much does the 7900 XT free at sampler entry?** 12 MiB of R9700 process allocation in the matched 960x544 comparison—negligible. Cold CLIP residency on the 7900 XT was about 15.93 GiB; after warm reuse only about 360 MiB remained there at sampler entry.
5. **Does matched dual run the target reliably?** Yes, D2 passed 4/4.
6. **Dual performance cost?** Warm wall median was 120.101 s dual versus 120.099 s single (+0.0017%); warm sampler median was +0.122%. Cold dual was 13.78% faster. This is inside the 20% cutoff, but offers no recurring sampler-memory or warm-time benefit.

## Causal observations

The known failure requested a 3.30 GiB allocation with 2.97 GiB free in the Turbo LoRA nested `F.linear` path. In the reconstructed controlled graph, `max` produced an activation shape ending in 3,852,544 elements; `match` produced 1,824,768, a 52.6% reduction. That explains why reference sizing crossed the boundary without reducing output video resolution or duration.

S2 is a real LoRA memory A/B. Turbo strength zero was rejected because the implementation still executes `F.linear(F.linear(x, down), up)` and applies the scale afterward. The S2 graph removes the LoRA branch before those operations while retaining the same five-step Turbo sampler/scheduler. Relative to S1, it saved about 0.435 GiB at entry and 1.054 GiB peak allocated, and reduced warm wall time by 3.33%. It is useful causal evidence, not the recommended production setting.

The dual cold run partially placed the 25 GB FP8 Qwen encoder on the 7900 XT, with the remainder offloaded. On warm runs ComfyUI had already removed almost all encoder residency before sampling. This directly explains why explicit second-GPU placement did not create useful R9700 sampling headroom under `--disable-smart-memory`.

## Historical dual lane

D0 verified the historical device topology and encoder placement, but was abandoned before sampler entry. Its pinned older runtime and memory-first loading exceeded the user’s 20%-over-single cold-wall threshold before sampling began. It is not a render failure and is not used in causal comparisons. See `logs/D0-abandoned.json`.

## Output validation

All 16 successful MP4 artifacts reported the expected H.264 geometry, 24 fps, 124 video frames, and stereo 32 kHz audio, and all fully decoded with `ffmpeg -v error` without errors. S1/S2/D2 were 960x544; D1 was 864x480. Raw Comfy history, probe data, output paths, and exact measurements are retained in `logs/*.json` and `tables/results.csv`.

## Recommendation

Use one R9700, retain Turbo, and set R2V references to `match`. Do not dedicate the RX 7900 XT to Qwen for this workload: its warm performance is unchanged and its measured R9700 headroom gain is only 12 MiB. Reserve `max` for an explicit identity-fidelity comparison after testing a smaller output or a custom-capped reference size.

The original submitted graph was not recoverable from Comfy history. The controlled run reconstructs the model, LoRA, reference asset, dimensions, frame count, and schedule from the live runtime and journal; the prompt was held fixed within this experiment but may differ from the original failed prompt.
