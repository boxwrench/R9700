# MiniMax H3 dual-GPU residency experiment — historical record

This path is preserved for compatibility with older links, but the 2026-08-12 dual-GPU Qwen-residency experiment is no longer the current operational recommendation.

The current H3 result is the 2026-08-18 single-R9700 wall-time campaign, which found that explicitly offloading Qwen3-VL-32B after conditioning and before sampling reduced sampler time from 29.51 s to 22.15 s in the paired A/B, with a 4.03 s offload cost and a net 8.1% wall-time improvement. A production sanity run reproduced the ~22.4 s sampler state.

Read the current record:

- [`minimax-h3-r9700-optimization-20260818.md`](minimax-h3-r9700-optimization-20260818.md)

The original dual-GPU experiment, including its useful `--disable-smart-memory` failure mode, host-RAM observations, exact workflows, and measurement table, is archived unchanged in substance here:

- [`archive/dual-gpu-residency-20260812.md`](archive/dual-gpu-residency-20260812.md)

Historical conclusion: the corrected RX 7900 XT encoder-residency lane improved changed-prompt wall time by 7.9% in that experiment and reduced observed host-RAM peak, but it remained below the adoption threshold and did not replace single-R9700 as the default.
