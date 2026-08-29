# Next steps

The requested memory/performance characterization is complete. No further render is required for the publication claim.

Optional follow-up work should answer a different question rather than repeat this matrix:

1. Perceptual A/B of `match` versus a custom intermediate reference cap, using the same seed, to quantify identity fidelity.
2. Test the smaller NVFP4 Qwen encoder only if encoder cold-start time becomes operationally important; sampler headroom is already known not to improve materially through second-GPU placement.
3. Test reference-video or multi-reference workloads separately. Do not generalize the single-image result to those larger conditioning contexts.

The production recommendation remains single R9700, 960x544/124f, `match + Turbo`.
