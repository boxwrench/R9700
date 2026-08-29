# Status — complete, 2026-08-29

- S1 `match + Turbo`, 960x544/124f, single R9700: 4/4 pass.
- S2 valid LoRA-branch removal: 4/4 pass.
- Reduction ladder: not required; full target already stable.
- D0 historical topology: mapped correctly, abandoned before sampling after exceeding the 20% cold-wall limit.
- D1 matched-dual sanity, 864x480/124f: 4/4 pass.
- D2 matched-dual target, 960x544/124f: 4/4 pass.
- Matched dual R9700 sampler-entry saving: 12 MiB; warm wall overhead: +0.0017%.
- Recommendation: single R9700, `ref_image_size=match`, Turbo retained.
- Production `comfyui-h3.service`: restored and verified healthy on `127.0.0.1:8190`; service and launcher were not edited.

See `RESULTS.md`, `tables/results.csv`, and `logs/*.json`.
