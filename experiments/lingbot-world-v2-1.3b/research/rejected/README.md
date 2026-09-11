# Rejected / closed paths — do not repeat without new evidence

Each entry below cost real GPU time and ended with a measured verdict. Full
evidence lives in [../FINDINGS.md](../FINDINGS.md) (section links are line-
anchored to entry titles, not line numbers, which shift as the log grows).

## Closed with measurement (code reverted, world state untouched)

| path | verdict | one-line lesson | evidence |
|---|---|---|---|
| camera-conditioning hoist | REJECT | ~25 ms removable compute, but +365 MB intermediates regressed next-ready ~8 ms | FINDINGS "camcond hoist: REJECTED" |
| regional torch.compile | STOP | 9281 launches/action were not the bottleneck; full-block compile fragmented (27 graphs) then went non-finite | FINDINGS "regional torch.compile Phase 0" |
| static metadata sync removal | REJECT | 704 → 456 syncs, ~0 ms gain; rope tolists were CPU-side, drains were critical-path GPU work | FINDINGS "static host-sync removal" |
| Python-shadowed dynamic KV cursor | REJECT | 704 → 254 syncs, bit-exact over 51 actions / 39 evictions, ~0 ms gain; sync-removal work closed | FINDINGS "Python KV cursor" |
| sync-inline 2× upscale mode | REJECT (mode) | +31 ms next-ready; async mode retained instead | FINDINGS "presentation-only 2x upscale" |
| native high-res DiT as product path | NOT RECOMMENDED | +61–273% tokens/frame breaks the loop budget; presentation upscale won | [../RESOLUTION_PATHS.md](../RESOLUTION_PATHS.md) §7 |

## Closed earlier (see FINDINGS / ledger for numbers)

- 14B GGUF path; same-GPU VAE/clean overlap (DiT 4.17 s → 19.83 s under
  contention); broad clean-pass shortcut (clean K/V differ, load-bearing);
  zero-latent tiling (approximate, rejected); BF16 VAE (8× less accurate);
  chunk 2/3 and resolution-upscaling games; 1.3B quantization; Vulkan;
  broad sweeps — see PERFORMANCE.md "Rejected with evidence" and the
  research index.
- Official v2 deployment refiner: unreleased, unplannable
  (RESOLUTION_PATHS.md §3). FlashVSR port: assessed, hard ROCm work, not
  started (RESOLUTION_PATHS.md §4).

## Rules for reopening

1. A newer ROCm/Torch/Inductor that changes codegen does NOT reopen a
   rejection by itself — re-run the bounded probe first.
2. Reopen only with a new mechanism the original probe did not test
   (e.g. compile *after* a future sync-topology change is a new question,
   but the 2026-09-10 topology made it moot — say which changed).
3. Quote the old numbers in the new proposal so the comparison is explicit.
