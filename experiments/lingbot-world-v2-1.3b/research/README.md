# Research archive — LingBot World v2 1.3B

Chronological and topical records. The authoritative present-tense docs live
one level up (`../README.md`, `../PERFORMANCE.md`, …); nothing here is required
reading to run the project. Do not update these files with new results — append
new findings to `../PERFORMANCE.md` or a dated note here.

## Index

| record | why it exists |
|---|---|
| [FINDINGS.md](FINDINGS.md) | running lab notebook, oldest-first: gfx1201 Conv3d root-cause (F1–F9), TAE investigation, 3-step-A acceptance, viewer fixes, `play.sh`, window-12 acceptance, timecond acceptance, camcond rejection |
| [INTERACTIVE.md](INTERACTIVE.md) | era narrative: persistent runner, chunk/streaming passes, VAE FP16, Phase A overhead, fifth-forward impossibility proof, TAE + 3-step-A acceptance sections. Frozen at the 2.81 s era except its last two sections; numbers with 464×832 / 1508-token geometry are superseded |
| [LATENCY_LEDGER.md](LATENCY_LEDGER.md) | 26-lever analysis + optimization-history table + standing rules. Frozen at 2.81 s; the history table's later rows live on in `../PERFORMANCE.md` |
| [WAN_OPTIMIZATION_CANDIDATES.md](WAN_OPTIMIZATION_CANDIDATES.md) | ecosystem-idea backlog (compile, CK backend, …), frozen at 2.81 s. Treat as ideas, not measurements |
| [HANDOFF.md](HANDOFF.md) | cold-start note from the 2.81 s era. Superseded by `../README.md` + `../QUICKSTART.md`; kept for its machine-lore details, several now stale (head hash, perf table, 27144-token window) |
| [logs/](../logs/) | raw captured tool output, never edited after the fact (tracked in git) |
| [../patches/](../patches/) | upstream compatibility patches for the pinned checkout |

## One-off harnesses (kept in place, indexed here — not moved)

`bench.py` (upstream-pipeline benchmark; driven by `run_baseline.sh`, which
reproduces the pinned 480*832/chunk-4 bring-up baseline — historical, keep
flags as-is), `conv3d_micro.py`, `vae_repro.py`, `split_sweep.py`,
`sweep2.sh`, `sweep_chunks.sh`, `precision_ab.py`, `profile_decoder.py`,
`fifth_forward_analysis.py`. Later measurement harnesses live outside the repo
with their result data, not here.

## Negative results (kept deliberately)

Fifth-forward removal (impossible, V rel 1.13), partial fifth prefix,
same-GPU overlap, zero-tiling, BF16 VAE, camera-conditioning hoist (bit-identical
yet −8 ms with +365 MB), cold single-latent canonical decode (invalid probe),
chunk sweeps (closed at 1), 14B GGUF track (closed). Deleting these would
invite re-running them.
