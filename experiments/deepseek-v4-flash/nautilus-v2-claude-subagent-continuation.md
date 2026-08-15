# Nautilus v2: Claude subagent continuation

## Purpose and provenance

This documents a continuation of `optimization-plan.md` under a different
execution model. The plan, phases, and promotion thresholds are unchanged.
What changed is who runs each experiment: instead of a single operator or a
Gemini-based worker, each bounded chunk is dispatched to a Claude Code
subagent (model `haiku` for mechanical/bounded work, `sonnet` for chunks
needing more in-flight judgment) launched by an orchestrating Claude Code
session running on Nautilus itself, with GPT-5.6 Sol retained as the judge
for each review gate.

The full working package — worker protocol, per-chunk handoff prompts, Sol
review gate definitions, evidence packet and result ledger templates, and
in-progress state — lives outside this repo at
`~/ai/experiments/deepseek-nautilus-v2/` (not tracked here; it is a working
tree, not a publication). This file is the pointer from the public record
back to that working tree, and the landing point for what graduates out of
it.

## What lands here, and where

Two kinds of result come out of the nautilus-v2 working tree, and they land
in different places for the same reason `rdna-boosts-experiments.md` is kept
separate from `optimization-plan.md`: a hardware-wide finding shouldn't be
stranded inside a DeepSeek-specific document where nobody doing unrelated
R9700 work (e.g. the video-generation lanes documented elsewhere in this
repo) would think to look for it.

- **Workload-specific results** (batch/ubatch, thread count, KV cache type,
  DSpark/ngram-mod speculative parameters, CPU-MoE placement, context size) —
  these extend `optimization-plan.md`'s existing phases and, once a result is
  promoted past matched-control verification, `profiles.tsv` and
  `data/experimental/deepseek-v4-flash.tsv` following the schema in
  `docs/data-schema.md`. Only verified, reproduced numbers land in the TSV —
  no placeholder or projected values, consistent with this repo's existing
  "unsupported data is `unknown`, not fabricated" convention.

- **Host-wide results** (CPU affinity/SMT policy, PCIe ASPM, amd-pstate/EPP,
  GPU DPM performance state, allocator, compiler/build flags, SDMA toggle,
  mmap/load-mode) — these are candidates for a new hardware-wide companion
  doc, `host-tuning-experiments.md`, sibling to `rdna-boosts-experiments.md`.
  Nothing lands there yet; it gets created once the nautilus-v2 working tree
  actually promotes a host-wide candidate past its chunk's Sol review gate
  (Review A for Chunk 1, Review B for Chunk 2). A host-wide win found while
  testing DeepSeek should not be assumed safe as a standing system default
  for other workloads without separate confirmation — the doc will say so
  explicitly per row.

## Status

As of this writing, the nautilus-v2 working tree has completed preflight
(Chunk 0: machine/CPU/GPU/power state capture, Gold binary and launch
command verified against `serve-best.sh` on disk) and is finishing the
same-session Gold benchmark confirmation before Chunk 1 (fast host-side
wins: CPU topology/SMT, PCIe ASPM, amd-pstate/EPP, batch/ubatch) is
dispatched. No results have been promoted yet, so no data has landed in
this repo beyond this pointer file.

## Source pins

Same Gold stack as the rest of this campaign: upstream HIP llama.cpp commit
`7b13a8404d7e219c13d1a243e2a21a857a6e99d9` (`gfx1201`), DeepSeek V4 Flash
`UD-Q4_K_XL` target, DSpark `Q8_0` drafter, 64K context, Q8/Q8 GPU KV, 42
target / 0 draft CPU-MoE, 12 threads. See
`~/ai/experiments/deepseek-nautilus-v2/state/GOLD_COMMAND.txt` in the working
tree for the exact launch command as extracted from `serve-best.sh`.
