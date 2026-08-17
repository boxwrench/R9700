# R9700 / Qwen3.8 Research Handoff

*Frozen 2026-08-17. Written to restart the program without conversation history.*

---

## Current State

| | |
|---|---|
| **Track A** — Vulkan / Q4_K_XL / native MTP | `PAUSED AT ENTRY 19 EARLY GATE` |
| **Track B** — ROCmFPX / native NVFP4 | `ARCHIVED / REOPENABLE` ([closeout](research-program/tracks/B-rocmfpx-nvfp4/CLOSEOUT.md)) |
| **Latest accepted entries** | 1–18B locked; **19 logged but `BLOCKED`** |
| **Production `llama.cpp`** | `/ai/github/llama.cpp` @ `ad1de39` / `b10448` — **clean, untouched** |
| **Production baseline** | **52.96–53.08 tok/s** native MTP |

**Production stack:** Radeon AI PRO R9700 (`gfx1201`) / Vulkan / RADV /
Qwen3.8-27B-UD-Q4_K_XL / native MTP / `n_max=2`, `p_min=0.3` / `f16` KV / FA on /
ubatch 512 / parallel 1.

Authoritative log: [`docs/qwen3-8-27b-experiment-log.md`](docs/qwen3-8-27b-experiment-log.md).
Live status: [`research-program/STATUS.md`](research-program/STATUS.md).

---

## What We Learned

**Entries 1–17 remain established.** Vulkan beats HIP on gfx1201 for this model;
`f16` KV is the right balance; `n_max=2, p_min=0.3` is the stable operating
point. Verification-side micro-optimization is **exhausted** — Entries 16 and 17
both closed at effectively neutral wall time against a 41.26 ms verifier floor.
That is what pushed the program to the proposer side.

**The proposer's LM head is genuinely expensive.** The Q6_K head
(`[248320, 5120]`, 1.04 GB) costs ~1652 µs per dispatch, and reducing its row
count cuts that hard: ~452 µs at 64K, ~234 µs at 32K. This is measured and
solid — and it is **mechanism-level only**. It has never been converted into an
end-to-end win.

**Two separate implementations of vocabulary trimming have now failed, and
neither failure was about vocabulary.** Entry 18 failed in the reconstruction
plumbing; Entry 19 is blocked in logits extraction. The reduced head, the `d2t`
map, and the row slicing have been correct throughout.

**Track B is finished as an inquiry.** Native NVFP4 runs on gfx1201, the uniform
derivative is the better artifact within that track, observed runtime is
effectively W4A16 rather than the advertised W4A4, and none of it beat Track A.

---

## Trusted Performance Baseline

| Measurement | Value | Status |
|---|---|---|
| FULL native MTP, corrected | **52.96–53.08 tok/s** | authoritative |
| FULL native MTP, historical | ~53.2–53.8 tok/s | independent corroboration |
| Serial decode, historical | ~29.4 tok/s | authoritative |
| Head cost — FULL 248K | ~1652 µs/dispatch | mechanism-level only |
| Head cost — 64K | ~452 µs/dispatch | mechanism-level only |
| Head cost — 32K | ~234 µs/dispatch | mechanism-level only |

The corrected baseline landing on the historical figure — without being tuned
toward it — is the main reason to trust the Entry 19 `llama-context.cpp`
correction.

**Head timings do not imply an end-to-end speedup.** No trimmed configuration
has ever produced one.

---

## Entry 18 — Failed Reconstruction Implementation

`FAILED IMPLEMENTATION / CONCEPT REMAINS OPEN`

The unseen `n_max=2` holdout collapsed: zero accepted drafts on both trimmed
arms, throughput roughly halved (16.82 → 8.32 / 8.69 tok/s).

The failure is in the `FILL(-INFINITY)` + `SET_ROWS` full-vocabulary
reconstruction and the backend sampling path that consumed it — **not** in the
reduced-vocabulary concept. Scattering into a 248,320-entry buffer distorted the
candidate softmax below `p_min`, and graph-split latency added an unamortized
28–33 ms per round.

**Entry 18B** then localized it precisely: reduced heads are genuinely
materialized, copied rows produce matching logits, `d2t` is valid I64 and maps
correctly, and `FILL` + `SET_ROWS` work *in isolation*. The architecture only
fails when composed against the real sampler/scheduler.

> [!WARNING]
> **Do not attempt to repair the `SET_ROWS` reconstruction path.** It is closed
> on architectural grounds, not on a fixable bug.

Entry 18's −48% to −50% result **must not** be cited as evidence against
reduced-vocabulary MTP heads. It is evidence against one implementation.

---

## Entry 19 — Blocked

`BLOCKED AT EARLY GATE` — blocked, not failed. No holdout was run.

Direct sampling: sample in the reduced space, map the local index through `d2t`,
never allocate or scatter a full-vocabulary destination.

**Confirmed working:** reduced heads materialize at the right widths
(`nextn.shared_head_head rows=65536` / `rows=32768` vs `model.output rows=248320`);
`d2t` maps correctly (local `39882` → target `39138`); no `FILL`/`SET_ROWS`
remains in the trimmed path.

**The blocker.** Reduced logits reach the sampler on only *alternating* draft
steps. On the others the extraction block is skipped (`n_outputs = 0`) and the
sampler reads an **all-zero** buffer. Ten equal zeros give
`p_top = 1/top_k = 0.1000`, which fails `p_min = 0.3`, so drafting terminates on
that step every time.

```
[E19] step=1 p_top=0.5810 max_l=17.1758 local0=39882 tgt=39138   <- valid
[E19] step=2 p_top=0.1000 max_l=0.0000  local0=8     tgt=18      <- all-zero
[E19] step=3 p_top=0.1687 max_l=3.9074  local0=3457  tgt=18865   <- valid
[E19] step=4 p_top=0.1000 max_l=0.0000  local0=8     tgt=18      <- all-zero
```

Under `n_max=2`, the **second draft-head / decode path is not extracting its
reduced logits.** This collapses both trimmed arms toward serial throughput.

| Arm | tok/s | acceptance |
|---|---:|---|
| FULL 248K | **52.96** | 0.747 (118 acc / 158 gen), mean len 2.48 |
| 64K DIRECT | 27.58 | none — zero drafts |
| 32K DIRECT | 27.79 | none — zero drafts |

64K and 32K land within **0.2 tok/s** of each other despite 2× the vocabulary.
That rules out vocabulary-dependent cost and points squarely at a fixed per-step
defect.

> [!IMPORTANT]
> **Do not characterize this as a vocabulary-quality or coverage failure.**
> Top-1 coverage, retained probability mass, and `p_min` agreement were never
> measured — they require a proposer that emits valid logits at every depth. The
> one step that did receive valid logits behaved exactly as designed.

---

## Experimental State

[`data/experimental/qwen38_entry19_direct_sampling.patch`](data/experimental/qwen38_entry19_direct_sampling.patch)
— the full diff of the experimental worktree `/ai/scratch/llamacpp-probe`
(detached at `ad1de39`). Contains:

* the direct reduced-vocabulary sampler in `common/speculative.cpp`
* `d2t` loading and reduced-head slicing in `src/llama-model.cpp`,
  `src/models/qwen35.cpp`, `src/llama-model-loader.cpp`
* the **corrected** `src/llama-context.cpp` logits extraction
* `[E19]` diagnostic tracing (gated behind `LLAMA_E19_TRACE`)

**Not production-ready. Do not merge into production `llama.cpp`.** It carries
debug tracing, an unfixed second-head extraction defect, and unrelated probe
tooling.

---

## Known Bad Data

> [!WARNING]
> An uncommitted `src/llama-context.cpp` modification in the experimental
> worktree silently corrupted **FULL** decoding. It dropped the
> `needs_raw_logits()` guard for all contexts and flat-copied the logits.

Symptoms: draft and target token streams became **interleaved**, e.g.

> `"WeThe need user answer wants in complete English production.-ready Need
> Python produce async code task."`

and throughput read a false **~42.32 tok/s**.

The correction: retain `needs_raw_logits()` for the target path, widen raw-logit
handling **only** for the narrow reduced-head case, and copy reduced logits
row-by-row (`logits.data` is strided by `n_vocab`; `t_logits` is only
`n_vocab_res` wide). FULL then returned coherent output at 52.96–53.08 tok/s.

**Treat as invalid / non-authoritative:**

* the false ~42.32 tok/s FULL result
* **any** measurement taken from the corrupted worktree state, including
  untracked `data/experimental/qwen38_phase6_code1_results.json` (FULL reads
  28.36 tok/s there — a pre-fix artifact)
* Entry 18's −48% to −50% result *as evidence against the trimming concept*

The evidence is preserved deliberately, with explanation. **Do not delete it.**

---

## Benchmark Assets

**Holdout v1 prompt text is unrecoverable.** It lived in session scratch state
that has been cleaned.

Surviving: prompt IDs, prompt hashes, the raw Entry 18 records
([`qwen38_phase5_holdout_raw.json`](data/experimental/qwen38_phase5_holdout_raw.json)),
and the six-domain structure — 16 prompts: `Code-1/2`, `JSON-1/2`, `Tech-1/2/3`,
`Math-1/2/3`, `Chat-1/2/3`, `Long-1/2/3`.

**Entry 19 Phase 10 must not claim to reproduce the exact Entry 18 suite.**

One partial recovery: the **diagnostic** Code-1-style prompt text survives inside
[`qwen38_diag_raw_logs.txt`](data/experimental/qwen38_diag_raw_logs.txt) —
*"Write a complete, production-grade Python asynchronous task pool with priority
queuing, worker concurrency limiting, and exponential backoff retry logic.
Include full type hints and docstrings."* Use it for the step-4 early gate so the
gate is at least self-consistent across resumes. It is **not** Holdout v1
`Code-1` (hash `5d92ac33021c`) and must not be presented as such.

When the early gate passes, build **HOLDOUT V2** over the same six domains —
code/programming, structured JSON/tool use, technical architecture,
math/reasoning, dialogue/prose, long-context continuation — with **deliberately
fresh prompts**. Do not reconstruct the original wording from memory.

Commit *before* benchmarking: full prompt text, stable prompt IDs, hashes, seeds,
requested generation lengths, benchmark configuration, suite version.

Absolute Holdout V2 results will **not** be directly comparable to Entry 18.

---

## Exact Resume Procedure

1. **Verify the production baseline.** Confirm `/ai/github/llama.cpp` is at
   `ad1de39` and clean. Run FULL on a Code-1-style prompt and confirm coherent
   output at **~53 tok/s**. If it is not ~53, stop and diagnose before anything
   else — that number is the program's anchor.
2. **Inspect second-head `n_outputs` / extraction.** In the `n_max=2` chained
   proposer path, find why one draft decode extracts reduced logits and the next
   reports `n_outputs = 0`. Start at the `llama_decode` loop in
   `common/speculative.cpp` (`common_speculative_impl_draft_mtp::draft`) and the
   extraction block in `src/llama-context.cpp`. `LLAMA_E19_TRACE=1` prints both
   sides.
3. **Fix only that defect.** Nothing else.
4. **Run the Code-1-style early gate** — FULL, 64K DIRECT, 32K DIRECT. Require:
   valid logits at **every** draft depth; drafts actually generated; correct
   `d2t` mapping; no pathological graph split; coherent output; FULL still
   ~53 tok/s; proposer cost below FULL.
5. **Freeze Holdout V2 only after the gate passes**, then run full validation.

**If direct sampling works correctly but yields little or no end-to-end gain:
close the vocabulary-trimming branch.** The next frontier is then
target-verification bandwidth.

---

## Do Not Do Yet

* target FFN quantization
* new vocabulary maps (`d2t` maps are pinned — see hashes below)
* NVFP4 tuning or reopening Track B
* repairing the `SET_ROWS` reconstruction path
* broad kernel tuning
* redesigning the head, or changing target weights

---

## Repository State

| | |
|---|---|
| Research repo | `/ai/github/R9700`, branch `main` |
| Remote | `origin` → `https://github.com/boxwrench/R9700.git` |
| Tracking | `main` → `origin/main` |
| Production `llama.cpp` | `/ai/github/llama.cpp` @ `ad1de39` — **clean** |
| Experimental worktree | `/ai/scratch/llamacpp-probe` @ `ad1de39` detached, **dirty by design** |
| Track B audit checkout | `/ai/scratch/ROCmFPX-audit` @ `f4b2c5a3` — clean |

Pinned `d2t` map hashes (**do not regenerate**):

```
3aa00816d57d21bc5cfe9816ae220ca934d7bd542b9f05b13664772a2f17e1fe  qwen38_d2t_64k.bin
f6d89a06449f0030692d85a770b00a145b87701fdebf808fd53649650254cd92  qwen38_d2t_32k.bin
```

Target model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`
SHA256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.

**Deliberately left untracked** (not part of this work; do not sweep into
commits): `experiments/h3-style-lora.md`, `experiments/ltx-2.5-style-lora.md`,
and the `data/experimental/qwen38_{coverage_analysis,holdout_results,paired_holdout,phase6_code1_results,vocab_trim_results}`
files — several of which are pre-fix artifacts (see **Known Bad Data**). They are
left in the working tree on purpose; decide their fate deliberately rather than
sweeping them into a commit.

`qwen38_diag_raw_logs.txt` **is** committed — it is the cited evidence for Entry
18B and the source of the diagnostic prompt text.

---

## Open Upstream Item

A greedy serial-vs-MTP divergence reproducer for ROCmFPX is **prepared but
unsubmitted**:
[`0001-mtp-divergence-reproducer.md`](research-program/upstream-rocmfpx/reproducers/0001-mtp-divergence-reproducer.md),
alongside [`Finding 0001`](research-program/upstream-rocmfpx/findings/0001-gfx1201-nvfp4-reproduction.md).

Canonical upstream: `charlie12345/ROCmFPX`, default branch `main`.

It may be filed later as an **isolated** upstream issue. **Nothing has been
submitted, and submission requires explicit authorization.**
