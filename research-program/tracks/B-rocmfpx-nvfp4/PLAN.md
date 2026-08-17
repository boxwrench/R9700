# Track B — Staged Research Protocol

**Subject:** native NVFP4 execution of Qwen3.8-27B on AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4) via ROCmFPX.

**Governing rule:** reproduce Track B cleanly *before* importing anything from Track A.

**Prerequisite reading:** [`upstream-audit/2026-08-17-upstream-audit.md`](upstream-audit/2026-08-17-upstream-audit.md).
The audit changes the shape of this plan in one important way: upstream's NVFP4
work was developed and measured on **gfx1151 (RDNA3.5)**, which takes a
different HIP code path from **gfx1201 (RDNA4)**, and upstream's own build docs
state that *"published benchmark numbers and regression guards assume Strix Halo
/ gfx1151"*. **B1 is therefore original measurement, not confirmation of a known
result.** Plan for the possibility that it does not work at all on first
contact.

All stages obey [`../../shared/benchmark-contract.md`](../../shared/benchmark-contract.md)
and report using [`../../shared/metrics.md`](../../shared/metrics.md) and
[`../../shared/run-record-schema.md`](../../shared/run-record-schema.md).

---

## B0 — Immutable snapshot

**Goal:** freeze exactly what was run, so any later number can be re-derived.
**Exit condition:** [`snapshot.md`](snapshot.md) has no `PENDING` fields.
**No performance conclusion may be drawn in B0.** Nothing in this stage is a result.

Capture:

| Group | Fields |
|---|---|
| Source | ROCmFPX remote URL, branch, commit SHA, `git status --porcelain` (must be empty), submodule SHAs |
| Toolchain | ROCm version, HIP version, hipcc/clang version, Vulkan loader version, RADV/Mesa version, kernel, glibc |
| Host | CPU model, core count, RAM, kernel command line if relevant |
| GPU | device name, PCI address, `gfx` arch, VRAM, driver, **enumeration index under each backend**, clock/power mode if readable |
| Build | build directory, exact `cmake` configure line, exact build command, compiler, build duration, **emitted code-object arch as verified after build** |
| Model | filename, full path, byte size, SHA256, GGUF architecture, quantization ftype, tensor count, per-type tensor breakdown |
| Runtime | complete launch command, every environment variable that alters behaviour, ctx, batch, ubatch, parallel, flash-attention, KV type, MTP `n-max` / `p-min` |
| Harness | script path and repo SHA, prompt ID, seed, token counts |

**Mandatory checks specific to this hardware and this upstream:**

1. **Device isolation.** The R9700 is *not* index 0 — this machine enumerates
   the RX 7900 XT first under both Vulkan and HIP. Pin the device explicitly and
   record which index resolved to `gfx1201`.
2. **Code-object arch verification.** Upstream documents that a `gfx1200` build
   on a `gfx1201` card *"links and loads a model, then segfaults or hangs with no
   error message"* (issues #18, #37). Record the verified emitted arch, not the
   requested one.
3. **NVFP4 graph restrictions.** `src/llama-graph.cpp` *asserts* — does not
   gracefully fall back — when a scale-carrying NVFP4 FFN tensor is combined with
   a bias or a LoRA adapter. Confirm no LoRA is loaded and record whether the
   model's FFN tensors carry biases.
4. **Scale side tensors.** Record whether the model ships `output.scale` /
   `output.input_scale` and the per-layer `*.scale` / `*.input_scale` set, since
   the loader's behaviour depends on it and it was the subject of three recent
   upstream fixes.

---

## B1 — Upstream reproduction

**Goal:** run ROCmFPX the way upstream intended, on the R9700, with none of our changes.
**Entry condition:** B0 complete.
**Exit condition:** a reproducible `gfx1201` native-NVFP4 baseline exists, or a
documented, minimal failure that becomes an upstream finding.

### Model selection, in priority order

1. The exact model and config used for the reported R9700 result — **if and only
   if the user supplies its provenance.** The audit found nothing in-tree that
   substantiates the ~72 tok/s figure; its model, backend, and settings are
   unknown, so it is not currently a reproduction target.
2. `RadixArk/Qwen3.8-27B-NVFP4` — the only real model named in the NVFP4 commits
   (`7b02624`, `5290625`). **Not present locally**; acquiring it is a large
   download and an explicit user decision.
3. A locally produced NVFP4 conversion via
   `llama-quantize --pure --token-embedding-type q5_K in.gguf out.gguf NVFP4`.
   **This is a fallback, not an equivalent** — it is a different checkpoint from
   upstream's and must be labelled as such in every result.

### Measurements, each recorded separately

* prompt processing (PP) throughput
* **serial / no-MTP** decode throughput
* **native MTP** decode throughput
* VRAM high-water mark
* correctness sanity: coherent generation; greedy determinism across repeats
* repeatability: ≥5 repetitions, report the distribution, not a single figure

**Constraints.** Stock upstream at the pinned SHA. Zero Track A patches. Zero
vocabulary trimming. Upstream-default speculative policy for the MTP arm unless
upstream documents otherwise.

**Correctness gate before any timing is believed.** `7b02624` states that
without the lm_head scale applied *"the logits would be off by a constant factor
and silently degrade"* — a failure mode that produces plausible text and normal
timings. Confirm the scale is actually loaded and applied before recording any
throughput number.

### Failure is a valid outcome

If B1 fails — assertion, hang, garbage output, or a load error — that is a
**gfx1201 finding**, and it is the most likely single thing to come out of this
track. Route it to [`../../upstream-rocmfpx/findings/`](../../upstream-rocmfpx/findings/)
using the finding template and stop; do not patch around it to obtain a number.

---

## B2 — Matched comparison

**Goal:** compare the Track A foundation against the Track B foundation fairly.
**Entry condition:** B1 reproduces stably.

Same physical R9700. Hold constant as far as technically possible: prompts,
generated token count, seeds, context length, flash-attention setting,
parallelism, and — for the MTP arm — `n-max` and `p-min`.

**Keep the serial and MTP comparisons strictly separate.** A serial number from
one track and an MTP number from the other is not a comparison.

Report per arm:

| | PP | serial decode | MTP decode | MTP multiplier | VRAM |
|---|---|---|---|---|---|
| Track A — Q4_K_XL / Vulkan | | | | | |
| Track B — NVFP4 / ROCmFPX | | | | | |

plus, for the MTP arms: aggregate acceptance, $p_0$, joint-$p_1$,
conditional-$p_1$, accepted drafts/round, and committed tokens/round.

**Confounds that must be stated, not buried.** The two tracks differ in
quantization format *and* bits-per-weight *and* possibly backend *and* possibly
checkpoint provenance. NVFP4 is 4.50 bpw (36 B per 64 weights); Q4_K_XL is a
mixed recipe whose FFN tensors Track A measured at IQ4_XS 4.25 bpw and Q5_K 5.50
bpw. A throughput difference is therefore **not** attributable to "NVFP4" as
such. Report the delta; do not name a cause without isolating one.

**No conclusion may rest on acceptance rate alone.** Acceptance is an input to
throughput, not a substitute for it.

---

## B3 — Track B cost decomposition

**Goal:** find out where NVFP4 puts the bottleneck on gfx1201.
**Entry condition:** B1 reproduces; B2 measured.

Decompose, using the same accounting Track A used:

* serial target forward cost
* MTP round cost
* verification cost (the multi-column target forward)
* proposer cost, and the head cost within it
* attention cost
* the major matmul families, by shape and type
* host-side and unexplained residual — **left labelled UNEXPLAINED**

**Do not assume Track A's bottlenecks apply.** Track A concluded, after four
built-and-measured attempts, that its verification kernel was bandwidth- and
latency-bound at N=3 and that work-per-byte reductions did not help. That
conclusion was derived for IQ4_XS and Q5_K on the Vulkan DMMV path. It is a
**hypothesis** for NVFP4, not an inherited result.

The framing question for this stage is: **where did NVFP4 move the bottleneck?**

### Specific questions the audit raises for B3

1. **Does the redundant-dequant structure recur?** NVFP4 on Vulkan uses the
   generic `dequantize4()` in `dequant_funcs.glsl` inside the same
   `mul_mat_vec.comp` template Track A profiled, so at `NUM_COLS = N` each
   fragment is dequantized N times. Measure whether it is limiting *here* — the
   byte count and the ALU cost both differ from IQ4_XS (`ue4m3_to_fp32` is a LUT
   lookup, not an arithmetic unpack).
2. **Do the scale multiplies fuse?** Every scale-carrying NVFP4 matmul emits an
   extra elementwise `GGML_OP_MUL` node (`llama-graph.cpp:1188-1192`). Track A
   entry 17 found the MUL_MAT+ADD fusion predicate is gated on
   `ggml_nrows(mul) != 1`, so it does not fuse at N>1. Count the dispatches and
   measure their cost during MTP verification.
3. **Does HIP's MMVQ path change the answer?** HIP has an NVFP4 Q8_1 vec-dot
   (`mmvq.cu:162`) that Vulkan on RADV structurally cannot use, because the R9700
   reports `integer_dot_product = false`. This is a real architectural fork, and
   it is why B4 must revisit backend choice rather than inherit Track A's.

---

## B4 — Track B native tuning

**Goal:** extract what the Track B stack itself offers, before importing anything.
**Entry condition:** B3 complete.

Explore Track-B-native knobs only:

* **backend choice — Vulkan vs HIP.** Track A's "Vulkan beats HIP on gfx1201"
  finding was established for Q4_K/IQ4_XS and does **not** transfer; the MMVQ
  asymmetry above is a concrete reason to expect it may invert.
* ROCmFPX quantization routing and its ROCmFP4 / ROCmFP4_FAST targets
* NVFP4-specific kernel selection and any upstream-recommended toggles
* MTP settings re-swept natively (`n-max`, `p-min`) — re-derived, not inherited
* gfx1201-relevant build and runtime choices

**One variable at a time.** **Do not import Track A patches in this stage.**

---

## B5 — Integration candidates

**Goal:** test whether Track A mechanisms transfer.
**Entry condition:** a stable, characterized Track B baseline exists.

Each candidate is a **fresh experiment with its own hypothesis and A/B**, not a
port. Nothing is assumed to carry over.

### Highest-priority mechanism-level candidate

**64K / 32K draft-vocabulary trimming.** In Track A this is *promising and
explicitly not accepted* — it has no true unseen `n-max=2` speculative holdout,
its multi-draft proposer timing residual is unreconciled, and 32K-vs-64K is
undecided. It must arrive in Track B carrying those caveats.

**Do not assume 32K wins.** The head cost that made trimming attractive in
Track A was a Q6_K `[248320, 5120]` matvec at 1.04 GB; under NVFP4 the lm_head
is itself 4-bit and carries `output.scale` / `output.input_scale`, so both the
byte count and the arithmetic differ. The trimming ratio that pays here is an
open question.

### Other portable methodology (low risk, port freely)

* positional acceptance metrics ($p_0$, joint-$p_1$, conditional-$p_1$)
* round-cost accounting
* the benchmark harness and validation ladder
* the Qwen3.6-style matched-control methodology

### Do NOT automatically port

* IQ4_XS Vulkan shader patches (type-specific; NVFP4 is a different type)
* the MUL_MAT+ADD Vulkan fusion patch
* workgroup / `ROWS` experiments (tuned against IQ4_XS occupancy)

These were measured against a specific type on a specific path, and three of
the four were **rejected** in Track A on their own merits.

---

## B6 — Upstream contribution

**Entry condition:** a gfx1201-specific bug or optimization is found and verified.

Assemble, per [`../../upstream-rocmfpx/README.md`](../../upstream-rocmfpx/README.md):

exact upstream SHA · hardware · toolchain · model hash · exact command · raw
before/after · minimal reproducer · correctness evidence · component result ·
end-to-end result.

Sequence: **finding → reproducer → minimal patch → PR.**

Never open a PR that is a large private fork. Given the audit's finding that
gfx1201 is effectively untested upstream while gfx1151 is the documented
reference, a *correctness* finding on gfx1201 is the most probable and most
useful first contribution.

---

## Stage gates at a glance

| Stage | Entry condition | Exit condition |
|---|---|---|
| B0 | — | `snapshot.md` has no `PENDING` field |
| B1 | B0 complete | stable baseline **or** documented minimal failure |
| B2 | B1 reproduces | matched serial and MTP arms, confounds stated |
| B3 | B2 measured | bottleneck located; residual labelled UNEXPLAINED |
| B4 | B3 complete | native knobs swept, one variable at a time |
| B5 | stable characterized baseline | each candidate A/B'd as a fresh experiment |
| B6 | verified finding | reproducer + minimal patch prepared |

**No stage may be skipped, and a failed stage stops the track rather than being
worked around.**
