# DeepSeek V4 decode optimization plan

## Objective and fixed controls

Improve repeatable single-user decoding beyond the saved 8.137 tokens/s result
while preserving the current UD-Q4_K_XL target, 32K context, Q8 target/draft KV,
DeepSeek Harness compatibility, exact long-context retrieval, and 5/5 deterministic
quality score.

Keep these fixed unless a phase explicitly changes one of them:

- R9700 is the only visible GPU.
- RX 7900 XT is not used during phases 1 through 3.
- Target quant remains UD-Q4_K_XL; quant-search profiles are out of scope.
- DSpark Q8 remains the model drafter.
- Context is 32,768, parallelism is one, and batch/ubatch are 2,048.
- Flash attention, Q8/Q8 GPU KV, 16 threads, sampling parameters, prompts, and
  benchmark token counts remain unchanged.
- The Vulkan llama.cpp build stays fixed through phases 1 through 3, except for
  the explicitly gated confidence-scheduling update.

## Phase 1: exchange draft VRAM for target experts

Run the placement rows in their listed order:

1. `41 target / 0 draft` CPU-MoE baseline.
2. `40 / 99`.
3. `39 / 99`.
4. `38 / 99`.
5. `37 / 99`.

Here a smaller target CPU-MoE count means more target expert layers reside on
the R9700. `99` deliberately requests that every available draft MoE layer stay
in system RAM. The earlier two-GPU failure does not disqualify these profiles;
it occurred before the R9700 Vulkan namespace was isolated.

Reject a profile if it fails to load, crashes or corrupts output, misses exact
context retrieval, scores below 5/5, materially grows swap, or cannot complete
two measured repetitions. Continue to the next row after recoverable failures.

Promote by repeatable decode throughput, using DSpark acceptance, prompt speed,
VRAM headroom, and tail-latency stability as tie-breakers. Treat a difference
below 3% as noise until confirmed with additional repetitions.

## Phase 1B: spend recovered headroom on context

The saved long-context test used 24,603 tokens, about 75% of the configured 32K
slot. After selecting a phase-1 placement, copy its target/draft CPU-MoE values
into the phase-1B rows and test actual prompt occupancy at 90%.

Run the ladder in order: 32K verification, 36K, 40K, 48K, then 64K. Enable one
new row only after the previous row passes retrieval and quality, completes
without allocation warnings worsening into a failure, and retains at least
1.5 GiB of measured R9700 headroom at the end of the filled-context request.
Stop the ladder at the first failure or unsafe margin; do not skip over it.

Track both configured context and actual probe tokens. DeepSeek V4's compressed
KV is not the dominant allocation on the current Vulkan path: the unfused
Lightning Indexer creates prompt-length-dependent compute scratch, so a healthy
load at a large `--ctx-size` is not proof that a prompt of that length will fit.
Keep a fast-context and a maximum-context profile if spending headroom on more
target experts and spending it on context produce different Pareto winners.

## Phase 2: DSpark plus draftless speculation

Copy the winning phase-1 placement into all phase-2 rows. Compare:

1. DSpark-only control.
2. DSpark plus `ngram-mod` using match/min/max `24/48/64`.
3. The hybrid winner with target backend sampling enabled, only if row 2 helps.

The ordinary decode prompt measures regression. A second workload set must
include code editing, rewriting supplied prose, and summarization because
`ngram-mod` accelerates reusable sequences already present in the context. Keep
separate aggregate results for general chat and reuse-heavy agent work; do not
hide a general-chat regression inside a combined average.

Promote the hybrid mode for regular use only if it is neutral on ordinary chat
and materially faster on at least one representative reuse-heavy workload.

## Phase 3: confidence-scheduled DSpark

This phase is blocked by capability, not by the experiment harness. The current
llama.cpp binary supports DSpark but does not expose
`--spec-draft-conf-min`. Before enabling phase-3 rows:

1. Build or install a candidate llama.cpp version that exposes the option.
2. Confirm that its DeepSeek V4 and DSpark changes are compatible with the Q8
   draft GGUF and that the draft contains a usable confidence head.
3. Run the phase-2 winner once with confidence scheduling disabled. Its result
   must remain within 3% of the frozen-build control and pass all quality gates.
4. Only then test a seven-token ceiling at thresholds 0.25, 0.50, and 0.75.

Record actual drafted/accepted length distributions if the server exposes them.
Select total decode throughput rather than raw acceptance percentage: an overly
conservative threshold can raise acceptance while reducing useful speculation.

## Phase 4: Vulkan versus HIP/ROCm

The primary HIP candidate is Stew Forster's llama.cpp `rdna-boosts` branch,
pinned initially to `ed89854b2aeb0e333dd61424f14af2aedaca126e` (2026-08-13).
Its 23-commit patch series is based on upstream `a94d563ed801d1da1b8c2432946de07d0231bb3d`
and contains explicit `gfx1201` work. Do not test a moving branch without first
recording a new pin and reviewing its delta.

This changes phase 4 from a single backend comparison into an attribution
ladder. Hold the expensive model runs until the phase-1 through phase-3 winner
is known, then build side-by-side binaries rather than replacing the known-good
Vulkan installation:

1. **Frozen Vulkan control:** current commit `7b13a8404d7e219c13d1a243e2a21a857a6e99d9`,
   RADV, and the promoted model profile.
2. **Matched upstream HIP control:** upstream llama.cpp at the fork base
   `a94d563e`, built with ROCm 7.2.1, `GGML_HIP=ON`, `GGML_BLAS=OFF`, and
   `AMDGPU_TARGETS=gfx1201`.
3. **RDNA-boost HIP candidate:** fork commit `ed89854b` with otherwise identical
   CMake and runtime settings.
4. **Matched fork Vulkan control, only if attribution remains ambiguous:** build
   `ed89854b` with Vulkan to separate newer common llama.cpp changes from HIP
   and RDNA kernel changes.

The matched upstream HIP row is essential. Comparing only the old Vulkan binary
with the fork would combine backend, upstream, and patch-series changes and
would not tell us what produced a gain. Record complete CMake options, compiler,
binary hash, GPU architecture, driver, ROCm version, and device visibility for
every binary.

### Phase 4A: capability and correctness gate

Before a timed run:

- Confirm only the R9700 is visible to the candidate process and that llama.cpp
  identifies it as `gfx1201`.
- Run the fork's backend operation tests for the quantized matmul, Flash
  Attention, DeepSeek V4 hyper-connection, and Lightning Indexer paths that are
  present in the build.
- Load the target without DSpark at 8K, then enable the Q8 DSpark drafter. Keep
  the promoted target/draft CPU-MoE placement unchanged.
- Inspect the server log. The current Vulkan baseline reports Lightning Indexer
  and DeepSeek V4 HC pre/comb/post as unsupported. Record whether HIP actually
  assigns each fused operation to the R9700; source-level support alone is not
  sufficient.
- Compare deterministic outputs, the five-item quality probe, and an exact
  retrieval prompt before accepting any throughput result.

### Phase 4B: fixed-profile backend A/B

For each backend run:

- Short and 256-token decode repetitions.
- The exact ~24K-token retrieval probe to capture deep-context prefill.
- Five-item deterministic quality probe.
- DSpark acceptance, peak R9700 VRAM, host RAM/swap, server warnings, and crash
  recovery behavior.

ROCm only wins if it improves repeatable decode without a severe 24K prefill
regression or loss of stability. Preserve Vulkan as the recovery baseline until
HIP passes the full gate.

The fork changes many GPU operations but it does not remove the DDR5 bottleneck
created by 41 host-resident target MoE layers. Report target-only and DSpark
results separately, and never infer whole-model improvement from a microbenchmark.
The most relevant fork changes for the current MXFP4/Q8_0 DeepSeek model are:

- per-graph reuse of Q8_1-quantized matmul inputs;
- MoE top-k weight scaling folded into the expert down projection;
- shared-expert output-chain fusion;
- RMS-normalization and gating paths that directly produce Q8_1 inputs;
- paired quantized K/V projection execution;
- RDNA4 Flash Attention tuning; and
- the generic single-GPU graph-wrapper reduction if the selected runtime path
  would otherwise create a Meta device.

SSM-specific fusions, the `gfx1151` Q8_0 launch table, integrated-APU host-buffer
workaround, three-plus-GPU split fix, and Q6_K-specific kernels are not primary
DeepSeek V4 experiments for this UD-Q4_K_XL model.

### Phase 4C: fork-only controlled experiments

Run these only after `ed89854b` passes phase 4B. Change one variable at a time:

1. **RDNA4 WMMA Flash Attention:** compare the default with
   `GGML_CUDA_FA_WMMA_256=0`. Keep Q8/Q8 KV and all placement fixed. This is an
   attribution toggle, not a proposed default.
2. **KV format:** compare the selected Q8/Q8 profile with BF16/BF16 only if it
   fits at 32K with at least 1.5 GiB R9700 headroom. The fork's native BF16
   Flash Attention work is relevant here; measure retrieval/quality, prefill,
   decode, and context-dependent VRAM. Do not sacrifice the 32K requirement.
3. **HIP graph capture:** compare builds with `GGML_HIP_GRAPHS=ON` and `OFF` if
   logs or traces suggest launch overhead remains material. The two builds must
   otherwise be identical.
4. **Post-backend placement ladder:** rerun target/draft CPU-MoE placement around
   the promoted Vulkan winner because HIP allocations may change the safe expert
   boundary. Stop immediately on swap growth, failed allocation, or less than
   1.5 GiB VRAM headroom.
5. **Speculation attribution:** compare target-only, DSpark with three-token
   ceiling, and the already-promoted speculation mode. Record draft throughput,
   acceptance, mean accepted length, and total verified decode throughput.

Do not add a Q6_K target merely to exercise the fork's Q6_K kernel. Quant changes
remain outside this campaign unless a separate model-quality experiment later
justifies one.

## Phase 5: Lucebox ROCmFPX with a system-RAM expert tier

This is a separate backend campaign, not another llama.cpp profile. Its purpose
is to test whether the Lucebox DeepSeek-specific HIP path can retain its fused
decode, DSpark verification, and indexed sparse prefill advantages when the
large shared Strix Halo memory pool is replaced by the R9700's 32 GiB of local
VRAM plus this machine's 192 GiB of DDR5.

Do not interpret ROCm managed-memory oversubscription as a successful host tier.
On Strix Halo the GPU directly reads a 256 GB/s unified LPDDR5X pool. On this
machine, R9700 access to DDR5 crosses PCIe. Repeatedly faulting or streaming the
whole target across that link is expected to erase the benefit. A viable design
must keep dense/attention/shared tensors and a bounded hot-expert cache in VRAM,
while cold routed experts remain in DDR5 and are either executed on the CPU or
prefetched selectively and asynchronously.

Run the gates in order:

1. **Static capability audit, with no model download or GPU load.** Pin the
   Lucebox commit and ROCm version; build for `gfx1201`; verify that the DeepSeek
   ROCmFPX kernels compile for the R9700; identify the exact host-resident expert
   path. The published standalone DeepSeek recipe is for `gfx1151`, while the
   published R9700 DeepSeek entry is a Strix-plus-R9700 burn-in configuration.
   A generic R9700 number elsewhere in the hardware table is not evidence that
   the complete 284B target works on a standalone R9700.
2. **Loader and placement smoke test.** Proceed only if weights can be placed
   intentionally rather than by GPU-memory overcommit. Log VRAM, pinned host
   RAM, ordinary host RAM, PCIe link width/speed, page faults, swap, and actual
   tensor/expert placement. Reserve at least 32 GiB of host RAM for the OS,
   DeepSeek Harness, runtime allocations, and context scratch.
3. **Exact quality baseline.** Start at 8K with the model-default six experts,
   exact prefill, target-only decoding, and greedy sampling. Compare output and
   the deterministic quality suite with the saved UD-Q4_K_XL result. ROCmFPX is
   allowed here only because it is inseparable from Lucebox's packed kernels;
   it does not replace the primary quant campaign.
4. **Add optimizations one at a time.** Add DSpark fused verification at q=4,
   then indexed sparse prefill. Record target-only and speculative throughput,
   draft acceptance, TTFT/prefill, VRAM, host bandwidth, PCIe traffic, and
   quality after each change.
5. **Reach the actual context requirement.** After the 8K path is stable, test
   16K and 32K at 90% prompt occupancy. A result below 32K is informative but
   cannot become the regular-use winner. Continue with 40K, 48K, and 64K only
   while the same retrieval, quality, swap, and 1.5 GiB VRAM-headroom gates used
   in phase 1B continue to pass.
6. **Approximate speed profile last.** Only after the six-expert 32K result is
   known, test top-k four. Label it approximate and quality-changing. Never
   compare its speed directly with the six-expert llama.cpp baseline without
   displaying that difference.

The published Lucebox 32 tok/s result is a reference ceiling, not our expected
result: it used a 102.3 GB mixed ROCmFPX target, an 11.3 GB DSpark draft, 8K
context, sparse prefill, q=4 verification, and top-k four on unified-memory
Strix Halo. Our promotion target is repeatable performance above 8.137 tok/s at
32K or greater, default top-k six, passing retrieval and quality, with no swap
growth or GPU-reset instability.

The disabled matrix for this campaign is in
`lucebox-host-ram-profiles.tsv`. It is intentionally not consumed by
`run-all.sh`; Lucebox needs its own process launcher, metrics parser, placement
telemetry, and recovery validation before any row can be enabled.

## Agent implementation: DeepSeek Harness

DeepSeek Harness is the primary interactive and agent implementation for every
backend that passes its API compatibility gate. It is above the inference
engine, so the same pinned harness profile and task corpus can exercise
llama.cpp/Vulkan, llama.cpp/HIP, or Lucebox without changing backend placement.
Hermes is retained only as a migration/control client.

Keep two measurement planes separate:

1. **Backend plane:** `run-all.sh`, direct HTTP probes, retrieval, quality,
   prompt/decode throughput, DSpark acceptance, VRAM/RAM/swap, and recovery.
   DeepSeek Harness must not sit in this path because its system prompt, tool
   schemas, session log, and retries would confound the serving result.
2. **Agent plane:** run the promoted backend through a pinned DeepSeek Harness
   headless profile. Measure completed tasks, tool-call validity, corrective
   turns, input/output tokens, time to completion, context growth, and recovery
   from a failed tool call. Compare Hermes only on this plane.

Every backend finalist must pass both planes before becoming the regular-use
configuration. See `deepseek-harness.md`, `deepseek-harness-settings.example.yaml`,
and the disabled `agent-harness-profiles.tsv` matrix.

## Deferred and excluded work

- **Deferred:** shared target/draft placement across R9700 and RX 7900 XT.
- **Deferred:** KTransformers dynamic expert placement and its CUDA-oriented V4
  serving path.
- **Isolated exception:** Lucebox ROCmFPX is allowed only in phase 5 because its
  packed format and HIP kernels are one serving path. It cannot silently replace
  the UD-Q4_K_XL quality baseline.
- **Excluded from phases 1 through 4:** changing the primary target quant.
- **Separate hardware campaign:** reusable RDNA4 and RX 7900 XT/RDNA3 kernel
  experiments are catalogued in `rdna-boosts-experiments.md`; they do not alter
  the R9700-only DeepSeek controls above.

## Source pins for the RDNA campaign

- Fork: <https://github.com/stew675/llama.cpp/tree/rdna-boosts>
- Initial fork head: `ed89854b2aeb0e333dd61424f14af2aedaca126e`
- Fork base: `a94d563ed801d1da1b8c2432946de07d0231bb3d`
- Patch count at review: 23 fork commits; the branch was 15 upstream commits
  behind `master` on 2026-08-14.
- Local toolchain at review: ROCm 7.2.1 / HIP 7.2, with the R9700 targeted as
  `gfx1201`.

## Resume and recovery

Use `optimization-results` as the result directory. Each completed profile has
its own status and artifacts, and reruns skip it. Failed loads remain resumable.
The existing recovery script terminates only the managed DeepSeek process,
checks R9700 VRAM, and resets only the isolated R9700 when reset is explicitly
enabled and no foreign process owns its render node.
