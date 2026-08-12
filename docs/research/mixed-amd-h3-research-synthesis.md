# Mixed AMD H3 multi-GPU research synthesis

Date added: 2026-08-12

This summary compares the two generated research documents in this directory.
They were written from the same question: can one MiniMax H3 video be made
faster by using the Radeon AI PRO R9700 and Radeon RX 7900 XT together?

## Short decision

The reports agree that two-way tensor parallelism is a poor immediate latency
bet for this mixed consumer-Radeon pair. They also agree that sequence/context
parallelism, especially Ulysses-style unified sequence parallelism (USP), is
the technically interesting path if the communication layer can be made to
work.

My decision is to pursue it anyway, as a gated research prototype rather than
as a production replacement for the current single-R9700 lane. “Do not pursue”
is the papers' recommendation for investing in TP latency work on today's
stack; it is not a reason to close the broader SP/USP question. The worthwhile
outcomes include:

- a measured answer about mixed gfx1100/gfx1201 collectives;
- a correctness-tested prototype that may scale to longer or higher-resolution
  H3 clips;
- a capacity win even if the 864x480, four-step Turbo clip does not get faster;
- reusable evidence for the next ROCm/RCCL release.

The existing experiment already establishes the practical baseline: placing
Qwen on the 7900 XT while keeping H3 on the R9700 produced corrected
single-video improvements of 4.7% cold, 3.5% for a same-prompt new seed, and
7.9% for a changed prompt. That is component placement and residency, not
parallel denoising. Single-R9700 remains the default.

## Where the reports converge

Both documents identify the same constraints:

1. H3 is a single-stream dense transformer with joint text, audio, and video
   conditioning. A correct split must preserve global attention and the packed
   sequence.
2. Tensor parallelism would require repeated all-reduce/all-gather operations
   through roughly 50 transformer layers. At only four Turbo steps, collective
   setup and synchronization have little time to amortize.
3. Ulysses/USP is a better architectural match because it shards activations
   and preserves the full attention problem, but it still needs reliable
   all-to-all or equivalent peer communication.
4. The mixed pair has no public, known-good H3 implementation. Existing H3
   multi-GPU demonstrations are on homogeneous datacenter AMD systems or
   different NVIDIA environments.
5. The R9700 and 7900 XT are heterogeneous in compute and VRAM. An equal split
   can leave the faster R9700 waiting for the smaller 7900 XT; an efficient
   weighted split would be custom research.
6. The golden ComfyUI service must remain untouched. Any prototype should run
   in an isolated, spawn-safe external service and fall back cleanly.

## What each document contributes

The Compass artifact is the more implementation-oriented plan. It recommends
an inexpensive Phase-0 RCCL/P2P preflight, followed by a BF16 one-block USP
correctness test, then a full external H3 service, Turbo integration, and
controlled benchmarks. It estimates that the short Turbo lane is the hardest
case for latency and that longer or higher-resolution clips may offer a better
SP capacity case.

The deep-research report gives a broader comparison of tensor, sequence,
pipeline, frame-split, component-placement, CPU-encoder, and GGUF options. It
is especially useful as a warning that data-parallel multi-job throughput is
not the same as lowering the latency of one video. It also recommends starting
with a small concurrency/correctness test before touching ComfyUI.

## Pursuit plan

The research path is intentionally narrower than “build TP immediately”:

1. **Communication preflight:** record device mapping, peer-copy bandwidth,
   RCCL all-reduce/all-gather/all-to-all behavior, and failure logs on the
   actual mixed pair. Test mitigations only in the isolated lane.
2. **One-block prototype:** use a BF16 reference and a small H3-shaped block.
   Compare single-device and two-device outputs, separate compute time from
   communication time, and prove concurrent GPU work rather than inferring it
   from device visibility.
3. **Small full-forward service:** use spawn-safe processes, real H3 packing,
   and short clips before attempting the 124-frame workload. Keep the current
   Qwen placement available as a control.
4. **Turbo integration:** add the v4 LoRA and four-step dual-clock sampler only
   after the reference path is correct. Keep this outside the golden service.
5. **Decision benchmark:** compare single GPU, component placement, and any
   true SP lane at identical prompt, seed, shape, and validation criteria.
   Report latency and capacity separately.

Success is not required to mean a faster five-second clip. A clean, correct
prototype that fits a longer or larger clip is a valid result. A latency claim
should require repeated controlled runs and a measured improvement, not a
projection from ideal two-way compute.

## Evidence and citation cautions

These are generated research/planning documents, not peer-reviewed results, and
their projections are not measurements from this workstation. The repository's
dual-GPU report and normalized TSV are the local experimental record.

The documents disagree sharply on the packed video-token count: one estimates
about 13,000 latent video tokens for the 864x480/124-frame workload, while the
other uses about 12.85 million. Those values cannot both describe the same
latent sequence. Until the actual ComfyUI packed tensor shape is instrumented,
all activation-size and communication projections should be treated as
hypotheses. The first estimate is more consistent with the stated latent grid,
but this summary does not promote it to a measured fact.

Some external citations were not independently re-audited when these files were
imported; one cited discussion is even dated after this repository snapshot.
Use the linked sources as leads, verify their revisions and dates, and do not
use either paper alone to justify a runtime or hardware change.

## Source documents

- [Compass feasibility and implementation plan](compass-mixed-amd-h3-feasibility.md)
- [Deep-research parallel-inference report](deep-research-mixed-amd-h3-parallel-inference.md)

The originals are retained unchanged. Their SHA-256 values are recorded in
[checksums/research.sha256](../../checksums/research.sha256).
