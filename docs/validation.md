# Validation and reproducibility notes

The canonical videos remain on the benchmark workstation as validation
references. They are not tracked here. Their expected hashes and source paths
are in [`data/artifacts.tsv`](../data/artifacts.tsv).

The 2026-08-12 source report records the following for all three baseline
artifacts:

- H.264 video decoded fully without error.
- Audio was stereo AAC and non-silent.
- The queue was idle after completion.
- The run window had no GPU reset, VM fault, ring timeout, or OOM.
- Geometry, frame count, frame rate, duration, audio sample rate, and SHA-256
  were recorded.

H3 Standard and H3 Turbo delivered 864×480, 124 frames, 24 fps, 5.167 seconds,
with stereo AAC at 32 kHz. LTX-2.5 delivered 896×512, 121 frames, 24 fps,
5.042 seconds, with stereo AAC at 48 kHz.

This repository does not claim a quality score, peak-resource statistic, or
multi-trial confidence interval. Those values were not measured consistently
for the canonical comparison.
