# Reproducibility — LingBot World v2 1.3B on R9700

## Hardware

- AMD Radeon AI PRO R9700, 32 GiB, `gfx1201` — must be HIP device **1**
  (`HIP_VISIBLE_DEVICES=1`); device 0 on this host is an RX 7900 XT
- Host at validation time: Ryzen 7 9800X3D, 188 GiB RAM, Ubuntu 24.04
- The R9700 is shared with production ComfyUI processes: check
  `rocm-smi --showpids` before benchmarking or killing anything

## Environment

- ROCm 7.2.1 (`/opt/rocm-7.2.1`), PyTorch `2.9.1+rocm7.2.1`, HIP 7.2.53211
  (all three recorded live from the benchmark process)
- venv `/ai/envs/lingbot-world-v2` — ROCm torch build, no CUDA torch, no
  flash-attn (self-attention runs the in-tree PyTorch SDPA fallback)
- `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib` is mandatory: `/opt/rocm` on this
  host lacks `libroctx64.so.4`, which this torch build links
- `WAN_VAE_CONV3D_TEMPORAL_SPLIT=1` (default): gfx1201 MIOpen workaround —
  unsplit FP32 Conv3d with output temporal extent ≥ 4 silently falls to a
  naive kernel (~0.16 vs ~2.6 TFLOP/s). `=0` restores exact upstream behavior
- MIOpen ships no gfx1201 database: per-shape autotune costs land in the
  user FindDb (`~/.config/miopen/`) once, then persist

## Upstream and models (all read-only at runtime)

- Upstream checkout `/ai/repos/lingbot-world-v2` (symlink into `/ai/github`),
  branch `rocm/r9700-1.3b-bringup` @ `7cf8109`
- Assembled checkpoint `/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled`:
  local `config.json` plus symlinks — 6 DiT shards + index from
  `/ai/models/lingbot-world-v2-1.3b-causal-fast/`, T5 (`models_t5_umt5-xxl-enc-bf16.pth`),
  VAE (`Wan2.1_VAE.pth`) and tokenizer (`google/`) from
  `/ai/models/lingbot-world-v2-14b-assets/`
- TAEW2.1: `/ai/models/taehv-011dfc2/` (`taehv.py` + `taew2_1.pth`); weight
  SHA-256 `d26151e7…c469c797e` is enforced at load (see `TAEW2_1_SHA256` in
  `interactive_world.py`)
- Conditioning cache `~/.cache/lingbot-world-cond/`, keyed by
  image × resolution × horizon (default 90 latent frames)
- DiT shape (live): dim 1536, 30 layers, 12 heads, head_dim 128, BF16
- Weight-file SHA manifest is not yet catalogued — provenance above is exact
  paths, the gap is hashes

## What is configurable vs pinned

Configurable (CLI/env, all validated): image, prompt, output dir, `--size`
area target, `--chunk_size`, `--local_attn_size`/`--sink_size`,
`--denoise_schedule`, `--presentation_decoder`, `--schedule`,
`--timecond_cache`, `--max_lat_frames`, `--seed`, move/turn amounts, fps.
Pinned (do not substitute without re-running validation): upstream commit,
checkpoint assembly, TAE weights, ROCm/Torch builds.

## Benchmark recipe (no new model runs needed to read this)

1. Same image, prompt, seed (42), `--size 384*672`, `x`-only stay script
   (≥ window length + margin: 24 stays covers window 12, 30 covers 18).
2. Warm the conditioning cache first (one throwaway run per image).
3. Compare saturated-window actions only (idx ≥ 12 for window 12, ≥ 18
   for window 18): P50/P95 of `first_frame_latency`, `latency_seconds`,
   `next_ready_latency` from `actions.jsonl`, plus `peak_alloc_gib`.
4. Full validation for state-affecting changes: 90-action adversarial script
   (travel/yaw/reversal/stay/oscillation/large-turn + exact pose returns at
   revisit gaps <6 / 6–12 / >12), all-finite check, revisit-L1 by horizon,
   sharpness trend, sequential warm canonical decode of saved `x0`.

Reference command (reproduces the 0.40/0.53/0.53 budget on this hardware):

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib HIP_VISIBLE_DEVICES=1
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1
/ai/envs/lingbot-world-v2/bin/python interactive_world.py \
  --image <frame> --prompt "<text>" --output <dir> \
  --size 384*672 --local_attn_size 12 --sink_size 6 \
  --denoise_schedule 3-step-A --presentation_decoder taew2_1 \
  --schedule decode-first --timecond_cache \
  --script "$(python3 -c "print(' '.join(['x']*30))")" --no_chunk_mp4
```

Expect ~35 s init warm-cached, then ~0.53 s/action saturated.
