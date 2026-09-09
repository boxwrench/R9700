# LingBot-World-V2 1.3B causal-fast on R9700 (gfx1201) — in progress

Working notes and raw logs for the native-BF16 single-R9700 bring-up.
The final report lands in `docs/lingbot-world-v2-1.3b-r9700-bringup-20260909.md`.

## Layout

- `bench.py` — instrumented single-GPU harness around the upstream
  `WanI2VCausal` pipeline. Times model load, T5 encode, VAE encode/decode
  and every DiT forward separately, records allocated/reserved/peak VRAM
  and system RSS, and separates cold from warm runs.
- `patches/` — the compatibility patch applied to the upstream checkout.
- `logs/` — raw captured output. Nothing here is edited after the fact.

## Environment

- Upstream checkout: `/ai/github/lingbot-world-v2`, branch
  `rocm/r9700-1.3b-bringup`, based on upstream `45fa4067`.
- venv: `/ai/envs/lingbot-world-v2` (ROCm 7.2.1 wheels, no CUDA torch,
  no flash-attn).
- Assembled checkpoint: `/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled`
  — symlinks to the 1.3B DiT shards plus the VAE/T5/tokenizer from the
  14B release, which the 1.3B repository does not ship.
- `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib` is required: `/opt/rocm` on this
  host does not carry `libroctx64.so.4`, which this torch build links.
- `HIP_VISIBLE_DEVICES=1` selects the R9700 (device 0 is an RX 7900 XT).

## Status

Phase 0-4 done through model load. Generation is blocked on VAE cost,
under investigation — see `logs/vae-isolated-memory.txt`.
