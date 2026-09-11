# Quickstart — LingBot World v2 1.3B on R9700

From a cold shell to a playable world in two commands. Validated on
`gfx1201` / ROCm 7.2.1; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the
full stack and model provenance.

## 1. Environment

All three variables are required (why: [REPRODUCIBILITY.md](REPRODUCIBILITY.md#environment)):

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export HIP_VISIBLE_DEVICES=1   # device 0 is a different GPU; the R9700 is 1
export WAN_VAE_CONV3D_TEMPORAL_SPLIT=1
```

Python: `/ai/envs/lingbot-world-v2/bin/python` (ROCm torch, no CUDA torch, no
flash-attn — attention runs the PyTorch SDPA path).

## 2. Models (read-only)

| artifact | path |
|---|---|
| assembled 1.3B checkpoint | `/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled` (symlinks: 1.3B DiT shards + T5/VAE/tokenizer from the 14B assets) |
| upstream checkout (pinned) | `/ai/repos/lingbot-world-v2`, branch `rocm/r9700-1.3b-bringup` @ `7cf8109` |
| TAEW2.1 presentation weights | `/ai/models/taehv-011dfc2/taew2_1.pth` (SHA enforced at load) |

Nothing is downloaded at runtime. First ever run for a new image also pays a
one-time conditioning encode (~4 min), cached to `~/.cache/lingbot-world-cond/`
keyed by image, resolution, and horizon.

## 3. Launch

Press-and-play (viewer + session, forge test scene):

```sh
./play.sh
# open http://localhost:8734/ and press keys once the world is ready
```

Note: `play.sh` runs its own defaults (window 18, no timecond memo, no
upscale) — great for playing, not the benchmarked product numbers below.

Validated product stack (2× Lanczos presentation):

```sh
./run_product_2x.sh   # IMAGE= PROMPT= OUT= overrides; extra args append
```

Reference without upscale (same world state, native 368×672 output only):

```sh
./run_product.sh
```

Raw form (reference defaults — conservative, slower; fast flags in
[PERFORMANCE.md](PERFORMANCE.md#stacks)):

```sh
/ai/envs/lingbot-world-v2/bin/python interactive_world.py \
  --image <frame> --prompt "<text>" --output <dir> \
  --script "w w q w"   # optional: run non-interactively, then exit
```

## 4. Controls

Terminal prompt and browser viewer send the same commands. Viewer keys:
`WASD`/arrows to move, `X` to hold position; the page snaps to the live edge
and only newest frames queue. Held keys do not stack up: repeats are
suppressed in the page and the session executes the newest queued key
(latest-key-wins), so a turn pressed mid-generation applies immediately.

| key | action | key | action |
|---|---|---|---|
| `w` | forward | `s` | back |
| `a` | strafe left | `d` | strafe right |
| `q` | turn left (yaw) | `e` | turn right (yaw) |
| `r` | up | `f` | down |
| `t` | tilt up (pitch) | `g` | tilt down |
| `x` | stay | | |

Session commands: `<action> <amount>` (metres, or degrees for turns),
`script ...`, `reset` (wipe world, keep loaded model), `stats`, `help`,
`quit`. Axes are OpenCV (+x right, +y down, +z forward). The model accepts
camera motion only — no attack/jump/interact input exists in the weights.

## 5. What to expect

- Boot: `world ready in ~35 s` with a warm conditioning cache.
- Each action: 4 new frames (the session's very first action yields 1 while
  the streaming decoder primes), ~0.4 s to first RGB on the fast stack.
- Session artifacts in the output dir: `metadata.json`, `initial.png`,
  per-action `chunk_*.mp4`, `session.mp4`, `frames/`, and `actions.jsonl`
  (one JSON record per action with timings, KV position, pose, VRAM).

## If something is wrong

| symptom | first check |
|---|---|
| `import torch` fails on `libroctx64` | `LD_LIBRARY_PATH` (see §1) |
| OOM / session horizon reached | default horizon is 90 latent frames; `reset` or raise `--max_lat_frames` (new horizon re-encodes conditioning) |
| viewer says keys rejected / 503 | the session isn't listening — (re)start it with stdin from the key FIFO (`play.sh` handles this) |
| world races ahead on one press | update to this branch: key-repeat suppression + latest-key-wins are the fix |
| slow model load, idle GPU, one hot CPU core | check no other process holds the GPU; also see root `TROUBLESHOOTING.md` |

Rule of the box: check `rocm-smi --showpids` before killing anything —
production ComfyUI processes share the R9700. Kill by PID, never loose `pkill -f`.
