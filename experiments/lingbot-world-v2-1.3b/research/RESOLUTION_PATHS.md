# R9700 resolution paths: LingBot v1/v2 architecture & deployment investigation

Product question: can we materially improve displayed resolution while keeping
the ~525 ms closed-loop (first RGB ~390 ms)? Preferred shape is path B:
368x672 sim unchanged, presentation-only refinement to ~720p.

Authoritative R9700 baseline: 368x672 / latent 46x84 / 966 tok/frame,
local_attn_size=12, sink 6, 3-step-A 999->908->768, exact clean t=0 KV,
TAEW2.1 FP16, timecond memoization. Closed: camcond hoist, torch.compile,
static/dynamic sync removal — do not revisit.

## 1. v1 720p Fast: verified reality vs documentation ambiguity

Verified: ONE `lingbot-world-fast` checkpoint serves both 480p and 720p. HF card
row `| LingBot-World-Fast | Camera Poses | 480P & 720P |` and card text "Our model
supports video generation at both 480P and 720P" (same text on base-cam card).
Fast config is byte-identical to each base expert
(`WanModel dim 5120, 40 layers/heads, ffn 13824`); Fast is a distilled single
expert (paper s3.4.1: "we initialize our causal student model using the
high-noise expert"), not the MoE pair.

Resolution is runtime geometry, not separate weights: `SIZE_CONFIGS` /
`MAX_AREA_CONFIGS` map `--size` to pixel area; latent grid is solved from
area x aspect with VAE stride (4,8,8) + patch (1,2,2). At 16:9 720p
(1280x720): latent 90x160 -> 3600 tokens/frame. At 480x832: latent 60x104 ->
1560 tokens/frame (~2.3x fewer). RoPE slices per-(f,h,w) at runtime, so the
mechanism is resolution-agnostic; per-resolution shift is re-tuned
(480p shift 3.0 vs 720p shift 5.0); Fast runs 4-step DMD
(`timesteps_index [0,179,358,679]`), chunk_size 3.

Ambiguity that remains: the causal-adaptation/DMD training resolution mix is
unstated (paper Table 1 says 720p; `run_fast.sh` demos only 480x832). No
published Fast-720p fps/VRAM/quality delta; only "16 fps throughput when
processing 480p on one GPU node". 720p via `--size 1280*720` is supported but
undemoed. Cost estimate for us (not a measurement): 720p-class tokens/frame
are ~2.3-3.7x our 966, scaling DiT work roughly linearly — incompatible with
the 525 ms loop on 1x R9700.

Primary sources: `Robbyant/lingbot-world` (`generate_fast.py`, `wan/configs/`,
`wan_i2v_A14B.py`, `image2video_fast.py`), HF `robbyant/lingbot-world-fast` +
`lingbot-world-base-cam` cards, arXiv:2601.20540 ss3.3.1/3.4.1/3.4.2/4.1.1.

## 2. v2 native resolution vs final 720p deployment resolution

The DiT is native 480p; 720p/60fps is the refined deployed stream, not DiT
native generation. Evidence, all fetched and quoted:

- Abstract (arXiv:2607.07534): "our system guarantees rapid response time,
  sufficient to drive 720p video streams at 60 fps" — subject is the system.
  Same abstract pairs the 14B model with "a lightweight 1.3B counterpart,
  which supports effortless deployment on a single GPU."
- v2 README (verified L151): "We do NOT plan to release our deployment code.
  If you would like to deploy our model yourself, please refer to the
  LingBot-World deployment in SGLang or flashdreams." Open inference is ALL
  480p (`--size 480*832` for both 14B and 1.3B).
- Paper S4.3.1: "lightweight spatio-temporal refiner after remote VAE
  decoding ... spatial ... upsampling ... temporal ... synthesizing
  intermediate frames ... TensorRT engines, multi-GPU". No per-stage ms.
- SGLang cookbook: 832x480 @ fps 25 base, upscale is "server-side super
  resolution AFTER decode", "keep interpolation disabled when measuring true
  generated FPS".
- Press (MarkTechPost 2026-07-09): "The 60 fps figure describes the deployed
  stream, which passes a spatio-temporal refiner."

Do not conflate the 720p final stream with 720p diffusion-native generation.

## 3. v2 spatial/temporal refiner: facts and release status

BLOCKED. Beyond the S4.3.1 paragraph above, nothing is public:

- Code: `Robbyant/lingbot-world-v2` tree HEAD has 85 paths, zero matching
  refin*/upscal*/super-res/VSR/RIFE/interp (verified via GitHub API this
  session). `generate.py` contains no refiner stage.
- Weights: all v2 HF repos contain DiT shards + `Wan2.1_VAE.pth` + umt5-xxl
  only (1.3B = 6 shards + index). No refiner checkpoint on HF or ModelScope.
- SGLang's upscaling is generic user-supplied postproc (Real-ESRGAN path +
  RIFE frame interpolation), NOT the Robbyant refiner.
- License on released v2 artifacts: CC-BY-NC-SA-4.0.

Inferred (unverifiable): ~480p -> 720p spatial (~1.5x) + temporal interp to
60 fps, post-VAE, TensorRT, multi-GPU, "only limited additional latency" (no
numbers). Cannot be the next experiment.

## 4. Transferable FlashDreams / SGLang components

Transferable PATTERNS (not drop-in code):

- LingBot presets (`flashdreams@6e2d3ad integrations_v2/lingbot/config.py`):
  base window 63/sink 0 + WanVAE decoder; streaming preset = TAEHV decoder +
  window 15/sink 3, `len_t=3`, 4-step `[1000,821,642,321]`.
- Window+sink policy, lazy-encode black frames, interactive KV-window event
  protocol, BF16 flag, VAE spatial-parallel decode, post-decode async present.
- SGLang 2.0 deploy shape: 8-GPU Ulysses, VAE parallel decode,
  `--enable-torch-compile false`, int4 PRQ KV quant (recent chunks stay BF16).

NOT transferable without real work:

- TAE weights: FlashDreams defaults to `lightx2v/Autoencoders lighttaew2_1`
  vs our `madebyollin/taehv taew2_1` — same family, NOT same weights
  (key-remap + channel truncation); numeric interchange untested.
- FlashVSR IS wired into flashdreams (`cam2v --postprocess-preset
  flashvsr-v1.1-sparse-1.5`, adapts 9-frame first / 12-frame steady blocks)
  but is absent from SGLang (Real-ESRGAN only). FlashVSR (arXiv:2510.12747,
  weights `JunhaoZhuang/FlashVSR-v1.1`) is a 3rd-party 2x streaming VSR, not
  Robbyant code: 1-step, 8-frame blocks, TAEHV-subclass decoder, ~17 FPS at
  768x1408 on A100 with VAE decode ~70% of cost. Port blockers on gfx1201:
  AdaIN `.cu` needs a HIP port, Triton sparse CUDA guards/SM tuning,
  CUDA-only flash_attn, cuDNN SDPA path, Inductor/CUDAGraph revalidation.
  No TensorRT/NVENC in the flashdreams implementation itself.
- SGLang pipeline internals + Ulysses kernels + quant lib are multi-GPU /
  CUDA-oriented, not portable to a single R9700.

Assessment: FlashVSR is a concrete candidate COMPONENT (it already consumes
LingBot-class streams in flashdreams), but it is a hard port, not a trial.
Do not attempt the port in the bounded experiment.

## 5. v1 -> v2 runtime differences worth preserving or revisiting

Local checkouts: `/ai/github/lingbot-world-v2` (upstream) and
`/ai/repos/lingbot-world-v2` (our fork, HEAD 7cf8109).

- CHANGED: control channels 6/7 incl. action path (v1) -> fixed 6
  camera-only (v2); action inputs (`action.npy`, WASD->camera maps) REMOVED.
- CHANGED: chunk_size 3 (v1 hardcode) -> 4 (v2 default; `run_fast.sh` uses 4).
- CHANGED: demo window global/0 (v1) -> 18/6 (v2 `run_fast.sh`).
- NEW in v2: `WanI2VCausal` dispatcher + `_generate_causal_pretrain`
  (40-step CFG path); our sync-skip kwargs (`grid_sizes_py`,
  `cross_attn_first_call`) are local additions on top.
- SAME: RoPE/KV math, VAE stride + temporal split, `t*0.0` clean re-feed,
  SDPA fallback, T5 handling, `in_dim 36`.
- Borrowable without switching to v1: global-attention consistency baseline,
  cheaper 3-channel conditioning precedent, WASD/action decoupling maps.
  None of these raise resolution; do NOT switch back to v1.

## 6. Dynamic KV and WorldKV status

- Paper S4.3.2 claims a scheduler that adapts "the cache on the fly
  according to the current control signal"; S6 admits the model "does not
  truly remember ... region that leaves context window and is later revisited
  tends to be regenerated rather than recalled".
- Released code is static FIFO only (`--local_attn_size/--sink_size`, left-
  shift eviction; evicted K/V never stored). No dynamic/retrieval/compression
  code in the release.
- WorldKV (`cvlab-kaist/WorldKV`, arXiv:2605.22718, third-party, tested on
  v1-14B only): banks evicted chunks per-layer (CPU), retrieves top-k by
  camera pose (translation-L2 + rotation-geodesic), re-inserts as
  [sink|retrieved|recent] without re-encoding, optional RoPE correction, plus
  optional anchor+pool token compression (3T->1.5T). Reported ~2x full-KV
  speed at near-full-KV quality on v1-14B. Deployment-scheduler equivalence
  unconfirmable (that stack is closed). Research only; do not integrate now.
  (Relevance note: it targets long-horizon recall, not our latency/resolution
  question, and our 12-frame loop is already at its latency floor.)

## 7. Ranked resolution paths for OUR system

### #1 Simple non-neural upscale, presentation-only (RECOMMENDED first)

- In/out: 368x672 TAE RGB -> ~720p-class present (aspect 0.548 vs 0.5625
  needs an explicit letterbox/crop decision at present time, not in sim).
- World state: ZERO change (never fed back; DiT/KV/poses/latents untouched).
- DiT work: ZERO added tokens (stays 966/frame).
- Temporal risk: LOW (deterministic; only aliasing/shimmer on fine detail).
- ROCm difficulty: TRIVIAL (display-side resize, no weights/kernels).
- Weights/code/license: none needed.
- Unknowns: perceived-sharpness ceiling vs neural SR; panel alignment.

### #2 FlashVSR-like deployment component, async sidecar

- In/out: LR RGB video -> 2x RGB (8-frame blocks in flashdreams wiring).
- World state: ZERO iff kept strictly after-decode/display-only.
- DiT work: ZERO DiT tokens; adds its own diffusion+decoder cost (~24 GB
  class VRAM,--single-R9700 fit unmeasured).
- Temporal risk: MEDIUM (1-step streaming; lookahead-vs-loop interaction with
  our chunk=1/window-12 unmeasured; flicker risk if mismatched).
- ROCm difficulty: HARD (AdaIN HIP port, Triton/CUDA guards, flash_attn,
  backends revalidation). No port in the bounded test.
- Weights/code/license: `JunhaoZhuang/FlashVSR-v1.1` (check license before
  any product use); 3rd-party, not Robbyant; ~17 FPS 768x1408/A100 publ.
- Unknowns: 1xR9700 fps/VRAM at 368x672->~720p; TAE-weight interchange.

### #3 Official/near-official LingBot spatial refiner — BLOCKED

- No code, no weights, no arch/params/shapes/license; TensorRT + multi-GPU
  closed stack. Do not plan on it.

### #4 Native higher-res 1.3B inference — NOT recommended

- 480p-class = 1560 tok/frame (+61%); 720p-class = 3600 (+273%). Direct
  multiplier on every step of the 525 ms loop; breaks the budget on 1xR9700
  and changes world state (grid, KV footprint, Plucker scale). The `max_area`
  knob makes it technically easy and perf-infeasible. Training-res mix for
  the 1.3B unpublished; shift would need per-res re-tune.

No generic-AI-upscaler filler was added: only components already used around
LingBot-class models (or trivially safe present-path ops) are ranked.

## 8. ONE bounded next hardware experiment

Presentation-only 368x672 -> ~720p bicubic/lanczos, off the critical path.
Sim/DiT/TAEW2.1 frozen at 368x672; tap decoded RGB packets; add an async
present worker (separate queue/thread) upscaling to panel ~720p with an
explicit crop/pad rule. SR output never re-enters world state.

- Budget: added per-packet present p95 <= 15 ms; next-ready must not regress
  vs ~525 ms within noise.
- Report three latencies separately: (a) first-frame (first sim frame ->
  first upscaled present; includes queue fill, NOT a gate), (b) steady-state
  per-packet present median/p95 (the gate), (c) next-ready (the loop guard).
- Run sync-present vs async-queue-present; keep whichever meets (b)+(c).
- Pass = (b) p95 <= 15 ms AND (c) unchanged; (a) reported only.
- If neither meets it: stop. No kernel tuning, no DiT changes (rejected
  items stay closed). A ~25 ms candidate is admissible ONLY async AND only
  with a substantial visual win, executed strictly outside next-action
  readiness.
- Output: the 3 numbers + worker mode + crop/scale rule + panel res. No
  quality claim beyond "presentation-only, no world-state effect".

## 9. Primary sources

- arXiv:2601.20540 (v1 paper, ss2.2/3.3.1/3.4.1/3.4.2/4.1.1)
- arXiv:2607.07534 (v2 paper; abstract + S1/S3.3/S4.1/S4.3.1/S4.3.2/S5/S6)
- `Robbyant/lingbot-world` (`generate_fast.py` L158-168/276-277/374-378/
  479-480, `wan/configs/__init__.py` L12-30, `wan_i2v_A14B.py` L16-36,
  `model_fast.py` L22-51, `image2video_fast.py` L438-448)
- `Robbyant/lingbot-world-v2` @main tree (85 paths, zero refiner paths —
  verified via GitHub API 2026-09-10); README L151 non-release quote
- HF `robbyant/lingbot-world-fast`, `lingbot-world-base-cam` cards;
  HF `lingbot-world-v2-1.3b-causal-fast` (rev 7e36a5f, DiT+VAE+T5 only)
- SGLang cookbook `docs.sglang.io/cookbook/diffusion/LingBot-World/
  LingBot-World-2.0` (832x480/fps-25 base, after-decode SR, 8-GPU Ulysses,
  int4 PRQ KV quant, compile off)
- `NVIDIA/flashdreams@6e2d3ad` (`integrations_v2/lingbot/config.py`,
  `recipes/taehv/`, `cam2v --postprocess-preset flashvsr-v1.1-sparse-1.5`)
- FlashVSR paper arXiv:2510.12747; weights `JunhaoZhuang/FlashVSR-v1.1`
- WorldKV arXiv:2605.22718; `cvlab-kaist/WorldKV` (2c9bcbb/046f6d1)
- MarkTechPost 2026-07-09 (deployed-stream reading of 60 fps)
- All v2 released artifacts: CC-BY-NC-SA-4.0

## 10. Verified facts vs unknowns

Verified this session by direct fetch/inspection: abstract 720p-system
wording; README L151 non-release quote; 85-path tree with zero refiner
paths; open-inference 480p for both v2 checkpoints; SGLang after-decode SR
posture. Carried from file:line evidence: v1 one-ckpt dual-res + config
identity + token math + shift-per-res + VAE-merge layout; FlashDreams
window/sink presets + TAE family mismatch + FlashVSR wiring + CUDA-only
blockers; v1->v2 control/chunk/schedule deltas; release KV = static FIFO.

Open unknowns (disclosed, not blocking the recommendation): v1 DMD
resolution mix; Fast-720p latency/VRAM/quality delta; refiner arch/params/
shapes/license/release plans; per-stage refiner latency + demo GPU count;
DiT-native fps; lighttaew-vs-taew numeric interchange; 1xR9700 FlashVSR
fps/VRAM; SGLang pipeline internals; WorldKV on v2-1.3B/1xR9700 behavior.
