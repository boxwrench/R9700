# Session 2026-09-26 — Keith music video + spy scenes + attention gates

Authored from the Nautilus session. Text record only; no weights or media
are stored in this repo. Absolute paths below refer to the Nautilus box.

## Delivered

- 66-second music video, 1280x1728, soundtrack-locked:
  `/ai/artifacts/runs/minimax-h3/minimax-music3/keith-full-66s-cugan2x.mp4`
  (117 MB, CUGAN SE 2x upscale in 208 s over 1582 frames)
- 30-second spy short, 1728x960, three stitched scenes:
  `/ai/artifacts/runs/minimax-h3/minimax-h3/spy-full-30s-cugan2x.mp4`
- Review shelf (also Desktop/keith-review): gate clips 00-03, both masters,
  66 s soundtrack FLAC, spy scenes 1-3 plus facelock 1-2.
- Soundtrack: MiniMax Music 3, hardcore hip-hop 172 BPM D minor, deep male
  rap lead, soul-chop undertow, Bay Area lyrics. Solid portion cut at 66 s
  (tail drifted past 1:06). Full 90s render kept as
  `keith-killa-90s-v2_00001.flac`; working cut is `keith-killa-90s-v2-66s.flac`.

## Locked generation recipe (faces pass)

- Base `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`, Qwen3VL 32B fp8
  encoder, official Turbo 4-step LoRA
  (`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_noprefix.safetensors`) at
  strength 1.0, 7 steps, TurboSampler, simple scheduler.
- Ref2Video with Picture 1 = face identity, Picture 2 = body/wardrobe/scene,
  Audio 1 = song segment, `ref_image_size=max`, 640x864 portrait, 243 frames.
- Per-clip wall ~630 s on the R9700. Dialogue needs single-speaker staging
  (visible mouth moves, listener lips closed and still) or lines swap voices.

## Gate results (identical clip 2, seed 777004, user judged)

| Candidate | Wall | Verdict |
|---|---|---|
| Official Turbo 7-step (locked) | 629.8 s | PASS, production standard |
| pdmd 4-step (533 MB) | 326.4 s | REJECT, distortion |
| pdmd 2-step (364 MB) | 181.1 s | REJECT, distortion |
| ELM longlive 4-step (1.9 GB) | 322.9 s | REJECT, distortion |
| Acc-8Step pruned (1.7 GB) | — | BLOCKED, Turbo node rejects its adaln entries on this base |
| w6a8 ref2va base (14.89 GB, on disk) | — | BLOCKED, ComfyUI 0.33.2 loader predates `w6a8_int8` quant |

## SageAttention RDNA4 (decided 2026-10-01: ADOPTED as standard)

- Source: https://github.com/IxMxAMAR/SageAttention-RDNA4 (Windows-focused).
- Local build: `/ai/tools/sage-rdna4/SageAttention-RDNA4`, wheel
  `dist/sageattention-2.2.0+amd.gfx12.1-cp312-cp312-linux_x86_64.whl` (31 MB),
  installed into `/ai/environments/comfyui-h3`, log `/tmp/sage_build.log`.
- Findings: imports with `SAGEATTENTION_ALLOW_UNVERIFIED_TORCH=1`, runs on
  the R9700 at 0.038 rel-err on random tensors, but only via the fallback
  kernel. The fast hand kernel never loads on Linux (loader references
  `ctypes.WinDLL`). That one-line platform bug is the fork-or-PR item.
- Service trial: `/home/boxwrench/.config/systemd/user/comfyui-h3.service.d/20-sage-attention.conf`
  adds `--use-sage-attention` plus the bypass env. Verified active in the
  unit log ("Using sage attention"). Remove the file + daemon-reload +
  restart to return to SDPA. A/B gate (identical clip 2) was running.
- Gate result (warm, identical recipe, seed 777005): 400.5 s vs 629.8 s
  SDPA, 1.57x pace. File `keith-turbo-clip2-sage_00001_.mp4` (Desktop
  `06-sage-gate.mp4`). User verdict 2026-10-01: quality identical.
  Standard moved to Sage; per-clip expectation drops from ~10.5 to ~6.7 min.
- Open item: hand-written SK1 kernel never loads on Linux
  (`ctypes.WinDLL` in loader). Fix + upstream PR pending; fallback kernel
  (PR #368 path) is what the gate measured.
- 2026-10-02: Linux loader port written, validated, and proposed upstream
  as https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1 (PR #1, one
  file, +80/-12, branch `linux-sk1-loader` on boxwrench/SageAttention-RDNA4).
  Evidence: 41/41 backend tests pass, handle_equal True on one HIP image,
  strict-mode H3 render completes (fallback would raise), warm medians
  Sage 83.5 s vs SDPA 109.4 s on neutral 124f content, same-seed PSNR
  15.2/18.9/17.1 dB per the repo's trajectory-divergence caveat. No
  likeness frames used anywhere in the evidence. Local notes:
  `/ai/tools/sage-rdna4/sage-PR.md`.
- Author's own numbers: 5-14% faster steps in real renders; VRAM savings
  unpublished, estimated low single-digit GB on our 16k-token workload.

## Upscale lanes (measured on 5.17 s finale)

- CUGAN SE 2x (nihui ncnn-vulkan, R9700): 17 s, pixel-faithful (rms 11.9).
  Full 66 in 208 s. Production choice.
- SeedVR2 1080p: 355 s (75 min extrapolated). Cleaner detail, slower.
- ComfyUI RealESRGAN 4x: 1750 s (6+ h extrapolated). Out.
- Video2X CUGAN: black output on RADV, both GPUs. Out.
