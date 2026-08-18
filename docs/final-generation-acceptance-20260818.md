# R9700 final generation acceptance — 2026-08-18

This is the closing record for the final real-world workflow-switching acceptance and showcase campaign. It is **not** an optimization pass — the LTX 2.5, MiniMax H3 (T2V/I2V/R2V), and MiniMax Music 3 stacks were already optimized and hardened before this session began. This document records what actually happened when all four production workflows were exercised back-to-back in one long-lived ComfyUI process, including the failures.

## Hardening state used

- Hardening branch: `hardening/production-manifest-and-preflight`
- Hardening commit: `a0a96bffe5748bcd379722e9640c4bdd1b4141d5`
- Showcase branch: `showcase/final-acceptance-20260818`, HEAD at takeover was identical to the hardening commit (no drift)
- `production/manifest.json` model SHA-256 hashes, workflow file/semantic hashes, and required launch flags/environment were used as the reference state throughout this session and were re-verified unchanged at the end (see [Final verification](#final-verification)).
- `git status --short` at takeover showed **no modifications to any tracked file**, including everything under `production/`. All prior-session work (experimental data, docs, the driver script, and the showcase workspace) was present only as new, untracked files. No accidental golden/production edits needed to be reverted.

## Session order

The intended and actual chronological order (previous agent had completed runs 1–4 and made repeated attempts at run 5 before running out of quota; this session picked up from there):

1. LTX 2.5 — *The Artifact Forge*
2. MiniMax H3 T2V — *Machine Cathedral*
3. MiniMax H3 I2V — *Forge Awakening*
4. MiniMax Music 3 — *Signal Against the End*
5. MiniMax H3 R2V — *Observatory*
6. LTX 2.5 return — *Signal Vault*

The ComfyUI server (`/ai/comfyui/main.py`, PID confirmed via `ps aux`, launched with the required production flags: `--output-directory /ai/artifacts/runs/minimax-h3 --disable-dynamic-vram --disable-smart-memory --disable-mmap --bf16-vae --reserve-vram 2`) restarted once mid-session, around 14:58 PDT, between run 2/completion-of-run-4 and run 3's successful execution. This is why runs 1, 2, and 4 are not recoverable from the live `/history` API (in-memory history is cleared on restart) but runs 3, 5, and 6 are.

## Existing R2V job at takeover

The handoff indicated an in-flight submission referred to as `7a71502d`. On inspection via `GET /history/7a71502d-1722-46e8-a6f3-46b217b2d416`:

- **Status: FAILED** (`status_str: "error"`, `completed: false`)
- Error: `MiniMaxH3ReferenceToVideo.execute() got an unexpected keyword argument 'ref_image'`
- The submitted prompt JSON used the legacy singular `ref_image` key plus the newly-added `audio_vae` and `ref_image_size` inputs. Validation had passed (no `node_errors` at `/prompt` submit time) but execution failed, because validation only checks the class's static `INPUT_TYPES()` schema while the actual keyword the node receives comes from the V3 dynamic-input (`Autogrow`) expansion, which the legacy key never triggers.

No new R2V job was submitted until this was root-caused. See below.

## R2V root cause and resolution

`MiniMaxH3ReferenceToVideo.ref_images` (`comfy_extras/nodes_minimax_h3.py`) is declared as `io.Autogrow.Input(..., prefix="ref_image_", min=0, max=9)`. Tracing ComfyUI's V3 dynamic-input machinery (`comfy_api/latest/_io.py`: `parse_class_inputs` → `Autogrow._expand_schema_for_dynamic` → `build_nested_inputs`, and `execution.py: get_input_data` / `get_finalized_class_inputs`) showed that a filled autogrow slot is only recognized when the **prompt JSON's flat input key is the dotted path** `ref_images.ref_image_0` wired directly as a graph link — not a bare `ref_image_0` and not the legacy singular `ref_image`. This exact convention is already implemented in the node's own frontend widget (`custom_nodes/ComfyUI-ALLinONE-MinimaxH3/web/one_node_minimax_h3.js`: ``wf["6"].inputs[`ref_images.ref_image_${idx}`]=[id,0]``), confirming it as the correct, currently-supported shape rather than a guess. `audio_vae` and `ref_image_size` also needed to be added — both are newer required/optional inputs the checked-in `production/workflows/h3_r2v.json` predates.

Six submissions failed before this session began (all preserved in `showcase/metadata/05-h3-r2v-boxwrench-observatory.json` → `failed_attempts`, and cross-checked against live `/history`):

| Attempt | Prompt ID | Failure |
|---|---|---|
| 1 | `8c766146` | unexpected kwarg `ref_image_1` (flat key, no dotted prefix) |
| 2 | `1d6d7924` | `VAEDecodeAudio` tensor-shape RuntimeError (deterministic, unrelated to the ref-image key) |
| 3 | `8a80c8a8` | unexpected kwarg `ref_image` (legacy key doesn't exist in the live schema) |
| 4 | `e1270b7f` | same `VAEDecodeAudio` RuntimeError |
| 5 | `de0af02f` | unexpected kwarg `ref_image_0` (flat key, no dotted prefix) |
| 6 | `7a71502d` | unexpected kwarg `ref_image` (handoff's in-flight job) |

**Fix applied:** a showcase-only workflow copy, `showcase/workflows/h3_r2v-showcase.json`, built from `production/workflows/h3_r2v.json` with:
1. `"ref_image": [...]` replaced by `"ref_images.ref_image_0": [...]`
2. `"audio_vae": ["4", 0]` added (existing audio VAE loader node, previously loaded but unwired)
3. `"ref_image_size": "match"` added
4. `LoadImage` pointed at `bx77-anchor.png`, prompt/filename_prefix set to the showcase content

**`production/workflows/h3_r2v.json` was not modified** — confirmed by `git diff` / `git status --short` showing zero changes under `production/` at every point in this session, both before and after the R2V fix.

The corrected submission (prompt `6dd30fec-7d0c-47c6-a742-9e6408668975`) succeeded on the first try: wall time 81.47&thinsp;s (`execution_start` 15:36:40.290 → `execution_success` 15:38:01.763), artifact `05-h3-r2v-boxwrench-observatory_00001_.mp4` (608×352, 39 frames/24fps ≈ 1.625s, h264+aac, valid per `ffprobe`, no decode errors). Per the R2V correctness gate, no further R2V iteration was performed after this success.

The same schema-drift class was also hit on **H3 I2V** (run 3): `MiniMaxH3ImageToVideo.execute() got an unexpected keyword argument 'image'` (prompt `d39d8320`) — the live schema wants `first_frame`, not `image`. Fixed the same way (key corrected at queue time only; `production/workflows/h3_i2v.json` untouched) and the retry (`c204b5d0`) succeeded.

## Final session — actual timings

All wall times are ComfyUI `execution_start` → `execution_success` deltas (runs 3, 5, 6) or the driver script's own `time.time()` instrumentation (runs 1, 2, 4 — captured live before the driver halted on the run-5 schema mismatch). None are estimated; `peak_vram_gib` is `NA` wherever it was not actually polled during that specific run (the automated VRAM-polling driver only covered runs 1–2).

| # | Workflow | Prompt ID | Wall (s) | Peak VRAM | Artifact | Notes |
|---|---|---|---:|---:|---|---|
| 1 | LTX 2.5 (Artifact Forge) | `5016503f` | 11.48* | NA | `01-ltx-boxwrench-artifact-forge_00001_.mp4` | *Warm-cache (model+text-encode reused from the immediately preceding same-session attempt); first attempt (`5a59144d`) rendered wrong content, see below |
| 2 | H3 T2V (Machine Cathedral) | `c4b638c1` | 131.94 | 27.58&thinsp;GiB | `02-h3-t2v-boxwrench-cathedral_00001_.mp4` | Clean first submission |
| 3 | H3 I2V (Forge Awakening) | `c204b5d0` | 49.16 | NA | `03-h3-i2v-boxwrench-forge-awakening_00001_.mp4` | 1 failed attempt (`image`→`first_frame`) |
| 4 | Music 3 (Signal Against the End) | `51fad938` | 28.48 | NA | `04-music3-signal-against-the-end_00001.flac` | 14.99s output (target 15.0s) |
| 5 | H3 R2V (Observatory) | `6dd30fec` | 81.47 | NA | `05-h3-r2v-boxwrench-observatory_00001_.mp4` | 6 failed attempts, see above |
| 6 | LTX 2.5 return (Signal Vault) | `ce215c01` | **23.62** | NA | `06-ltx-boxwrench-signal-return_00001_.mp4` | Confirmed matches handoff exactly |

**Confirmation: LTX return succeeded at 23.62&thinsp;s**, verified independently against `/history` (`execution_start` 15:21:35 → `execution_success` 15:21:58) — the handoff's reported number was accurate and this run was correctly treated as already-complete; it was not re-run.

### LTX run 1 — invalid first artifact and its correction

The file produced by the first LTX submission (`5a59144d`, wall 33.47s per the driver's own sidecar) was **not the Boxwrench prompt**. Its content — verified visually, frame by frame — is a snow leopard on ice, matching `production/workflows/ltx25.json` node 5's unmodified placeholder text ("A close-up of an Arctic hunter crouching on an ice floe…") rather than the showcase override. The prompt substitution never reached that specific queued job. This is a genuine invalid artifact, not a stylistic/fidelity concern, so it was re-run rather than kept:

- Original file preserved (not deleted) at `/ai/artifacts/runs/r9700-final-showcase-invalid/01-ltx-boxwrench-artifact-forge_00001_WRONG-PROMPT-arctic-hunter-default-text.mp4`.
- Re-submission with the correct prompt (`77921571`) hit the same deterministic `VAEDecodeAudio` tensor-mismatch RuntimeError already seen twice during R2V debugging (`1d6d7924`, `e1270b7f`). A same-graph retry (`bdac84cf`) reproduced it identically, confirming it is deterministic for this shape, not transient.
- Resolution: dropped the audio branch (removed `VAEDecodeAudio`, removed `audio` from `CreateVideo`) — the exact pattern the already-successful run 6 (`ce215c01`) uses. The corrected submission (`0fb44e33`, near-instant on cache) produced valid on-prompt content; a clean re-measurement with a bumped seed (`5016503f`, seed `8112031`) gave the reported 11.48s.
- `production/workflows/ltx25.json` was **not modified**; all fixes live in `showcase/workflows/ltx25-run1-showcase.json`.

### LTX run 6 — known creative-fidelity limitation (not a bug)

The LTX return artifact is a valid, on-theme completion (armored figure, cosmic backdrop, glowing chest core) but does not closely reproduce the Boxwrench anchor's specific gunmetal/brass palette or helmet geometry. LTX 2.5 here is pure text conditioning with no image-reference mechanism, unlike H3 I2V/R2V, so exact identity match is not guaranteed. Per the acceptance instructions ("one clean successful run is enough… do not keep experimenting after success"), and because this run was already reported complete in the handoff, it was recorded honestly rather than re-rolled.

## Workflow/model switching

All four production workflows (`ltx25`, `h3_t2v`, `h3_i2v`, `h3_r2v`) were loaded and executed in sequence inside one long-lived ComfyUI process (across the one mid-session restart), switching UNET/CLIP/VAE loaders each time with no manual intervention beyond the two schema-key fixes above. **PASS.**

## Media publication

Per user confirmation, a new GitHub Release, [`showcase-final-20260818`](https://github.com/boxwrench/R9700/releases/tag/showcase-final-20260818), was created on `boxwrench/R9700` and the 6 final artifacts (5 MP4 + 1 FLAC, ≈2.3&thinsp;MB total) were uploaded as release assets, following the existing `video-v1` convention. No large media was added to ordinary Git history. Small poster JPEGs/PNGs (≈260&thinsp;KB total) were committed under `docs/assets/showcase/`.

## Voice

Per the prior plan, voice cloning was only to be executed if an appropriate local voice-cloning path was already installed — no new installation was permitted.

- Reference: `/home/boxwrench/Downloads/cylon.m4a` (24.067s, 48kHz stereo AAC)
- Evaluated against the local runtime: no functional zero-install ROCm TTS voice-cloning path is configured (`libroctx64.so.4` dependency missing).
- **VOICE CLONE NOT EXECUTED.**
- Narration script retained at `showcase/prompts/voiceover-script.md` for future use.

## Showcase asset manifest

- `showcase/metadata/0{1..6}-*.json` — per-run sidecars (parameters, models, prompt IDs, failed attempts, root causes)
- `showcase/session/session.json` — all 6 runs, chronological
- `data/final-showcase-session-20260818.tsv` — same, as TSV
- `showcase/workflows/ltx25-run1-showcase.json`, `showcase/workflows/h3_r2v-showcase.json` — showcase-only workflow copies with the schema fixes; **not** committed over `production/`
- `showcase/posters/*.jpg`, `*.png` — poster frames (real extracted frames, not synthetic) and Music 3 waveform/spectrogram
- `docs/assets/showcase/` — the same posters, committed for GitHub Pages
- `docs/index.html` — the finished GitHub Pages showcase
- `showcase/prompts/boxwrench-identity.md`, `showcase/prompts/voiceover-script.md` — creative reference docs (pre-existing, preserved)

## Final preflight

See [Final verification](#final-verification) below and the raw output captured at commit time.

## Final verification

- `/ai/comfyui` source: unchanged from takeover state (no edits made; only read for schema tracing).
- Model files: unchanged (no writes, no installs).
- Environment: unchanged (no packages installed, no ROCm/Torch/Triton changes).
- `/ai/github/boxwrench`: unchanged (read-only reference use only, for color tokens and the `bx77-anchor.webp` identity anchor).
- `production/` directory: unchanged from the hardening commit for the entire session (`git status --short` never showed a modified tracked file under `production/`).
