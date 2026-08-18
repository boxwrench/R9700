# LTX-2.5 visual-style LoRA procedure

Status: planned procedure. This experiment has not been executed yet.

This is the preferred long-term path for a reproducible LTX-2.5 style adapter
because Lightricks publishes an official `ltx-trainer` with LoRA, audio/video,
image, video, and conditioning support.

Sources:

- [Official LTX-2 repository](https://github.com/Lightricks/LTX-2)
- [LTX trainer quick start](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/quick-start.md)
- [LTX dataset preparation](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md)
- [LTX training modes](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-modes.md)
- [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)

## Hardware decision

The official trainer currently documents Linux with CUDA and recommends an
80 GB NVIDIA GPU for the standard configuration. It provides a low-VRAM
configuration aimed at the approximately 32 GB class using quantization and
memory optimizations.

The R9700/ROCm machine should be used first for local inference validation.
Training should initially run on a compatible NVIDIA cloud machine unless we
deliberately choose to port and test the trainer for ROCm. Do not alter the
working R9700 ComfyUI environment for this procedure.

## Experiment objective

Create an adapter named `boxwrench_style_ltx25_v1` that changes the visual
language of LTX-2.5 while preserving prompt following, motion, audio/video
alignment, and the base model’s normal generation behavior.

The first adapter is a visual-style adapter. Voice training is a separate
branch.

## Dataset plan

Start with a rights-reviewed pilot:

- 40–80 still images;
- 12–24 short video clips;
- 8–12 held-out validation samples;
- trigger token: `boxwrench_style_ltx25_v1`.

LTX’s trainer supports images, videos, or a mixture. If images and videos are
mixed, preprocess separate frame buckets and use batch size 1. The official
documentation also notes that separate still and video LoRAs are the cleaner
fully-supported option. For this project, start with a still-only adapter and
then make a separate video adapter if needed.

Example metadata:

```json
{
  "caption": "boxwrench_style_ltx25_v1, a brass robot repairing a machine in a dark workshop, warm rim light, medium shot",
  "video": "videos/0001.mp4"
}
```

For an image-only record, use the same `video` column with an image path, or
follow the trainer’s current image-dataset convention. Keep the original
source manifest separate from the trainer metadata.

For every asset, preserve:

```text
asset_id
source_url
creator
source_license
attribution_required
synthetic_or_camera_source
model_and_revision_if_synthetic
prompt_if_synthetic
sha256
```

Stable Diffusion images and Wan videos are usable as source media only when the
underlying media license permits it. Their LoRA weights are not transferable to
LTX-2.5.

## Phase 0 — select the correct LTX-2.5 base

Do not train against the local INT8-convrot distilled inference artifact by
default. First obtain and record the trainer-compatible LTX-2.5 training base.
For a split pack, the trainer expects separate files for:

```text
transformer
LTX-2.5 Gemma 4 text encoder
video VAE
audio VAE
```

Record:

```text
transformer_filename:
transformer_sha256:
text_encoder_filename:
text_encoder_sha256:
video_vae_filename:
video_vae_sha256:
audio_vae_filename:
audio_vae_sha256:
ltx_repository_commit:
trainer_commit:
```

LTX-2.5 text embeddings are not interchangeable with older LTX versions. A
new `.precomputed` directory is required when changing the checkpoint,
encoder, trigger token, or encoding options.

## Phase 1 — install the official trainer

Run this on the CUDA training machine:

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
uv sync
cd packages/ltx-trainer
```

Copy the current low-VRAM configuration to a run-specific file. Do not edit
the repository’s shipped example in place.

Starting configuration:

```text
training_mode: lora
rank: 16
alpha: 16
batch size: 1
gradient checkpointing: enabled
8-bit optimizer: enabled where supported
transformer quantization: use the low-VRAM default initially
validation: enabled
```

Use `t2v_lora` when the final use is text-to-video. Use `i2v_lora` when the
adapter must work with both text-to-video and image-to-video. The trainer’s
preprocessing supports a trigger token, so use the same token in all training
captions and validation prompts.

## Phase 2 — prepare and preprocess data

Split long source videos into coherent scenes when necessary:

```bash
uv run python scripts/split_scenes.py input.mp4 scenes/ \
  --filter-shorter-than 5s
```

Create or review `dataset.json`. The official trainer recognizes `video`,
`audio`, and `caption` columns. For video with embedded audio, audio latents
are extracted automatically. For standalone speech files, use the `audio`
column.

Preprocess a 49-frame pilot bucket:

```bash
uv run python scripts/process_dataset.py dataset.json \
  --resolution-buckets "960x544x49" \
  --model-path /models/ltx-2.5/transformer.safetensors \
  --text-encoder-path /models/ltx-2.5/gemma4-text-encoder.safetensors \
  --video-vae-path /models/ltx-2.5/video-vae.safetensors \
  --audio-vae-path /models/ltx-2.5/audio-vae.safetensors \
  --lora-trigger boxwrench_style_ltx25_v1
```

Review the decoded preprocessed samples before training. If the bucket causes
OOM, reduce spatial dimensions or frame count. If the preprocessing options or
base model changes, use a new `.precomputed` directory or pass `--overwrite`.

## Phase 3 — visual-style pilot

Start with the still-only adapter. Use a one-frame bucket such as:

```text
960x544x1
```

Run a 100-step smoke test first. It must:

- complete without OOM or NaN;
- write a valid checkpoint;
- produce a validation sample;
- load the saved LoRA in an LTX inference pipeline.

Then train the pilot and retain intermediate checkpoints. Evaluate the same
validation matrix at each checkpoint:

- portrait or character;
- product/object close-up;
- environment;
- low light;
- daylight;
- wide shot;
- medium shot;
- motion-oriented prompt;
- three fixed seeds per prompt;
- adapter strengths 0.5, 0.75, and 1.0.

Measure visual style consistency, prompt adherence, anatomy, motion, audio
quality, audio/video sync, wall time, peak VRAM, and failures.

## Continuing and branching runs

Once the 100-step smoke test passes, continue the same still-image run rather
than restarting from zero. Resume to 250, 500, and then 1,000 steps only when
the validation samples are still improving. Use the official trainer full
resume checkpoint if available so the adapter weights, optimizer state, random
state, and step number are restored.

A finished `.safetensors` adapter may be sufficient for inference but may not
contain optimizer and scheduler state. Loading it into a new run is a warm
start, not an exact resume. Record which one was used. Example checkpoint
names:

```text
boxwrench_style_ltx25_v1_image_step0100
boxwrench_style_ltx25_v1_image_step0250
boxwrench_style_ltx25_v1_image_step0500
```

Resume only when the base checkpoint, rank, LoRA targets, trigger token,
trainer commit, and precomputed embeddings remain compatible. A changed base,
bucket, encoder, trigger token, or preprocessing option requires a new
preprocessing run and should normally be a new training run. Do not overwrite
the best checkpoint.

## Phase 4 — video-style adapter

If the still adapter looks useful but motion is weak, train a separate adapter:

```text
parent: boxwrench_style_ltx25_v1_image_step0500
boxwrench_style_ltx25_video_v1
```

Use short clips with varied content and a fixed video bucket. The video adapter
can be branched from the best still checkpoint when the base and LoRA targets
are compatible. Keep a separate video-from-base control when possible so we can
measure the value of continuation. Do not silently mix images and videos in
one batch. The official trainer supports mixed data, but it requires separate
frame buckets and batch size 1; two separate adapters make the comparison
easier.

Keep the still adapter unchanged. If the video branch changes the visual style
or prompt adherence, retain the still checkpoint and treat the video result as
a separate experiment rather than replacing it.

## Voice experiment

Yes, a consistent voice can be explored, but LTX-2.5 is not a dedicated TTS
voice-cloning system. A LoRA may learn timbre, speaking rhythm, and audio
texture, but it can also learn TTS artifacts rather than the intended voice.

Use one voice source and many different utterances:

- several minutes of clean speech for the pilot;
- many scripts with varied phonemes and sentence lengths;
- exact transcript for every recording;
- one TTS model, voice identifier, and settings;
- consistent sample rate and loudness;
- no music, reverb, or sound effects in the voice-only branch.

The TTS model and voice must be licensed for this use. Record its name,
revision, voice ID, prompt, seed, settings, and output hash. If the voice is
intended to represent a real person, obtain that person’s permission.

There are two LTX experiments:

### A. Joint audiovisual style plus voice

Use video clips with the generated speech embedded as their audio track. Train
the normal joint video/audio LoRA. This tests whether the adapter can preserve
the voice while generating matching visuals.

### B. Audio-only voice adapter

Use the trainer’s audio-only dataset path with `audio` and `caption` records.
This isolates the voice question from visual style. It is the cleaner research
test, but it should not be described as a production voice-cloning model until
held-out speech is evaluated.

For production consistency, the safer architecture remains:

```text
LTX-2.5 generates video and ambience
dedicated TTS system generates the voice
audio mixer combines voice, ambience, and effects
```

## Required run artifacts

Save beside every run:

```text
run config
dataset.json and dataset hash
license manifest
base model manifest and hashes
trainer and repository commits
preprocessing command
training log
checkpoint hashes
resume checkpoint or parent run
optimizer/RNG state availability
validation prompts and seeds
before/after videos
audio waveforms or spectrograms
VRAM and wall-time measurements
failure notes
```

