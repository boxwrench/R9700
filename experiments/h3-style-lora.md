# MiniMax H3 visual-style LoRA procedure

Status: planned procedure. This experiment has not been executed yet.

This procedure is for training a visual-style adapter for MiniMax H3. It is
intentionally separate from the existing H3 Turbo adapter. The goal is to
change visual language while preserving prompt following, motion, and native
audio as much as possible.

## Important scope and compatibility notes

MiniMax H3 is a 33B joint audio-video transformer. Its shared transformer
predicts video and audio latents, while modality-specific behavior is present
in the input/output and AdaLN branches. The official release says the complete
weights support further development, including fine-tuning, but does not ship a
complete public LoRA trainer.

Sources:

- [Official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [Official H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)

Current community training tools are experimental. Inline Studio advertises H3
image and short-video LoRA training, but lists AMD/ROCm as untested. Do not
replace the working ComfyUI environment with the trainer environment. Use a
separate virtual environment and record the trainer commit.

## Experiment objective

Create an adapter named `boxwrench_style_h3_v1` that makes unrelated subjects
share a defined visual language without making every output look like the
training subjects.

The first run is visual-only. Audio training is a separate follow-up so that a
bad audio result cannot be mistaken for a visual-style result.

## Dataset plan

Start with a small, rights-reviewed pilot:

- 40–80 varied still images;
- 12–24 short video clips for the motion follow-up;
- 8–12 held-out validation samples that never enter training;
- one unique trigger token: `boxwrench_style_h3_v1`.

The still set should vary the subject, composition, lighting, camera distance,
and environment. If every image contains the same character, the adapter will
learn a character LoRA rather than a style LoRA.

Use captions that describe the scene and subject. Do not put the style name in
every caption; the trigger token should carry that information.

Example caption:

```text
boxwrench_style_h3_v1, a small brass robot repairing a machine in a dark workshop, warm rim light, medium shot
```

For every asset, preserve this metadata outside the trainer UI:

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

Stable Diffusion images and Wan videos may be used as source media when their
individual licenses permit it. A Stable Diffusion or Wan LoRA file itself is
not transferable to H3. All media must be encoded again for H3 by the selected
trainer.

## Phase 0 — freeze the base

Do not train against an unknown or changing base. Record:

```text
base_family: MiniMax H3 FL2VA or Ref2VA
base_filename:
base_sha256:
text_encoder_filename:
text_encoder_sha256:
video_vae_filename:
video_vae_sha256:
audio_vae_filename:
audio_vae_sha256:
trainer_repository:
trainer_commit:
```

The local inference base currently documented in this repository is an FP8
ComfyUI artifact. A trainer-compatible BF16 or 4-bit base may be different.
The training base and inference base must be explicitly tested as compatible;
do not assume that an adapter trained against one quantization can be loaded
into another.

## Phase 1 — environment smoke test

Use a separate checkout and environment. One current community option is:

```bash
git clone https://github.com/inlineresearch/Inline-Studio.git
cd Inline-Studio/core
./webui.sh --install --extra training
./webui.sh
```

On this AMD system, treat the installation as an experiment. Inline Studio’s
own documentation lists Linux ROCm as untested. Stop if the environment would
overwrite the ComfyUI PyTorch installation or if the H3 base is not recognized.

The first smoke run should be:

```text
dataset: 4 images
resolution: 512px training bucket
rank: 16
alpha: 16
batch size: 1
gradient checkpointing: enabled
audio training: disabled
steps: 100
validation: enabled
```

Success means that the run completes, writes a valid LoRA, and the adapter can
be loaded against the frozen base. Loss alone is not a success criterion.

## Phase 2 — visual-style pilot

Use the full still-image pilot only after Phase 1 passes.

Recommended starting configuration:

```text
rank: 16
alpha: 16
batch size: 1
gradient checkpointing: enabled
optimizer: 8-bit AdamW if supported by the trainer
resolution: 512px for the first AMD run
audio training: disabled
checkpoint interval: 250 or 500 steps
```

Do not decide the final checkpoint by step number alone. Save several
checkpoints and compare them on the same validation matrix.

## Continuing and branching runs

Do not throw away a useful adapter after the first pilot. After the 100-step
smoke test passes, resume the same run to 250, 500, and then 1,000 steps as
needed. Use the trainer full resume checkpoint when available; a true resume
restores the adapter weights, optimizer state, random state, and step number.

An exported `.safetensors` LoRA may be usable for inference without containing
the optimizer state. Loading that file into a new run is a warm start, not an
exact resume. Record which of those two paths was used.

Keep every useful checkpoint and give it an unambiguous name, for example:

```text
boxwrench_style_h3_v1_image_step0100
boxwrench_style_h3_v1_image_step0250
boxwrench_style_h3_v1_image_step0500
```

Resume only when the base model, rank, LoRA targets, trigger token, trainer
commit, and relevant preprocessing settings remain compatible. If one of
those changes, start a new run or explicitly record a warm start. Never
silently replace the best checkpoint.

Validation matrix:

- portrait or character;
- product/object close-up;
- landscape or environment;
- low light;
- bright daylight;
- wide shot;
- medium shot;
- motion-oriented prompt;
- three fixed seeds per prompt;
- adapter strengths 0.5, 0.75, and 1.0.

Record prompt adherence, style consistency, anatomy, temporal behavior when
used for video, audio validity, generation time, peak VRAM, and any GPU or
service errors.

## Phase 3 — motion/style follow-up

Only add video after the image adapter is useful. Use short clips with varied
subjects and coherent motion. Keep video resolution, frame count, frame rate,
and crop policy fixed for the run.

Do not mix stills and clips without recording the sampling ratio. If the trainer
allows both, run a still-only and a clip-only control so we can tell whether a
result came from style learning or motion learning.

The motion run should be a new branch from the best image checkpoint, for
example:

```text
parent: boxwrench_style_h3_v1_image_step0500
name: boxwrench_style_h3_motion_v1
```

This lets the video run build on the learned visual language without
destroying the image-only result. Keep a separate video-from-base control when
possible; it tells us whether an improvement came from learning style or only
from continued training. Do not overwrite the image-only adapter.

Start the voice/audio branch from the frozen base unless we deliberately want
to test multimodal continuation. Keeping it separate prevents audio problems
from contaminating the visual adapter.

## Voice experiment

Yes, TTS-generated speech could help train a consistent voice-like result, but
it will teach the acoustic signature of the TTS system as well as the voice.
It may reproduce robotic timing, vocoder artifacts, pronunciation habits, or
background processing. It is not equivalent to training a dedicated voice
cloning model.

For the first voice dataset:

- use one speaker and one TTS model revision;
- use many different scripts and phonetic combinations;
- keep speech dry, clean, and isolated from music and sound effects;
- keep loudness, sample rate, channel layout, and silence policy consistent;
- retain the text transcript for every clip;
- record the TTS model, voice identifier, settings, and license;
- do not mix multiple voices in the same run.

For H3, use a separate audio-enabled experiment only if the selected trainer
supports audio targets. The joint model means a visual-only LoRA should not be
advertised as a voice LoRA. The safest production path is still:

```text
H3 generates video and ambience
dedicated TTS system generates the consistent voice
audio mixer combines voice, ambience, and effects
```

The H3 audio LoRA is worth testing as a research branch, not as the only voice
solution.

## Required run artifacts

Save these beside each run:

```text
config.yaml or exported trainer settings
dataset manifest and sha256
base model manifest and sha256
trainer commit
environment versions
training log
checkpoint hashes
resume checkpoint or parent run
optimizer/RNG state availability
validation prompts and seeds
before/after media
audio waveform or spectrogram checks
failure notes
```

Never publish the dataset as a single archive without its source and license
manifest.

