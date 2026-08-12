# Comparison protocol

## What stays fixed

For a comparison series, keep these constant unless the model makes a value impossible:

- prompt card and lane;
- target duration: 5 seconds;
- frame rate: 24 fps;
- aspect ratio: 16:9;
- numeric seed: `8112026` where the workflow accepts it;
- prompt enhancement: disabled;
- no LoRA, style adapter, interpolation, upscaling, or post-generation audio;
- model-native sampler and recommended step count, both disclosed;
- first saved decodable artifact after a true application restart for a cold run.

Use the nearest valid native dimensions and frame count when a model has architectural constraints. Report requested and delivered values. A native comparison is valid when every deviation is explicit; it is not an exact-workload comparison.

## Lane rules

### I2V lane

Use the prompt card's named reference image at its original 1672×941 resolution unless the workflow performs a documented model-native resize. Do not crop, recolor, retouch, or pre-animate the source.

### T2V lane

Use the exact self-contained T2V prompt. Do not attach a reference image. The repeated chassis description is the identity anchor and must remain unchanged.

I2V and T2V results receive separate tables, rankings, and claims.

## Prompt handling

- Copy the positive prompt verbatim.
- Use the common negative prompt verbatim when supported.
- If negative conditioning is unsupported, record `unsupported`; do not move it into the positive prompt.
- If native audio is unsupported, record `unsupported`; do not add stock audio before evaluation.
- Do not request rendered text, captions, a logo, or a watermark from the model.
- A retry is a new run, never a replacement for a failed result.

## Common negative prompt

```text
text, subtitles, captions, logo, watermark, interface overlay, duplicate character, extra limbs, missing limbs, fused fingers, malformed hands, warped armor, melting metal, changing helmet, changing visor color, floating tools, camera cuts, jump cuts, flicker, jitter, temporal smearing
```

## Timing and reporting

The headline measurement is prompt submission to saved, decodable artifact. Also report delivered duration and wall seconds per output second.

Keep these distinct:

- application startup time;
- prompt-to-artifact wall time;
- restart-to-artifact total;
- delivered video duration;
- model sampling time, only when directly measured.

Do not infer component timings from incomplete logs. Branding and social-media transcodes happen after the timed artifact and are excluded.

## Quality review

Score the unbranded original from 0–4 on each applicable dimension:

1. BX-77 identity and material consistency;
2. temporal stability;
3. requested motion and physical contact;
4. camera and composition adherence;
5. fine-detail integrity;
6. audio-event alignment, when natively supported.

Publish the category scores rather than collapsing everything into one opaque quality number. Reviewers should not see the model name until scores are locked.

## Integrity gates

A result is invalid if it is corrupt, has the wrong prompt card or lane, uses undisclosed enhancement, changes the source frame, includes an undisclosed adapter, or lacks enough provenance to reproduce the run. A visible failure may remain in the record as a valid failed generation.
