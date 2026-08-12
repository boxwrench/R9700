# Boxwrench Video Comparison Prompt Standard v1

This is the canonical branded companion suite to the neutral brass-robot benchmark. It provides five repeatable Boxwrench scenes, each with an exact image-to-video (I2V) prompt and a self-contained text-to-video (T2V) prompt.

## The five tests

| ID | Scene | Primary stress | Reference image |
|---|---|---|---|
| BX-C01 | Anchor Scan | identity, reflective armor, subtle motion | `references/bx77-anchor.webp` |
| BX-C02 | Forge Awakening | full-body articulation, sparks, steam | `references/forge-awakening.webp` |
| BX-C03 | Convergence Repair | hands, tool contact, rain, particles | `references/convergence-wars.webp` |
| BX-C04 | Call to the Past | rotating machinery, beam stability | `references/call-to-past-signal-acquired.webp` |
| BX-C05 | Last Citadel Walk | gait, tracking camera, reflections | `references/last-citadel.webp` |

The selected frames span portrait, full-body, close interaction, mechanical effects, and locomotion. They share the same BX-77 design language without repeating one composition.

## Start here

1. Read `STANDARD.md` once.
2. Choose one prompt card from `prompts/`.
3. Use either its I2V lane or T2V lane exactly as written.
4. Record the run with `RUN-RECORD-TEMPLATE.md`.
5. Apply the optional mark only after the timed artifact has been saved; see `BRANDING.md`.

Do not compare an I2V run against a T2V run. Do not silently rewrite prompts for a model. Unsupported negative prompting or audio is recorded as unsupported.

## Version identity

- Suite: `boxwrench-video-comparison-v1`
- Prompt-card version: `1.0.0`
- Narrative source: `Boxwrench Lorebook.md`
- Source images: `public/images/lore/`
- Integrity manifest: `SHA256SUMS`

The Hugging Face workflow guidance influenced the provenance fields: record the exact model repository, revision, filename, and file hash rather than identifying a run only by a friendly model name.
