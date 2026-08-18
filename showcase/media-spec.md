# Showcase media slot specification

Authoritative asset requirements for the R9700 GitHub Pages showcase
(`docs/index.html`). The page layout is locked against these slots; media
generated later must fit them without changing the page.

- Page: <https://boxwrench.github.io/R9700/>
- Design language: the Boxwrench design system (paper ground, ink type, brass
  rules, zero border radius, 2px ink edges). Art is the only place colour and
  atmosphere enter the page, so every slot below carries real visual weight.
- **Nothing in this file has been generated yet.** The slots currently hold the
  existing session artifacts, the BX-77 anchor, or a labelled placeholder.

## Conventions

| Term | Meaning |
|---|---|
| Display slot | The CSS container on the page. Fixed. Media is `object-fit: cover`ed into it. |
| Source dimensions | What the generator should actually produce. |
| Mobile crop | The aspect ratio the same asset is cropped to below 760px. Keep the subject inside this crop. |
| Safe area | The region that survives both the desktop and the mobile crop. |

All video slots share one frame geometry so the session reads as a single
sequence. Do not vary aspect ratio between runs.

Poster frames are extracted from the video itself (`ffmpeg -vf "select=eq(n\,20)"`)
and stored in `showcase/posters/` and `docs/assets/showcase/`. Every video slot
needs a poster; the page renders the poster before playback and it is what most
readers will actually see.

---

## HERO — Plate 01

- **Role:** the page's primary art moment. Establishes BX-77 and the workshop /
  forge register before any numbers appear. Sits directly under the H1.
- **Display slot:** 16:7 (≈2.286:1) desktop, full canvas width (max 1180px).
- **Source dimensions:** 2560 × 1120 (16:7). Minimum acceptable 1680 × 735.
- **Mobile crop:** 3:2. Centre-weighted; the current page uses
  `object-position: 50% 38%`, so keep the head and chest core in the upper-middle third.
- **Safe area:** central 3:2 region must be a complete composition on its own.
- **Format:** WebP (lossy, q≈82) or JPEG. Target under 400 KB.
- **Visual intent:** BX-77 at rest or mid-work in the artifact lab — dark
  technical editorial, machine/forge/artifact-lab, warm forge light against cold
  diagnostic light, gunmetal and brass. Cinematic, high detail, one subject.
  Wide negative space on one side is welcome; the heading sits above, not over.
- **Hard constraint:** no text, no UI, no numbers, no fabricated instrumentation
  readouts inside the image. All real measurements live in page chrome.
- **Currently occupied by:** `docs/assets/showcase/hero-bx77-anchor-placeholder.webp`
  — a copy of the Boxwrench anchor `bx77-anchor.webp` (1672 × 941, 16:9),
  cropped by the slot. Labelled "Plate 01 — placeholder" on the page.

---

## RUN 01 — LTX 2.5 · "The Artifact Forge"

- **Display slot:** 16:9, full canvas width.
- **Target display dimensions:** 1280 × 720 presentation frame.
- **Source dimensions:** 16:9 native preferred. The current production LTX
  config renders 768 × 448 (1.714:1), which is *not* 16:9 and is cropped ~2% by
  the slot. If the workflow is re-parameterised, prefer 1280 × 720 or 960 × 540;
  otherwise 768 × 432 removes the crop.
- **Duration target:** 2–4 s (current: 41 frames @ 24 fps ≈ 1.71 s). Longer
  reads better in a scroll sequence; 3 s is the sweet spot.
- **Audio:** none. The audio branch is dropped for LTX — see the field notes
  disclosure on the page.
- **Poster:** frame ~20, extracted from the final video.
- **Visual intent:** the forge. Sparks, heat, working metal. The loudest of the
  six video slots.
- **Currently occupied by:** `01-ltx-boxwrench-artifact-forge_00001_.mp4`
  (release `showcase-final-20260818`), poster
  `docs/assets/showcase/01-ltx-boxwrench-artifact-forge.jpg` (768 × 448).

---

## RUN 02 — MiniMax H3 · Text to Video · "Machine Cathedral"

- **Display slot:** 16:9, full canvas width.
- **Target display dimensions:** 1280 × 720.
- **Source dimensions:** current production H3 config renders 608 × 352
  (1.727:1). Prefer 640 × 360 to land exactly on 16:9 at the same cost class.
- **Duration target:** 2–4 s (current: 39 frames @ 24 fps ≈ 1.63 s).
- **Audio:** H3 emits an audio track. Keep it; the player exposes it.
- **Poster:** frame ~20.
- **Visual intent:** architectural scale — the cathedral of machines. Wide,
  static or slow camera. Deliberately the least kinetic slot on the page.

---

## RUN 03 — MiniMax H3 · Image to Video · "Forge Awakening"

- **Display slot:** 16:9, full canvas width.
- **Target display dimensions:** 1280 × 720.
- **Source dimensions:** as run 02. Prefer 640 × 360.
- **Duration target:** 2–4 s (current: 39 frames @ 24 fps ≈ 1.63 s).
- **Conditioning:** `first_frame` = the BX-77 anchor. The first frame of the
  video *is* the anchor, so the poster should be taken later in the clip
  (≥ frame 20) to show that motion actually happened.
- **Audio:** present.
- **Visual intent:** the anchor coming to life — the same subject as the hero,
  now moving. Continuity with the hero plate matters more here than novelty.

---

## RUN 04 — MiniMax Music 3 · "Signal Against the End"

This run gets a distinct treatment: a dark record-sleeve block, not a video card.

### Cover plate (sleeve)

- **Display slot:** 2.4:1 desktop, 3:2 mobile. Inset inside the dark block, on a
  near-black ground with a `--chrome` 2px edge.
- **Source dimensions:** 2400 × 1000 (2.4:1). Minimum 1680 × 700.
- **Mobile crop:** 3:2, centre.
- **Format:** WebP or JPEG, under 350 KB.
- **Visual intent:** a record sleeve, not a visualisation. Heavy-metal artifact
  register: the transmitter, the last star, cold logic against entropy. Darker
  and more graphic than the video slots — it is allowed to be the second loud
  moment on the page.
- **Hard constraint:** no rendered song title, no band name, no text of any kind
  inside the image. The title is set in Share Tech Mono above it.
- **Currently occupied by:** labelled placeholder, no image.

### Player

- **Audio:** FLAC, served from the GitHub release. Rendered with a native
  `<audio controls preload="none">`.
- **Duration target:** 15 s is the current production ceiling
  (`max_duration: 15.0`); the delivered track is 14.99 s. If a longer piece is
  ever wanted, the layout tolerates it — the slot does not change.
- **Caption slot:** directly under the player, one line, `--ink-muted`/`--chrome`.

### Waveform

- **Display slot:** full width of the dark block, natural height.
- **Source dimensions:** 1200 × 300 (4:1), rendered from the delivered FLAC
  (`ffmpeg -filter_complex showwavespic`).
- **Presentation:** the page inverts the image in CSS so it reads as light-on-dark.
  Render it as dark-on-light, exactly as the current asset is.
- **Currently occupied by:**
  `docs/assets/showcase/04-music3-signal-against-the-end-waveform.png` (1200 × 300).

### Spectrum

- Behind the "Technical detail" disclosure, not in the main visual flow.
- **Source dimensions:** 1200 × 400 (3:1), `showspectrum`.
- **Currently occupied by:**
  `docs/assets/showcase/04-music3-signal-against-the-end-spectrum.png` (1200 × 400).

### Lyric excerpt

- One line, set in Newsreader italic against a brass left rule.
- Current: *"Carry the signal, deny the end…"*

---

## RUN 05 — MiniMax H3 · Reference to Video · "Observatory"

- **Display slot:** 16:9, full canvas width.
- **Target display dimensions:** 1280 × 720.
- **Source dimensions:** as run 02. Prefer 640 × 360.
- **Duration target:** 2–4 s (current: 39 frames @ 24 fps ≈ 1.63 s).
- **Conditioning:** `ref_images.ref_image_0` = the BX-77 anchor,
  `ref_image_size: match`. Identity preservation is the whole point of this run,
  so the subject must stay recognisably BX-77.
- **Audio:** present.
- **Visual intent:** the observatory — the subject placed somewhere cold, high,
  and wide. Should feel like a different location from runs 01–03 while keeping
  the same character.

---

## RUN 06 — LTX 2.5 · Return Run · "Signal Vault"

- **Display slot:** 16:9, full canvas width.
- **Target display dimensions:** 1280 × 720.
- **Source dimensions:** as run 01. Prefer 768 × 432 or 1280 × 720.
- **Duration target:** 2–4 s (current: 41 frames @ 24 fps ≈ 1.71 s).
- **Audio:** none (audio branch dropped).
- **Known limitation:** LTX 2.5 here is pure text conditioning with no image
  reference, so it will not reproduce the anchor's exact armour geometry or
  palette. Write the prompt for a scene that reads as BX-77's *world* rather
  than demanding a literal identity match — this slot closes the sequence and
  should not look like a failed identity test.
- **Visual intent:** the vault. Enclosed, still, terminal. The quietest of the
  six, deliberately — it is the last frame before the closing plate.

---

## CLOSING ART — Plate 02

- **Role:** the sign-off. Reserved second high-quality plate.
- **Display slot:** 3:1 desktop, 3:2 mobile, full canvas width, with the
  9px offset `--chrome` lift shadow (same treatment as the hero).
- **Source dimensions:** 2160 × 720 (3:1). Minimum 1500 × 500.
- **Mobile crop:** 3:2, centre.
- **Format:** WebP or JPEG, under 400 KB.
- **Visual intent:** a wide, quiet editorial banner. BX-77 departing, powering
  down, or a wide empty shot of the lab with the work finished. Restrained.
  The hero opens loud; this should close soft.
- **Hard constraint:** no text in the image. The "BX-77 // END OF FIELD REPORT"
  line is set in type beneath it.
- **Currently occupied by:** labelled placeholder, no image.

---

## Delivery checklist

For each asset, before it replaces a placeholder:

1. Correct source aspect ratio for its slot (no letterboxing baked in).
2. Subject survives the mobile crop.
3. No text, numbers, UI, or fabricated instrumentation inside the media.
4. Video: poster frame extracted from the delivered file, not from a different take.
5. Video: duration within the slot's target range.
6. Stills: WebP or JPEG, sized as specified, under the stated weight.
7. Large media (video, FLAC) goes to a GitHub release, not into Git history.
   Posters and stills are small enough to commit under `docs/assets/showcase/`.
8. Descriptive `alt` text written for the actual delivered image, replacing the
   placeholder alt on the page.
