# BX-C05 — Last Citadel Walk

Primary test: locomotion, mass, backward tracking, occlusion, rain, and floor reflections.

Reference: `../references/last-citadel.webp`

## I2V positive prompt

```text
Boxwrench walks toward the camera through the abandoned Aegis Citadel with three slow, heavy, evenly timed steps. Each armored foot plants fully before the next lifts, producing a small splash and a delayed ripple across the wet floor. His shoulders counter-swing subtly while the helmet and single crimson visor remain level and stable. The camera tracks backward at matching speed along a straight path; pillars create clean parallax and never bend. No camera cut. Audio: three weighty metal footfalls, shallow water splashes, servo movement, distant rain, and a cavernous mechanical echo.
```

## T2V positive prompt

```text
Boxwrench (BX-77), a seven-foot sentient heavy armored robot with a mirror-polished chrome-titanium chassis, deep brass command plating, heavy gold-headed rivets, a brass-framed hexagonal chest reactor, a high armored collar, layered shoulder pauldrons, a raised gold helmet crest, and one dark horizontal visor containing a single crimson scanning light, walks through the abandoned Aegis Citadel. Cinematic 1990s science-fiction cel-animation language rendered with detailed industrial materials, hard inked edges, restrained black, chrome, brass, and one crimson accent. He takes three slow, heavy, evenly timed steps toward the camera. Each foot plants fully before the next lifts, producing a small splash and delayed ripple across the wet floor. His shoulders counter-swing subtly while the helmet and visor remain level. The camera tracks backward at matching speed on a straight path; tall pillars create clean parallax. No camera cut. Audio: three weighty metal footfalls, shallow water splashes, servo movement, distant rain, and a cavernous mechanical echo.
```

Use the common negative prompt from `STANDARD.md`.

Watch for: skating feet, extra or mistimed steps, leg intersections, changing proportions, bent architecture, reflection mismatch, or camera speed drift.
