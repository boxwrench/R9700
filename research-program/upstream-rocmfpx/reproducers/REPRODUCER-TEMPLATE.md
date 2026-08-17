# REPRODUCER: <one-line title>

> Copy to `reproducers/YYYY-MM-DD-<slug>.md`. A reproducer is judged by whether
> a maintainer who has never seen this hardware can run it and see the same
> thing. Optimize for that, not for completeness.

---

## Problem

One paragraph. What goes wrong, on what, and why it matters.

---

## Minimal environment

State the **smallest** configuration that still reproduces. If you have not
tried to reduce it, say so — an unreduced reproducer is still useful, but it
should not claim to be minimal.

```yaml
hardware:           # and whether other GPUs must be present or absent
gfx_arch:
backend:
rocm:
mesa_radv:
kernel:
model_required:     # yes/no - a reproducer needing no model download is far more useful
model_file:
model_sha256:
```

**Reduction attempted:**

- [ ] Smaller model, or no model at all
- [ ] Single GPU visible
- [ ] Fewer layers / shorter context
- [ ] Unit test instead of full inference
- [ ] Other backend, to show the contrast

---

## Exact commit

```
repo:   # full remote URL
sha:    # full 40-char SHA
branch:
```

Confirm the working tree is clean, or list every local modification. A
reproducer against a dirty tree is not a reproducer.

---

## Exact command

```bash
# Build
# Run - verbatim, including every environment variable
```

---

## Expected behavior

What should happen, and why you believe that. Where possible, show the
contrasting case that *does* work (another arch, another backend, another type)
with its own command and output.

## Observed behavior

What happens instead.

```
# verbatim output, trimmed only for length, with trimming marked
```

For a crash, include the signal and a backtrace with at least the first frame
inside project code.

---

## Reproduction rate

```
attempts:
reproduced:
rate:
```

An intermittent failure must be labelled intermittent. Report the rate even
when it is 100%.

---

## Regression range

```
last_known_good:    # SHA, or UNKNOWN
first_known_bad:    # SHA, or UNKNOWN
bisected:           # yes/no
```

If not bisected, say so rather than implying a range you did not establish.

---

## Raw output

| Description | Path |
|---|---|
| | |

---

## Notes for maintainers

Anything that would save the reader time: known-adjacent issues, why an obvious
workaround does not work, or which part you are least sure about.
