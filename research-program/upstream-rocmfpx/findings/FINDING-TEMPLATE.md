# FINDING: <one-line title>

> Copy to `findings/YYYY-MM-DD-<slug>.md` and fill in. Delete any field you
> cannot fill **only** after replacing it with `UNKNOWN` — a silently removed
> field reads as "not applicable" when it usually means "not checked".

---

## Summary

**Date:**
**Author:**
**Status:** `DRAFT` | `CONFIRMED` | `RETRACTED`
**Type:** `correctness` | `crash` | `performance` | `build` | `documentation`

One or two sentences: what is wrong or what is available, and for whom.

---

## Environment

```yaml
rocmfpx_repo:       # full remote URL - charlie12345 and ciru-ai both publish a ROCmFPX
rocmfpx_sha:        # full 40-char SHA
rocmfpx_branch:
working_tree_clean: # true/false; if false, list the modifications
hardware:           # e.g. AMD Radeon AI PRO R9700
gfx_arch:           # e.g. gfx1201
gpu_index:          # enumeration index under the backend used
code_object_arch:   # arch VERIFIED emitted by the build
backend:            # vulkan | hip | cpu
rocm:
hip:
mesa_radv:
kernel:
cpu:
build_command:      # exact
```

## Model

```yaml
model_repo:
model_file:
model_sha256:
model_bytes:
model_arch:
quant:
model_provenance:   # upstream-published | locally-converted | unknown
```

If no model is involved (a build or unit-test finding), write `n/a` and say so.

---

## Exact command

```bash
# Verbatim, including every environment variable. Not a paraphrase.
```

---

## Observation

### Expected

What upstream documents, or what the equivalent configuration does elsewhere.
Cite the doc, commit, or comparison run.

### Measured

What actually happened. Numbers with units. For crashes, the signal and the
first frame in project code.

### Raw logs

| Description | Path |
|---|---|
| | |

Raw logs are evidence: preserve them unedited, do not paste only the lines that
support the conclusion.

---

## Scope — what was ruled in and out

The single most common way a finding is wrong is that it is real but not
*specific*. Record what was tested:

| Question | Answer | Evidence |
|---|---|---|
| Reproduces on a second run? | | |
| Specific to this `gfx` arch? | | tested on: |
| Specific to this backend? | | tested on: |
| Specific to this quantization type? | | tested on: |
| Specific to this model? | | tested on: |
| Present on the CPU backend? | | |
| Regression, or never worked? | first bad commit: | |

---

## Interpretation

What you believe is happening, and — separately — what you have actually shown.
Label each claim `SOURCE FACT`, `MEASURED`, `INFERRED`, or `UNKNOWN`.

Do not name a cause you have not isolated. "Unexplained" is a valid conclusion
and a more useful one than a plausible guess.

---

## Confidence

`HIGH` | `MEDIUM` | `LOW` — and the reason. State what would change your mind.

---

## Recommended next action

- [ ] Minimal reproducer written (`../reproducers/`)
- [ ] Upstream issue searched for duplicates
- [ ] Patch drafted (`../patches/`)
- [ ] Correctness invariance demonstrated
- [ ] Component **and** end-to-end results present
- [ ] Ready to file upstream
