# Local Model Inventory — Track B candidates

**Date:** 2026-08-17
**Scope:** inventory. Written as inventory-only; partly superseded once the
download was authorized.

> [!NOTE]
> **SUPERSEDED IN PART.** This document was written while the NVFP4 model was
> absent and B1 was blocked on it. The user subsequently authorized the
> download. The section below records the outcome; the inventory of other local
> models further down is unchanged and still accurate.

---

## ~~Result: no native NVFP4 Qwen3.8-27B model is present locally~~ — RESOLVED

Originally: `RadixArk/Qwen3.8-27B-NVFP4` — the only real model named in
upstream's NVFP4 commits (`7b02624`, `5290625`) — was not on this machine, and
acquiring it was an explicit user decision, so it was not started.

**It has since been acquired, converted, and quantized.** Identity and
provenance evidence: [`2026-08-17-b1-gates.md`](2026-08-17-b1-gates.md) (Gate 2).
Results: [`2026-08-17-b1-results.md`](2026-08-17-b1-results.md).

| Artifact | Bytes | SHA256 (prefix) |
|---|---:|---|
| `/ai/models/Qwen3.8-27B-NVFP4/` (HF safetensors) | ~21 GB | — |
| `…-GGUF/Qwen3.8-27B-NVFP4.gguf` (mixed) | 28,230,539,776 | `cfcff7f6e965a207…` |
| `…-GGUF/Qwen3.8-27B-NVFP4-uniform.gguf` | 15,547,030,016 | `f529c734266ff579…` |

Conversion required a new CPU-only virtualenv at `/ai/environments/gguf-convert`
(torch 2.13.0+cpu, transformers 5.15.0). The three pre-existing
`/ai/environments/*` venvs all fail with `ImportError: libroctx64.so.4` — their
ROCm-torch builds target an older ROCm than the installed 7.2.1. **None of them
was modified**; a fresh environment was created alongside them.

---

## Other Qwen3.8-27B models present

| Path | Bytes | Quant | Relevance |
|---|---:|---|---|
| `/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,923,394,624 | UD-Q4_K_XL | **Track A foundation.** The B2 matched-comparison baseline. |
| `/ai/models/Qwen3.8-27B-UD-Q6_K_XL/Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25,924,152,384 | UD-Q6_K_XL | Track A Q6 precision arm. |
| `/ai/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge/…PromptForge.gguf` | 15,982,093,472 | unknown | See below. |

SHA256 was **not** computed — these are 16–26 GB files and no run needs the
hashes yet. B0 must hash whichever model it actually uses
([`../scripts/02_hash_model.sh`](../scripts/02_hash_model.sh)).

`/ai/models/Qwen3.8-27B-UD-Q6_K_XL/mmproj-F16.gguf` is **49 bytes** — a
truncated or placeholder file. Noted in passing; it is not a Track B concern,
but it would fail confusingly if something tried to load it.

---

## The ActiveFPX model — a candidate, with a caveat

`/ai/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge/` (from `jcbtc/…` on
HuggingFace) is a 16 GB GGUF plus two `.pfs` sidecars (FFN 17.1 GB, GDN 4.0 GB).

It is **"ActiveFPX", not NVFP4**, and its name matches the *other* ROCmFPX
remote: the local `/ai/github/ROCmFPX` checkout points at `ciru-ai/ROCmFPX` and
sits at `a71e6c8 feat(hip): add ActiveFPX PromptForge routes for Qwen3.8-27B`.

**INFERRED:** this model belongs to the `ciru-ai` line, and pairs with that
checkout rather than with the `charlie12345` NVFP4 work this track audited. Its
quantization type has not been read from its GGUF metadata.

**It is not a substitute for a native NVFP4 model**, and using it would not be
the B1 reproduction the plan calls for. It becomes relevant only if the user
confirms the two remotes are the same project — which is the open question in
[the audit](../upstream-audit/2026-08-17-upstream-audit.md) §0.

---

## Other NVFP4 artifacts found (not Track B candidates)

* `/ai/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` —
  a text encoder, wrong model and wrong format.
* `mmq-instance-nvfp4.cu` under `/ai/github/llama.cpp` and
  `/ai/github/atomic-llama-cpp-turboquant` — source, not models. NVFP4 support
  exists in other local llama.cpp trees too.
* `/ai/deepseek-v4-hermes/backends/*/build-hip-gfx1201/…nvfp4.cu.o` — build
  objects from an unrelated campaign. Worth noting only because they confirm
  **gfx1201 HIP builds are routine on this host.**

---

## Consequence for B1

B1 cannot proceed until a reproduction model exists. In the plan's priority order:

1. The model behind the reported R9700 result — **provenance unknown**; nothing
   in the audited tree substantiates that figure.
2. `RadixArk/Qwen3.8-27B-NVFP4` — **not present**; requires a large download the
   user must authorize.
3. A local NVFP4 conversion from an existing checkpoint via
   `llama-quantize --pure --token-embedding-type q5_K in.gguf out.gguf NVFP4`.
   Possible today with the `build-vulkan` binaries already compiled, but it
   produces a **different checkpoint from upstream's** and must be labelled
   `model_provenance: locally-converted` everywhere it appears.

This is the decision point where Track B stops and waits.
