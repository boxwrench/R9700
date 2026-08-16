#!/usr/bin/env python3
"""Prompt-length-bucketed prefill/decode/accept-rate benchmark for the llama-router.

Usage: MODEL=<model-id> python3 bench.py
"""
import base64, json, os, statistics, sys, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = open("/ai/pi/config/llama-api-key").read().strip()
MODEL = os.environ["MODEL"]
SCRATCH = os.path.dirname(os.path.abspath(__file__))

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")


def post(body, timeout=600):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(prompt_text, max_tokens=256, image=None, seed=42):
    if image:
        content = [{"type": "text", "text": prompt_text},
                   {"type": "image_url", "image_url": {"url": image}}]
    else:
        content = prompt_text
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.6, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = post(body)
    t = r.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"),
        "prefill_tps": t.get("prompt_per_second"),
        "decode_tps": t.get("predicted_per_second"),
        "predicted_n": t.get("predicted_n"),
        "draft_n": t.get("draft_n"),
        "draft_accepted_n": t.get("draft_n_accepted"),
    }


def pad(n_words):
    return (FILLER * (n_words // 20 + 1))[:n_words * 6]


BUCKETS = [
    ("short (~150 tok)", pad(110) + "\n\nSummarize the passage above, then explain speculative decoding."),
    ("medium (~1.5K tok)", pad(1100) + "\n\nSummarize the passage above, then explain speculative decoding."),
    ("long (~8K tok)", pad(6000) + "\n\nSummarize the passage above, then explain speculative decoding."),
    ("very-long (~33K tok)", pad(25000) + "\n\nSummarize the passage above, then explain speculative decoding."),
]

results = {}
print(f"=== model: {MODEL} ===", flush=True)
NONCE = os.environ.get("NONCE", "0")
for name, prompt in BUCKETS:
    # Leading nonce so a re-run doesn't reuse the previous run's cached KV.
    r = run(f"Session {NONCE} bucket {name}. " + prompt)
    results[name] = r
    acc = (r["draft_accepted_n"] / r["draft_n"] * 100) if r.get("draft_n") else float("nan")
    print(f"{name:22s} prompt_n={r['prompt_n']:>6} prefill={r['prefill_tps']:8.1f} "
          f"decode={r['decode_tps']:6.2f} accept={acc:5.1f}%", flush=True)

# vision
img = base64.b64encode(open(os.path.join(SCRATCH, "vtest.png"), "rb").read()).decode()
r = run("Describe this image in detail.", image=f"data:image/png;base64,{img}")
results["vision"] = r
print(f"{'vision':22s} prompt_n={r['prompt_n']:>6} decode={r['decode_tps']:6.2f}", flush=True)

# 10-sample mean on the medium bucket
# Each sample gets a unique *leading* sentence so the prompt cache misses and
# prefill is actually measured. Varying only the seed keeps the prompt identical,
# which makes llama.cpp reuse the cached KV and report a meaningless ~34 tok/s
# prefill rate computed over the 1-2 genuinely new tokens.
print("\n--- 10-sample mean (medium bucket, cache-busted) ---", flush=True)
pre, dec, dn, da = [], [], 0, 0
for i in range(10):
    uniq = f"Run identifier {i} zeta-{i*7919}. "
    r = run(uniq + BUCKETS[1][1], seed=100 + i)
    pre.append(r["prefill_tps"]); dec.append(r["decode_tps"])
    if r.get("draft_n"):
        dn += r["draft_n"]; da += r["draft_accepted_n"]
    print(f"  [{i+1:2d}] prefill={r['prefill_tps']:8.1f}  decode={r['decode_tps']:6.2f}", flush=True)

print(f"\nprefill mean={statistics.mean(pre):.2f} sd={statistics.pstdev(pre):.2f}")
print(f"decode  mean={statistics.mean(dec):.2f} sd={statistics.pstdev(dec):.2f}")
if dn:
    print(f"overall accept rate={da/dn:.3f}  ({da}/{dn})")
