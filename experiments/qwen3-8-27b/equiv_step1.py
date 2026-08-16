#!/usr/bin/env python3
"""Step 1: three-arm speculative-decoding equivalence control.

Same prompt, same greedy settings, three decode paths:
  A none         -- ordinary autoregressive decode (reference)
  B draft-mtp    -- MTP speculative decoding, n-max 2
  C ngram-simple -- n-gram speculative decoding, draft depth 2

Each arm runs twice from independent server launches to establish
reproducibility before any cross-arm comparison is made.

If A == C but B != A, a generic "n+1 verification batch" explanation is
insufficient and the cause is MTP-specific.
"""
import hashlib, json, os, subprocess, sys, time, urllib.request

BIN = "/ai/github/llama.cpp/build/bin/llama-server"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.path.dirname(MODEL) + "/mmproj-F16.gguf"
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("EQ_OUT", "equiv_step1.jsonl"))
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
MAX_TOKENS = 256

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT_BODY = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
# Trial 3 is the shortest known-divergent prompt: first divergence at token 7.
TRIAL = 3
PROMPT = f"Trial {TRIAL} marker-{TRIAL*7919}. " + PROMPT_BODY

ARMS = {
    "A_none":        [],
    "B_mtp":         ["--spec-type", "draft-mtp",
                      "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.3"],
    "C_ngram":       ["--spec-type", "ngram-simple",
                      "--spec-ngram-simple-size-n", "2",
                      "--spec-ngram-simple-size-m", "2",
                      "--spec-draft-n-max", "2"],
}


def launch(extra, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ, "--alias", "eq",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "163840", "--ubatch-size", "512",
    ] + extra
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if proc.poll() is not None:
            return None, "server exited during load"
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    return proc, None
        except Exception:
            time.sleep(1)
    return proc, "health check timed out"


def stop(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def generate():
    body = {"model": "eq", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": 0, "ignore_eos": True, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    ch = j["choices"][0]
    toks = [t["token"] for t in (ch.get("logprobs", {}).get("content") or [])]
    return ch["message"].get("content") or "", toks


if __name__ == "__main__":
    recs = []
    for arm, extra in ARMS.items():
        for rep in (1, 2):
            proc, err = launch(extra, os.path.join(SCRATCH, f"eq1-{arm}-{rep}.log"))
            if err:
                stop(proc)
                print(f"FAIL {arm} rep{rep}: {err}", flush=True)
                continue
            try:
                generate()                      # discard warmup
                text, toks = generate()
            finally:
                stop(proc)
            rec = {"arm": arm, "rep": rep,
                   "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                   "n_tokens": len(toks), "tokens": toks, "text": text}
            recs.append(rec)
            print(f"{arm} rep{rep}: hash={rec['hash']} ntok={len(toks)}", flush=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print("\n=== reproducibility ===", flush=True)
    by = {}
    for r in recs:
        by.setdefault(r["arm"], []).append(r)
    for arm, rs in by.items():
        ok = len(rs) == 2 and rs[0]["hash"] == rs[1]["hash"]
        print(f"  {arm}: {'reproducible' if ok else 'NOT reproducible'} {[r['hash'] for r in rs]}")

    print("\n=== vs A_none ===", flush=True)
    ref = by["A_none"][0]
    for arm in ("B_mtp", "C_ngram"):
        if arm not in by:
            continue
        cur = by[arm][0]
        a, b = ref["tokens"], cur["tokens"]
        idx = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        same = ref["hash"] == cur["hash"]
        print(f"  {arm}: {'IDENTICAL' if same else 'DIFFERS'} "
              f"first_divergent_token_index={idx} "
              f"common_prefix_tokens={idx if idx is not None else min(len(a), len(b))}")
        if idx is not None:
            print(f"     A={a[idx]!r}  {arm}={b[idx]!r}")
