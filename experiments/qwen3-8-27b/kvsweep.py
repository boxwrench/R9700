#!/usr/bin/env python3
"""KV-cache precision experiment for Qwen3.8-27B on the Vulkan backend.

Everything except --cache-type-k/--cache-type-v is frozen at the production
configuration. Each configuration gets a fresh llama-server so the Prometheus
counters (/metrics) and the VRAM reading are scoped to that configuration.

Generation is greedy (temperature 0, top-k 1) with ignore_eos so every
configuration emits exactly the same number of tokens and the committed output
can be compared byte-for-byte.

Two runs per KV type:
  mtp  -- speculative decoding on (the production path)
  base -- speculative decoding off (the reference decoder for that KV type)

The correctness gate compares mtp vs base *within* a KV type. Speculative
decoding is supposed to be output-identical to the reference decoder it
verifies against, so a mismatch there is a real bug. Comparisons *across* KV
types are also recorded, but a mismatch there is expected numerics, not a
defect: quantizing the cache changes the attention arithmetic.

Usage: python3 kvsweep.py
"""
import hashlib, json, os, statistics, subprocess, sys, time, urllib.request

BIN = "/ai/github/llama.cpp/build/bin/llama-server"
MODEL = "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MMPROJ = "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf"
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kv_results.jsonl")
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
SAMPLES = 5
MAX_TOKENS = 256
R9700_CARD = "card1"

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."

# Frozen production configuration. Only cache-type-k/v vary.
BASELINE = {
    "ctx-size": 163840,
    "ubatch-size": 512,
    "spec-draft-n-max": 2,
    "spec-draft-p-min": 0.3,
}

KV_TYPES = ["f16", "q8_0", "q4_0"]


def vram_used():
    """Bytes of VRAM in use on the R9700."""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                         capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        parts = line.split(",")
        if parts and parts[0].strip() == R9700_CARD:
            return int(parts[2])
    return None


def launch(kv, spec, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ,
        "--alias", "kv",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on",
        "--cache-type-k", kv, "--cache-type-v", kv,
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0",
        "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", str(BASELINE["ctx-size"]),
        "--ubatch-size", str(BASELINE["ubatch-size"]),
    ]
    if spec:
        args += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(BASELINE["spec-draft-n-max"]),
                 "--spec-draft-p-min", str(BASELINE["spec-draft-p-min"])]
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


def metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {"per_pos": {}}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "per_pos_total{position=" in line:
            pos = int(line.split('position="')[1].split('"')[0])
            out["per_pos"][pos] = float(line.rsplit("}", 1)[1])
            continue
        name, _, val = line.partition(" ")
        out[name.replace("llamacpp:", "")] = float(val)
    return out


def generate(i):
    """Greedy, fixed-length generation. Unique leading text busts the prompt cache."""
    body = {"model": "kv",
            "messages": [{"role": "user", "content": f"Trial {i} marker-{i*7919}. " + PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    return j["timings"], j["choices"][0]["message"].get("content") or ""


def run(kv, spec):
    label = f"{kv}/{'mtp' if spec else 'base'}"
    log_path = os.path.join(SCRATCH, f"kvlog-{kv}-{'mtp' if spec else 'base'}.log")
    proc, err = launch(kv, spec, log_path)
    if err:
        stop(proc)
        rec = {"kv": kv, "spec": spec, "status": "failed", "error": err}
        print(f"FAIL {label}: {err}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    try:
        generate(0)                       # discard: first-call effects
        vram = vram_used()                # measured with the cache allocated and warm
        m0 = metrics()
        dec, pre, ntok, hashes = [], [], [], []
        for i in range(1, SAMPLES + 1):
            t, content = generate(i)
            dec.append(t["predicted_per_second"])
            pre.append(t["prompt_per_second"])
            ntok.append(t["predicted_n"])
            hashes.append(hashlib.sha256(content.encode()).hexdigest()[:16])
        m1 = metrics()
    except Exception as exc:
        stop(proc)
        rec = {"kv": kv, "spec": spec, "status": "error", "error": repr(exc)}
        print(f"ERROR {label}: {exc!r}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    stop(proc)

    drafted = m1.get("spec_decode_num_draft_tokens_total", 0) - m0.get("spec_decode_num_draft_tokens_total", 0)
    accepted = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
    steps = m1.get("spec_decode_num_drafts_total", 0) - m0.get("spec_decode_num_drafts_total", 0)
    per_pos = {p: m1["per_pos"].get(p, 0) - m0["per_pos"].get(p, 0) for p in m1["per_pos"]}

    rec = {
        "kv": kv, "spec": spec, "status": "ok",
        "decode_mean": statistics.mean(dec), "decode_sd": statistics.pstdev(dec),
        "decode_sem": statistics.pstdev(dec) / len(dec) ** 0.5,
        "prefill_mean": statistics.mean(pre), "prefill_sd": statistics.pstdev(pre),
        "vram_bytes": vram,
        "predicted_n": sorted(set(ntok)),
        "drafted": drafted, "accepted": accepted, "verif_steps": steps,
        "acceptance": (accepted / drafted) if drafted else None,
        "mean_accepted_len": (1.0 + accepted / steps) if steps else None,
        "positional": {str(p): (per_pos[p] / steps if steps else None) for p in sorted(per_pos)},
        "hashes": hashes,
        "samples": len(dec),
    }
    acc = f"{rec['acceptance']:.3f}" if rec["acceptance"] is not None else "n/a"
    print(f"OK {label}: decode={rec['decode_mean']:.2f}+-{rec['decode_sem']:.2f} "
          f"prefill={rec['prefill_mean']:.1f} vram={vram/2**30:.2f}GiB acc={acc}", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    todo = sys.argv[1:] or KV_TYPES
    for kv in todo:
        for spec in (True, False):
            run(kv, spec)
