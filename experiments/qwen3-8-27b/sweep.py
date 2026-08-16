#!/usr/bin/env python3
"""Staged parameter sweep for Qwen3.8-27B on the Vulkan backend.

Each configuration gets a fresh llama-server so the Prometheus counters
(/metrics) are scoped to that configuration. Results are appended to a JSONL
file so a crashed stage can be resumed without rerunning earlier work.

Usage: python3 sweep.py <stage-name> <json-list-of-config-dicts>
"""
import json, os, statistics, subprocess, sys, time, urllib.request

BIN = "/ai/github/llama.cpp/build/bin/llama-server"
MODEL = "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MMPROJ = "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf"
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.jsonl")
SAMPLES = 10

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."

# Frozen baseline. Sweep entries override individual keys.
BASELINE = {
    "ctx-size": 163840,
    "ubatch-size": 512,
    "spec-draft-n-max": 2,
    "spec-draft-p-min": 0.3,
}


def launch(cfg, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ,
        "--alias", "sweep",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--spec-type", "draft-mtp",
        "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", str(cfg["ctx-size"]),
        "--ubatch-size", str(cfg["ubatch-size"]),
        "--spec-draft-n-max", str(cfg["spec-draft-n-max"]),
        "--spec-draft-p-min", str(cfg["spec-draft-p-min"]),
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
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
        proc.wait(timeout=60)
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
    body = {"model": "sweep",
            "messages": [{"role": "user", "content": f"Trial {i} marker-{i*7919}. " + PROMPT}],
            "max_tokens": 256, "temperature": 0.6, "seed": 100 + i,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["timings"]


def run(stage, cfg):
    full = dict(BASELINE)
    full.update(cfg)
    tag = f"{stage} ctx={full['ctx-size']} ub={full['ubatch-size']} nmax={full['spec-draft-n-max']} pmin={full['spec-draft-p-min']}"
    log_path = f"/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad/sweeplog-{stage}.log"
    proc, err = launch(full, log_path)
    if err:
        stop(proc)
        rec = {"stage": stage, "config": full, "status": "failed", "error": err}
        print(f"FAIL {tag}: {err}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    try:
        generate(0)               # discard: first-call effects
        m0 = metrics()
        dec, pre = [], []
        for i in range(1, SAMPLES + 1):
            t = generate(i)
            dec.append(t["predicted_per_second"])
            pre.append(t["prompt_per_second"])
        m1 = metrics()
    except Exception as exc:
        stop(proc)
        rec = {"stage": stage, "config": full, "status": "error", "error": repr(exc)}
        print(f"ERROR {tag}: {exc!r}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    stop(proc)

    drafted = m1.get("spec_decode_num_draft_tokens_total", 0) - m0.get("spec_decode_num_draft_tokens_total", 0)
    accepted = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
    steps = m1.get("spec_decode_num_drafts_total", 0) - m0.get("spec_decode_num_drafts_total", 0)
    per_pos = {p: m1["per_pos"].get(p, 0) - m0["per_pos"].get(p, 0) for p in m1["per_pos"]}

    rec = {
        "stage": stage, "config": full, "status": "ok",
        "decode_mean": statistics.mean(dec), "decode_sd": statistics.pstdev(dec),
        "decode_sem": statistics.pstdev(dec) / len(dec) ** 0.5,
        "prefill_mean": statistics.mean(pre), "prefill_sd": statistics.pstdev(pre),
        "drafted": drafted, "accepted": accepted, "verif_steps": steps,
        "acceptance": (accepted / drafted) if drafted else None,
        "mean_accepted_len": (1.0 + accepted / steps) if steps else None,
        "positional": {str(p): (per_pos[p] / steps if steps else None) for p in sorted(per_pos)},
        "samples": len(dec),
    }
    print(f"OK {tag}: decode={rec['decode_mean']:.2f}+-{rec['decode_sem']:.2f} "
          f"acc={rec['acceptance']:.3f} mal={rec['mean_accepted_len']:.3f}", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    stage = sys.argv[1]
    configs = json.loads(sys.argv[2])
    for cfg in configs:
        run(stage, cfg)
