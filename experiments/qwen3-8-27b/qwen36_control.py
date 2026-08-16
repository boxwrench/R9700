#!/usr/bin/env python3
"""Qwen3.6 vs Qwen3.8 native MTP control.

Runs the established fixed-prompt harness against a model with MTP off and on,
so the two releases can be compared on proposer quality rather than raw speed.

Matched to the Qwen3.8 f16 reference arm: greedy, ignore_eos, exactly 256
tokens, five measured repetitions after a discarded warmup, fresh server per
arm, cache-busted prompts.

Env:
  EQ_MODEL  path to the .gguf            (default: Qwen3.6 MTP UD-Q4_K_XL)
  EQ_TAG    label used in output paths   (default: qwen36)
  EQ_NMAX   MTP draft depth              (default: 2; phase 3 sweeps this)
"""
import hashlib, json, os, statistics, subprocess, sys, time, urllib.request

BIN = "/ai/github/llama.cpp/build/bin/llama-server"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.6-27B-MTP-UD-Q4_K_XL/Qwen3.6-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.path.dirname(MODEL) + "/mmproj-F16.gguf"
TAG = os.environ.get("EQ_TAG", "qwen36")
NMAX = os.environ.get("EQ_NMAX", "2")
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"qwen36_control_{TAG}.jsonl")
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
SAMPLES = 5
MAX_TOKENS = 256
R9700_CARD = "card1"

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
WARM = "Warmup unrelated prompt alpha-31337. Reply with a single short sentence."


def vram_used():
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                         capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        p = line.split(",")
        if p and p[0].strip() == R9700_CARD:
            return int(p[2])
    return None


def launch(spec, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ, "--alias", "ctl",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "163840", "--ubatch-size", "512",
    ]
    if spec:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", NMAX,
                 "--spec-draft-p-min", "0.3"]
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
        try:
            out[name.replace("llamacpp:", "")] = float(val)
        except ValueError:
            pass
    return out


def generate(i, text=None):
    body = {"model": "ctl",
            "messages": [{"role": "user",
                          "content": (text if text else f"Trial {i} marker-{i*7919}. " + PROMPT)}],
            "max_tokens": MAX_TOKENS if text is None else 16,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "ignore_eos": True, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    ch = j["choices"][0]
    return j["timings"], (ch["message"].get("content") or "")


def run(spec):
    label = f"{TAG}-{'mtp' if spec else 'base'}-n{NMAX if spec else 0}"
    log_path = os.path.join(SCRATCH, f"ctl-{label}.log")
    proc, err = launch(spec, log_path)
    if err:
        stop(proc)
        rec = {"model": MODEL, "tag": TAG, "spec": spec, "n_max": NMAX,
               "status": "failed", "error": err}
        print(f"FAIL {label}: {err}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    try:
        generate(0, WARM)                 # unrelated warmup
        vram = vram_used()
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
        rec = {"model": MODEL, "tag": TAG, "spec": spec, "status": "error", "error": repr(exc)}
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
        "model": MODEL, "tag": TAG, "spec": spec, "n_max": int(NMAX) if spec else 0,
        "status": "ok",
        "decode_mean": statistics.mean(dec), "decode_sd": statistics.pstdev(dec),
        "decode_sem": statistics.pstdev(dec) / len(dec) ** 0.5,
        "prefill_mean": statistics.mean(pre), "prefill_sd": statistics.pstdev(pre),
        "vram_bytes": vram, "predicted_n": sorted(set(ntok)),
        "drafted": drafted, "accepted": accepted, "verif_steps": steps,
        "acceptance": (accepted / drafted) if drafted else None,
        "mean_accepted_len": (1.0 + accepted / steps) if steps else None,
        # positional survival; these sum to expected accepted drafts per round
        "positional": {str(p): (per_pos[p] / steps if steps else None) for p in sorted(per_pos)},
        "expected_drafts_per_round": (accepted / steps) if steps else None,
        "samples": len(dec), "hashes": hashes,
    }
    acc = f"{rec['acceptance']:.4f}" if rec["acceptance"] is not None else "n/a"
    print(f"OK {label}: decode={rec['decode_mean']:.2f}+-{rec['decode_sem']:.2f} "
          f"prefill={rec['prefill_mean']:.1f} vram={vram/2**30:.2f}GiB acc={acc}", flush=True)
    if rec["positional"]:
        print(f"   positional: {  {k: round(v,4) for k,v in rec['positional'].items() if v is not None} }", flush=True)
        print(f"   expected accepted drafts/round: {rec['expected_drafts_per_round']:.4f}", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    arms = sys.argv[1:] or ["base", "mtp"]
    for a in arms:
        run(a == "mtp")
