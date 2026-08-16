#!/usr/bin/env python3
"""Qwen3.8 MTP round-cost decomposition (steps 2 and 3).

Arms:
  T0  serial decode, n_rs_seq=0 (K=1)          -- reference cost per token
  T1  serial decode, forced n_rs_seq=2 (K=3)   -- isolates snapshot overhead
  T2  ngram-simple depth 2, forced n_rs_seq=2  -- multi-token verification
                                                  without a 425M proposer
  T3  native MTP n-max 2                       -- production path

Every arm runs the same fixed 256-token harness: greedy, ignore_eos, five
repetitions after a discarded unrelated warmup, fresh server, cache_prompt
disabled.

draft-mtp reports cumulative dur(b,g,a) via SPC_TRC, so the statistics line is
snapshotted after the warmup and again after the measured block and differenced.
t_draft_us brackets a llama_synchronize (common_sampler_sample synchronizes
ctx_dft), so dur(g) covers completed draft GPU work. Target verification is NOT
inside any of these timers -- it happens in the server loop -- so it is derived
by subtraction here and measured directly in step 4.
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request

BIN = "/ai/scratch/llamacpp-probe/build/bin/llama-server"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.path.dirname(MODEL) + "/mmproj-F16.gguf"
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "round_cost.jsonl")
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
SAMPLES = 5
MAX_TOKENS = 256

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
WARM = "Warmup unrelated prompt alpha-31337. Reply with a single short sentence."

ARMS = {
    "T0_serial_k1":  {"spec": [], "rs": None},
    "T1_serial_k3":  {"spec": [], "rs": 2},
    "T2_ngram_k3":   {"spec": ["--spec-type", "ngram-simple",
                               "--spec-ngram-simple-size-n", "2",
                               "--spec-ngram-simple-size-m", "2",
                               "--spec-draft-n-max", "2"], "rs": 2},
    "T3_mtp":        {"spec": ["--spec-type", "draft-mtp",
                               "--spec-draft-n-max", "2",
                               "--spec-draft-p-min", "0.3"], "rs": None},
}

STATS = re.compile(
    r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), "
    r"#gen tokens =\s*(\d+), #acc tokens =\s*(\d+)"
    r"(?:.*?dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms)?")


def launch(cfg, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ, "--alias", "rc",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "163840", "--ubatch-size", "512", "-lv", "4",
    ] + cfg["spec"]
    env = dict(os.environ)
    if cfg["rs"] is not None:
        env["LLAMA_FORCE_N_RS_SEQ"] = str(cfg["rs"])
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
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


def generate(i, warm=False):
    body = {"model": "rc",
            "messages": [{"role": "user",
                          "content": WARM if warm else f"Trial {i} marker-{i*7919}. " + PROMPT}],
            "max_tokens": 16 if warm else MAX_TOKENS,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "ignore_eos": True, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["timings"]


def metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.partition(" ")
        if "spec_decode" in name and "per_pos" not in name:
            try:
                out[name.replace("llamacpp:", "")] = float(val)
            except ValueError:
                pass
    return out


def last_stats(path):
    """Most recent cumulative speculative statistics line, if any."""
    found = None
    for line in open(path, errors="replace"):
        m = STATS.search(line)
        if m:
            found = {"type": m.group(1),
                     "n_begin": int(m.group(2)), "n_draft": int(m.group(3)),
                     "n_accept": int(m.group(4)),
                     "gen_drafts": int(m.group(5)), "acc_drafts": int(m.group(6)),
                     "gen_tokens": int(m.group(7)), "acc_tokens": int(m.group(8)),
                     "dur_b": float(m.group(9)) if m.group(9) else None,
                     "dur_g": float(m.group(10)) if m.group(10) else None,
                     "dur_a": float(m.group(11)) if m.group(11) else None}
    return found


def run(name, cfg):
    log_path = os.path.join(SCRATCH, f"rc-{name}.log")
    proc, err = launch(cfg, log_path)
    if err:
        stop(proc)
        print(f"FAIL {name}: {err}", flush=True)
        return {"arm": name, "status": "failed", "error": err}
    try:
        generate(0, warm=True)
        s0 = last_stats(log_path)
        m0 = metrics()
        dec, pre, pred_ms, pred_n, prompt_ms = [], [], [], [], []
        for i in range(1, SAMPLES + 1):
            t = generate(i)
            dec.append(t["predicted_per_second"])
            pre.append(t["prompt_per_second"])
            pred_ms.append(t["predicted_ms"])
            pred_n.append(t["predicted_n"])
            prompt_ms.append(t["prompt_ms"])
        s1 = last_stats(log_path)
        m1 = metrics()
    except Exception as exc:
        stop(proc)
        print(f"ERROR {name}: {exc!r}", flush=True)
        return {"arm": name, "status": "error", "error": repr(exc)}
    stop(proc)

    def d(k):
        return (m1.get(k, 0) - m0.get(k, 0))
    rounds = d("spec_decode_num_drafts_total")
    drafted = d("spec_decode_num_draft_tokens_total")
    accepted = d("spec_decode_num_accepted_tokens_total")

    dur = {}
    if s1 and s0 and s1.get("dur_g") is not None and s0.get("dur_g") is not None:
        for k in ("b", "g", "a"):
            dur[k] = s1[f"dur_{k}"] - s0[f"dur_{k}"]
        for k in ("n_begin", "n_draft", "n_accept"):
            dur[k] = s1[k] - s0[k]
    elif s1 and s1.get("dur_g") is not None:
        dur = {"b": s1["dur_b"], "g": s1["dur_g"], "a": s1["dur_a"],
               "n_begin": s1["n_begin"], "n_draft": s1["n_draft"],
               "n_accept": s1["n_accept"], "note": "no warmup snapshot; cumulative"}

    tot_pred_ms = sum(pred_ms)
    tot_pred_n = sum(pred_n)
    rec = {
        "arm": name, "status": "ok",
        "decode_mean": statistics.mean(dec), "decode_sd": statistics.pstdev(dec),
        "decode_sem": statistics.pstdev(dec) / len(dec) ** 0.5,
        "prefill_mean": statistics.mean(pre),
        "total_predicted_ms": tot_pred_ms, "total_predicted_n": tot_pred_n,
        "ms_per_output_token": tot_pred_ms / tot_pred_n if tot_pred_n else None,
        "rounds": rounds, "drafted": drafted, "accepted": accepted,
        "ms_per_round": (tot_pred_ms / rounds) if rounds else None,
        "output_tokens_per_round": (tot_pred_n / rounds) if rounds else None,
        "dur": dur, "samples": len(dec),
    }
    print(f"OK {name}: decode={rec['decode_mean']:.2f} ms/tok={rec['ms_per_output_token']:.3f} "
          f"rounds={int(rounds)} ms/round={rec['ms_per_round'] if rec['ms_per_round'] is None else round(rec['ms_per_round'],3)}",
          flush=True)
    if dur:
        n = dur.get("n_draft") or 1
        na = dur.get("n_accept") or 1
        print(f"   dur(b,g,a) = {dur['b']:.1f}, {dur['g']:.1f}, {dur['a']:.1f} ms over "
              f"calls {dur.get('n_begin')}, {dur.get('n_draft')}, {dur.get('n_accept')}", flush=True)
        print(f"   draft gen {dur['g']/n:.3f} ms/call | accept {dur['a']/na:.3f} ms/call "
              f"| begin {dur['b']/max(dur.get('n_begin') or 1,1):.3f} ms/call", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    todo = sys.argv[1:] or list(ARMS)
    for name in todo:
        run(name, ARMS[name])
