#!/usr/bin/env python3
"""n_rs_seq sufficiency test.

A  baseline      : no speculation, n_rs_seq = 0        -> K = 1
B  forced snap   : no speculation, n_rs_seq = 2 forced -> K = 3, no drafter
C  normal MTP    : draft-mtp n-max 2, n_rs_seq = 2     -> K = 3, with drafting

Decides whether K=3 snapshot execution alone reproduces the MTP output shift,
or whether actual speculative snapshot/rollback traffic is also required.

Fresh server per arm. Warmup uses an unrelated prompt and --no-context-shift
plus cache_prompt=false so the measured generation never reuses cached KV.
"""
import hashlib, json, os, re, subprocess, sys, time, urllib.request

BIN = "/ai/scratch/llamacpp-probe/build/bin/llama-server"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.path.dirname(MODEL) + "/mmproj-F16.gguf"
TAG = os.environ.get("EQ_TAG", "q4")
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
MAX_TOKENS = 256
FULL_VOCAB = 248320          # target-model candidate count; distinguishes target from draft samples

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT_BODY = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
TRIAL = 3
PROMPT = f"Trial {TRIAL} marker-{TRIAL*7919}. " + PROMPT_BODY
WARM = "Warmup unrelated prompt alpha-31337. Reply with a single short sentence."

ARMS = {
    "A_base":   {"extra": [], "force_rs": None},
    "B_forced": {"extra": [], "force_rs": 2},
    "C_mtp":    {"extra": ["--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.3"],
                 "force_rs": None},
}
REF = "A_base"


def launch(cfg, log_path):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ, "--alias", "eq",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--jinja", "--reasoning-preserve", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "163840", "--ubatch-size", "512",
    ] + cfg["extra"]
    env = dict(os.environ, LLAMA_PROBE_GDN="1", LLAMA_PROBE_LOGITS="1")
    if cfg["force_rs"] is not None:
        env["LLAMA_FORCE_N_RS_SEQ"] = str(cfg["force_rs"])
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


def generate(text, max_tokens=MAX_TOKENS):
    body = {"model": "eq", "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": 0, "ignore_eos": True, "logprobs": True, "top_logprobs": 5,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    ch = j["choices"][0]
    return (ch["message"].get("content") or "",
            [t["token"] for t in (ch.get("logprobs", {}).get("content") or [])])


def metrics():
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.partition(" ")
        if "spec_decode" in name:
            try:
                out[name.replace("llamacpp:", "")] = float(val)
            except ValueError:
                pass
    return out


LP = re.compile(r"\[LOGIT\] step=(\d+) sampler_idx=(-?\d+) n_cand=(\d+)(.*)")
SNAP = re.compile(r"\[GDNSNAP\] il=\s*(\d+) n_rs_seq=(\d+) K=(\d+) n_seq_tokens=(\d+) n_seqs=(\d+) "
                  r"branch=(\w+) state_ne=\[([^\]]+)\] D=(\d+) n_embd_s=(\d+) "
                  r"ssm_states_all_ne=\[([^\]]+)\]")
GDN = re.compile(r"\[GDN\] il=\s*(\d+) ubatch_n_tokens=(\d+).*branch=(\w+) path=(\w+) state_ne=\[([^\]]+)\]")


def target_samples(path):
    """Sampler calls against the full target vocabulary, in committed order."""
    out = []
    for line in open(path, errors="replace"):
        m = LP.search(line)
        if not m or int(m.group(3)) != FULL_VOCAB:
            continue
        body, _, marg = m.group(4).partition("||")
        top = []
        for part in body.split("|"):
            part = part.strip()
            if ":" in part:
                t, _, l = part.partition(":")
                try:
                    top.append((int(t), float(l)))
                except ValueError:
                    pass
        out.append({"gstep": int(m.group(1)), "idx": int(m.group(2)), "top": top,
                    "margin": float(marg.split("=")[1]) if "=" in marg else None})
    return out


def scan(path):
    snaps, gdns, draft_samples = {}, {}, 0
    forced = None
    for line in open(path, errors="replace"):
        if "[PROBE] forcing" in line:
            forced = line.strip()
        m = SNAP.search(line)
        if m:
            snaps.setdefault((m.group(2), m.group(3), m.group(4), m.group(6),
                              m.group(7), m.group(9), m.group(10)), 0)
            snaps[(m.group(2), m.group(3), m.group(4), m.group(6),
                   m.group(7), m.group(9), m.group(10))] += 1
            continue
        m = GDN.search(line)
        if m:
            key = (m.group(2), m.group(3), m.group(4), m.group(5))
            gdns[key] = gdns.get(key, 0) + 1
            continue
        m = LP.search(line)
        if m and int(m.group(3)) != FULL_VOCAB:
            draft_samples += 1
    return {"snap": snaps, "gdn": gdns, "draft_samples": draft_samples, "forced": forced}


def run(arm, cfg):
    log_path = os.path.join(SCRATCH, f"rs-{TAG}-{arm}.log")
    proc, err = launch(cfg, log_path)
    if err:
        stop(proc)
        raise SystemExit(f"FAIL {arm}: {err}")
    try:
        generate(WARM, 16)
        n_warm = len(target_samples(log_path))
        text, toks = generate(PROMPT)
        m = metrics()
    finally:
        stop(proc)
    info = scan(log_path)
    return {"arm": arm, "text": text, "tokens": toks,
            "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "n_warm_target_samples": n_warm,
            "targets": target_samples(log_path), "metrics": m,
            "log": log_path, **info}


if __name__ == "__main__":
    res = {}
    for arm, cfg in ARMS.items():
        r = run(arm, cfg)
        res[arm] = r
        print(f"\n=== {arm}: hash={r['hash']} ntok={len(r['tokens'])} "
              f"target_samples={len(r['targets'])} (warm {r['n_warm_target_samples']})", flush=True)
        if r["forced"]:
            print(f"    {r['forced']}", flush=True)
        print(f"    draft-head sampler calls: {r['draft_samples']}", flush=True)
        print(f"    spec metrics: {r['metrics'] or 'none'}", flush=True)
        if r["gdn"]:
            print("    build_delta_net dispatches (K=1 path):", flush=True)
            for (nt, br, pa, st), c in sorted(r["gdn"].items(), key=lambda x: int(x[0][0])):
                print(f"      n_tokens={nt:>4} branch={br:<8} path={pa:<14} state_ne=[{st}] x{c}", flush=True)
        if r["snap"]:
            print("    snapshot-path dispatches (K>1):", flush=True)
            for (rs, K, nt, br, st, ne, ss), c in sorted(r["snap"].items(), key=lambda x: int(x[0][2])):
                print(f"      n_rs_seq={rs} K={K} n_tokens={nt:>4} branch={br:<8} "
                      f"state_ne=[{st}] n_embd_s={ne} ssm_states_all=[{ss}] x{c}", flush=True)

    a = res[REF]["tokens"]
    print("\n=== hashes ===", flush=True)
    for arm in ARMS:
        print(f"  {arm:<10} {res[arm]['hash']}", flush=True)

    div = {}
    for arm in ARMS:
        if arm == REF:
            continue
        b = res[arm]["tokens"]
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
        div[arm] = i
        print(f"  {arm} vs {REF}: {'IDENTICAL' if i is None else f'first divergence @ token {i}'}", flush=True)
    same_bc = res["B_forced"]["hash"] == res["C_mtp"]["hash"]
    print(f"  B_forced == C_mtp: {same_bc}", flush=True)

    for arm, i in div.items():
        if i is None:
            continue
        ta, tb = res[REF]["targets"], res[arm]["targets"]
        n = res[REF]["n_warm_target_samples"] + i
        if n >= len(ta) or n >= len(tb):
            print(f"\n  {arm}: probe index {n} out of range", flush=True)
            continue
        ea, eb = ta[n], tb[n]
        print(f"\n--- {arm} vs {REF} at committed token {i} "
              f"({REF} gstep={ea['gstep']}, {arm} gstep={eb['gstep']})", flush=True)
        print(f"    {REF:<10} top1={ea['top'][0][0]} logit={ea['top'][0][1]:.6f} margin={ea['margin']:.6f}", flush=True)
        print(f"    {arm:<10} top1={eb['top'][0][0]} logit={eb['top'][0][1]:.6f} margin={eb['margin']:.6f}", flush=True)
        for rank in range(10):
            xa = f"{ea['top'][rank][0]}:{ea['top'][rank][1]:.6f}" if rank < len(ea["top"]) else "-"
            xb = f"{eb['top'][rank][0]}:{eb['top'][rank][1]:.6f}" if rank < len(eb["top"]) else "-"
            print(f"      rank{rank:<2} {REF}={xa:<24} {arm}={xb}", flush=True)
        la, lb = dict(ea["top"]), dict(eb["top"])
        sh = set(la) & set(lb)
        if sh:
            d = {t: lb[t] - la[t] for t in sh}
            mx = max(d, key=lambda t: abs(d[t]))
            print(f"    shared top-32: {len(sh)}  max|dlogit|={abs(d[mx]):.6f} (id {mx})  "
                  f"mean|dlogit|={sum(abs(v) for v in d.values())/len(d):.6f}", flush=True)
        ra = {t: k for k, (t, _) in enumerate(ea["top"])}
        rb = {t: k for k, (t, _) in enumerate(eb["top"])}
        print(f"    {REF} top1 rank under {arm}: {rb.get(ea['top'][0][0], '>31')}", flush=True)
        print(f"    {arm} top1 rank under {REF}: {ra.get(eb['top'][0][0], '>31')}", flush=True)

    # dispatch tables are keyed by tuples; stringify for JSON
    def jsonable(v):
        out = {}
        for kk, vv in v.items():
            if kk in ("text", "targets"):
                continue
            out[kk] = {" ".join(map(str, k2)): c for k2, c in vv.items()} \
                if kk in ("snap", "gdn") else vv
        return out

    json.dump({k: jsonable(v) for k, v in res.items()} |
              {"divergence": div, "B_equals_C": same_bc},
              open(os.path.join(HERE, f"rs_seq_test_{TAG}.json"), "w"), indent=2, default=str)
