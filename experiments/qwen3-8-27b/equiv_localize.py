#!/usr/bin/env python3
"""Localize speculative-vs-serial target divergence.

ARM A: ordinary decode (no speculation)
ARM B: ngram-simple speculative decoding, draft depth 2

n-gram is used rather than MTP so the learned proposer is removed entirely
while the divergent speculative target path is preserved.

Fresh server per arm. Warmup uses an UNRELATED prompt so the measured
generation always evaluates the target prompt fresh -- warming on the same
prompt lets the cache restore recurrent state and moves the divergence point.

Pass 1 (no args): find first divergence, dump GDN dispatch records.
Pass 2 (--dump-step N): re-run both arms dumping the full candidate array at
step N for exact cross-arm logit deltas.

Env: EQ_MODEL / EQ_TAG to switch quantization (Q6 control).
"""
import hashlib, json, os, re, struct, subprocess, sys, time, urllib.request

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

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
PROMPT_BODY = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
TRIAL = 3
PROMPT = f"Trial {TRIAL} marker-{TRIAL*7919}. " + PROMPT_BODY
WARM = "Warmup unrelated prompt alpha-31337. Reply with a single short sentence."

ARMS = {
    "A_none":  [],
    "B_ngram": ["--spec-type", "ngram-simple",
                "--spec-ngram-simple-size-n", "2",
                "--spec-ngram-simple-size-m", "2",
                "--spec-draft-n-max", "2"],
    "C_mtp":   ["--spec-type", "draft-mtp",
                "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.3"],
}
REF = "A_none"


def launch(extra, log_path, dump_step=None, dump_file=None):
    args = [
        BIN, "--model", MODEL, "--mmproj", MMPROJ, "--alias", "eq",
        "--device", "Vulkan1", "--n-gpu-layers", "999", "--parallel", "1",
        "--threads", "6", "--threads-batch", "6",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-unified", "--load-mode", "none", "--split-mode", "none",
        "--n-cpu-moe", "0", "--jinja", "--reasoning-preserve",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "163840", "--ubatch-size", "512",
    ] + extra
    env = dict(os.environ, LLAMA_PROBE_GDN="1", LLAMA_PROBE_LOGITS="1")
    if dump_step is not None:
        env["LLAMA_PROBE_DUMP_STEP"] = str(dump_step)
        env["LLAMA_PROBE_DUMP_FILE"] = dump_file
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
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    ch = j["choices"][0]
    return (ch["message"].get("content") or "",
            [t["token"] for t in (ch.get("logprobs", {}).get("content") or [])])


GDN_PAT = re.compile(
    r"\[GDN\] il=\s*(\d+) ubatch_n_tokens=(\d+) ubatch_n_seq_tokens=(\d+) ubatch_n_seqs=(\d+) "
    r"q_n_seq_tokens=(\d+) q_n_seqs=(\d+) branch=(\w+) path=(\w+) state_ne=\[([^\]]+)\]"
    r"(?: pos=\[([^\]]*)\])?(?: seq=\[([^\]]*)\])?")
LOGIT_PAT = re.compile(r"\[LOGIT\] step=(\d+) sampler_idx=(-?\d+) n_cand=(\d+)(.*)")


def parse_gdn(path):
    recs, seen = [], set()
    for line in open(path, errors="replace"):
        m = GDN_PAT.search(line)
        if not m:
            continue
        r = {"il": int(m.group(1)), "ubatch_n_tokens": int(m.group(2)),
             "ubatch_n_seq_tokens": int(m.group(3)), "ubatch_n_seqs": int(m.group(4)),
             "q_n_seq_tokens": int(m.group(5)), "q_n_seqs": int(m.group(6)),
             "branch": m.group(7), "path": m.group(8), "state_ne": m.group(9),
             "pos": m.group(10), "seq": m.group(11)}
        key = (r["ubatch_n_tokens"], r["q_n_seq_tokens"], r["branch"], r["path"], r["state_ne"])
        if key not in seen:
            seen.add(key)
            recs.append(r)
    return recs


def parse_logits(path):
    steps = []
    for line in open(path, errors="replace"):
        m = LOGIT_PAT.search(line)
        if not m:
            continue
        body, _, margin = m.group(4).partition("||")
        top = []
        for part in body.split("|"):
            part = part.strip()
            if ":" in part:
                tid, _, lg = part.partition(":")
                try:
                    top.append((int(tid), float(lg)))
                except ValueError:
                    pass
        steps.append({"step": int(m.group(1)), "sampler_idx": int(m.group(2)),
                      "n_cand": int(m.group(3)), "top": top,
                      "margin": float(margin.split("=")[1]) if "=" in margin else None})
    return steps


def read_dump(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        out = {}
        for _ in range(n):
            tid, lg = struct.unpack("<if", f.read(8))
            out[tid] = lg
    return out


def run_arm(arm, extra, dump_step=None):
    log_path = os.path.join(SCRATCH, f"loc-{TAG}-{arm}{'-d' if dump_step is not None else ''}.log")
    dump_file = os.path.join(SCRATCH, f"loc-{TAG}-{arm}.bin") if dump_step is not None else None
    proc, err = launch(extra, log_path, dump_step, dump_file)
    if err:
        stop(proc)
        raise SystemExit(f"FAIL {arm}: {err}")
    try:
        generate(WARM, 16)                       # unrelated warmup prompt
        n_warm_steps = len(parse_logits(log_path))
        n_warm_gdn = sum(1 for _ in open(log_path, errors="replace") if "[GDN]" in _)
        text, toks = generate(PROMPT)
    finally:
        stop(proc)
    return {"arm": arm, "text": text, "tokens": toks,
            "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "n_warm_steps": n_warm_steps, "n_warm_gdn": n_warm_gdn,
            "gdn": parse_gdn(log_path), "logits": parse_logits(log_path),
            "log": log_path, "dump": dump_file}


if __name__ == "__main__":
    dump_step = None
    if "--dump-step" in sys.argv:
        dump_step = int(sys.argv[sys.argv.index("--dump-step") + 1])

    res = {}
    for arm, extra in ARMS.items():
        r = run_arm(arm, extra, dump_step)
        res[arm] = r
        print(f"{arm}: hash={r['hash']} ntok={len(r['tokens'])} "
              f"sampler_steps={len(r['logits'])} (warmup {r['n_warm_steps']})", flush=True)
        print(f"  GDN distinct dispatches ({len(r['gdn'])}):", flush=True)
        for g in r["gdn"]:
            print(f"    ubatch_n_tokens={g['ubatch_n_tokens']:>4} q_n_seq_tokens={g['q_n_seq_tokens']:>4} "
                  f"branch={g['branch']:<8} path={g['path']:<30} state_ne=[{g['state_ne']}] "
                  f"pos=[{g['pos']}] seq=[{g['seq']}]", flush=True)

    a = res[REF]["tokens"]
    div = {}
    for arm in ARMS:
        if arm == REF:
            continue
        b = res[arm]["tokens"]
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
        div[arm] = i
        print(f"\n=== {arm} vs {REF}: "
              f"{'IDENTICAL' if i is None else f'first_divergent_token_index={i}'}", flush=True)
        if i is not None:
            print(f"  {REF} committed {a[i]!r}   {arm} committed {b[i]!r}", flush=True)
            print(f"  common prefix (tokens 0..{i-1}) identical", flush=True)

    diverging = [k for k, v in div.items() if v is not None]
    if not diverging:
        print("\nall arms identical -- nothing to localize", flush=True)
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("text", "logits")}
                   for k, v in res.items()} | {"divergence": div},
                  open(os.path.join(HERE, f"equiv_localize_{TAG}.json"), "w"), indent=2)
        raise SystemExit(0)

    target = diverging[0]
    idx = div[target]

    # Map committed token index -> sampler step (offset by warmup steps).
    for arm in (REF, target):
        r = res[arm]
        s = r["n_warm_steps"] + idx
        cand = [x for x in r["logits"] if x["step"] == s]
        print(f"\n--- {arm} sampler step {s} (committed token {idx})", flush=True)
        if not cand:
            print("    no probe record at this step", flush=True)
            continue
        e = cand[0]
        print(f"    sampler_idx(output index)={e['sampler_idx']} n_cand={e['n_cand']} margin={e['margin']}", flush=True)
        for i, (tid, lg) in enumerate(e["top"][:10]):
            print(f"      rank{i:<2} id={tid:<8} logit={lg:.6f}", flush=True)
        res[arm]["probe_at_divergence"] = e

    # cross-arm ranks
    ea = res[REF].get("probe_at_divergence")
    eb = res[target].get("probe_at_divergence")
    if ea and eb:
        ta = {t: i for i, (t, _) in enumerate(ea["top"])}
        tb = {t: i for i, (t, _) in enumerate(eb["top"])}
        a1, b1 = ea["top"][0][0], eb["top"][0][0]
        print(f"\n  A top1 id={a1} -> rank under B: {tb.get(b1 if False else a1, '>31')}", flush=True)
        print(f"  B top1 id={b1} -> rank under A: {ta.get(b1, '>31')}", flush=True)
        common = set(t for t, _ in ea["top"]) & set(t for t, _ in eb["top"])
        la = dict(ea["top"]); lb = dict(eb["top"])
        if common:
            diffs = {t: lb[t] - la[t] for t in common}
            mx = max(diffs, key=lambda t: abs(diffs[t]))
            print(f"  max |logit delta| over shared top-32: id={mx} delta={diffs[mx]:.6f}", flush=True)
            print(f"  mean |logit delta| over shared top-32: "
                  f"{sum(abs(v) for v in diffs.values())/len(diffs):.6f}", flush=True)

    if dump_step is not None:
        try:
            da, db = read_dump(res[REF]["dump"]), read_dump(res[target]["dump"])
            shared = set(da) & set(db)
            deltas = [abs(db[t] - da[t]) for t in shared]
            print(f"\n  FULL-VOCAB dump at step {dump_step}: {len(shared)} shared candidates", flush=True)
            print(f"    max |logit delta| = {max(deltas):.6f}", flush=True)
            print(f"    mean |logit delta| = {sum(deltas)/len(deltas):.6f}", flush=True)
        except Exception as exc:
            print(f"  full dump comparison unavailable: {exc!r}", flush=True)

    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("text", "logits")}
               for k, v in res.items()} | {"divergence": div, "localized": target},
              open(os.path.join(HERE, f"equiv_localize_{TAG}.json"), "w"), indent=2)
