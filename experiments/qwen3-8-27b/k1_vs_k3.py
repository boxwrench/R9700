#!/usr/bin/env python3
"""Fixed-proposal K=1 vs K=3 verifier agreement.

Phase 1: run production MTP once, capturing (prefix, d0, d1, accept/reject) for
         every speculative round. Proposals are then FROZEN.
Phase 2: re-score the identical prefixes and proposals under two target
         contexts -- V1 (n_rs_seq=0, K=1) and V3 (forced n_rs_seq=2, K=3, no
         drafter). Neither verifier generates a continuation.
Phase 3: validate that V3 reproduces the production K=3 accept/reject decisions
         before any K=1 vs K=3 comparison is interpreted.

Scoring uses /completions with an explicit token-id prompt and n_predict=1, so
the target distribution is evaluated at an exact prefix. cache_prompt is false
throughout: prompt-cache restoration is known to alter recurrent state on this
model and would invalidate the causal comparison.
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request

BIN = "/ai/scratch/llamacpp-probe/build/bin/llama-server"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.path.dirname(MODEL) + "/mmproj-F16.gguf"
TAG = os.environ.get("EQ_TAG", "q4")
PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
CAPTURE = os.path.join(HERE, f"k1k3_rounds_{TAG}.jsonl")
SCORES = os.path.join(HERE, f"k1k3_scores_{TAG}.jsonl")

MAX_ROUNDS = int(os.environ.get("K1K3_ROUNDS", "250"))
N_PROBS = 10

FILLER = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
          "profiles kernel launches and inspects the memory hierarchy of the accelerator. ")
BODY = (FILLER * 56)[:1100 * 6] + "\n\nSummarize the passage above, then explain speculative decoding."
# Several distinct prompts so rounds are not all from one continuation.
PROMPTS = [(f"p{t}", f"Trial {t} marker-{t*7919}. " + BODY) for t in (3, 5, 8, 11)]
WARM = "Warmup unrelated prompt alpha-31337. Reply with a single short sentence."

MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.3"]


def launch(extra, log_path, env_extra=None):
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
    env = dict(os.environ)
    env.update(env_extra or {})
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


def post(path, body, timeout=900):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(text, max_tokens):
    return post("/v1/chat/completions",
                {"model": "eq", "messages": [{"role": "user", "content": text}],
                 "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                 "seed": 0, "ignore_eos": True, "cache_prompt": False,
                 "chat_template_kwargs": {"enable_thinking": False}})


def score_prefix(tokens):
    """Target top-k at an exact token prefix. No continuation is generated."""
    r = post("/completions",
             {"prompt": tokens, "n_predict": 1, "n_probs": N_PROBS,
              "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
              "cache_prompt": False, "post_sampling_probs": False})
    probs = r.get("completion_probabilities") or []
    if not probs:
        return None
    top = probs[0].get("top_logprobs") or []
    cands = []
    for e in top:
        tid, lg = e.get("id"), e.get("logprob")
        if tid is not None and lg is not None:
            cands.append((int(tid), float(lg)))
    cands.sort(key=lambda x: -x[1])
    return {"top": cands,
            "top1": cands[0][0] if cands else None,
            "margin": (cands[0][1] - cands[1][1]) if len(cands) >= 2 else None}


SPEC = re.compile(r"\[SPEC\] prefix_len=(\d+) n_draft=(\d+) draft=\[([^\]]*)\] "
                  r"committed=\[([^\]]*)\] n_accepted=(\d+) prefix=\[([^\]]*)\]")


def parse_spec(path):
    out = []
    for line in open(path, errors="replace"):
        m = SPEC.search(line)
        if not m:
            continue
        draft = [int(x) for x in m.group(3).split(",") if x]
        committed = [int(x) for x in m.group(4).split(",") if x]
        prefix = [int(x) for x in m.group(6).split(",") if x]
        out.append({"prefix_len": int(m.group(1)), "n_draft": int(m.group(2)),
                    "draft": draft, "committed": committed,
                    "n_accepted": int(m.group(5)), "prefix": prefix})
    return out


# ---------------------------------------------------------------- phase 1
def phase1():
    log_path = os.path.join(SCRATCH, f"k1k3-capture-{TAG}.log")
    proc, err = launch(MTP_ARGS, log_path, {"LLAMA_PROBE_SPEC": "1"})
    if err:
        stop(proc)
        raise SystemExit(f"FAIL capture: {err}")
    try:
        chat(WARM, 16)
        for pid, text in PROMPTS:
            chat(text, 256)
            print(f"  captured through prompt {pid}", flush=True)
    finally:
        stop(proc)
    rounds = parse_spec(log_path)
    for i, r in enumerate(rounds):
        r["round_id"] = i
    with open(CAPTURE, "w") as f:
        for r in rounds:
            f.write(json.dumps(r) + "\n")
    print(f"phase 1: captured {len(rounds)} speculative rounds", flush=True)
    return rounds


# ---------------------------------------------------------------- phase 2
def score_rounds(rounds, label, extra, env_extra):
    log_path = os.path.join(SCRATCH, f"k1k3-{label}-{TAG}.log")
    proc, err = launch(extra, log_path, env_extra)
    if err:
        stop(proc)
        raise SystemExit(f"FAIL {label}: {err}")
    out = []
    t0 = time.time()
    try:
        for n, r in enumerate(rounds):
            # The captured prefix already contains the draft tokens (verified: its
            # tail equals `draft`). Trim them to recover the true common prefix.
            pre = r["prefix"][:r["prefix_len"] - r["n_draft"]]
            s0 = score_prefix(pre)
            d0 = r["draft"][0] if len(r["draft"]) > 0 else None
            d1 = r["draft"][1] if len(r["draft"]) > 1 else None
            s1 = score_prefix(pre + [d0]) if d0 is not None else None
            out.append({"round_id": r["round_id"], "verifier": label,
                        "d0": d0, "d1": d1,
                        "top1_0": s0["top1"] if s0 else None,
                        "margin_0": s0["margin"] if s0 else None,
                        "top_0": s0["top"][:5] if s0 else [],
                        "top1_1": s1["top1"] if s1 else None,
                        "margin_1": s1["margin"] if s1 else None,
                        "top_1": s1["top"][:5] if s1 else []})
            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f"    {label}: {n+1}/{len(rounds)} ({el/(n+1):.2f}s/round)", flush=True)
    finally:
        stop(proc)
    return out


# ---------------------------------------------------------------- analysis
def rate(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def summarize(rounds, scores, label):
    by = {s["round_id"]: s for s in scores}
    p0, p1c, joint = [], [], []
    m0, m1 = [], []
    for r in rounds:
        s = by.get(r["round_id"])
        if not s or s["d0"] is None or s["d1"] is None:
            continue
        a0 = (s["d0"] == s["top1_0"])
        a1 = (s["d1"] == s["top1_1"])
        p0.append(a0)
        p1c.append(a1)
        joint.append(a0 and a1)
        if s["margin_0"] is not None:
            m0.append(s["margin_0"])
        if s["margin_1"] is not None:
            m1.append(s["margin_1"])
    P0, P1, J = rate(p0), rate(p1c), rate(joint)
    return {"label": label, "n": len(p0), "P0": P0, "P1_counterfactual": P1,
            "Joint2": J, "Conditional_P1": (J / P0) if P0 else None,
            "expected_drafts": (P0 + J) if (P0 is not None and J is not None) else None,
            "mean_margin_pos0": statistics.mean(m0) if m0 else None,
            "mean_margin_pos1": statistics.mean(m1) if m1 else None}


def main():
    if os.path.exists(CAPTURE) and "--recapture" not in sys.argv:
        rounds = [json.loads(l) for l in open(CAPTURE)]
        print(f"phase 1: reusing {len(rounds)} captured rounds", flush=True)
    else:
        rounds = phase1()

    full = [r for r in rounds if r["n_draft"] == 2 and len(r["draft"]) == 2]
    print(f"rounds with a full 2-token proposal: {len(full)}", flush=True)

    # Structural check: the captured prefix must end with exactly the draft tokens,
    # otherwise the trim in score_rounds() recovers the wrong common prefix.
    bad = [r["round_id"] for r in full
           if r["prefix"][-r["n_draft"]:] != r["draft"]
           or len(r["prefix"]) != r["prefix_len"]]
    if bad:
        raise SystemExit(f"prefix/draft layout assumption violated in {len(bad)} rounds "
                         f"(first: {bad[:5]}) -- diagnose before scoring")
    print("prefix layout verified: captured prefix ends with the draft tokens", flush=True)
    if len(full) > MAX_ROUNDS:
        step = len(full) / MAX_ROUNDS
        full = [full[int(i * step)] for i in range(MAX_ROUNDS)]
    print(f"scoring {len(full)} rounds", flush=True)

    v3 = score_rounds(full, "V3", [], {"LLAMA_FORCE_N_RS_SEQ": "2"})
    v1 = score_rounds(full, "V1", [], {})
    with open(SCORES, "w") as f:
        for s in v3 + v1:
            f.write(json.dumps(s) + "\n")

    # ---- phase 3: replay validation
    by3 = {s["round_id"]: s for s in v3}
    agree0 = agree1 = tot0 = tot1 = 0
    mism = []
    for r in full:
        s = by3.get(r["round_id"])
        if not s:
            continue
        prod_a0 = r["n_accepted"] >= 1
        prod_a1 = r["n_accepted"] >= 2
        rep_a0 = (s["d0"] == s["top1_0"])
        rep_a1 = (s["d1"] == s["top1_1"])
        tot0 += 1
        agree0 += (prod_a0 == rep_a0)
        if prod_a0:
            tot1 += 1
            agree1 += (prod_a1 == rep_a1)
        if prod_a0 != rep_a0:
            mism.append((r["round_id"], prod_a0, rep_a0, r["prefix_len"]))
    a0r = agree0 / tot0 if tot0 else 0.0
    print(f"\n=== PHASE 3 replay validation")
    print(f"  V3 vs production, position 0: {agree0}/{tot0} = {a0r:.4f}")
    if tot1:
        print(f"  V3 vs production, position 1 (given d0 accepted): {agree1}/{tot1} = {agree1/tot1:.4f}")
    if mism[:5]:
        print(f"  first mismatches (round, prod_accept, replay_accept, prefix_len): {mism[:5]}")

    s1 = summarize(full, v1, "K=1")
    s3 = summarize(full, v3, "K=3")

    print(f"\n=== PRIMARY METRICS")
    print(f"{'metric':<22}{'K=1':>12}{'K=3':>12}")
    for k in ("P0", "P1_counterfactual", "Joint2", "Conditional_P1",
              "expected_drafts", "mean_margin_pos0", "mean_margin_pos1"):
        f1 = s1[k]
        f3 = s3[k]
        print(f"{k:<22}{(f'{f1:.4f}' if f1 is not None else '-'):>12}"
              f"{(f'{f3:.4f}' if f3 is not None else '-'):>12}")

    # ---- disagreement matrices
    b1 = {s["round_id"]: s for s in v1}
    cells0 = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    cellsJ = dict(cells0)
    diffs = []
    for r in full:
        x, y = b1.get(r["round_id"]), by3.get(r["round_id"])
        if not x or not y:
            continue
        k1a0, k3a0 = (x["d0"] == x["top1_0"]), (y["d0"] == y["top1_0"])
        k1j = k1a0 and (x["d1"] == x["top1_1"])
        k3j = k3a0 and (y["d1"] == y["top1_1"])
        cells0[(k1a0, k3a0)] += 1
        cellsJ[(k1j, k3j)] += 1
        if k1a0 != k3a0:
            l1 = dict(x["top_0"])
            l3 = dict(y["top_0"])
            d0 = x["d0"]
            r1 = [t for t, _ in x["top_0"]].index(d0) if d0 in l1 else None
            r3 = [t for t, _ in y["top_0"]].index(d0) if d0 in l3 else None
            diffs.append({"round_id": r["round_id"], "k1_accept": k1a0, "k3_accept": k3a0,
                          "k1_margin": x["margin_0"], "k3_margin": y["margin_0"],
                          "rank_d0_k1": r1, "rank_d0_k3": r3,
                          "dlogit_d0": (l3.get(d0) - l1.get(d0))
                                       if (d0 in l1 and d0 in l3) else None})

    print(f"\n=== POSITION-0 DISAGREEMENT MATRIX")
    print(f"  K1 accept / K3 accept : {cells0[(True, True)]}")
    print(f"  K1 accept / K3 reject : {cells0[(True, False)]}")
    print(f"  K1 reject / K3 accept : {cells0[(False, True)]}")
    print(f"  K1 reject / K3 reject : {cells0[(False, False)]}")
    print(f"\n=== JOINT-2 DISAGREEMENT MATRIX")
    print(f"  K1 survive / K3 survive : {cellsJ[(True, True)]}")
    print(f"  K1 survive / K3 fail    : {cellsJ[(True, False)]}")
    print(f"  K1 fail    / K3 survive : {cellsJ[(False, True)]}")
    print(f"  K1 fail    / K3 fail    : {cellsJ[(False, False)]}")

    if diffs:
        km = [d["k1_margin"] for d in diffs if d["k1_margin"] is not None]
        k3m = [d["k3_margin"] for d in diffs if d["k3_margin"] is not None]
        dl = [d["dlogit_d0"] for d in diffs if d["dlogit_d0"] is not None]
        print(f"\n=== LOGIT STATISTICS ON {len(diffs)} POSITION-0 DISAGREEMENTS")
        if km:
            print(f"  K1 margin  mean={statistics.mean(km):.4f} median={statistics.median(km):.4f}")
        if k3m:
            print(f"  K3 margin  mean={statistics.mean(k3m):.4f} median={statistics.median(k3m):.4f}")
        if dl:
            print(f"  |dlogit| on proposed token: mean={statistics.mean(map(abs, dl)):.4f} "
                  f"max={max(map(abs, dl)):.4f}")
        print(f"  K3 breaks a K1 acceptance: {sum(1 for d in diffs if d['k1_accept'] and not d['k3_accept'])}")
        print(f"  K3 rescues a K1 rejection: {sum(1 for d in diffs if not d['k1_accept'] and d['k3_accept'])}")
    else:
        print("\n=== no position-0 disagreements")

    json.dump({"n_rounds_scored": len(full), "replay_pos0_agreement": a0r,
               "K1": s1, "K3": s3,
               "pos0_matrix": {f"{k[0]}_{k[1]}": v for k, v in cells0.items()},
               "joint_matrix": {f"{k[0]}_{k[1]}": v for k, v in cellsJ.items()},
               "n_disagreements": len(diffs)},
              open(os.path.join(HERE, f"k1k3_summary_{TAG}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
