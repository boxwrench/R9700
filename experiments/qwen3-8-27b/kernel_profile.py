#!/usr/bin/env python3
"""Vulkan kernel delta profile: 1-token vs 3-token target verification.

The Vulkan perf logger accumulates across every graph compute and flushes once
at context destroy, so a single run's totals include prefill and warmup. Each
arm is therefore run at two iteration counts and differenced, which cancels the
shared prefix exactly and leaves per-iteration kernel cost.

Both arms are differenced over the same number of advanced sequence positions
(300), so V1 contributes 300 decode calls and V3 contributes 100.

usage: kernel_profile.py [n_prefill]
"""
import collections, json, os, re, subprocess, sys

BIN = "/ai/scratch/llamacpp-probe/build/bin/llama-vbench"
MODEL = os.environ.get("EQ_MODEL",
                       "/ai/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = "/tmp/claude-1000/-home-boxwrench-Desktop/c9916d9d-b35f-4c78-8161-17f92bff5f70/scratchpad"
DEV = "1"
N_RS = "2"
POSITIONS = 300          # advanced positions in the differenced window

# batch_n -> (low_iter, high_iter); (high-low)*batch_n == POSITIONS
PLAN = {1: (200, 500), 3: (100, 200)}

LINE = re.compile(r"^(.*?): (\d+) x ([\d.]+) us = ([\d.]+) us")
RESULT = re.compile(r"^RESULT .*mean=([\d.]+) .*median=([\d.]+)")


def run(batch_n, n_iter, n_prefill):
    env = dict(os.environ, GGML_VK_PERF_LOGGER="1",
               GGML_VK_PERF_LOGGER_FREQUENCY="1")
    log = os.path.join(SCRATCH, f"kp-b{batch_n}-i{n_iter}-p{n_prefill}.log")
    with open(log, "w") as f:
        subprocess.run([BIN, MODEL, str(n_prefill), str(batch_n), str(n_iter), DEV, N_RS],
                       stdout=f, stderr=subprocess.STDOUT, timeout=3600, env=env)
    ops, wall = collections.OrderedDict(), None
    for line in open(log, errors="replace"):
        m = RESULT.search(line)
        if m:
            wall = float(m.group(1))
            continue
        m = LINE.match(line.strip())
        if m:
            name = m.group(1).strip()
            cnt = int(m.group(2))
            tot = float(m.group(4))          # us
            e = ops.setdefault(name, [0, 0.0])
            e[0] += cnt
            e[1] += tot
    return ops, wall, log


def family(name):
    n = name.upper()
    if "FLASH_ATTN" in n:                      return "FLASH_ATTN_EXT"
    if "GATED_DELTA" in n or "DELTA_NET" in n: return "GATED_DELTA_NET"
    if "MUL_MAT_ID" in n:                      return "MUL_MAT_ID"
    if "MUL_MAT" in n:                         return "MUL_MAT"
    if "ROPE" in n:                            return "ROPE"
    if "RMS_NORM" in n or "NORM" in n:         return "NORM"
    if "SOFT_MAX" in n:                        return "SOFT_MAX"
    if n.startswith("CPY") or "DUP" in n or "CONT" in n or "SET_ROWS" in n or "GET_ROWS" in n:
        return "COPY/TRANSFORM"
    if "ADD" in n or "MUL" in n or "SUB" in n or "DIV" in n or "SCALE" in n:
        return "ELEMENTWISE"
    if "SILU" in n or "GELU" in n or "SWIGLU" in n or "SIGMOID" in n or "EXP" in n:
        return "ACTIVATION"
    return "OTHER"


def diff(lo, hi):
    out = {}
    for k, (c, t) in hi.items():
        c0, t0 = lo.get(k, (0, 0.0))
        out[k] = (c - c0, t - t0)
    return out


def main():
    n_prefill = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    per_call, walls, raw = {}, {}, {}
    for bn, (lo_i, hi_i) in PLAN.items():
        lo, wl, _ = run(bn, lo_i, n_prefill)
        hi, wh, _ = run(bn, hi_i, n_prefill)
        d = diff(lo, hi)
        calls = hi_i - lo_i
        per_call[bn] = {k: (c / calls, t / calls) for k, (c, t) in d.items()}
        walls[bn] = wh
        raw[bn] = {"lo_iter": lo_i, "hi_iter": hi_i, "calls": calls,
                   "wall_mean_ms": wh, "wall_lo_ms": wl}
        print(f"arm batch_n={bn}: differenced {hi_i}-{lo_i}={calls} decode calls, "
              f"wall {wh:.3f} ms/call", flush=True)

    fam = {}
    for bn in PLAN:
        agg = collections.defaultdict(lambda: [0.0, 0.0])
        for k, (c, t) in per_call[bn].items():
            f = family(k)
            agg[f][0] += c
            agg[f][1] += t / 1000.0        # ms per call
        fam[bn] = agg

    keys = sorted(set(fam[1]) | set(fam[3]),
                  key=lambda f: -(fam[3].get(f, [0, 0])[1] - fam[1].get(f, [0, 0])[1]))
    d_wall = walls[3] - walls[1]
    print(f"\n=== KERNEL DELTA PROFILE  (V1={walls[1]:.3f} ms, V3={walls[3]:.3f} ms, "
          f"wall delta={d_wall:+.3f} ms)")
    print(f"{'family':<20}{'V1 ms':>9}{'V1 disp':>9}{'V3 ms':>9}{'V3 disp':>9}{'delta ms':>10}{'% of d':>9}")
    tot1 = tot3 = 0.0
    for f in keys:
        c1, t1 = fam[1].get(f, [0.0, 0.0])
        c3, t3 = fam[3].get(f, [0.0, 0.0])
        tot1 += t1
        tot3 += t3
        dd = t3 - t1
        print(f"{f:<20}{t1:9.3f}{c1:9.1f}{t3:9.3f}{c3:9.1f}{dd:+10.3f}{dd/d_wall*100:8.1f}%")
    print(f"{'GPU kernel total':<20}{tot1:9.3f}{'':9}{tot3:9.3f}{'':9}{tot3-tot1:+10.3f}"
          f"{(tot3-tot1)/d_wall*100:8.1f}%")
    print(f"{'unexplained':<20}{walls[1]-tot1:9.3f}{'':9}{walls[3]-tot3:9.3f}{'':9}"
          f"{d_wall-(tot3-tot1):+10.3f}{(d_wall-(tot3-tot1))/d_wall*100:8.1f}%")

    # biggest individual contributors, ungrouped
    ind = []
    for k in set(per_call[1]) | set(per_call[3]):
        t1 = per_call[1].get(k, (0, 0.0))[1] / 1000.0
        t3 = per_call[3].get(k, (0, 0.0))[1] / 1000.0
        ind.append((t3 - t1, k, t1, t3,
                    per_call[1].get(k, (0, 0.0))[0], per_call[3].get(k, (0, 0.0))[0]))
    ind.sort(reverse=True)
    print(f"\n=== TOP INDIVIDUAL OPERATIONS BY DELTA")
    for dd, k, t1, t3, c1, c3 in ind[:15]:
        print(f"  {dd:+8.4f} ms ({dd/d_wall*100:5.1f}%)  V1 {t1:7.4f} x{c1:6.1f}  "
              f"V3 {t3:7.4f} x{c3:6.1f}   {k[:95]}")

    json.dump({"n_prefill": n_prefill, "walls": walls, "raw": raw,
               "family": {str(b): {f: v for f, v in fam[b].items()} for b in fam},
               "individual": {str(b): {k: list(v) for k, v in per_call[b].items()} for b in per_call}},
              open(os.path.join(HERE, f"kernel_profile_{n_prefill}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
