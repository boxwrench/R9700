#!/usr/bin/env python3
"""Aggregate B1 parsed records into the four-arm result table.

Reads JSONL from b1_parse.py. Computes distribution statistics and the derived
quantities defined in shared/metrics.md:

    conditional-p1        = joint-p1 / p0
    accepted drafts/round = p0 + joint-p1
    committed tokens/round= 1 + p0 + joint-p1
    MTP multiplier        = mtp decode mean / matched serial decode mean

Every derived value is labelled CALCULATED. Nothing here is measured.
Standard deviation uses n-1 and is reported as None (never 0) when n == 1.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ARMS = ["mixed-serial", "mixed-mtp", "uniform-serial", "uniform-mtp"]


def dist(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    n = len(vals)
    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "stdev": statistics.stdev(vals) if n > 1 else None,
        "min": min(vals),
        "max": max(vals),
    }


def fmt(d: dict, prec: int = 2) -> str:
    if not d or d.get("n", 0) == 0:
        return "n/a"
    s = d["stdev"]
    sd = "n/a" if s is None else f"{s:.{prec}f}"
    return f"{d['mean']:.{prec}f} ± {sd}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    args = ap.parse_args()

    by_arm: dict[str, list[dict]] = {a: [] for a in ARMS}
    for line in args.jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rep", "").startswith("warmup"):
            continue
        by_arm.setdefault(r["arm"], []).append(r)

    out: dict[str, dict] = {}
    for arm, recs in by_arm.items():
        if not recs:
            continue
        a: dict = {
            "n_reps": len(recs),
            "decode_tok_s": dist([r.get("decode_tok_s") for r in recs]),
            "pp_tok_s": dist([r.get("pp_tok_s") for r in recs]),
            "decode_ms": dist([r.get("decode_ms") for r in recs]),
            "total_ms": dist([r.get("total_ms") for r in recs]),
        }
        gen = {r.get("generated_tokens") for r in recs}
        pt = {r.get("prompt_tokens") for r in recs}
        a["generated_tokens"] = gen.pop() if len(gen) == 1 else sorted(gen)
        a["prompt_tokens"] = pt.pop() if len(pt) == 1 else sorted(pt)

        if any(r.get("rounds") is not None for r in recs):
            a["rounds"] = dist([r.get("rounds") for r in recs])
            a["drafts_generated"] = dist([r.get("drafts_generated") for r in recs])
            a["drafts_accepted"] = dist([r.get("drafts_accepted") for r in recs])
            a["draft_tokens_generated"] = dist([r.get("draft_tokens_generated") for r in recs])
            a["draft_tokens_accepted"] = dist([r.get("draft_tokens_accepted") for r in recs])
            p0 = dist([r.get("p0") for r in recs])
            jp1 = dist([r.get("joint_p1") for r in recs])
            a["p0"] = p0
            a["joint_p1"] = jp1
            if p0.get("mean"):
                a["conditional_p1_CALCULATED"] = jp1["mean"] / p0["mean"]
                # Accepted drafts per round is the sum over EVERY draft position,
                # not p0 + joint-p1. That two-term form is the n_max=2 special
                # case in metrics.md; B1 runs n_max=4, where positions 2 and 3
                # also contribute. Each entry of acc_rate_per_pos is
                # P(at least i+1 accepted), so the sum is the expectation of the
                # accepted-prefix length.
                pos = recs[0].get("acc_rate_per_pos") or []
                a["n_draft_positions"] = len(pos)
                a["accepted_drafts_per_round_CALCULATED"] = sum(pos)
                a["committed_tokens_per_round_CALCULATED"] = 1.0 + sum(pos)
                # Cross-check against the raw counters, which are independent of
                # the per-position rounding. A mismatch means the identity above
                # does not hold and the numbers must not be reported.
                rounds_mean = a["rounds"].get("mean")
                acc_mean = a["draft_tokens_accepted"].get("mean")
                if rounds_mean:
                    raw = acc_mean / rounds_mean
                    a["accepted_drafts_per_round_RAW_CHECK"] = raw
                    a["accepted_drafts_per_round_AGREES"] = abs(raw - sum(pos)) < 5e-3
            a["acc_rate_per_pos_first_rep"] = recs[0].get("acc_rate_per_pos")
            a["mean_acc_len_upstream"] = dist([r.get("mean_acc_len_upstream") for r in recs])
        out[arm] = a

    for model in ("mixed", "uniform"):
        s, m = f"{model}-serial", f"{model}-mtp"
        if s in out and m in out:
            sm, mm = out[s]["decode_tok_s"].get("mean"), out[m]["decode_tok_s"].get("mean")
            if sm and mm:
                out[m]["mtp_multiplier_CALCULATED"] = mm / sm

    for mode in ("serial", "mtp"):
        u, x = f"uniform-{mode}", f"mixed-{mode}"
        if u in out and x in out:
            um, xm = out[u]["decode_tok_s"], out[x]["decode_tok_s"]
            if um.get("mean") and xm.get("mean"):
                delta = um["mean"] - xm["mean"]
                spread = max(um.get("stdev") or 0.0, xm.get("stdev") or 0.0)
                out.setdefault("_deltas", {})[f"uniform_vs_mixed_{mode}"] = {
                    "delta_tok_s_CALCULATED": delta,
                    "ratio_CALCULATED": um["mean"] / xm["mean"],
                    "larger_spread": spread,
                    "exceeds_spread": abs(delta) > spread,
                }

    print(json.dumps(out, indent=2))

    print("\n" + "=" * 78, file=sys.stderr)
    hdr = f"{'metric':<26}" + "".join(f"{a:>17}" for a in ARMS if a in out)
    print(hdr, file=sys.stderr)
    for label, key, prec in (
        ("decode tok/s", "decode_tok_s", 2),
        ("PP tok/s", "pp_tok_s", 2),
        ("decode wall ms", "decode_ms", 1),
    ):
        row = f"{label:<26}" + "".join(
            f"{fmt(out[a][key], prec):>17}" for a in ARMS if a in out
        )
        print(row, file=sys.stderr)
    for label, key in (
        ("MTP multiplier", "mtp_multiplier_CALCULATED"),
        ("committed tok/round", "committed_tokens_per_round_CALCULATED"),
        ("p0", None),
    ):
        vals = []
        for a in ARMS:
            if a not in out:
                continue
            if key is None:
                v = out[a].get("p0", {}).get("mean")
            else:
                v = out[a].get(key)
            vals.append(f"{v:.3f}" if isinstance(v, float) else "-")
        print(f"{label:<26}" + "".join(f"{v:>17}" for v in vals), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
