#!/usr/bin/env python3
"""Aggregate and compare run records.

Reads JSONL (one record per line) or CSV. Reports, per group:
n, mean, stdev, median, min, max, p10, p90 -- and, when two groups share run
identifiers, the paired delta.

Deliberate omissions:
  * No p-values or significance claims. Reporting the distribution is what the
    benchmark contract asks for; a t-test on 5 runs of a GPU benchmark would
    imply more than the data supports.
  * No filling of missing values. A record missing the metric is counted as
    skipped and reported as such.
  * Sample stdev (n-1). With n=1 the stdev is reported as null, not 0.

Usage:
    06_compare_runs.py results.jsonl --metric decode_tok_s --group-by experiment_id
    06_compare_runs.py a.jsonl b.jsonl --metric decode_tok_s --group-by experiment_id \\
        --paired-on prompt_id,run_index

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    """Load JSONL or CSV. Malformed lines are reported, not silently dropped."""
    records: list[dict] = []
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return []

    if path.suffix.lower() == ".csv":
        for row in csv.DictReader(text.splitlines()):
            records.append(dict(row))
        return records

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"warning: {path}:{line_no}: skipping unparseable line ({exc.msg})",
                  file=sys.stderr)
            continue
        if isinstance(obj, dict):
            records.append(obj)
        else:
            print(f"warning: {path}:{line_no}: not a JSON object, skipped", file=sys.stderr)
    return records


def as_number(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 1]."""
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict:
    s = sorted(values)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else None,
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "p10": percentile(s, 0.10),
        "p90": percentile(s, 0.90),
    }


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--metric", required=True, help="record field to aggregate")
    ap.add_argument("--group-by", default="experiment_id", help="record field to group on")
    ap.add_argument("--paired-on", default=None,
                    help="comma-separated fields forming a pairing key across exactly two groups")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    records: list[dict] = []
    for path in args.inputs:
        records.extend(load_records(path))

    if not records:
        print("no records loaded", file=sys.stderr)
        return 1

    groups: dict[str, list[float]] = {}
    skipped = 0
    for rec in records:
        value = as_number(rec.get(args.metric))
        if value is None:
            skipped += 1
            continue
        key = str(rec.get(args.group_by, "<ungrouped>"))
        groups.setdefault(key, []).append(value)

    if not groups:
        print(f"no records carried a numeric '{args.metric}' "
              f"({skipped} records skipped)", file=sys.stderr)
        return 1

    summaries = {key: summarize(vals) for key, vals in sorted(groups.items())}

    paired = None
    if args.paired_on:
        keys = [k.strip() for k in args.paired_on.split(",") if k.strip()]
        group_names = sorted(groups)
        if len(group_names) != 2:
            print(f"warning: --paired-on needs exactly 2 groups, found {len(group_names)}; "
                  "skipping paired analysis", file=sys.stderr)
        else:
            a_name, b_name = group_names
            indexed: dict[str, dict[tuple, float]] = {a_name: {}, b_name: {}}
            for rec in records:
                value = as_number(rec.get(args.metric))
                if value is None:
                    continue
                g = str(rec.get(args.group_by, "<ungrouped>"))
                if g not in indexed:
                    continue
                pair_key = tuple(str(rec.get(k, "")) for k in keys)
                indexed[g][pair_key] = value
            common = sorted(set(indexed[a_name]) & set(indexed[b_name]))
            if not common:
                print("warning: no matching pair keys between the two groups", file=sys.stderr)
            else:
                deltas = [indexed[b_name][k] - indexed[a_name][k] for k in common]
                paired = {
                    "baseline": a_name,
                    "comparison": b_name,
                    "pairs": len(common),
                    "unpaired_baseline": len(indexed[a_name]) - len(common),
                    "unpaired_comparison": len(indexed[b_name]) - len(common),
                    "delta": summarize(deltas),
                }

    if args.json:
        print(json.dumps({
            "metric": args.metric,
            "group_by": args.group_by,
            "records_loaded": len(records),
            "records_skipped_missing_metric": skipped,
            "groups": summaries,
            "paired": paired,
        }, indent=2))
        return 0

    print(f"metric: {args.metric}    grouped by: {args.group_by}")
    print(f"records loaded: {len(records)}    skipped (metric absent): {skipped}")
    print()
    header = f"{'group':<32}{'n':>4}{'mean':>12}{'stdev':>12}{'median':>12}{'min':>12}{'max':>12}{'p10':>12}{'p90':>12}"
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print(f"{name[:31]:<32}{s['n']:>4}{fmt(s['mean']):>12}{fmt(s['stdev']):>12}"
              f"{fmt(s['median']):>12}{fmt(s['min']):>12}{fmt(s['max']):>12}"
              f"{fmt(s['p10']):>12}{fmt(s['p90']):>12}")

    for name, s in summaries.items():
        if s["n"] < 5:
            print(f"\nnote: group '{name}' has n={s['n']}; the benchmark contract asks for "
                  "at least 5 repetitions before a production-style comparison.")

    if paired:
        d = paired["delta"]
        print(f"\npaired delta ({paired['comparison']} - {paired['baseline']})")
        print(f"  pairs matched: {paired['pairs']}"
              f"   unpaired: {paired['unpaired_baseline']} / {paired['unpaired_comparison']}")
        print(f"  mean {fmt(d['mean'])}   stdev {fmt(d['stdev'])}   median {fmt(d['median'])}")
        print(f"  min  {fmt(d['min'])}    max   {fmt(d['max'])}")
        if d["stdev"] is not None and abs(d["mean"]) < d["stdev"]:
            print("  NOTE: |mean delta| is smaller than its stdev. This does not "
                  "establish a difference.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
