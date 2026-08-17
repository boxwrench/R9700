#!/usr/bin/env python3
"""Correctness/sanity checks over B1 arm logs.

Extracts the generated text from each llama-cli log and reports:
  * determinism within an arm (all reps byte-identical?)
  * MTP-vs-serial greedy agreement for the SAME model
  * NaN/inf warnings, malformed output, degenerate repetition

Mixed and uniform are deliberately NOT required to agree: uniform quantization
changes weights, so divergence there is expected and is not a defect.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# llama-cli with --no-display-prompt --simple-io writes the completion to
# stdout interleaved with the log. The generated text is what remains after
# dropping timestamped log lines and the trailing timing banner.
LOG_LINE = re.compile(r"^\d+\.\d+\.\d+\.\d+ [IWED] ")
# llama-cli writes the completion to stdout while the logger writes to the same
# stream, so a timestamped log record can begin partway through a line of
# generated text. Strip from the timestamp to end of line wherever it appears --
# comparing only what survives keeps the comparison stable across reps without
# pretending to reconstruct the exact original character stream.
EMBEDDED_LOG = re.compile(r"\d+\.\d+\.\d+\.\d+\s*\S{0,2}\s*[IWED] .*$")
BANNER = re.compile(r"^\[ Prompt:.*Generation:.*\]$")
NOISE = re.compile(
    r"^(build\s*:|model\s*:|modalities\s*:|available commands:|\s*/\w+|Exiting\.\.\.|"
    r"WARNING: radv|main:|>\s|\s*$)"
)


def extract_text(path: Path) -> str:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if LOG_LINE.match(line):
            continue
        line = EMBEDDED_LOG.sub("", line)
        if BANNER.match(line) or NOISE.match(line):
            continue
        if "statistics" in line or "eval time" in line or "total time" in line:
            continue
        out.append(line)
    return "\n".join(out).strip()


def degenerate(text: str, window: int = 8, threshold: int = 4) -> bool:
    """True if any window-length token run repeats >= threshold times."""
    toks = text.split()
    if len(toks) < window * threshold:
        return False
    counts = collections.Counter(
        " ".join(toks[i : i + window]) for i in range(len(toks) - window)
    )
    return bool(counts and counts.most_common(1)[0][1] >= threshold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rawdir", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    arms: dict[str, dict[str, str]] = collections.defaultdict(dict)
    flags: dict[str, list[str]] = collections.defaultdict(list)
    for p in sorted(args.rawdir.glob("*.rep*.log")):
        arm, rep = p.name.split(".")[0], p.name.split(".")[1]
        text = extract_text(p)
        arms[arm][rep] = text
        raw = p.read_text(errors="replace")
        if re.search(r"\bnan\b|\binf\b|-nan", raw, re.I):
            flags[arm].append(f"{rep}: nan/inf token present in log")
        if not text:
            flags[arm].append(f"{rep}: EMPTY generated text")
        if degenerate(text):
            flags[arm].append(f"{rep}: degenerate repetition detected")

    report: dict = {"arms": {}, "cross_arm": {}}
    for arm, reps in sorted(arms.items()):
        vals = set(reps.values())
        report["arms"][arm] = {
            "n_reps": len(reps),
            "deterministic": len(vals) == 1,
            "n_distinct_outputs": len(vals),
            "chars": len(next(iter(reps.values()))) if reps else 0,
            "flags": flags.get(arm, []),
            "sample": (next(iter(reps.values()))[:200] if reps else ""),
        }

    # greedy MTP-vs-serial agreement, per model
    for model in ("mixed", "uniform"):
        s, m = f"{model}-serial", f"{model}-mtp"
        if s in arms and m in arms:
            st = next(iter(arms[s].values()))
            mt = next(iter(arms[m].values()))
            common = min(len(st), len(mt))
            pref = 0
            while pref < common and st[pref] == mt[pref]:
                pref += 1
            report["cross_arm"][f"{model}: mtp==serial"] = {
                "identical": st == mt,
                "common_prefix_chars": pref,
                "serial_chars": len(st),
                "mtp_chars": len(mt),
            }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for arm, r in report["arms"].items():
            status = "DETERMINISTIC" if r["deterministic"] else f"NON-DETERMINISTIC ({r['n_distinct_outputs']} variants)"
            print(f"{arm:16s} n={r['n_reps']}  {status}  {r['chars']} chars")
            for f in r["flags"]:
                print(f"    FLAG {f}")
        print()
        for k, v in report["cross_arm"].items():
            print(f"{k}: identical={v['identical']}  common_prefix={v['common_prefix_chars']}/{v['serial_chars']} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
