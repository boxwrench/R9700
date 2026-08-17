#!/usr/bin/env python3
"""Parse B1 arm logs into JSONL run records.

Every emitted value carries the file, line number, and verbatim source line it
came from. A field that no pattern matched is emitted as null and counted as
unmatched -- it is never defaulted, interpolated, or inferred from another
field. Derived quantities (conditional-p1, MTP multiplier) are computed by
06_compare_runs.py or the report step, not here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Each pattern must anchor on text llama-cli actually prints. If upstream
# changes a log line, the field goes null and is reported -- it does not
# silently pick up a neighbouring number.
PATTERNS = {
    "pp_tok_s": re.compile(
        r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)"
    ),
    "decode": re.compile(
        r"^\s*eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)"
    ),
    "total": re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens"),
    "vram_model": re.compile(r"load_tensors:\s+(\S+) model buffer size =\s*([\d.]+) MiB"),
    "vram_kv": re.compile(r"llama_kv_cache:\s+(\S+) KV buffer size =\s*([\d.]+) MiB"),
    "vram_rs": re.compile(r"llama_memory_recurrent:\s+(\S+) RS buffer size =\s*([\d.]+) MiB"),
    "vram_compute": re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s*([\d.]+) MiB"),
    "stats": re.compile(
        r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), "
        r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), "
        r"#acc tokens =\s*(\d+), #mean acc len =\s*([\d.]+), "
        r"#acc rate/pos = \(([^)]*)\)"
    ),
}


def parse_log(path: Path) -> dict:
    rec: dict = {"file": str(path), "evidence": {}}
    unmatched: list[str] = []
    # The device-resident buffers are the last-reported ones: llama-cli reserves
    # a scratch context before loading, which prints a 0.00 MiB set first.
    vram: dict[str, float] = {}

    text = path.read_text(errors="replace").splitlines()
    for lineno, line in enumerate(text, 1):
        def ev(field, value):
            rec[field] = value
            rec["evidence"][field] = {"line": lineno, "text": line.strip()}

        m = PATTERNS["pp_tok_s"].search(line)
        if m:
            ev("pp_ms", float(m.group(1)))
            ev("prompt_tokens", int(m.group(2)))
            ev("pp_tok_s", float(m.group(3)))
            continue
        m = PATTERNS["decode"].search(line)
        if m:
            ev("decode_ms", float(m.group(1)))
            ev("generated_tokens", int(m.group(2)))
            ev("decode_tok_s", float(m.group(3)))
            continue
        m = PATTERNS["total"].search(line)
        if m:
            ev("total_ms", float(m.group(1)))
            continue
        for key, field in (
            ("vram_model", "model"),
            ("vram_kv", "kv"),
            ("vram_rs", "rs"),
            ("vram_compute", "compute"),
        ):
            m = PATTERNS[key].search(line)
            if m:
                mib = float(m.group(2))
                # keep the largest observed per (buffer, backend) -- the
                # pre-load reservation prints 0.00 MiB for the same names
                k = f"{field}:{m.group(1)}"
                if mib >= vram.get(k, -1.0):
                    vram[k] = mib
                break
        m = PATTERNS["stats"].search(line)
        if m:
            ev("spec_impl", m.group(1))
            ev("rounds", int(m.group(4)))          # #calls accept == verification rounds
            ev("drafts_generated", int(m.group(5)))
            ev("drafts_accepted", int(m.group(6)))
            ev("draft_tokens_generated", int(m.group(7)))
            ev("draft_tokens_accepted", int(m.group(8)))
            ev("mean_acc_len_upstream", float(m.group(9)))
            pos = [float(x) for x in m.group(10).split(",") if x.strip()]
            ev("acc_rate_per_pos", pos)
            # index 0 = P(>=1 accepted) = p0; index 1 = P(>=2 accepted) = joint-p1
            rec["p0"] = pos[0] if len(pos) > 0 else None
            rec["joint_p1"] = pos[1] if len(pos) > 1 else None
            continue
        if "error" in line.lower() or "failed" in line.lower():
            unmatched.append(f"{lineno}: {line.strip()}")

    rec["vram_mib"] = vram
    rec["notable_lines"] = unmatched
    missing = [f for f in ("decode_tok_s", "pp_tok_s", "generated_tokens") if rec.get(f) is None]
    rec["missing_fields"] = missing
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--arm", help="label to attach to every record")
    args = ap.parse_args()

    problems = 0
    for p in sorted(args.logs):
        rec = parse_log(p)
        name = p.name
        rec["arm"] = args.arm or name.split(".")[0]
        rec["rep"] = name.split(".")[1] if "." in name else None
        if rec["missing_fields"]:
            problems += 1
            print(f"UNMATCHED in {p}: {rec['missing_fields']}", file=sys.stderr)
        print(json.dumps(rec))
    if problems:
        print(f"{problems} log(s) had unmatched required fields", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
