#!/usr/bin/env python3
"""Extract benchmark metrics from raw llama.cpp / ROCmFPX logs, with provenance.

Design rule, from shared/benchmark-contract.md: a number with no identifiable
source line does not enter a run record. Every value this script emits carries
the file, line number, and verbatim text it was parsed from. Anything it does
not recognise is reported as unmatched -- never inferred, never defaulted.

Usage:
    05_extract_basic_metrics.py LOG [LOG ...]            # human-readable
    05_extract_basic_metrics.py --json LOG [LOG ...]     # JSONL, one per file
    05_extract_basic_metrics.py --strict LOG             # exit 1 if nothing matched

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Each pattern must anchor on text the tool actually prints. When a tool's
# output format changes, the pattern stops matching and the metric is reported
# as absent -- which is the correct failure mode. Do not add loose patterns
# that would match several different quantities.
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # llama-bench markdown table rows, e.g. "| pp512 | 1234.56 ± 7.89 |"
    ("pp_tok_s",
     re.compile(r"\|\s*pp\d+\s*\|.*?\|\s*([0-9]+\.[0-9]+)\s*±\s*([0-9]+\.[0-9]+)\s*\|"),
     "llama-bench table row (pp)"),
    ("decode_tok_s",
     re.compile(r"\|\s*tg\d+\s*\|.*?\|\s*([0-9]+\.[0-9]+)\s*±\s*([0-9]+\.[0-9]+)\s*\|"),
     "llama-bench table row (tg)"),

    # llama-cli / llama-completion timing block
    ("pp_tok_s",
     re.compile(r"prompt eval time =.*?,\s*([0-9]+\.[0-9]+)\s*tokens per second"),
     "llama.cpp timing: prompt eval"),
    ("decode_tok_s",
     re.compile(r"^\s*eval time =.*?,\s*([0-9]+\.[0-9]+)\s*tokens per second"),
     "llama.cpp timing: eval"),

    # Speculative / MTP counters
    ("draft_accepted",
     re.compile(r"n_drafted\s*=\s*\d+.*?n_accept\s*=\s*(\d+)"),
     "speculative summary: n_accept"),
    ("draft_generated",
     re.compile(r"n_drafted\s*=\s*(\d+)"),
     "speculative summary: n_drafted"),
    ("acceptance_rate",
     re.compile(r"accept(?:ance)?\s*(?:rate)?\s*=\s*([0-9]+\.[0-9]+)"),
     "speculative summary: acceptance"),

    # VRAM
    ("vram_mib",
     re.compile(r"(?:VRAM|buffer size)\s*[=:]\s*([0-9]+\.?[0-9]*)\s*MiB"),
     "backend buffer report"),
]


@dataclass
class Extracted:
    metric: str
    value: float
    stdev: float | None
    source_file: str
    source_line_no: int
    source_line: str
    pattern_label: str
    classification: str = "MEASURED"


def extract(path: Path) -> tuple[list[Extracted], int]:
    """Return (matches, total_lines). Never raises on malformed content."""
    results: list[Extracted] = []
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return [], 0

    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for metric, pattern, label in PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            try:
                value = float(m.group(1))
            except (ValueError, IndexError):
                continue
            stdev = None
            if pattern.groups >= 2:
                try:
                    stdev = float(m.group(2))
                except (ValueError, IndexError, TypeError):
                    stdev = None
            results.append(Extracted(
                metric=metric,
                value=value,
                stdev=stdev,
                source_file=str(path),
                source_line_no=line_no,
                source_line=line.strip(),
                pattern_label=label,
            ))
    return results, len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true",
                    help="emit JSONL, one object per input file")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any input file yielded no metrics")
    args = ap.parse_args()

    any_empty = False

    for path in args.logs:
        matches, total_lines = extract(path)
        if not matches:
            any_empty = True

        if args.json:
            print(json.dumps({
                "source_file": str(path),
                "lines_scanned": total_lines,
                "metrics": [asdict(m) for m in matches],
                "matched": len(matches),
                # Explicit: absence of a metric means "not found", not "zero".
                "note": ("no recognised metric lines found" if not matches
                         else "values carry source-line provenance"),
            }))
            continue

        print(f"--- {path} ({total_lines} lines) ---")
        if not matches:
            print("  no recognised metric lines found")
            print("  (this is reported, not guessed around -- check the log format)")
            continue
        for m in matches:
            stdev = f" ± {m.stdev}" if m.stdev is not None else ""
            print(f"  {m.metric} = {m.value}{stdev}   [{m.classification}]")
            print(f"    from {m.source_file}:{m.source_line_no}  ({m.pattern_label})")
            print(f"    > {m.source_line}")

    if args.strict and any_empty:
        print("strict: at least one input yielded no metrics", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
