#!/usr/bin/env python
"""Matched contact sheet: same timestamps across configurations, adjacent.

Layout: one ROW per timestamp, one COLUMN per configuration, so the same moment
of video sits side by side across configs. Frames are extracted at identical
wall-clock timestamps from every input.

Usage:
  contact_sheet.py -o sheet.png -t 0.25,1.5,2.75,4.0 LABEL=video.mp4 [LABEL=video.mp4 ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-t", "--timestamps", default="0.25,1.5,2.75,4.0")
ap.add_argument("--width", type=int, default=432, help="per-tile width")
ap.add_argument("inputs", nargs="+", help="LABEL=path.mp4")
args = ap.parse_args()

times = [float(x) for x in args.timestamps.split(",")]
pairs = []
for spec in args.inputs:
    if "=" not in spec:
        sys.exit(f"expected LABEL=path, got {spec!r}")
    label, path = spec.split("=", 1)
    if not Path(path).exists():
        sys.exit(f"missing: {path}")
    pairs.append((label, path))


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


HDR, LBL = 46, 34
f_hdr, f_lbl = font(24), font(18)

with tempfile.TemporaryDirectory() as td:
    grid: list[list[Image.Image]] = []
    for ti, t in enumerate(times):
        row = []
        for ci, (label, path) in enumerate(pairs):
            out = f"{td}/{ti}_{ci}.png"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", str(t), "-i", path,
                 "-frames:v", "1", "-y", out],
                check=True)
            im = Image.open(out).convert("RGB")
            w = args.width
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            row.append(im)
        grid.append(row)

    tw, th = grid[0][0].size
    sheet = Image.new("RGB",
                      (tw * len(pairs), HDR + (th + LBL) * len(times)),
                      (18, 18, 20))
    d = ImageDraw.Draw(sheet)

    for ci, (label, _) in enumerate(pairs):
        d.text((ci * tw + 10, 12), label, fill=(255, 255, 255), font=f_hdr)

    for ti, t in enumerate(times):
        y = HDR + ti * (th + LBL)
        d.text((10, y + 8), f"t = {t:.2f} s", fill=(150, 200, 255), font=f_lbl)
        for ci, im in enumerate(grid[ti]):
            sheet.paste(im, (ci * tw, y + LBL))

    sheet.save(args.output)
    print(f"wrote {args.output}  ({sheet.width}x{sheet.height}, "
          f"{len(times)} timestamps x {len(pairs)} configs)")
