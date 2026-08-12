#!/usr/bin/env python3
"""Small dependency-free integrity and data-quality check for this repository."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
WORKFLOWS = {
    "workflows/minimax-h3/MiniMax-H3-Native-FP8.json": "23353acceadd769c352bc5a2fd367712ca448de1505c0946d570cd4d7d10b277",
    "workflows/minimax-h3/MiniMax-H3-Turbo-v4-FP8.json": "da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f",
    "workflows/ltx-2.5/LTX-2.5-Brass-Robot-Comparison.json": "db11db4591a280c5e668882084a3b40f62c0d1287fe841541deaae7e7736a4bf",
    "workflows/minimax-h3/minimax-h3-native-fp8.json": "23353acceadd769c352bc5a2fd367712ca448de1505c0946d570cd4d7d10b277",
    "workflows/minimax-h3/minimax-h3-turbo-v4-fp8.json": "da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f",
    "workflows/minimax-h3/minimax-h3-quality-native-fp8-20-api.json": "66d8ea1c62d5d25b6bbdca2390fd1c3f393505af9f83ea29c7a127a216155423",
    "workflows/minimax-h3/MiniMax-H3-Turbo-v4-FP8-DualGPU-Qwen-on-7900XT.json": "70754239ca071fae7e3c2df4a07c6991c32dd3a451f10a41b9cee1ada4e64d06",
    "workflows/minimax-h3/h3-dualgpu-shakedown-5s-turbo-v4-api.json": "1dd5380b0a44c0d3a1067cda6905b4ef18972f72c7c58696d493d2c00675fedf",
    "workflows/minimax-h3/minimax-h3-turbo-v4-4-control-api.json": "64b57c6b8322bd55c183543eb022895e55cdfebd89095b97556f046f319db262",
}
REQUIRED_RESULTS = {
    "run_id", "date", "model", "lane", "model_repository", "model_revision",
    "model_file", "model_sha256", "quantization", "prompt_id", "seed",
    "cold_state", "startup_seconds", "prompt_to_artifact_seconds",
    "restart_to_artifact_seconds", "width", "height", "frames", "fps",
    "duration_seconds", "wall_seconds_per_output_second", "sampler", "scheduler",
    "steps", "native_audio", "workflow_path", "workflow_sha256",
    "artifact_source_path", "artifact_sha256", "validation_status",
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SECRET_PATTERNS = [
    re.compile(rb"/home/boxwrench"),
    re.compile(rb"LICENSE-AUTHORIZATION"),
    re.compile(rb"-----BEGIN [^-]+ PRIVATE KEY-----"),
    re.compile(rb"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"\bsk-[A-Za-z0-9]{20,}"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_workflows(errors: list[str]) -> None:
    for rel, expected in WORKFLOWS.items():
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing workflow: {rel}")
            continue
        if sha256(path) != expected:
            errors.append(f"workflow hash mismatch: {rel}")
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            errors.append(f"workflow JSON parse failed: {rel}: {exc}")


def check_tsv_files(errors: list[str]) -> None:
    for path in ROOT.rglob("*.tsv"):
        if ".git" in path.parts:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        if not rows:
            errors.append(f"empty TSV: {path.relative_to(ROOT)}")
            continue
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            errors.append(f"inconsistent TSV column count: {path.relative_to(ROOT)}")


def check_results(errors: list[str]) -> None:
    path = ROOT / "data/results.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        errors.append("data/results.tsv is empty")
        return
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        errors.append("data/results.tsv has inconsistent column counts")
        return
    header = rows[0]
    missing = REQUIRED_RESULTS.difference(header)
    if missing:
        errors.append(f"data/results.tsv missing columns: {', '.join(sorted(missing))}")
    for number, row in enumerate(rows[1:], start=2):
        record = dict(zip(header, row))
        try:
            prompt = float(record["prompt_to_artifact_seconds"])
            duration = float(record["duration_seconds"])
            reported = float(record["wall_seconds_per_output_second"])
            startup = float(record["startup_seconds"])
            restart = float(record["restart_to_artifact_seconds"])
            if abs(prompt / duration - reported) > 0.02:
                errors.append(f"results row {number}: derived ratio mismatch")
            if abs(startup + prompt - restart) > 0.02:
                errors.append(f"results row {number}: restart time mismatch")
            for field in ("seed", "width", "height", "frames", "fps"):
                int(record[field])
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"results row {number}: invalid numeric field: {exc}")


def check_links(errors: list[str]) -> None:
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in MARKDOWN_LINK.findall(text):
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "/")):
                continue
            if not (path.parent / target).exists():
                errors.append(f"broken relative Markdown link: {path.relative_to(ROOT)} -> {target}")


def check_safety(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {path.relative_to(ROOT)}")
        data = path.read_bytes()
        if path.name == "verify.py":
            continue
        if path.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".mkv"}:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                errors.append(f"sensitive-content pattern in {path.relative_to(ROOT)}: {pattern.pattern.decode(errors='replace')}")


def main() -> int:
    errors: list[str] = []
    check_workflows(errors)
    check_tsv_files(errors)
    check_results(errors)
    check_links(errors)
    check_safety(errors)
    if errors:
        print("verification failed")
        print("\n".join(f"- {item}" for item in errors))
        return 1
    print(f"verification passed: {len(WORKFLOWS)} JSON workflows, normalized TSV, links, size, and safety scans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
