#!/usr/bin/env python
"""Submit a workflow to the isolated ComfyUI and report timing + VSA evidence.

Usage: run_workflow.py <workflow.json> [--port 8191] [--server-log PATH]

Fails loudly on silent dense fallback: if the graph contains an H3VSA node, the
server log MUST show "[H3-VSA] ACTIVE" and MUST NOT show a miss/fallback line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("workflow")
ap.add_argument("--port", type=int, default=8191)
ap.add_argument("--server-log", default="/ai/lab/experiments/fasth3-vsa/logs/server.log")
args = ap.parse_args()

BASE = f"http://127.0.0.1:{args.port}"


def api(path: str, payload=None):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


graph = json.load(open(args.workflow))
has_vsa = any(n.get("class_type") == "H3VSA" for n in graph.values())
try:
    log_mark = sum(1 for _ in open(args.server_log, errors="replace"))
except OSError:
    log_mark = 0

print(f"workflow : {args.workflow}")
print(f"H3VSA    : {'yes' if has_vsa else 'no (dense control)'}")
t0 = time.time()
resp = api("/prompt", {"prompt": graph})
pid = resp["prompt_id"]
print(f"prompt_id: {pid}\nrunning...", flush=True)

hist = None
while True:
    time.sleep(2)
    h = api(f"/history/{pid}")
    if pid in h and h[pid].get("status", {}).get("completed") is not None:
        hist = h[pid]
        break
    if time.time() - t0 > 3600:
        sys.exit("TIMEOUT after 3600 s")

wall = time.time() - t0
status = hist.get("status", {})
ok = status.get("status_str") == "success"

# ComfyUI's own execution_start -> execution_success boundary (v1 standard)
ts = {}
for m in status.get("messages", []):
    if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict) and "timestamp" in m[1]:
        ts.setdefault(m[0], m[1]["timestamp"])
exec_ms = None
if "execution_start" in ts and "execution_success" in ts:
    exec_ms = (ts["execution_success"] - ts["execution_start"]) / 1000.0

print("\n" + "=" * 62)
print("status              :", status.get("status_str"))
print("client wall         : %.3f s" % wall)
if exec_ms is not None:
    print("execution_start->success: %.3f s   <-- v1 benchmark boundary" % exec_ms)
for node_id, out in (hist.get("outputs") or {}).items():
    for vids in (out.get("images", []) + out.get("videos", []) + out.get("gifs", [])):
        print("artifact            :", vids.get("filename"))

# ---- VSA evidence from the server log ----
print("-" * 62)
lines = open(args.server_log, errors="replace").read().splitlines()[log_mark:]
vsa_lines = [l for l in lines if "[H3-VSA]" in l]
for l in vsa_lines[:20]:
    print("LOG:", l.strip())
if len(vsa_lines) > 20:
    print(f"LOG: ... {len(vsa_lines) - 20} more [H3-VSA] lines")

verdict_ok = ok
if has_vsa:
    active = [l for l in vsa_lines if "ACTIVE" in l]
    misses = [l for l in vsa_lines if re.search(r"fallback|falling back|not a MiniMaxH3Model|no gate", l, re.I)]
    print("-" * 62)
    print("VSA ACTIVE lines    :", len(active))
    print("VSA fallback lines  :", len(misses))
    if not active:
        print("!! FAIL: no '[H3-VSA] ACTIVE' -- attention silently ran DENSE")
        verdict_ok = False
    if misses:
        print("!! FAIL: fallback detected:")
        for m in misses[:5]:
            print("   ", m.strip())
        verdict_ok = False
    if active and not misses:
        print("OK: VSA active on every reported call, no dense fallback")

print("=" * 62)
print("VERDICT:", "PASS" if verdict_ok else "FAIL")
sys.exit(0 if verdict_ok else 1)
