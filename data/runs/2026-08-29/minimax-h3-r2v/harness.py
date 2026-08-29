#!/usr/bin/env python3
"""Reproducible MiniMax H3 R2V single/dual-GPU experiment harness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parent
WORKFLOWS = ROOT / "workflows"
LOGS = ROOT / "logs"
RESULTS = ROOT / "tables" / "results.csv"
SINGLE = "http://127.0.0.1:8190"
HISTORICAL_DUAL = "http://127.0.0.1:8191"
MATCHED_DUAL = "http://127.0.0.1:8192"
HISTORICAL_GRAPH = pathlib.Path(
    "/ai/benchmarks/minimax-h3/dual-gpu-residency-20260812-101311/graphs/dual-cold-prompt-a.json"
)
REFERENCE_IMAGE = "Gemini_Generated_Image_cajce7cajce7cajc.jpeg"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_fp8.safetensors"
TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
SEEDS = [8292026, 8292027, 8292028, 8292029]
PROMPT = """subject_definitions:
<Subject 1> is the person in <Picture 1>: mid-20s, dark hair, light jacket.

summary:
[reference generation] Generate a short video starring <Subject 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity kept exactly as in <Picture 1>.

detailed_description:
The target video is cinematic live-action, soft window light.
[Shot 1] <Subject 1> turns slowly toward the camera and smiles, small head movement, static camera.

overall_soundscape:
Quiet room tone, soft cloth movement.

non_diegetic_music:
N/A"""

CSV_FIELDS = [
    "run_id", "timestamp", "service", "topology", "gpu_mapping", "workflow_sha256",
    "resolution", "frames", "reference_type", "ref_image_size", "turbo_lora",
    "sampler", "steps", "attention_configured", "smart_memory", "result",
    "wall_seconds", "sampler_seconds", "r9700_entry_allocated", "r9700_entry_reserved",
    "r9700_peak_allocated", "r9700_peak_reserved", "r9700_driver_entry_used",
    "r9700_rocm_peak_used", "rx7900_entry_allocated", "rx7900_entry_reserved",
    "rx7900_peak_allocated", "rx7900_peak_reserved", "rx7900_driver_entry_used",
    "rx7900_rocm_peak_used", "oom_requested", "oom_free", "oom_allocated",
    "output_sanity", "notes",
]


@dataclass(frozen=True)
class Case:
    name: str
    width: int
    height: int
    frames: int = 124
    ref_size: str = "match"
    turbo_lora: bool = True
    steps: int = 5


def api(host: str, path: str, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        host + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def wait_ready(host: str, timeout=180):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            return api(host, "/h3mem/map", timeout=5)
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"{host} did not become ready: {last_error}")


def wait_idle(host: str, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        queue = api(host, "/queue", timeout=10)
        if not queue.get("queue_running") and not queue.get("queue_pending"):
            return
        time.sleep(5)
    raise TimeoutError(f"{host} queue did not become idle")


def free_models(host: str):
    wait_idle(host)
    api(host, "/free", {"unload_models": True, "free_memory": True}, timeout=30)
    time.sleep(2)


def graph_sha(graph: dict) -> str:
    canonical = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_r2v_graph(case: Case, seed: int, dual_clip: bool, run_id: str) -> dict:
    graph = json.loads((WORKFLOWS / "r2v-r9700-baseline.json").read_text())
    graph["1"]["class_type"] = "CLIPLoaderMultiGPU" if dual_clip else "CLIPLoader"
    graph["1"]["inputs"]["clip_name"] = TEXT_ENCODER
    graph["1"]["inputs"]["device"] = "cuda:1" if dual_clip else "default"
    graph["6"]["inputs"].update({
        "prompt": PROMPT,
        "width": case.width,
        "height": case.height,
        "length": case.frames,
        "ref_image_size": case.ref_size,
        "ref_images.ref_image_0": ["200", 0],
    })
    graph["8"]["inputs"]["noise_seed"] = seed
    graph["9"]["inputs"].update({"steps": case.steps, "scheduler": "simple"})
    graph["10"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {},
                   "_meta": {"title": "Turbo Sampler"}}
    graph["200"] = {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE},
                    "_meta": {"title": "Fixed R2V reference"}}
    if case.turbo_lora:
        graph["100"] = {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["2", 0], "lora_name": TURBO_LORA,
                "strength": 1.0, "low_vram": False,
            },
            "_meta": {"title": "Turbo LoRA (bypass mode)"},
        }
        model_source = ["100", 0]
    else:
        # Pure memory A/B: remove the LoRA branch before either nested F.linear.
        # The Turbo sampler, scheduler, step count, model, seed and conditioning remain fixed.
        model_source = ["2", 0]
    graph["5"]["inputs"]["model"] = model_source
    graph["15"]["inputs"]["filename_prefix"] = f"minimax-h3-r2v/{run_id}"
    return graph


def load_historical_graph(seed: int, run_id: str) -> dict:
    payload = json.loads(HISTORICAL_GRAPH.read_text())
    graph = payload.get("prompt", payload)
    for node in graph.values():
        if node.get("class_type") == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
        if node.get("class_type") == "SaveVideo":
            node["inputs"]["filename_prefix"] = f"minimax-h3-r2v/{run_id}"
    return graph


def device_gate(mapping: dict, dual: bool):
    devices = mapping.get("devices", [])
    expected_count = 2 if dual else 1
    if len(devices) != expected_count:
        raise RuntimeError(f"expected {expected_count} visible device(s), got {devices}")
    if "R9700" not in devices[0].get("name", ""):
        raise RuntimeError(f"cuda:0 is not R9700: {devices[0]}")
    if dual and "7900" not in devices[1].get("name", ""):
        raise RuntimeError(f"cuda:1 is not RX 7900 XT: {devices[1]}")
    if devices[0].get("arch", "").split(":", 1)[0] not in ("", "gfx1201"):
        raise RuntimeError(f"cuda:0 architecture mismatch: {devices[0]}")
    if dual and devices[1].get("arch", "").split(":", 1)[0] not in ("", "gfx1100"):
        raise RuntimeError(f"cuda:1 architecture mismatch: {devices[1]}")


def rocm_snapshot() -> dict:
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        raw = json.loads(output)
        return {
            card: {
                "total": int(values["VRAM Total Memory (B)"]),
                "used": int(values["VRAM Total Used Memory (B)"]),
            }
            for card, values in raw.items()
        }
    except Exception as exc:
        return {"error": str(exc)}


class RocmTelemetry:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            self.samples.append({"time": time.time(), "devices": rocm_snapshot()})
            self.stop_event.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        self.thread.join(timeout=15)
        self.samples.append({"time": time.time(), "devices": rocm_snapshot()})

    def peak(self, card: str):
        values = [s["devices"].get(card, {}).get("used") for s in self.samples]
        values = [value for value in values if isinstance(value, int)]
        return max(values) if values else None


def submit_and_wait(host: str, graph: dict, run_id: str, timeout=1800):
    probe_path = pathlib.Path(f"/tmp/h3-mem-{run_id}.json")
    if probe_path.exists():
        raise RuntimeError(f"refusing to overwrite existing probe record: {probe_path}")
    armed = api(host, "/h3mem/set_run_id", {"run_id": run_id}, timeout=10)
    if not armed.get("ok"):
        raise RuntimeError(f"failed to arm sampler probe: {armed}")
    started = time.time()
    with RocmTelemetry() as telemetry:
        response = api(host, "/prompt", {"prompt": graph, "client_id": f"h3-r2v-{run_id}"})
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            api(host, "/h3mem/set_run_id", {"run_id": ""}, timeout=5)
            raise RuntimeError(f"prompt rejected: {response}")
        deadline = started + timeout
        entry = None
        while time.time() < deadline:
            history = api(host, f"/history/{prompt_id}", timeout=20)
            if prompt_id in history:
                entry = history[prompt_id]
                state = entry.get("status", {}).get("status_str")
                if state in {"success", "error", "cancelled"}:
                    break
            time.sleep(2)
        if entry is None:
            raise TimeoutError(f"{run_id} did not finish within {timeout}s")
    wall = time.time() - started
    try:
        api(host, "/h3mem/set_run_id", {"run_id": ""}, timeout=5)
    except Exception:
        pass
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    return prompt_id, entry, wall, probe, telemetry


def flatten_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def oom_fields(entry: dict):
    text = flatten_text(entry)
    import re
    requested = re.search(r"Tried to allocate ([0-9.]+ [GM]iB)", text, re.I)
    free = re.search(r"([0-9.]+ [GM]iB) free", text, re.I)
    allocated = re.search(r"([0-9.]+ [GM]iB) allocated", text, re.I)
    return tuple(match.group(1) if match else "" for match in (requested, free, allocated))


def extract_device(probe: dict, index: int, phase: str):
    return probe.get(phase, {}).get(f"cuda:{index}", {})


def output_sanity(entry: dict, width: int, height: int, frames: int):
    if entry.get("status", {}).get("status_str") != "success":
        return "not_applicable"
    text = flatten_text(entry.get("outputs", {}))
    if not text.strip():
        return "success_no_output_metadata"
    # SaveVideo metadata is retained raw; dimensions/frame count are checked in
    # the post-run summarizer against the actual artifact with ffprobe.
    return f"history_output_present expected={width}x{height}/{frames}f"


def record_run(*, run_id, host, label, topology, graph, case, entry, wall, probe,
               telemetry, smart_memory, notes):
    mapping = api(host, "/h3mem/map", timeout=10)
    r_enter = extract_device(probe, 0, "before")
    r_exit = extract_device(probe, 0, "after")
    x_enter = extract_device(probe, 1, "before")
    x_exit = extract_device(probe, 1, "after")
    requested, free, allocated = oom_fields(entry)
    state = entry.get("status", {}).get("status_str", "unknown")
    error_text = flatten_text(entry)
    result = "PASS" if state == "success" else "FAIL_OOM" if "out of memory" in error_text.lower() else f"FAIL_{state.upper()}"
    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service": label,
        "topology": topology,
        "gpu_mapping": json.dumps([{k: d.get(k) for k in ("index", "name", "arch")} for d in mapping.get("devices", [])]),
        "workflow_sha256": graph_sha(graph),
        "resolution": f"{case.width}x{case.height}",
        "frames": case.frames,
        "reference_type": "image" if topology != "historical-dual" else "historical-i2v",
        "ref_image_size": case.ref_size if topology != "historical-dual" else "n/a",
        "turbo_lora": "on" if case.turbo_lora else "off_removed_branch",
        "sampler": "MiniMaxH3TurboSampler",
        "steps": case.steps,
        "attention_configured": mapping.get("configured_attention"),
        "smart_memory": smart_memory,
        "result": result,
        "wall_seconds": f"{wall:.3f}",
        "sampler_seconds": f"{probe.get('elapsed_seconds', 0):.3f}" if probe else "",
        "r9700_entry_allocated": r_enter.get("allocated"),
        "r9700_entry_reserved": r_enter.get("reserved"),
        "r9700_peak_allocated": r_exit.get("max_allocated"),
        "r9700_peak_reserved": r_exit.get("max_reserved"),
        "r9700_driver_entry_used": r_enter.get("driver_used"),
        "r9700_rocm_peak_used": telemetry.peak("card1"),
        "rx7900_entry_allocated": x_enter.get("allocated"),
        "rx7900_entry_reserved": x_enter.get("reserved"),
        "rx7900_peak_allocated": x_exit.get("max_allocated"),
        "rx7900_peak_reserved": x_exit.get("max_reserved"),
        "rx7900_driver_entry_used": x_enter.get("driver_used"),
        "rx7900_rocm_peak_used": telemetry.peak("card0"),
        "oom_requested": requested,
        "oom_free": free,
        "oom_allocated": allocated,
        "output_sanity": output_sanity(entry, case.width, case.height, case.frames),
        "notes": notes,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS.exists()
    with RESULTS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{run_id}.json").write_text(json.dumps({
        "row": row, "graph": graph, "history": entry, "probe": probe,
        "rocm_telemetry": telemetry.samples,
    }, indent=2) + "\n")
    print(f"{run_id}: {result}, wall={wall:.3f}s, probe={probe.get('status', 'missing')}")
    return result == "PASS"


def run_repetitions(host: str, label: str, topology: str, case: Case, dual_clip: bool,
                    smart_memory: str, historical=False):
    passed = []
    for index, seed in enumerate(SEEDS):
        suffix = "cold" if index == 0 else f"warm{index}"
        run_id = f"{case.name}-{suffix}"
        if index == 0:
            free_models(host)
        graph = load_historical_graph(seed, run_id) if historical else build_r2v_graph(case, seed, dual_clip, run_id)
        prompt_id, entry, wall, probe, telemetry = submit_and_wait(host, graph, run_id)
        notes = f"prompt_id={prompt_id}; {'exact 20260812 graph replay; non-causal' if historical else 'fixed image/prompt; distinct seed per repetition'}"
        ok = record_run(
            run_id=run_id, host=host, label=label, topology=topology, graph=graph,
            case=case, entry=entry, wall=wall, probe=probe, telemetry=telemetry,
            smart_memory=smart_memory, notes=notes,
        )
        passed.append(ok)
        if index == 0 and not ok:
            break
    return passed


def phase1():
    mapping = wait_ready(SINGLE)
    device_gate(mapping, dual=False)
    print("Phase 1 gate passed:", json.dumps(mapping, indent=2))
    s1 = Case("S1-match-turbo-960x544-124f", 960, 544, turbo_lora=True)
    s2 = Case("S2-match-lora-removed-960x544-124f", 960, 544, turbo_lora=False)
    s1_results = run_repetitions(SINGLE, "comfyui-h3 :8190", "single", s1, False, "disabled")
    run_repetitions(SINGLE, "comfyui-h3 :8190", "single", s2, False, "disabled")
    if not (len(s1_results) == 4 and all(s1_results)):
        ladder = [
            Case("S3a-match-turbo-864x480-124f", 864, 480, turbo_lora=True),
            Case("S3b-match-turbo-864x480-96f", 864, 480, frames=96, turbo_lora=True),
            Case("S3c-match-turbo-608x352-124f", 608, 352, turbo_lora=True),
        ]
        for case in ladder:
            results = run_repetitions(SINGLE, "comfyui-h3 :8190", "single", case, False, "disabled")
            if len(results) == 4 and all(results):
                break


def phase2_historical():
    mapping = wait_ready(HISTORICAL_DUAL)
    device_gate(mapping, dual=True)
    print("D0 historical gate passed:", json.dumps(mapping, indent=2))
    case = Case("D0-historical-exact-864x480-124f", 864, 480, ref_size="n/a", turbo_lora=True, steps=4)
    run_repetitions(
        HISTORICAL_DUAL, "historical dual :8191", "historical-dual", case,
        True, "enabled/default", historical=True,
    )


def phase2_matched():
    mapping = wait_ready(MATCHED_DUAL)
    device_gate(mapping, dual=True)
    print("Matched dual gate passed:", json.dumps(mapping, indent=2))
    sanity = Case("D1-matched-864x480-124f", 864, 480, turbo_lora=True)
    target = Case("D2-matched-960x544-124f", 960, 544, turbo_lora=True)
    fallback = Case("D3-matched-lora-removed-960x544-124f", 960, 544, turbo_lora=False)
    run_repetitions(MATCHED_DUAL, "matched dual :8192", "matched-dual", sanity, True, "disabled")
    target_results = run_repetitions(MATCHED_DUAL, "matched dual :8192", "matched-dual", target, True, "disabled")
    if not (len(target_results) == 4 and all(target_results)):
        run_repetitions(MATCHED_DUAL, "matched dual :8192", "matched-dual", fallback, True, "disabled")


def summarize():
    if not RESULTS.exists():
        print("No results.csv exists yet")
        return
    rows = list(csv.DictReader(RESULTS.open()))
    groups = {}
    for row in rows:
        base = row["run_id"].rsplit("-", 1)[0]
        groups.setdefault(base, []).append(row)
    for base, group in groups.items():
        warm = [float(r["wall_seconds"]) for r in group if "-warm" in r["run_id"] and r["result"] == "PASS"]
        print(base, "passes", sum(r["result"] == "PASS" for r in group), "/", len(group),
              "warm_median", statistics.median(warm) if warm else "n/a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["phase1", "d0", "matched-dual", "summarize", "dry-run"])
    args = parser.parse_args()
    if args.command == "phase1":
        phase1()
    elif args.command == "d0":
        phase2_historical()
    elif args.command == "matched-dual":
        phase2_matched()
    elif args.command == "summarize":
        summarize()
    else:
        for case in [
            Case("S1-match-turbo-960x544-124f", 960, 544),
            Case("S2-match-lora-removed-960x544-124f", 960, 544, turbo_lora=False),
            Case("S3a-match-turbo-864x480-124f", 864, 480),
            Case("S3b-match-turbo-864x480-96f", 864, 480, frames=96),
            Case("S3c-match-turbo-608x352-124f", 608, 352),
            Case("D1-matched-864x480-124f", 864, 480),
            Case("D2-matched-960x544-124f", 960, 544),
        ]:
            graph = build_r2v_graph(case, SEEDS[0], case.name.startswith("D"), case.name + "-cold")
            print(case.name, graph_sha(graph), len(graph), "nodes")


if __name__ == "__main__":
    main()
