#!/usr/bin/env python3
"""Controlled single-vs-dual GPU MiniMax H3 Turbo engineering benchmark."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


PROMPT_A = (
    "A tiny brass robot carefully waters a glowing mushroom garden at night. "
    "Gentle rain, soft mechanical movements, warm lantern light. Audio: quiet "
    "rain, small servo sounds, and a soft bell chime. No text or logos."
)
PROMPT_B = (
    "A tiny brass robot carefully polishes a glowing crystal garden at dawn. "
    "Gentle mist, soft mechanical movements, cool window light. Audio: quiet "
    "wind, small servo sounds, and a soft bell chime. No text or logos."
)

SINGLE_GRAPH = Path(
    "/ai/benchmarks/minimax-h3/workflows/deferred/"
    "minimax-h3-quality-turbo-v4-4-api.NEVER-SUBMITTED.json"
)
DUAL_GRAPH = Path(
    "/ai/lab/experiments/minimax-h3/dual-gpu/workflows/"
    "h3-dualgpu-shakedown-5s-turbo-v4-api.json"
)
SINGLE_OUTPUT = Path("/ai/artifacts/runs/minimax-h3")
DUAL_OUTPUT = Path("/ai/artifacts/runs/minimax-h3-dualgpu")
AMD_SMI = "/opt/rocm/bin/amd-smi"
RUN_ID = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
RESULT_ROOT = Path(f"/ai/benchmarks/minimax-h3/dual-gpu-residency-{RUN_ID}")

EXPECTED_HASHES = {
    Path("/ai/comfyui/user/default/workflows/MiniMax-H3-Turbo-v4-FP8.json"):
        "da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f",
    Path("/ai/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"):
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    Path("/ai/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"):
        "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993",
    Path("/ai/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors"):
        "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
}

TOPOLOGIES = [
    {
        "name": "single",
        "unit": "comfyui-h3.service",
        "other_unit": "comfyui-h3-dualgpu.service",
        "port": 8190,
        "graph": SINGLE_GRAPH,
        "output_root": SINGLE_OUTPUT,
        "expected_devices": 1,
    },
    {
        "name": "dual",
        "unit": "comfyui-h3-dualgpu.service",
        "other_unit": "comfyui-h3.service",
        "port": 8191,
        "graph": DUAL_GRAPH,
        "output_root": DUAL_OUTPUT,
        "expected_devices": 2,
    },
]

RUNS = [
    ("cold-prompt-a", PROMPT_A, 8112026),
    ("same-prompt-new-seed", PROMPT_A, 8112027),
    ("changed-prompt-b", PROMPT_B, 8112027),
]


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def epoch_ms() -> int:
    return time.time_ns() // 1_000_000


def run_cmd(args: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True, timeout=timeout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_http(port: int, should_be_up: bool, timeout: int = 300) -> dict | None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = http_json(f"http://127.0.0.1:{port}/system_stats", timeout=3)
            if should_be_up:
                return last
        except Exception:
            if not should_be_up:
                return None
        time.sleep(1)
    state = "up" if should_be_up else "down"
    raise RuntimeError(f"port {port} did not become {state}; last={last}")


def normalize_graph(graph: dict) -> dict:
    normalized = copy.deepcopy(graph["prompt"])
    normalized["2"]["class_type"] = "ENCODER_LOADER"
    normalized["2"]["inputs"].pop("device", None)
    normalized["14"]["inputs"]["filename_prefix"] = "OUTPUT_PREFIX"
    return normalized


def preflight() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    (RESULT_ROOT / "runs").mkdir()
    (RESULT_ROOT / "telemetry").mkdir()
    (RESULT_ROOT / "journals").mkdir()
    (RESULT_ROOT / "graphs").mkdir()

    hashes = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        hashes[str(path)] = actual
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {path}: {actual} != {expected}")

    single = json.loads(SINGLE_GRAPH.read_text())
    dual = json.loads(DUAL_GRAPH.read_text())
    if normalize_graph(single) != normalize_graph(dual):
        raise RuntimeError("single/dual graphs differ beyond encoder loader/device/output prefix")

    for graph in (single, dual):
        prompt = graph["prompt"]
        assert prompt["5"]["inputs"]["width"] == 864
        assert prompt["5"]["inputs"]["height"] == 480
        assert prompt["5"]["inputs"]["length"] == 124
        assert prompt["8"]["inputs"]["steps"] == 4
        assert prompt["13"]["inputs"]["fps"] == 24.0
        assert prompt["15"]["inputs"]["strength"] == 1.0

    inventory = json.loads(run_cmd([AMD_SMI, "list", "--json"], timeout=20).stdout)
    bdfs = {item["gpu"]: item["bdf"] for item in inventory}
    if bdfs != {0: "0000:03:00.0", 1: "0000:06:00.0"}:
        raise RuntimeError(f"unexpected physical GPU inventory: {bdfs}")

    record = {
        "run_id": RUN_ID,
        "created": iso_now(),
        "classification": "engineering decision run; not publication-grade",
        "prompts": {"A": PROMPT_A, "B": PROMPT_B},
        "seeds": {"cold": 8112026, "same_prompt": 8112027, "changed_prompt": 8112027},
        "hashes": hashes,
        "single_graph_sha256": sha256(SINGLE_GRAPH),
        "dual_graph_sha256": sha256(DUAL_GRAPH),
        "physical_gpu_inventory": inventory,
    }
    (RESULT_ROOT / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")


class Telemetry:
    def __init__(self, topology: str):
        self.topology = topology
        self.path = RESULT_ROOT / "telemetry" / f"{topology}.jsonl"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _loop(self) -> None:
        with self.path.open("a", buffering=1) as out:
            while not self.stop_event.is_set():
                sample = {"timestamp": iso_now(), "epoch_ms": epoch_ms(), "topology": self.topology}
                try:
                    result = run_cmd([AMD_SMI, "metric", "--gpu", "all", "--json"], timeout=10)
                    raw = json.loads(result.stdout)
                    selected = []
                    for gpu in raw.get("gpu_data", []):
                        selected.append({
                            "gpu": gpu.get("gpu"),
                            "gfx_activity_percent": gpu.get("usage", {}).get("gfx_activity", {}).get("value"),
                            "power_w": gpu.get("power", {}).get("socket_power", gpu.get("power", {})).get("value")
                                if isinstance(gpu.get("power", {}), dict) else None,
                            "temp_edge_c": gpu.get("temperature", {}).get("edge", {}).get("value"),
                            "temp_hotspot_c": gpu.get("temperature", {}).get("hotspot", {}).get("value"),
                            "temp_mem_c": gpu.get("temperature", {}).get("mem", {}).get("value"),
                            "vram_used_mb": gpu.get("mem_usage", {}).get("used_vram", {}).get("value"),
                            "vram_total_mb": gpu.get("mem_usage", {}).get("total_vram", {}).get("value"),
                            "gtt_used_mb": gpu.get("mem_usage", {}).get("used_gtt", {}).get("value"),
                        })
                    sample["gpus"] = selected
                except Exception as exc:
                    sample["amd_smi_error"] = repr(exc)
                try:
                    meminfo = {}
                    for line in Path("/proc/meminfo").read_text().splitlines():
                        key, value = line.split(":", 1)
                        meminfo[key] = int(value.strip().split()[0])
                    sample["host_mem_available_kb"] = meminfo["MemAvailable"]
                    sample["host_mem_total_kb"] = meminfo["MemTotal"]
                except Exception as exc:
                    sample["meminfo_error"] = repr(exc)
                out.write(json.dumps(sample, separators=(",", ":")) + "\n")
                self.stop_event.wait(1)


def extract_artifact(history_item: dict, output_root: Path) -> Path:
    found = []
    def walk(value):
        if isinstance(value, dict):
            if "filename" in value:
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(history_item.get("outputs", {}))
    if len(found) != 1:
        raise RuntimeError(f"expected one artifact, found {found}")
    item = found[0]
    artifact = output_root / item.get("subfolder", "") / item["filename"]
    if not artifact.is_file():
        raise RuntimeError(f"artifact missing: {artifact}")
    return artifact


def validate_artifact(artifact: Path) -> dict:
    probe = json.loads(run_cmd([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_name,codec_type,width,height,"
        "r_frame_rate,avg_frame_rate,nb_frames,duration,channels,sample_rate",
        "-of", "json", str(artifact),
    ], timeout=30).stdout)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    if not (
        video["codec_name"] == "h264" and video["width"] == 864 and video["height"] == 480
        and video["r_frame_rate"] == "24/1" and int(video["nb_frames"]) == 124
        and audio["codec_name"] == "aac" and int(audio["channels"]) == 2
    ):
        raise RuntimeError(f"media validation failed: {probe}")
    run_cmd(["ffmpeg", "-v", "error", "-i", str(artifact), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], timeout=180)
    volume = run_cmd(["ffmpeg", "-hide_banner", "-i", str(artifact), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"], check=False, timeout=60)
    volume_text = volume.stdout + volume.stderr
    mean_match = re.search(r"mean_volume:\s+(-?[0-9.]+) dB", volume_text)
    max_match = re.search(r"max_volume:\s+(-?[0-9.]+) dB", volume_text)
    if not mean_match or not max_match or float(max_match.group(1)) <= -90:
        raise RuntimeError(f"silent or unreadable audio: {volume_text[-1000:]}")
    black = run_cmd(["ffmpeg", "-hide_banner", "-i", str(artifact), "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"], check=False, timeout=120)
    black_text = black.stdout + black.stderr
    if "black_start:" in black_text:
        raise RuntimeError(f"black interval detected in {artifact}: {black_text[-1500:]}")
    return {
        "ffprobe": probe,
        "sha256": sha256(artifact),
        "audio_mean_db": float(mean_match.group(1)),
        "audio_peak_db": float(max_match.group(1)),
        "decode_ok": True,
        "black_interval_detected": False,
    }


def graph_for_run(base: dict, topology: str, state: str, prompt_text: str, seed: int) -> dict:
    graph = copy.deepcopy(base)
    graph["prompt"]["5"]["inputs"]["prompt"] = prompt_text
    graph["prompt"]["6"]["inputs"]["noise_seed"] = seed
    graph["prompt"]["14"]["inputs"]["filename_prefix"] = (
        f"dual-gpu-residency-{RUN_ID}/{topology}-{state}"
    )
    return graph


def execute_prompt(topology: dict, base_graph: dict, state: str, prompt_text: str, seed: int) -> dict:
    port = topology["port"]
    graph = graph_for_run(base_graph, topology["name"], state, prompt_text, seed)
    graph_path = RESULT_ROOT / "graphs" / f"{topology['name']}-{state}.json"
    graph_path.write_text(json.dumps(graph, indent=2) + "\n")

    accepted_client_ms = epoch_ms()
    response = http_json(f"http://127.0.0.1:{port}/prompt", graph, timeout=30)
    prompt_id = response["prompt_id"]
    print(f"RUN_START topology={topology['name']} state={state} prompt_id={prompt_id}", flush=True)

    deadline = time.monotonic() + 900
    history_item = None
    while time.monotonic() < deadline:
        history = http_json(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=10)
        history_item = history.get(prompt_id)
        if history_item and history_item.get("status", {}).get("completed"):
            break
        time.sleep(1)
    if not history_item or not history_item.get("status", {}).get("completed"):
        raise RuntimeError(f"prompt timeout: {prompt_id}")
    if history_item["status"].get("status_str") != "success":
        raise RuntimeError(f"prompt failed: {prompt_id}: {history_item['status']}")

    events = {}
    for name, payload in history_item["status"].get("messages", []):
        if "timestamp" in payload:
            events[name] = payload["timestamp"]
    wall_ms = events["execution_success"] - events["execution_start"]
    artifact = extract_artifact(history_item, topology["output_root"])
    validation = validate_artifact(artifact)
    record = {
        "topology": topology["name"],
        "state": state,
        "prompt_label": "A" if prompt_text == PROMPT_A else "B",
        "prompt": prompt_text,
        "seed": seed,
        "prompt_id": prompt_id,
        "client_submit_epoch_ms": accepted_client_ms,
        "events": events,
        "wall_ms": wall_ms,
        "wall_seconds": wall_ms / 1000,
        "artifact": str(artifact),
        "artifact_validation": validation,
        "history": history_item,
    }
    run_path = RESULT_ROOT / "runs" / f"{topology['name']}-{state}.json"
    run_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"RUN_DONE topology={topology['name']} state={state} wall={wall_ms/1000:.3f}s artifact={artifact}", flush=True)
    return record


def capture_journals(topology: dict, start_iso: str, end_iso: str) -> tuple[str, str]:
    unit_log = run_cmd([
        "journalctl", "--user", "-u", topology["unit"], "--since", start_iso,
        "--until", end_iso, "--no-pager", "-o", "short-iso-precise",
    ], check=False, timeout=60).stdout
    kernel_log = run_cmd([
        "journalctl", "-k", "--since", start_iso, "--until", end_iso,
        "--no-pager", "-o", "short-iso-precise",
    ], check=False, timeout=60).stdout
    (RESULT_ROOT / "journals" / f"{topology['name']}-service.log").write_text(unit_log)
    (RESULT_ROOT / "journals" / f"{topology['name']}-kernel.log").write_text(kernel_log)
    return unit_log, kernel_log


def assert_health(topology: dict, unit_log: str, kernel_log: str) -> None:
    fatal_service = re.compile(
        r"out of memory|OutOfMemory|OOM|Traceback|execution error|NaN|invalid device",
        re.IGNORECASE,
    )
    fatal_kernel = re.compile(
        r"amdgpu.*(?:VM fault|ring timeout|GPU reset|fatal error)|KFD.*error|out of memory|oom-kill",
        re.IGNORECASE,
    )
    if fatal_service.search(unit_log):
        raise RuntimeError(f"service health stop condition in {topology['name']} log")
    if fatal_kernel.search(kernel_log):
        raise RuntimeError(f"kernel health stop condition in {topology['name']} log")
    if topology["name"] == "single":
        if "CLIP/text encoder model load device: cuda:0" not in unit_log:
            raise RuntimeError("single-GPU encoder placement not proven")
    else:
        if "CLIP/text encoder model load device: cuda:1" not in unit_log:
            raise RuntimeError("dual-GPU encoder placement not proven")
    for loaded_mb in ("14960.20 MB loaded, full load: True", "19984.52 MB loaded, full load: True"):
        if loaded_mb not in unit_log:
            raise RuntimeError(f"full-load evidence missing: {topology['name']}: {loaded_mb}")


def run_topology(topology: dict) -> dict:
    print(f"TOPOLOGY_START {topology['name']}", flush=True)
    run_cmd(["systemctl", "--user", "stop", topology["unit"]], check=False, timeout=60)
    run_cmd(["systemctl", "--user", "stop", topology["other_unit"]], check=False, timeout=60)
    wait_http(topology["port"], False, timeout=60)

    telemetry = Telemetry(topology["name"])
    telemetry.start()
    start_iso = iso_now()
    start_ms = epoch_ms()
    run_cmd(["systemctl", "--user", "start", topology["unit"]], timeout=60)
    stats = wait_http(topology["port"], True, timeout=300)
    ready_ms = epoch_ms()
    devices = stats["devices"]
    if len(devices) != topology["expected_devices"]:
        raise RuntimeError(f"device count mismatch for {topology['name']}: {devices}")
    if "R9700" not in devices[0]["name"]:
        raise RuntimeError(f"cuda:0 is not R9700: {devices}")
    if topology["name"] == "dual" and "7900 XT" not in devices[1]["name"]:
        raise RuntimeError(f"cuda:1 is not 7900 XT: {devices}")

    base_graph = json.loads(topology["graph"].read_text())
    records = []
    try:
        for state, prompt_text, seed in RUNS:
            records.append(execute_prompt(topology, base_graph, state, prompt_text, seed))
    finally:
        end_iso = iso_now()
        telemetry.stop()
        unit_log, kernel_log = capture_journals(topology, start_iso, end_iso)
        run_cmd(["systemctl", "--user", "stop", topology["unit"]], check=False, timeout=60)
    assert_health(topology, unit_log, kernel_log)
    topology_record = {
        "topology": topology["name"],
        "service": topology["unit"],
        "service_start_request_epoch_ms": start_ms,
        "service_ready_epoch_ms": ready_ms,
        "service_startup_seconds": (ready_ms - start_ms) / 1000,
        "restart_to_first_artifact_seconds": (records[0]["events"]["execution_success"] - start_ms) / 1000,
        "devices": devices,
        "runs": records,
    }
    (RESULT_ROOT / f"{topology['name']}-summary.json").write_text(json.dumps(topology_record, indent=2) + "\n")
    print(f"TOPOLOGY_DONE {topology['name']}", flush=True)
    return topology_record


def summarize_telemetry(topology: str) -> dict:
    samples = [json.loads(line) for line in (RESULT_ROOT / "telemetry" / f"{topology}.jsonl").read_text().splitlines() if line]
    summary = {"sample_count": len(samples), "gpus": {}, "host_ram_peak_used_mb": None}
    for sample in samples:
        total = sample.get("host_mem_total_kb")
        avail = sample.get("host_mem_available_kb")
        if total is not None and avail is not None:
            used_mb = (total - avail) / 1024
            current = summary["host_ram_peak_used_mb"]
            summary["host_ram_peak_used_mb"] = used_mb if current is None else max(current, used_mb)
        for gpu in sample.get("gpus", []):
            index = str(gpu["gpu"])
            target = summary["gpus"].setdefault(index, {})
            for key in ("gfx_activity_percent", "power_w", "temp_edge_c", "temp_hotspot_c", "temp_mem_c", "vram_used_mb", "gtt_used_mb"):
                value = gpu.get(key)
                if isinstance(value, (int, float)):
                    target[f"peak_{key}"] = max(target.get(f"peak_{key}", value), value)
    return summary


def main() -> int:
    print(f"BENCHMARK_ROOT={RESULT_ROOT}", flush=True)
    preflight()
    summaries = []
    failure = None
    try:
        for topology in TOPOLOGIES:
            summaries.append(run_topology(topology))
    except Exception as exc:
        failure = repr(exc)
        print(f"BENCHMARK_FAILED {failure}", file=sys.stderr, flush=True)
        raise
    finally:
        run_cmd(["systemctl", "--user", "stop", "comfyui-h3-dualgpu.service"], check=False, timeout=60)
        run_cmd(["systemctl", "--user", "start", "comfyui-h3.service"], check=False, timeout=60)
        try:
            wait_http(8190, True, timeout=300)
        except Exception as restore_exc:
            (RESULT_ROOT / "RESTORE-ERROR.txt").write_text(repr(restore_exc) + "\n")

    result = {
        "run_id": RUN_ID,
        "result_root": str(RESULT_ROOT),
        "completed": iso_now(),
        "summaries": summaries,
        "telemetry": {topology["name"]: summarize_telemetry(topology["name"]) for topology in TOPOLOGIES},
        "failure": failure,
    }
    (RESULT_ROOT / "raw-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BENCHMARK_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
