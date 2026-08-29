"""Passive, gated sampler-memory probe for the MiniMax H3 experiments.

The module patches ``comfy.samplers.CFGGuider.inner_sample`` once at import.
That method is entered after ``prepare_sampling`` has loaded the denoiser and
immediately before conditioning is processed and the sampler is invoked. The
patch does nothing unless a run id has been armed through the HTTP endpoint.
"""

import json
import os
import re
import threading
import time
from pathlib import Path

import torch
from aiohttp import web
from server import PromptServer
import comfy.samplers


_LOCK = threading.Lock()
_CURRENT_RUN_ID = None
_ACTIVE_RUN_ID = None
_LAST_PROBE = {}
_ORIGINAL_INNER_SAMPLE = comfy.samplers.CFGGuider.inner_sample
_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_run_id(value):
    value = _SAFE_RUN_ID.sub("_", str(value or "").strip())[:160]
    return value or None


def _device_snapshot():
    devices = {}
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        key = f"cuda:{index}"
        try:
            free, driver_total = torch.cuda.mem_get_info(index)
            props = torch.cuda.get_device_properties(index)
            devices[key] = {
                "name": torch.cuda.get_device_name(index),
                "arch": getattr(props, "gcnArchName", None),
                "total": int(props.total_memory),
                "driver_free": int(free),
                "driver_total": int(driver_total),
                "driver_used": int(driver_total - free),
                "allocated": int(torch.cuda.memory_allocated(index)),
                "reserved": int(torch.cuda.memory_reserved(index)),
                "max_allocated": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved": int(torch.cuda.max_memory_reserved(index)),
            }
        except Exception as exc:
            devices[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return devices


def _sync_all():
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _write_probe(record):
    global _LAST_PROBE
    _LAST_PROBE = record
    path = Path(f"/tmp/h3-mem-{record['run_id']}.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _claim_run():
    global _CURRENT_RUN_ID, _ACTIVE_RUN_ID
    with _LOCK:
        if not _CURRENT_RUN_ID or _ACTIVE_RUN_ID:
            return None
        run_id = _CURRENT_RUN_ID
        _CURRENT_RUN_ID = None
        _ACTIVE_RUN_ID = run_id
        return run_id


def _finish_run(run_id):
    global _ACTIVE_RUN_ID
    with _LOCK:
        if _ACTIVE_RUN_ID == run_id:
            _ACTIVE_RUN_ID = None


def _probed_inner_sample(self, *args, **kwargs):
    run_id = _claim_run()
    if not run_id:
        return _ORIGINAL_INNER_SAMPLE(self, *args, **kwargs)

    record = {
        "schema_version": 2,
        "run_id": run_id,
        "probe_point": "CFGGuider.inner_sample",
        "configured_attention": os.environ.get("H3_FLEX_ATTENTION"),
        "enter_time": time.time(),
    }
    try:
        _sync_all()
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        record["before"] = _device_snapshot()
        _write_probe(record)
        print(f"[H3MemProbe] {run_id} enter: {json.dumps(record['before'])}", flush=True)

        result = _ORIGINAL_INNER_SAMPLE(self, *args, **kwargs)
        record["status"] = "success"
        return result
    except BaseException as exc:
        record["status"] = "exception"
        record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            _sync_all()
        except Exception as exc:
            record["sync_error"] = f"{type(exc).__name__}: {exc}"
        record["exit_time"] = time.time()
        record["elapsed_seconds"] = record["exit_time"] - record["enter_time"]
        record["after"] = _device_snapshot()
        try:
            _write_probe(record)
            print(f"[H3MemProbe] {run_id} exit: {json.dumps(record)}", flush=True)
        except Exception as exc:
            print(f"[H3MemProbe] {run_id} write failed: {type(exc).__name__}: {exc}", flush=True)
        _finish_run(run_id)


@PromptServer.instance.routes.post("/h3mem/set_run_id")
async def set_run_id(request):
    global _CURRENT_RUN_ID
    try:
        data = await request.json()
        run_id = _sanitize_run_id(data.get("run_id"))
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    with _LOCK:
        if _ACTIVE_RUN_ID and run_id:
            return web.json_response(
                {"ok": False, "error": "probe already active", "active_run_id": _ACTIVE_RUN_ID},
                status=409,
            )
        _CURRENT_RUN_ID = run_id
    return web.json_response({"ok": True, "run_id": run_id, "active_run_id": _ACTIVE_RUN_ID})


@PromptServer.instance.routes.get("/h3mem/last")
async def get_last(_request):
    return web.json_response(_LAST_PROBE)


@PromptServer.instance.routes.get("/h3mem/map")
async def get_map(_request):
    return web.json_response({
        "ok": bool(torch.cuda.is_available()),
        "devices": [
            {"index": int(key.split(":")[1]), **value}
            for key, value in _device_snapshot().items()
        ],
        "current_run_id": _CURRENT_RUN_ID,
        "active_run_id": _ACTIVE_RUN_ID,
        "configured_attention": os.environ.get("H3_FLEX_ATTENTION"),
        "probe_point": "CFGGuider.inner_sample",
    })


if not getattr(comfy.samplers.CFGGuider.inner_sample, "_h3_mem_probe", False):
    _probed_inner_sample._h3_mem_probe = True
    comfy.samplers.CFGGuider.inner_sample = _probed_inner_sample
    print("[H3MemProbe] installed gated CFGGuider.inner_sample probe", flush=True)
else:
    print("[H3MemProbe] probe already installed; leaving existing patch intact", flush=True)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
