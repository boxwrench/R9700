"""Torch-level VRAM probe exposed as HTTP routes.

rocm-smi reports device-wide usage at coarse sampling intervals. Acceptance for
the memory-cleanup pass needs torch's own peak allocated and peak reserved, so
this registers two routes on the ComfyUI server and touches no graph node.

  GET /memprobe/reset  -> zeroes torch peak counters
  GET /memprobe        -> {allocated, reserved, max_allocated, max_reserved} bytes
"""
import logging

import torch
from server import PromptServer

log = logging.getLogger(__name__)


@PromptServer.instance.routes.get("/memprobe/reset")
async def _memprobe_reset(request):
    from aiohttp import web
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/memprobe")
async def _memprobe(request):
    from aiohttp import web
    if not torch.cuda.is_available():
        return web.json_response({"ok": False})
    return web.json_response({
        "ok": True,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    })


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
