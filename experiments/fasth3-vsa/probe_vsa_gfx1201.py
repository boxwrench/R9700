#!/usr/bin/env python
"""Standalone gfx1201 probe for FastVideo's Triton VSA kernel.

No ComfyUI, no H3 weights. Answers exactly one question: does
``vsa.video_sparse_attn`` compile and execute correctly on gfx1201, at
H3-representative shapes and dtypes?

Correctness is checked against a dense reference at topk == n_blocks, where VSA
is mathematically required to equal dense attention over the same tokens.
"""
from __future__ import annotations

import math
import os
import sys
import traceback

import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/ai/cache/triton")


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


banner("environment")
print("torch      :", torch.__version__)
print("hip        :", torch.version.hip)
import triton  # noqa: E402

print("triton     :", triton.__version__)
print("device     :", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability :", cap, "-> vsa picks", "CUDA/SM90" if cap == (9, 0) else "TRITON")

banner("import vsa")
import vsa  # noqa: E402

print("vsa module        :", vsa.__file__)
_bsa = repr(vsa.block_sparse_attn)
print("block_sparse_attn :", _bsa)
print("block_sparse_fwd  :", vsa.block_sparse_fwd, "(None == no CUDA kernel, expected on ROCm)")
assert vsa.block_sparse_fwd is None, "CUDA kernel selected on ROCm -- unexpected"
assert "triton" in _bsa.lower(), f"Triton path not selected: {_bsa}"
assert "vsa_cuda" not in sys.modules, "vsa_cuda was imported -- CUDA path leaked in"
print("OK: Triton path selected, no CUDA kernel imported")


def dense_reference(q, k, v, variable_block_sizes, compress_attn_weight=None):
    """VSA with every block selected == dense attention + the compress branch."""
    b, h, s, d = q.shape
    be = 64
    nb = s // be
    qc = (q.view(b, h, nb, be, d).float().sum(3) / variable_block_sizes.view(1, 1, -1, 1)).to(q.dtype)
    kc = (k.view(b, h, nb, be, d).float().sum(3) / variable_block_sizes.view(1, 1, -1, 1)).to(k.dtype)
    vc = (v.view(b, h, nb, be, d).float().sum(3) / variable_block_sizes.view(1, 1, -1, 1)).to(v.dtype)
    oc, _ = vsa.torch_attention(qc, kc, vc)
    oc = oc.view(b, h, nb, 1, d).repeat(1, 1, 1, be, 1).view(b, h, s, d)
    od, _ = vsa.torch_attention(q, k, v)
    return oc * compress_attn_weight + od if compress_attn_weight is not None else oc + od


def run_case(heads: int, seq_len: int, topk_ratio: float, use_gate: bool, check: bool) -> bool:
    b, d, be = 1, 128, 64
    nb = seq_len // be
    topk = max(1, int(math.ceil(nb * topk_ratio)))
    label = f"H={heads} S={seq_len} blocks={nb} topk={topk} ({topk_ratio:g}) gate={use_gate}"
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(b, heads, seq_len, d, device=dev, dtype=dt)
    k = torch.randn(b, heads, seq_len, d, device=dev, dtype=dt)
    v = torch.randn(b, heads, seq_len, d, device=dev, dtype=dt)
    vbs = torch.full((nb,), be, device=dev, dtype=torch.int32)
    gate = torch.rand(b, heads, seq_len, d, device=dev, dtype=dt) if use_gate else None
    try:
        torch.cuda.synchronize()
        out = vsa.video_sparse_attn(q, k, v, vbs, topk, (4, 4, 4), compress_attn_weight=gate)
        torch.cuda.synchronize()
    except Exception:
        print(f"FAIL  {label}")
        traceback.print_exc()
        return False
    finite = torch.isfinite(out).all().item()
    line = f"OK    {label}  out={tuple(out.shape)} {out.dtype} finite={finite}"
    if check:
        ref = dense_reference(q, k, v, vbs, gate)
        num = (out.float() - ref.float()).norm().item()
        den = ref.float().norm().item() or 1.0
        rel = num / den
        line += f"  rel_L2_vs_dense={rel:.3e} {'PASS' if rel < 2e-2 else 'MISMATCH'}"
        finite = finite and rel < 2e-2
    print(line, flush=True)
    return finite


banner("correctness: topk == n_blocks must equal dense attention")
ok = True
ok &= run_case(heads=8, seq_len=4096, topk_ratio=1.0, use_gate=False, check=True)
ok &= run_case(heads=8, seq_len=4096, topk_ratio=1.0, use_gate=True, check=True)

banner("execution at H3-representative shapes (56 heads, dim 128, bf16)")
for s in (4096, 8192, 16384):
    ok &= run_case(heads=56, seq_len=s, topk_ratio=0.10, use_gate=True, check=False)

banner("result")
print("peak VRAM: %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("VERDICT:", "PASS - Triton VSA runs on gfx1201" if ok else "FAIL - see traceback above")
sys.exit(0 if ok else 1)
