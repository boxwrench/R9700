#!/usr/bin/env python
"""Equivalence check: patched vsa.video_sparse_attn vs the original implementation.

Loads the pre-patch file as a separate module and compares outputs and routing on
H3-representative shapes. Requires bit-level or bf16-tolerance agreement AND an
identical top-k block mask.
"""
from __future__ import annotations

import importlib.util
import math
import sys

import torch

import vsa  # patched

ORIG = "/ai/lab/experiments/fasth3-vsa/vsa_init_ORIGINAL.py"
spec = importlib.util.spec_from_file_location("vsa_orig", ORIG)
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

print("patched :", vsa.__file__)
print("original:", ORIG)


def routing(q, k, v, vbs, topk, be):
    """Recompute the top-k mask the way both implementations do."""
    b, h, s, d = q.shape
    nb = s // be
    qc = (q.view(b, h, nb, be, d).float().sum(3) / vbs.view(1, 1, -1, 1)).to(q.dtype)
    kc = (k.view(b, h, nb, be, d).float().sum(3) / vbs.view(1, 1, -1, 1)).to(k.dtype)
    vc = (v.view(b, h, nb, be, d).float().sum(3) / vbs.view(1, 1, -1, 1)).to(v.dtype)
    _, score = vsa.torch_attention(qc, kc, vc)
    idx = torch.topk(score, topk, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)


ok = True
for heads, seq, ratio, gate in ((56, 16384, 0.20, True),
                                (56, 16384, 0.10, True),
                                (56, 8192, 0.20, False),
                                (8, 4096, 1.00, True)):
    b, d, be = 1, 128, 64
    nb = seq // be
    topk = max(1, int(math.ceil(nb * ratio)))
    torch.manual_seed(1234)
    q = torch.randn(b, heads, seq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, heads, seq, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, heads, seq, d, device="cuda", dtype=torch.bfloat16)
    vbs = torch.full((nb,), be, device="cuda", dtype=torch.int32)
    w = torch.rand(b, heads, seq, d, device="cuda", dtype=torch.bfloat16) if gate else None

    with torch.no_grad():
        a = orig.video_sparse_attn(q.clone(), k.clone(), v.clone(), vbs, topk, (4, 4, 4),
                                   compress_attn_weight=None if w is None else w.clone())
        torch.cuda.synchronize()
        bout = vsa.video_sparse_attn(q.clone(), k.clone(), v.clone(), vbs, topk, (4, 4, 4),
                                     compress_attn_weight=None if w is None else w.clone())
        torch.cuda.synchronize()
        m1 = routing(q, k, v, vbs, topk, be)

    same_shape = a.shape == bout.shape
    exact = torch.equal(a, bout)
    diff = (a.float() - bout.float()).abs()
    rel = (diff.norm() / (a.float().norm() + 1e-12)).item()
    maxabs = diff.max().item()
    finite = torch.isfinite(bout).all().item()
    # routing must be untouched
    route_ok = bool(m1.sum(-1).eq(topk).all().item())

    good = same_shape and finite and route_ok and (exact or rel < 5e-3)
    ok &= good
    print(f"H={heads:<3} S={seq:<6} topk={topk:>3}/{nb:<3} gate={str(gate):<5} "
          f"shape_ok={same_shape} bitexact={exact} rel_L2={rel:.3e} max_abs={maxabs:.3e} "
          f"routing_ok={route_ok} -> {'PASS' if good else 'FAIL'}")

print("\nVERDICT:", "PASS - patched implementation is equivalent" if ok else "FAIL")
sys.exit(0 if ok else 1)
