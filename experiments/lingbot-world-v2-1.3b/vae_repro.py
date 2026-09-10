#!/usr/bin/env python
"""Isolated Wan2.1 VAE decode reproducer for the gfx1201 Conv3d investigation.

Decodes the same latent three times in one process with synchronised timers,
optionally instrumenting every Conv3d call so first-call and repeat-call cost
can be separated per input shape. Nothing about the VAE itself is changed:
FP32 weights, upstream chunking, upstream module code.
"""
import argparse, json, os, sys, time, warnings, hashlib
from collections import OrderedDict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

VAE_PTH = '/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled/Wan2.1_VAE.pth'


def sync():
    torch.cuda.synchronize()


def instrument(model, table):
    """Wrap every Conv3d so we record per-(in-shape, weight-shape) timing and
    the peak allocation delta the call causes (a proxy for workspace)."""
    for mod in model.modules():
        if not isinstance(mod, nn.Conv3d):
            continue
        fwd = mod.forward

        def make(fwd=fwd, mod=mod):
            def inner(x, *a, **kw):
                key = (tuple(x.shape), tuple(mod.weight.shape),
                       tuple(mod.stride), tuple(mod.padding), str(x.dtype))
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                sync(); t0 = time.perf_counter()
                out = fwd(x, *a, **kw)
                sync(); dt = time.perf_counter() - t0
                transient = (torch.cuda.max_memory_allocated() - before) / 2**30
                e = table.setdefault(key, {'times': [], 'transient_gib': 0.0})
                e['times'].append(dt)
                e['transient_gib'] = max(e['transient_gib'], transient)
                return out
            return inner
        mod.forward = make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--frames', type=int, default=4, help='latent frames')
    ap.add_argument('--lat_h', type=int, default=60)
    ap.add_argument('--lat_w', type=int, default=104)
    ap.add_argument('--disable-miopen', action='store_true')
    ap.add_argument('--shapes', action='store_true')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    sys.path.insert(0, '/ai/repos/lingbot-world-v2')
    from wan.modules.vae2_1 import Wan2_1_VAE

    if args.disable_miopen:
        torch.backends.cudnn.enabled = False

    rec = OrderedDict(
        lane=args.lane,
        env={k: v for k, v in os.environ.items()
             if k.startswith(('MIOPEN', 'ROCM', 'HIP', 'PYTORCH'))},
        cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        miopen_immediate=getattr(getattr(torch.backends, 'miopen', None),
                                 'immediate', 'n/a'),
        torch=torch.__version__, hip=torch.version.hip,
        gpu=torch.cuda.get_device_properties(0).gcnArchName,
    )

    vae = Wan2_1_VAE(vae_pth=VAE_PTH, device='cuda')
    table = {}
    if args.shapes:
        instrument(vae.model, table)

    z = torch.zeros(16, args.frames, args.lat_h, args.lat_w, device='cuda')
    runs = []
    out_ref = None
    for i in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        v = vae.decode([z])[0]
        sync(); dt = time.perf_counter() - t0
        runs.append(dict(
            index=i, seconds=dt,
            peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        if out_ref is None:
            out_ref = v.float().cpu()
        print(f'[{args.lane}] decode {i}: {dt:.2f}s  '
              f'peak_alloc {runs[-1]["peak_alloc_gib"]:.2f} GiB  '
              f'peak_reserved {runs[-1]["peak_reserved_gib"]:.2f} GiB', flush=True)
        del v
    rec['runs'] = runs

    f = out_ref
    rec['output'] = dict(
        shape=list(f.shape), dtype=str(f.dtype),
        mean=float(f.mean()), std=float(f.std()),
        min=float(f.min()), max=float(f.max()),
        finite=bool(torch.isfinite(f).all()),
        sha256=hashlib.sha256(f.numpy().tobytes()).hexdigest()[:32])
    print(f'[{args.lane}] output mean={f.mean():.6f} std={f.std():.6f} '
          f'sha={rec["output"]["sha256"]}', flush=True)

    if args.shapes:
        shapes = []
        for k, e in table.items():
            t = e['times']
            shapes.append(dict(
                in_shape=list(k[0]), weight_shape=list(k[1]),
                stride=list(k[2]), padding=list(k[3]), dtype=k[4],
                count=len(t), first=t[0],
                repeat_mean=(sum(t[1:]) / len(t[1:]) if len(t) > 1 else None),
                total=sum(t), transient_gib=e['transient_gib']))
        shapes.sort(key=lambda s: -s['total'])
        rec['unique_conv3d_shapes'] = len(shapes)
        rec['conv3d_total_seconds'] = sum(s['total'] for s in shapes)
        rec['conv3d_shapes'] = shapes
        print(f'[{args.lane}] unique Conv3d shapes: {len(shapes)}  '
              f'conv3d total {rec["conv3d_total_seconds"]:.2f}s '
              f'(over {args.runs} decodes)', flush=True)
        for s in shapes[:8]:
            print('   in=%s w=%s stride=%s n=%d first=%.3fs repeat=%s total=%.2fs '
                  'transient=%.2fGiB' % (
                      s['in_shape'], s['weight_shape'], s['stride'], s['count'],
                      s['first'],
                      'n/a' if s['repeat_mean'] is None else '%.3fs' % s['repeat_mean'],
                      s['total'], s['transient_gib']), flush=True)

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(rec, fh, indent=2)


if __name__ == '__main__':
    main()
