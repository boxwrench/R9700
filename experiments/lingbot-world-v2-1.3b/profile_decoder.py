#!/usr/bin/env python
"""Rank the operators in one steady-state chunk_size=1 VAE decode.

Reproduces exactly what interactive_world.py does per action: a warm causal
feature cache, then a single decoder pass over one latent frame producing four
pixel frames. Module-level hooks give operator type, shapes, dtypes, call
counts and synchronised latency; a torch-profiler pass on the same call picks
up whatever lives between modules (copies, elementwise, layout changes).
"""
import argparse, collections, json, os, sys, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

sys.path.insert(0, '/ai/repos/lingbot-world-v2')
VAE_PTH = '/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled/Wan2.1_VAE.pth'


def sync():
    torch.cuda.synchronize()


class Hooks:
    def __init__(self, root):
        self.rows = collections.OrderedDict()
        self.handles = []
        self.enabled = False
        for name, m in root.named_modules():
            if len(list(m.children())):          # leaves only
                continue
            self.handles.append(m.register_forward_pre_hook(self._pre(name)))
            self.handles.append(m.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def f(mod, inp):
            if not self.enabled:
                return
            sync()
            mod.__t0 = time.perf_counter()
            mod.__a0 = torch.cuda.memory_allocated()
        return f

    def _post(self, name):
        def f(mod, inp, out):
            if not self.enabled:
                return
            sync()
            dt = time.perf_counter() - mod.__t0
            da = (torch.cuda.memory_allocated() - mod.__a0) / 2**30
            x = inp[0] if isinstance(inp, tuple) and torch.is_tensor(inp[0]) else None
            o = out if torch.is_tensor(out) else None
            w = getattr(mod, 'weight', None)
            key = (name, type(mod).__name__,
                   tuple(x.shape) if x is not None else None,
                   tuple(w.shape) if torch.is_tensor(w) else None,
                   tuple(o.shape) if o is not None else None,
                   str(x.dtype) if x is not None else None)
            r = self.rows.setdefault(key, {'n': 0, 'total': 0.0, 'alloc': 0.0,
                                           'stride': getattr(mod, 'stride', None),
                                           'padding': getattr(mod, 'padding', None),
                                           'groups': getattr(mod, 'groups', None)})
            r['n'] += 1
            r['total'] += dt
            r['alloc'] = max(r['alloc'], da)
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat_h', type=int, default=58)
    ap.add_argument('--lat_w', type=int, default=104)
    ap.add_argument('--warm', type=int, default=3, help='latent frames to warm the causal cache')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    from wan.modules.vae2_1 import Wan2_1_VAE
    vae = Wan2_1_VAE(vae_pth=VAE_PTH, device='cuda')
    m = vae.model
    m.clear_cache()
    scale = vae.scale

    def decode_one(z1):
        """One latent frame through the decoder, cache preserved."""
        zz = z1.unsqueeze(0)
        zz = zz / scale[1].view(1, m.z_dim, 1, 1, 1) + scale[0].view(1, m.z_dim, 1, 1, 1)
        x = m.conv2(zz)
        m._conv_idx = [0]
        return m.decoder(x[:, :, 0:1], feat_cache=m._feat_map, feat_idx=m._conv_idx)

    torch.manual_seed(0)
    z = torch.randn(16, args.warm + args.reps + 1, args.lat_h, args.lat_w, device='cuda')

    for i in range(args.warm):                     # warm the causal cache
        decode_one(z[:, i])
    sync()

    hooks = Hooks(m)
    times = []
    hooks.enabled = True
    for i in range(args.reps):
        sync(); t0 = time.perf_counter()
        out = decode_one(z[:, args.warm + i])
        sync(); times.append(time.perf_counter() - t0)
    hooks.enabled = False
    hooks.remove()

    total = sum(times) / len(times)
    print(f'steady-state decode of 1 latent frame -> {tuple(out.shape)}')
    print(f'wall: {total:.4f} s   (reps {[round(t,4) for t in times]})\n')

    rows = []
    for (name, kind, ish, wsh, osh, dt), r in hooks.rows.items():
        rows.append(dict(module=name, op=kind, in_shape=list(ish) if ish else None,
                         weight_shape=list(wsh) if wsh else None,
                         out_shape=list(osh) if osh else None, dtype=dt,
                         calls=r['n'] // args.reps,
                         total_s=r['total'] / args.reps,
                         mean_ms=r['total'] / r['n'] * 1000,
                         alloc_gib=r['alloc'],
                         stride=list(r['stride']) if r['stride'] else None,
                         padding=list(r['padding']) if isinstance(r['padding'], tuple) else r['padding'],
                         groups=r['groups']))
    rows.sort(key=lambda r: -r['total_s'])

    by_op = collections.Counter()
    for r in rows:
        by_op[r['op']] += r['total_s']

    print('by operator class:')
    for k, v in by_op.most_common():
        print(f'  {k:20s} {v:7.4f} s  {100*v/total:5.1f}%')

    print('\nranked operators (>=1% each):')
    print(f'  {"op":14s} {"in":24s} {"weight":18s} {"out":24s} {"n":>3s} '
          f'{"mean ms":>9s} {"total s":>9s} {"%":>6s}')
    cum = 0.0
    for r in rows:
        pct = 100 * r['total_s'] / total
        if pct < 1.0:
            continue
        cum += pct
        print(f'  {r["op"]:14s} {str(r["in_shape"]):24s} {str(r["weight_shape"]):18s} '
              f'{str(r["out_shape"]):24s} {r["calls"]:3d} {r["mean_ms"]:9.2f} '
              f'{r["total_s"]:9.4f} {pct:5.1f}%')
    print(f'  -> listed operators cover {cum:.1f}% of decode wall time')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(dict(wall_seconds=total, reps=times, out_shape=list(out.shape),
                           by_op={k: v for k, v in by_op.items()}, rows=rows), f, indent=2)


if __name__ == '__main__':
    main()
