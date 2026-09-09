#!/usr/bin/env python
"""Bounded WAN_VAE_CONV3D_TEMPORAL_SPLIT sweep.

One process per split value would be cleaner, but the split is read at import
time into a module global, so this script rebinds it explicitly between lanes
and rebuilds the VAE each time. Lane 0 (upstream, unsplit) runs first and
provides the reference output every other lane is compared against.
"""
import argparse, json, statistics, sys, time, warnings
warnings.filterwarnings('ignore')

import torch

sys.path.insert(0, '/ai/repos/lingbot-world-v2')
import wan.modules.vae2_1 as vae_mod
from wan.modules.vae2_1 import Wan2_1_VAE

VAE_PTH = '/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled/Wan2.1_VAE.pth'


def lane(split, z, ref, runs):
    vae_mod._CONV3D_TEMPORAL_SPLIT = split
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vae = Wan2_1_VAE(vae_pth=VAE_PTH, device='cuda')

    times, out = [], None
    for i in range(runs):
        if i == 1:                      # reset stats after the cold run
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = vae.decode([z])[0]
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)

    warm = times[1:] or times
    r = dict(
        split=split, cold=times[0], warm_median=statistics.median(warm),
        warm_all=warm,
        peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        mean=float(out.float().mean()), std=float(out.float().std()))
    f = out.float()
    if ref is not None:
        d = (f - ref).abs()
        r['max_abs_err'] = float(d.max())
        r['rel_err'] = float(d.max() / ref.abs().max())
        r['mean_abs_err'] = float(d.mean())
        r['bit_identical'] = bool(torch.equal(f, ref))
    print(f"  split={split}: cold {r['cold']:7.2f}s  warm {r['warm_median']:7.2f}s  "
          f"peak_alloc {r['peak_alloc_gib']:5.2f} GiB  "
          f"peak_reserved {r['peak_reserved_gib']:5.2f} GiB  "
          f"rel_err {r.get('rel_err', 0):.2e}", flush=True)
    ret = f.clone() if ref is None else None
    del vae, out, f
    torch.cuda.empty_cache()
    return r, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    z = torch.randn(16, args.frames, 60, 104, device='cuda')

    print('Wan2.1 VAE decode, 480x832, FP32, upstream chunking')
    results = []
    r0, ref = lane(0, z, None, args.runs)      # upstream control -> reference
    results.append(r0)
    for s in (1, 2, 3):
        r, _ = lane(s, z, ref, args.runs)
        results.append(r)

    base = r0['warm_median']
    print('\n  split | cold      | warm      | speedup | peak alloc | peak resv | rel err')
    for r in results:
        print(f"  {r['split']:5d} | {r['cold']:8.2f}s | {r['warm_median']:8.2f}s | "
              f"{base / r['warm_median']:6.2f}x | {r['peak_alloc_gib']:9.2f}G | "
              f"{r['peak_reserved_gib']:8.2f}G | {r.get('rel_err', 0):.2e}")

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(results, fh, indent=2)


if __name__ == '__main__':
    main()
