#!/usr/bin/env python
"""Standalone microbenchmark for the two Conv3d shapes that dominate the
Wan2.1 VAE decode on gfx1201.

Shapes and layout are taken from the live VAE: CausalConv3d pads the input
before calling F.conv3d, so the convolution itself sees padding=0 and an
input whose temporal/spatial extent is already padded. We reproduce both the
padded input the kernel actually receives and the tensor's real strides.
"""
import argparse, json, os, statistics, sys, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# (label, in_shape_as_seen_by_F.conv3d, weight_shape)
SHAPES = {
    'dominant': ((1, 96, 4, 480, 832), (96, 96, 3, 3, 3)),
    'rgb_out':  ((1, 96, 4, 480, 832), (3, 96, 3, 3, 3)),
    'ref_192':  ((1, 192, 4, 240, 416), (192, 192, 3, 3, 3)),
    'ref_384':  ((1, 384, 2, 120, 208), (384, 384, 3, 3, 3)),
}


def bench(label, in_shape, w_shape, warmup, iters, pad):
    dev = 'cuda'
    x = torch.randn(*in_shape, device=dev, dtype=torch.float32)
    w = torch.randn(*w_shape, device=dev, dtype=torch.float32)
    b = torch.randn(w_shape[0], device=dev, dtype=torch.float32)
    info = dict(
        label=label, in_shape=list(in_shape), weight_shape=list(w_shape),
        in_strides=list(x.stride()), weight_strides=list(w.stride()),
        in_contiguous=x.is_contiguous(), weight_contiguous=w.is_contiguous(),
        padding=list(pad),
    )
    try:
        for _ in range(warmup):
            y = torch.nn.functional.conv3d(x, w, b, padding=pad)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = torch.nn.functional.conv3d(x, w, b, padding=pad)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        macs = (w_shape[0] * w_shape[1] * 27 *
                (in_shape[2] - 2 + 2 * pad[0]) *
                (in_shape[3] - 2 + 2 * pad[1]) *
                (in_shape[4] - 2 + 2 * pad[2]))
        info.update(
            out_shape=list(y.shape), warm_median=statistics.median(ts),
            warm_min=min(ts), warm_max=max(ts), samples=ts,
            gflop=2 * macs / 1e9,
            tflops=2 * macs / 1e12 / statistics.median(ts),
            peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
            checksum=float(y.float().sum()))
        print(f'  {label:10s} {list(in_shape)} x {list(w_shape)} -> {list(y.shape)}  '
              f'median {info["warm_median"]:.4f}s  {info["tflops"]:.2f} TFLOP/s  '
              f'sum={info["checksum"]:.3f}', flush=True)
        del y
    except Exception as e:
        info.update(error=f'{type(e).__name__}: {str(e)[:200]}')
        print(f'  {label:10s} FAILED {info["error"]}', flush=True)
    del x, w, b
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True)
    ap.add_argument('--shapes', default='dominant,rgb_out,ref_192,ref_384')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--disable-miopen', action='store_true')
    ap.add_argument('--padding', default='0,0,0')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    if args.disable_miopen:
        torch.backends.cudnn.enabled = False
    pad = tuple(int(v) for v in args.padding.split(','))

    print(f'lane={args.lane} cudnn.enabled={torch.backends.cudnn.enabled} '
          f'padding={pad} '
          f'MIOPEN_FIND_MODE={os.environ.get("MIOPEN_FIND_MODE", "(unset)")}',
          flush=True)
    rec = dict(lane=args.lane, cudnn_enabled=torch.backends.cudnn.enabled,
               padding=list(pad),
               env={k: v for k, v in os.environ.items() if k.startswith('MIOPEN')},
               torch=torch.__version__, hip=torch.version.hip,
               gpu=torch.cuda.get_device_properties(0).gcnArchName,
               results=[])
    for name in args.shapes.split(','):
        in_shape, w_shape = SHAPES[name]
        rec['results'].append(bench(name, in_shape, w_shape,
                                    args.warmup, args.iters, pad))
    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(rec, fh, indent=2)


if __name__ == '__main__':
    main()
