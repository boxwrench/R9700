#!/usr/bin/env python
"""A/B the Wan VAE decoder in FP32 / BF16 / FP16 on identical latents.

Every lane gets the same latent tensor and the same warm-up sequence, so the
causal feature cache is in the equivalent state before the measured frame.
FP32 is the reference; the others are compared against it.
"""
import argparse, json, sys, time, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

sys.path.insert(0, '/ai/repos/lingbot-world-v2')
VAE_PTH = '/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled/Wan2.1_VAE.pth'


def sync():
    torch.cuda.synchronize()


def build(dtype, keep_norm_fp32):
    from wan.modules.vae2_1 import Wan2_1_VAE, RMS_norm
    vae = Wan2_1_VAE(vae_pth=VAE_PTH, device='cuda')
    if dtype is not torch.float32:
        vae.model = vae.model.to(dtype)
        if keep_norm_fp32:
            for m in vae.model.modules():
                if isinstance(m, RMS_norm):
                    m.to(torch.float32)
    return vae


def run_lane(label, dtype, z, warm, reps, keep_norm_fp32=False, ref=None):
    vae = build(dtype, keep_norm_fp32)
    m = vae.model
    m.clear_cache()
    scale = vae.scale

    def decode_one(z1):
        zz = z1.unsqueeze(0).to(torch.float32)
        zz = zz / scale[1].view(1, m.z_dim, 1, 1, 1) + scale[0].view(1, m.z_dim, 1, 1, 1)
        zz = zz.to(dtype)
        x = m.conv2(zz)
        m._conv_idx = [0]
        return m.decoder(x[:, :, 0:1], feat_cache=m._feat_map, feat_idx=m._conv_idx)

    for i in range(warm):
        decode_one(z[:, i])
    sync()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(reps):
        sync(); t0 = time.perf_counter()
        out = decode_one(z[:, warm])          # always the same frame
        sync(); ts.append(time.perf_counter() - t0)
        # re-warm so every rep sees the same cache state
        if i < reps - 1:
            m.clear_cache()
            for j in range(warm):
                decode_one(z[:, j])
            sync()
    f = out.float()
    r = dict(lane=label, dtype=str(dtype), keep_norm_fp32=keep_norm_fp32,
             wall=min(ts), walls=ts,
             peak_gib=torch.cuda.max_memory_allocated() / 2**30,
             finite=bool(torch.isfinite(f).all()),
             has_nan=bool(torch.isnan(f).any()), has_inf=bool(torch.isinf(f).any()),
             min=float(f.min()), max=float(f.max()),
             mean=float(f.mean()), std=float(f.std()))
    if ref is not None:
        d = (f - ref).abs()
        r['max_abs_err'] = float(d.max())
        r['mean_abs_err'] = float(d.mean())
        r['rel_err'] = float(d.max() / ref.abs().max())
    print(f'  {label:22s} {r["wall"]:7.4f}s  peak {r["peak_gib"]:5.2f} GiB  '
          f'finite={r["finite"]}  mean {r["mean"]:+.5f} std {r["std"]:.5f}'
          + ('' if ref is None else
             f'  maxerr {r["max_abs_err"]:.4f} rel {r["rel_err"]:.2e}'), flush=True)
    del vae, m
    torch.cuda.empty_cache()
    return r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat_h', type=int, default=58)
    ap.add_argument('--lat_w', type=int, default=104)
    ap.add_argument('--warm', type=int, default=3)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    z = torch.randn(16, args.warm + 2, args.lat_h, args.lat_w, device='cuda')

    print('Wan2.1 VAE decoder, one latent frame at 464x832, warm causal cache\n')
    out = []
    r0, ref = run_lane('fp32 (reference)', torch.float32, z, args.warm, args.reps)
    out.append(r0)
    for label, dt, keep in (('bf16', torch.bfloat16, False),
                            ('bf16 + fp32 norms', torch.bfloat16, True),
                            ('fp16', torch.float16, False),
                            ('fp16 + fp32 norms', torch.float16, True)):
        try:
            r, _ = run_lane(label, dt, z, args.warm, args.reps, keep, ref)
            r['speedup'] = r0['wall'] / r['wall']
            out.append(r)
        except Exception as e:
            print(f'  {label:22s} FAILED {type(e).__name__}: {str(e)[:120]}', flush=True)
            out.append(dict(lane=label, error=f'{type(e).__name__}: {e}'))

    print('\nsummary')
    for r in out:
        if 'error' in r:
            print(f'  {r["lane"]:22s} FAILED'); continue
        sp = r.get('speedup')
        print(f'  {r["lane"]:22s} {r["wall"]:7.4f}s '
              f'{"" if sp is None else f"{sp:5.2f}x"}  '
              f'peak {r["peak_gib"]:5.2f} GiB  '
              f'rel_err {r.get("rel_err", 0):.2e}  finite={r["finite"]}')
    if args.out_json:
        json.dump(out, open(args.out_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
