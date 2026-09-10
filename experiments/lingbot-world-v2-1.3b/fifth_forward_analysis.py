#!/usr/bin/env python
"""Is the fifth (KV-writing) forward reusable from the fourth denoising forward?

The self-attention in model_fast.py writes roped_key / v into the KV cache on
*every* forward. So after four denoising forwards the cache already holds K/V
-- the question is whether those are the K/V the model is supposed to keep.

Denoising forward i runs on `latent`, a noisy sample at timestep t_i.
The fifth forward runs on `x0`, the accepted clean latent, at timestep 0.
If the cache contents after forward 4 and after forward 5 differ materially,
reuse is not merely lossy, it stores the wrong thing.
"""
import os, sys, argparse, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interactive_world as IW


def snapshot(kv, lo, hi):
    """Only the slots this chunk wrote, held on CPU -- the full cache is 4.66 GiB."""
    return [(c['k'][:, lo:hi].detach().float().cpu(),
             c['v'][:, lo:hi].detach().float().cpu()) for c in kv]


def compare(a, b, lo, hi, label):
    kd = vd = 0.0
    kn = vn = 0.0
    for (ka, va), (kb, vb) in zip(a, b):
        kd = max(kd, (ka - kb).abs().max().item()); kn = max(kn, kb.abs().max().item())
        vd = max(vd, (va - vb).abs().max().item()); vn = max(vn, vb.abs().max().item())
    print(f'  {label}')
    print(f'     K  max|diff| {kd:.4f}   max|ref| {kn:.4f}   relative {kd/max(kn,1e-9):.3f}')
    print(f'     V  max|diff| {vd:.4f}   max|ref| {vn:.4f}   relative {vd/max(vn,1e-9):.3f}')
    return dict(k_absdiff=kd, k_ref=kn, k_rel=kd / max(kn, 1e-9),
                v_absdiff=vd, v_ref=vn, v_rel=vd / max(vn, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/ai/repos/lingbot-world-v2/examples/03/image.jpg')
    ap.add_argument('--prompt', default='A serene lakeside scene.')
    ap.add_argument('--out_json', default=None)
    args_ns = ap.parse_args()

    import argparse as A
    ns = A.Namespace(
        image=args_ns.image, prompt=args_ns.prompt, output='/tmp/fifth_fwd',
        ckpt_dir='/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled',
        task='i2v-1.3B', size='480*832', chunk_size=1, local_attn_size=18,
        sink_size=6, max_lat_frames=90, shift=10.0,
        timesteps_index=[0, 250, 500, 750], seed=42, move_amount=1.0,
        turn_deg=8.0, fov_deg=60.0, fps=16, script=None, vae_dtype='fp16',
        overlap=0, queue_max=1, no_chunk_mp4=True, cond_cache=None)

    world = IW.World(ns)
    pipe = world.pipe
    print(f'world ready. chunk={world.chunk} frame_seqlen={world.frame_seqlen}\n')

    # ---- replicate one step, snapshotting the cache between forwards ----
    i0 = world.lat_frames_done
    rel = world._relative_poses('forward', 1.0)
    plucker = world._plucker(rel)
    cond = world.y[:, i0:i0 + world.chunk]
    latent = torch.randn(16, world.chunk, world.lat_h, world.lat_w,
                         dtype=torch.float32, generator=world.gen, device=world.device)
    kwargs = dict(context=[world.context[0]], seq_len=world.chunk * world.frame_seqlen,
                  y=[cond], dit_cond_dict={'c2ws_plucker_emb': plucker.chunk(1, dim=0)},
                  kv_cache=world.kv_cache, crossattn_cache=world.cross_cache,
                  current_start=i0 * world.frame_seqlen,
                  max_attention_size=world.kv_size, frame_seqlen=world.frame_seqlen)
    lo, hi = 0, world.chunk * world.frame_seqlen
    snaps = {}
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=pipe.param_dtype):
        for ti in range(len(world.timesteps)):
            ts = torch.stack([world.timesteps[ti]]).to(world.device)
            noise_pred = pipe.model(x=[latent], t=ts,
                                    cross_attn_first_call=(ti == 0), **kwargs)[0]
            x0 = pipe._convert_flow_pred_to_x0(
                flow_pred=noise_pred, xt=latent, timestep=world.timesteps[ti],
                scheduler=pipe.scheduler)
            snaps[f'after_denoise_{ti+1}'] = snapshot(world.kv_cache, lo, hi)
            if ti < len(world.timesteps) - 1:
                latent = pipe.scheduler.add_noise(
                    x0, torch.randn(x0.shape, generator=world.gen,
                                    device=x0.device, dtype=x0.dtype),
                    world.timesteps[ti + 1])
        pipe.model(x=[x0], t=torch.stack([world.timesteps[-1] * 0.0]).to(world.device),
                   cross_attn_first_call=False, **kwargs)
        snaps['after_fifth'] = snapshot(world.kv_cache, lo, hi)

    print('timesteps used:', [float(t) for t in world.timesteps],
          '   fifth forward timestep: 0.0\n')
    print('KV cache contents written by this chunk, compared against the state')
    print('the model actually keeps (after the fifth forward):\n')
    res = {}
    for k in ('after_denoise_1', 'after_denoise_2', 'after_denoise_3', 'after_denoise_4'):
        res[k] = compare(snaps[k], snaps['after_fifth'], lo, hi,
                         f'{k} vs after_fifth')
    print('\n  sanity: the fourth denoising forward vs the third')
    res['d4_vs_d3'] = compare(snaps['after_denoise_4'], snaps['after_denoise_3'], lo, hi,
                              'after_denoise_4 vs after_denoise_3')
    if args_ns.out_json:
        json.dump(res, open(args_ns.out_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
