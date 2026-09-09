#!/usr/bin/env python
"""Instrumented single-GPU baseline harness for lingbot-world-v2-1.3b-causal-fast.

Runs the upstream WanI2VCausal pipeline unchanged and wraps the component
entry points with synchronised timers so cold and warm runs can be reported
separately. Writes one JSON record per invocation.
"""
import argparse, json, os, sys, time, gc
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image


def sync():
    torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.spans = {}

    def add(self, name, dt):
        self.spans.setdefault(name, []).append(dt)

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)

        def inner(*a, **kw):
            sync(); t0 = time.perf_counter()
            out = fn(*a, **kw)
            sync(); self.add(name, time.perf_counter() - t0)
            return out
        setattr(obj, attr, inner)
        return fn


def mem():
    return dict(
        alloc_gib=torch.cuda.memory_allocated() / 2**30,
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
        peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )


def rss_gib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 2**20
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/ai/repos/lingbot-world-v2')
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--task', default='i2v-1.3B')
    ap.add_argument('--size', default='480*832')
    ap.add_argument('--frame_num', type=int, default=29)
    ap.add_argument('--chunk_size', type=int, default=4)
    ap.add_argument('--local_attn_size', type=int, default=18)
    ap.add_argument('--sink_size', type=int, default=6)
    ap.add_argument('--action_path', default=None)
    ap.add_argument('--image', default=None)
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sample_shift', type=float, default=None)
    ap.add_argument('--offload_model', type=int, default=0)
    ap.add_argument('--warm_repeats', type=int, default=1)
    ap.add_argument('--prewarm', type=int, default=0)
    ap.add_argument('--save_dir', default='output')
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    from wan.modules.attention import (FLASH_ATTN_2_AVAILABLE,
                                       FLASH_ATTN_3_AVAILABLE)

    cfg = WAN_CONFIGS[args.task]
    shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    action_path = args.action_path or os.path.join(args.repo, 'examples/03')
    image = args.image or os.path.join(args.repo, 'examples/03/image.jpg')
    prompt = args.prompt or (
        "A serene lakeside scene with a lone tree standing in calm water, "
        "surrounded by distant snow-capped mountains under a bright blue sky "
        "with drifting white clouds — gentle ripples reflect the tree and sky, "
        "creating a tranquil, meditative atmosphere.")

    rec = dict(
        tag=args.tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        args=vars(args),
        shift=shift,
        gpu=torch.cuda.get_device_properties(0).name,
        gcn_arch=torch.cuda.get_device_properties(0).gcnArchName,
        torch=torch.__version__,
        hip=torch.version.hip,
        flash_attn_2=FLASH_ATTN_2_AVAILABLE,
        flash_attn_3=FLASH_ATTN_3_AVAILABLE,
        runs=[],
    )

    img = Image.open(image).convert('RGB')

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
        convert_model_dtype=False,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        infer_mode='causal_fast')
    sync()
    rec['load_seconds'] = time.perf_counter() - t0
    rec['after_load_mem'] = mem()
    rec['after_load_rss_gib'] = rss_gib()

    dt = {}
    for n, p in pipe.model.named_parameters():
        dt[str(p.dtype)] = dt.get(str(p.dtype), 0) + p.numel()
    rec['dit_param_dtypes'] = dt
    rec['dit_param_bytes'] = sum(p.numel() * p.element_size()
                                 for p in pipe.model.parameters())
    rec['dit_device'] = str(next(pipe.model.parameters()).device)
    vdt = {}
    for p in pipe.vae.model.parameters():
        vdt[str(p.dtype)] = vdt.get(str(p.dtype), 0) + p.numel()
    rec['vae_param_dtypes'] = vdt

    timer = Timer()
    timer.wrap(pipe.model, 'forward', 'dit_forward')
    timer.wrap(pipe.vae, 'decode', 'vae_decode')
    timer.wrap(pipe.vae, 'encode', 'vae_encode')
    timer.wrap(pipe.text_encoder, '__call__', 't5_encode')

    if args.prewarm:
        sync(); tp = time.perf_counter()
        pipe.prewarm(img, max_area=MAX_AREA_CONFIGS[args.size],
                     frame_num=args.frame_num, chunk_size=args.chunk_size)
        sync(); rec['prewarm_seconds'] = time.perf_counter() - tp

    os.makedirs(args.save_dir, exist_ok=True)
    n_runs = 1 + args.warm_repeats
    for i in range(n_runs):
        kind = 'cold' if i == 0 else f'warm{i}'
        timer.spans.clear()
        torch.cuda.reset_peak_memory_stats()
        pre = mem()
        sync(); t0 = time.perf_counter()
        video = pipe.generate(
            prompt, img, action_path=action_path,
            chunk_size=args.chunk_size,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num, shift=shift, seed=args.seed,
            offload_model=bool(args.offload_model))
        sync(); wall = time.perf_counter() - t0
        post = mem()

        v = video.float()
        stats = dict(
            shape=list(v.shape),
            min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
            std=float(v.std()),
            has_nan=bool(torch.isnan(v).any()), has_inf=bool(torch.isinf(v).any()),
            per_frame_std=[float(v[:, f].std()) for f in range(v.shape[1])],
            frame_to_frame_l1=[float((v[:, f + 1] - v[:, f]).abs().mean())
                               for f in range(v.shape[1] - 1)],
        )

        f = video.shape[1]
        run = dict(
            kind=kind, wall_seconds=wall, frames=f, size=args.size,
            fps_effective=f / wall,
            spans={k: dict(n=len(x), total=sum(x), mean=sum(x) / len(x),
                           first=x[0], rest_mean=(sum(x[1:]) / len(x[1:])
                                                  if len(x) > 1 else None))
                   for k, x in timer.spans.items()},
            mem_before=pre, mem_after=post, rss_gib=rss_gib(),
            video_stats=stats,
        )
        # per-chunk latency: dit_forward spans grouped 5 per chunk
        d = timer.spans.get('dit_forward', [])
        per_chunk = [sum(d[j:j + 5]) for j in range(0, len(d), 5)]
        run['dit_forwards'] = len(d)
        run['per_chunk_dit_seconds'] = per_chunk
        run['time_to_first_chunk'] = sum(d[:5]) if len(d) >= 5 else None
        rec['runs'].append(run)

        out = os.path.join(args.save_dir,
                           f'{args.tag}_{kind}_{args.size.replace("*","x")}_'
                           f'{f}f_w{args.local_attn_size}_s{args.sink_size}.mp4')
        save_video(tensor=video[None], save_file=out, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        run['output'] = out
        print(f'[{kind}] {wall:.2f}s  {f} frames  -> {out}', flush=True)
        del video, v
        gc.collect()

    js = json.dumps(rec, indent=2)
    if args.out_json:
        with open(args.out_json, 'w') as fh:
            fh.write(js)
    print(js)


if __name__ == '__main__':
    main()
