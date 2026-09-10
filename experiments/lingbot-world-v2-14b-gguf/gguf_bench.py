#!/usr/bin/env python
"""Headless harness for the RealRebelAI LingBot-World-V2 GGUF ComfyUI pack on ROCm.

The pack's compute path is: their GGUF streaming loader (LBWorldLoader) plus
their vendored copy of upstream `wan/`, driven through upstream's own
`_generate_causal_fast`. This harness reproduces exactly that path without
standing up ComfyUI, by stubbing `folder_paths` and importing their node module
unchanged. Their `LBWorldSampler.sample` body is reimplemented here only to add
instrumentation -- the pre-flight, resolution presets and generate() call are
kept identical, and any deviation is commented.
"""
import argparse, gc, hashlib, json, math, os, statistics, sys, time, types, warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

LAB = '/ai/lab/lingbot-gguf-r9700'
PACK = os.path.join(LAB, 'ComfyUI_Rebels_LingBotWorld')

import numpy as np
import torch
from PIL import Image


def sync():
    torch.cuda.synchronize()


def mem():
    return dict(alloc=torch.cuda.memory_allocated() / 2**30,
                reserved=torch.cuda.memory_reserved() / 2**30,
                peak_alloc=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved=torch.cuda.max_memory_reserved() / 2**30)


def rss():
    out = {}
    with open('/proc/self/status') as f:
        for line in f:
            for k, n in (('VmRSS:', 'rss'), ('VmSize:', 'vsize'), ('VmPin:', 'pinned')):
                if line.startswith(k):
                    out[n] = int(line.split()[1]) / 2**20
    return out


PHASES = []


def phase(name, reset=False):
    m = mem(); m.update(rss()); m['phase'] = name
    PHASES.append(m)
    print(f'  [mem] {name:26s} alloc {m["alloc"]:6.2f} resv {m["reserved"]:6.2f} '
          f'peak {m["peak_alloc"]:6.2f} rss {m.get("rss", 0):6.2f} '
          f'pinned {m.get("pinned", 0):5.2f}', flush=True)
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return m


def install_stubs(models_dir, vae_dir, action_root):
    """Minimal `folder_paths` so their node module imports unmodified."""
    fp = types.ModuleType('folder_paths')
    fp.get_filename_list = lambda key: (
        [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        if key in ('unet_gguf', 'diffusion_models', 'unet')
        else [f for f in os.listdir(vae_dir)] if key == 'vae' else [])
    fp.get_folder_paths = lambda key: (
        [models_dir] if key in ('unet_gguf', 'diffusion_models', 'unet')
        else [vae_dir] if key == 'vae' else [])
    fp.get_full_path = lambda key, name: os.path.join(
        models_dir if key != 'vae' else vae_dir, name)
    fp.get_input_directory = lambda: action_root
    fp.models_dir = models_dir
    sys.modules['folder_paths'] = fp


class Timer:
    def __init__(self):
        self.spans = {}

    def add(self, n, dt):
        self.spans.setdefault(n, []).append(dt)

    def wrap(self, obj, attr, name, snapshot=False):
        fn = getattr(obj, attr)

        def inner(*a, **kw):
            if snapshot:
                phase(f'before {name}', reset=True)
            sync(); t0 = time.perf_counter()
            out = fn(*a, **kw)
            sync(); self.add(name, time.perf_counter() - t0)
            if snapshot:
                phase(f'after {name}')
            return out
        setattr(obj, attr, inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf')
    ap.add_argument('--models_dir', default='/ai/models/lingbot-gguf')
    ap.add_argument('--vae_dir', default='/ai/models/lingbot-world-v2-14b-assets')
    ap.add_argument('--vae_name', default='Wan2.1_VAE.pth')
    ap.add_argument('--action_root', default='/ai/lab/lingbot-gguf-r9700/input')
    ap.add_argument('--action_dir', default='03')
    ap.add_argument('--image', default='/ai/github/lingbot-world-v2/examples/03/image.jpg')
    ap.add_argument('--resolution', default='480x832 (needs tiny window)')
    ap.add_argument('--frame_num', type=int, default=29)
    ap.add_argument('--chunk_size', type=int, default=3)
    ap.add_argument('--local_attn_size', type=int, default=6)
    ap.add_argument('--sink_size', type=int, default=2)
    ap.add_argument('--shift', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pin_gb', type=int, default=2)
    ap.add_argument('--warm_repeats', type=int, default=1)
    ap.add_argument('--vae_split', default=None,
                    help='WAN_VAE_CONV3D_TEMPORAL_SPLIT for the vendored VAE')
    ap.add_argument('--profile_dequant', action='store_true')
    ap.add_argument('--tag', default='q4-baseline')
    ap.add_argument('--save_dir', default='/ai/outputs/lingbot-world-v2-14b-gguf')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    if args.out_json:
        args.out_json = os.path.abspath(args.out_json)
    args.save_dir = os.path.abspath(args.save_dir)
    if args.vae_split is not None:
        os.environ['WAN_VAE_CONV3D_TEMPORAL_SPLIT'] = str(args.vae_split)

    install_stubs(args.models_dir, args.vae_dir, args.action_root)
    sys.path.insert(0, PACK)          # their vendored wan/
    sys.path.insert(0, LAB)           # so _import_city96_dequant finds ComfyUI-GGUF
    import lbworld_nodes as LB

    rec = dict(tag=args.tag, timestamp=datetime.now(timezone.utc).isoformat(),
               args=vars(args), torch=torch.__version__, hip=torch.version.hip,
               gpu=torch.cuda.get_device_properties(0).gcnArchName,
               vae_split=os.environ.get('WAN_VAE_CONV3D_TEMPORAL_SPLIT', '(default)'),
               runs=[])

    gguf_path = os.path.join(args.models_dir, args.gguf)
    rec['gguf_bytes'] = os.path.getsize(gguf_path)

    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    rec['flash_attn'] = [FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE]

    phase('process start', reset=True)
    t0 = time.perf_counter()
    loader = LB.LBWorldLoader()
    (bundle,) = loader.load(args.gguf, args.vae_name,
                            args.local_attn_size, args.sink_size, pin_gb=args.pin_gb)
    sync()
    rec['load_seconds'] = time.perf_counter() - t0
    rec['pinned_bytes'] = LB._pinned['bytes']
    phase('after loader', reset=True)

    pipe, model, device = bundle['pipe'], bundle['model'], bundle['device']

    # weight residency accounting
    n_wrapped = sum(1 for m in model.modules() if isinstance(m, LB.GGUFLinearWrapper))
    q_bytes = sum(m.qdata.numel() for m in model.modules()
                  if isinstance(m, LB.GGUFLinearWrapper))
    dense_bytes = sum(p.numel() * p.element_size() for p in model.parameters()
                      if p.device.type != 'meta')
    rec['gguf_linear_wrappers'] = n_wrapped
    rec['quantized_bytes_host'] = q_bytes
    rec['dense_param_bytes'] = dense_bytes
    rec['meta_params_left'] = sum(1 for _, p in model.named_parameters()
                                  if p.device.type == 'meta')
    print(f'  wrapped Linear: {n_wrapped}  quant bytes host {q_bytes/2**30:.2f} GiB  '
          f'dense params {dense_bytes/2**30:.2f} GiB  meta left {rec["meta_params_left"]}',
          flush=True)

    timer = Timer()

    # Separate the two halves of the GGUF streaming cost inside their wrapper:
    # the host->device copy of the quantized bytes, and the dequant itself.
    # Both are synchronised, so this instrumentation is not free -- it is only
    # enabled when --profile_dequant is passed.
    if args.profile_dequant:
        LBW = LB.GGUFLinearWrapper
        _orig_fwd = LBW.forward

        def instrumented(self, x):
            q = self.qdata
            if q.device != x.device:
                sync(); t0 = time.perf_counter()
                q = q.to(x.device, non_blocking=True)
                sync(); timer.add('gguf_h2d', time.perf_counter() - t0)
            sync(); t0 = time.perf_counter()
            w = LB.dequant_bytes(q, self.qtype, self.oshape, x.dtype)
            sync(); timer.add('gguf_dequant', time.perf_counter() - t0)
            b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
            sync(); t0 = time.perf_counter()
            out = torch.nn.functional.linear(x, w, b)
            sync(); timer.add('gguf_matmul', time.perf_counter() - t0)
            del w
            return out
        LBW.forward = instrumented

    timer.wrap(pipe.model, 'forward', 'dit_forward')
    timer.wrap(pipe.vae, 'encode', 'vae_encode', snapshot=True)
    timer.wrap(pipe.vae, 'decode', 'vae_decode', snapshot=True)

    img = Image.open(args.image).convert('RGB')
    res_h, res_w = {'256x448 (lowest VRAM)': (256, 448), '320x544': (320, 544),
                    '384x656': (384, 656), '480x832 (needs tiny window)': (480, 832)
                    }[args.resolution]
    max_area = res_h * res_w
    action_path = os.path.join(args.action_root, 'lingbot_actions', args.action_dir)

    # Their pre-flight, reproduced verbatim so we record the same number it would
    # print in ComfyUI. NOTE: it uses (local + sink); upstream allocates
    # frame_seqlen * local_attn_size only -- sink frames live inside the window.
    ar = img.size[1] / img.size[0]
    lat_h = round(math.sqrt(max_area * ar) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / ar) // 8 // 2 * 2)
    tokens = max(int(lat_h * lat_w // 4), 1)
    their_kv_gb = 2 * 40 * (args.local_attn_size + args.sink_size) * tokens * 5120 * 2 / 1e9  # original, wrong
    true_kv_gb = 2 * 40 * args.local_attn_size * tokens * 5120 * 2 / 1e9
    rec.update(lat_h=lat_h, lat_w=lat_w, tokens_per_frame=tokens,
               preflight_kv_gb_their_formula=their_kv_gb,
               kv_gb_upstream_allocation=true_kv_gb)
    print(f'  lat {lat_h}x{lat_w}  {tokens} tok/frame  '
          f'their pre-flight KV {their_kv_gb:.2f} GB  actual upstream KV {true_kv_gb:.2f} GB',
          flush=True)

    # Their sampler registers embeddings through a shim keyed by sha256(text).
    # We have no ComfyUI CLIP node here, so encode with the real T5 once on CPU.
    from wan.modules.t5 import T5EncoderModel
    t5 = T5EncoderModel(
        text_len=pipe.config.text_len, dtype=pipe.config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.vae_dir, pipe.config.t5_checkpoint),
        tokenizer_path=os.path.join(args.vae_dir, pipe.config.t5_tokenizer))
    prompt = ("A serene lakeside scene with a lone tree standing in calm water, "
              "surrounded by distant snow-capped mountains under a bright blue sky "
              "with drifting white clouds — gentle ripples reflect the tree and sky, "
              "creating a tranquil, meditative atmosphere.")
    t_t5 = time.perf_counter()
    emb = t5([prompt], torch.device('cpu'))
    rec['t5_encode_seconds'] = time.perf_counter() - t_t5
    # _T5Shim.set takes the [L, C] tensor their _cond_to_embed would produce
    # from a ComfyUI CONDITIONING, not a list.
    pipe.text_encoder.set(hashlib.sha256(prompt.encode()).hexdigest(),
                          emb[0].to(device).contiguous())
    del t5
    gc.collect()
    phase('after T5 (CPU, released)', reset=True)

    os.makedirs(args.save_dir, exist_ok=True)
    from wan.utils.utils import save_video

    for i in range(1 + args.warm_repeats):
        kind = 'cold' if i == 0 else f'warm{i}'
        timer.spans.clear(); PHASES.clear()
        torch.cuda.reset_peak_memory_stats()
        phase(f'start {kind}')
        model.to(device)
        pipe.vae.model = pipe.vae.model.to(device)
        pipe.vae.mean = pipe.vae.mean.to(device)
        pipe.vae.std = pipe.vae.std.to(device)
        pipe.vae.scale = [pipe.vae.mean, 1.0 / pipe.vae.std]
        pipe.vae.device = device
        phase(f'{kind} after move to GPU')

        sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            video = pipe.generate(prompt, img, action_path,
                                  chunk_size=args.chunk_size, max_area=max_area,
                                  frame_num=args.frame_num, shift=args.shift,
                                  seed=args.seed, offload_model=True)
        sync(); wall = time.perf_counter() - t0
        phase(f'end {kind}')

        v = video.float()
        d = timer.spans.get('dit_forward', [])
        per_chunk = [sum(d[j:j + 5]) for j in range(0, len(d), 5)]
        stats = dict(
            shape=list(v.shape), height=int(v.shape[2]), width=int(v.shape[3]),
            min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
            std=float(v.std()), has_nan=bool(torch.isnan(v).any()),
            has_inf=bool(torch.isinf(v).any()),
            per_frame_std=[float(v[:, f].std()) for f in range(v.shape[1])],
            frame_to_frame_l1=[float((v[:, f + 1] - v[:, f]).abs().mean())
                               for f in range(v.shape[1] - 1)])
        run = dict(kind=kind, wall_seconds=wall, frames=int(video.shape[1]),
                   fps_effective=video.shape[1] / wall,
                   size_requested=f'{res_h}x{res_w}',
                   size_actual=f'{v.shape[2]}x{v.shape[3]}',
                   dit_forwards=len(d), per_chunk_dit_seconds=per_chunk,
                   time_to_first_chunk=(sum(d[:5]) if len(d) >= 5 else None),
                   dit_forward_seconds=list(d),
                   spans={k: dict(n=len(x), total=sum(x), mean=sum(x) / len(x),
                                  first=x[0]) for k, x in timer.spans.items()},
                   phases=list(PHASES), mem=mem(), host=rss(), video_stats=stats)
        rec['runs'].append(run)
        out = os.path.join(args.save_dir,
                           f'{args.tag}_{kind}_{v.shape[2]}x{v.shape[3]}_'
                           f'{video.shape[1]}f_w{args.local_attn_size}_s{args.sink_size}.mp4')
        t_save = time.perf_counter()
        save_video(tensor=video[None].float().cpu(), save_file=out, fps=16, nrow=1,
                   normalize=True, value_range=(-1, 1))
        run['save_seconds'] = time.perf_counter() - t_save
        run['output'] = out
        print(f'[{kind}] {wall:.2f}s  {video.shape[1]}f  {v.shape[2]}x{v.shape[3]}  -> {out}',
              flush=True)
        del video, v
        gc.collect(); torch.cuda.empty_cache()

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(rec, fh, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
