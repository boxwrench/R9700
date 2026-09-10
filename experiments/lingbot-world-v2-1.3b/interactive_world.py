#!/usr/bin/env python
"""Persistent interactive LingBot-World-V2 1.3B session on a single R9700.

One process, one model load, one world. The KV cache, the cross-attention
cache, the VAE decoder's causal feature cache and the camera pose are all
created once at init and mutated in place; every action advances them by one
causal chunk. Nothing is rebuilt between actions.

This deliberately breaks open upstream's `_generate_causal_fast`, which
allocates the caches, loops over every chunk and decodes once at the end. The
per-chunk maths is kept identical -- same four timesteps, same fifth
KV-writing forward, same Plucker conditioning, same scheduler -- with three
changes that interactivity requires and which are commented at their sites:
translation normalisation, incremental VAE decode, and the conditioning
tensor being pre-encoded for a bounded horizon instead of a known length.

Actions are camera motion. That is not a simplification: the causal-fast
checkpoint's only conditioning channel is a 6-dim Plucker ray embedding
(`control_dim = 6` in model_fast.py), and upstream sets `wasd_action = None`
unconditionally. There is no attack/jump/interact input in this model.
"""
import argparse, contextlib, json, math, os, queue, sys, threading, time
from datetime import datetime, timezone

import numpy as np
import torch

REPO = '/ai/repos/lingbot-world-v2'
sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Camera actions. OpenCV camera axes: +x right, +y down, +z forward.
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    'forward': np.array([0.0, 0.0, 1.0]),
    'back':    np.array([0.0, 0.0, -1.0]),
    'left':    np.array([-1.0, 0.0, 0.0]),
    'right':   np.array([1.0, 0.0, 0.0]),
    'up':      np.array([0.0, -1.0, 0.0]),
    'down':    np.array([0.0, 1.0, 0.0]),
}
# +y is down, so a negative yaw about y swings the forward axis toward -x.
ROTATIONS = {
    'turn_left':  ('y', -1.0),
    'turn_right': ('y', +1.0),
    'tilt_up':    ('x', -1.0),
    'tilt_down':  ('x', +1.0),
}
ALIASES = {
    'w': 'forward', 's': 'back', 'a': 'left', 'd': 'right',
    'q': 'turn_left', 'e': 'turn_right',
    'r': 'up', 'f': 'down',
    't': 'tilt_up', 'g': 'tilt_down',
    'x': 'stay', 'strafe_left': 'left', 'strafe_right': 'right',
}
ACTIONS = set(TRANSLATIONS) | set(ROTATIONS) | {'stay'}


def rot_matrix(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.eye(3)
    i, j = {'y': (0, 2), 'x': (1, 2)}[axis]
    R[i, i] = c
    R[j, j] = c
    R[i, j] = s
    R[j, i] = -s
    return R


def sync():
    torch.cuda.synchronize()


def gib(x):
    return x / 2**30


class IncrementalDecoder:
    """Upstream's `Wan2_1_VAE.decode`, minus the cache clears, as a generator.

    `decode` calls `clear_cache()` on entry and exit, which is right for a
    one-shot decode of the whole latent sequence and wrong here: the decoder's
    3D convolutions are causal and carry a per-conv feature cache across latent
    frames. Dropping that cache between chunks would restart the temporal
    context and put a visible seam at every chunk boundary. This keeps it.

    The loop is already frame-serial -- one decoder pass per *latent* frame,
    producing 1 pixel frame for the very first frame of a session and 4 for
    every one after. Yielding inside that loop is what makes the first frames
    of a chunk visible without waiting for the rest, and costs nothing.
    """

    def __init__(self, vae, dtype=torch.float32):
        self.vae = vae
        self.m = vae.model
        self.dtype = dtype
        self.m.clear_cache()
        self.started = False

    def iter_decode(self, z):
        """z: [C, T, H, W] -> yields [3, t, H*8, W*8] once per latent frame."""
        m = self.m
        scale = self.vae.scale
        # de-scaling stays in fp32 regardless of the decoder's own precision:
        # it is a per-channel affine on the latent, costs nothing, and keeps
        # the low-precision cast to exactly one place.
        z = z.unsqueeze(0).to(torch.float32)
        z = z / scale[1].view(1, m.z_dim, 1, 1, 1) + scale[0].view(1, m.z_dim, 1, 1, 1)
        x = m.conv2(z.to(self.dtype))
        for i in range(x.shape[2]):
            m._conv_idx = [0]
            out = m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map,
                            feat_idx=m._conv_idx)
            self.started = True
            yield out.float().clamp_(-1, 1).squeeze(0)

    def decode_chunk(self, z):
        return torch.cat(list(self.iter_decode(z)), dim=1)


class World:
    def __init__(self, args):
        self.args = args
        from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
        import wan
        from PIL import Image
        import torchvision.transforms.functional as TF

        self.cfg = cfg = WAN_CONFIGS[args.task]
        self.device = torch.device('cuda:0')
        t0 = time.perf_counter()

        self.pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
            local_attn_size=args.local_attn_size, sink_size=args.sink_size,
            infer_mode='causal_fast')
        sync()
        self.t_model_load = time.perf_counter() - t0
        pipe = self.pipe

        # ---- geometry, derived exactly as upstream does ----
        img = Image.open(args.image).convert('RGB')
        self.img_size = img.size
        max_area = MAX_AREA_CONFIGS[args.size]
        h0, w0 = img.size[1], img.size[0]
        ar = h0 / w0
        self.lat_h = round(np.sqrt(max_area * ar) // cfg.vae_stride[1]
                           // cfg.patch_size[1] * cfg.patch_size[1])
        self.lat_w = round(np.sqrt(max_area / ar) // cfg.vae_stride[2]
                           // cfg.patch_size[2] * cfg.patch_size[2])
        self.h = self.lat_h * cfg.vae_stride[1]
        self.w = self.lat_w * cfg.vae_stride[2]
        self.frame_seqlen = self.lat_h * self.lat_w // 4
        self.chunk = args.chunk_size
        self.max_lat_f = args.max_lat_frames - (args.max_lat_frames % self.chunk)

        # ---- prompt, then release the encoder ----
        t0 = time.perf_counter()
        pipe.text_encoder.model.to(self.device)
        self.context = pipe.text_encoder([args.prompt], self.device)
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        sync()
        self.t_t5 = time.perf_counter() - t0

        # ---- conditioning image, encoded once for the whole horizon ----
        # Upstream encodes concat([image, zeros(F-1)]) for a known F. Here the
        # session length is open-ended, so encode a bounded horizon once at
        # init and slice it per chunk. Identical arithmetic, paid once.
        #
        # On gfx1201 that encode is the single largest start-up cost -- about
        # 2.9 s per latent frame, so ~260 s for a 90-frame horizon -- and it
        # depends only on (image, resolution, horizon), never on the session.
        # Cache it. Tiling a converged tail frame instead was considered and
        # rejected: the zero-input latents decay but do not actually converge
        # (still drifting 8.3e-03 at latent frame 13), so tiling would be an
        # approximation where caching is exact.
        t0 = time.perf_counter()
        F = (self.max_lat_f - 1) * 4 + 1
        px = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        px = torch.nn.functional.interpolate(
            px[None].cpu(), size=(self.h, self.w), mode='bicubic').transpose(0, 1)

        import hashlib
        key = hashlib.sha256(
            open(args.image, 'rb').read() +
            f'{self.h}x{self.w}x{self.max_lat_f}'.encode()).hexdigest()[:24]
        cache_dir = args.cond_cache or os.path.join(
            os.path.expanduser('~'), '.cache', 'lingbot-world-cond')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{key}.pt')
        self.cond_cache_file = cache_file
        if os.path.isfile(cache_file):
            y = torch.load(cache_file, map_location=self.device)
            self.cond_cached = True
        else:
            y = pipe.vae.encode([torch.concat(
                [px, torch.zeros(3, F - 1, self.h, self.w)], dim=1).to(self.device)])[0]
            torch.save(y.cpu(), cache_file)
            y = y.to(self.device)
            self.cond_cached = False
        msk = torch.ones(1, F, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                            msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2)[0]
        self.y = torch.concat([msk, y])
        sync()
        self.t_vae_encode = time.perf_counter() - t0

        # ---- intrinsics, transformed to the working resolution ----
        from wan.utils.cam_utils import get_Ks_transformed
        fx = (832.0 / 2.0) / math.tan(math.radians(args.fov_deg) / 2.0)
        K = torch.tensor([[fx, fx, 832.0 / 2.0, 480.0 / 2.0]], dtype=torch.float32)
        self.K = get_Ks_transformed(K, height_org=480, width_org=832,
                                    height_resize=self.h, width_resize=self.w,
                                    height_final=self.h, width_final=self.w)[0].to(self.device)

        # ---- scheduler + noise ----
        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.shift)
        self.timesteps = pipe.scheduler.timesteps[args.timesteps_index]
        self.seed = args.seed
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.seed)

        # ---- the persistent caches: allocated once, never rebuilt ----
        self.kv_size = (self.frame_seqlen * args.local_attn_size
                        if args.local_attn_size > -1
                        else self.frame_seqlen * self.max_lat_f)
        m = pipe.model.config
        head_dim = m.dim // m.num_heads
        self.kv_cache = pipe._initialize_self_kv_cache(
            num_layers=m.num_layers,
            shape=[1, self.kv_size, m.num_heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        self.cross_cache = pipe._initialize_crossattn_cache(
            num_layers=m.num_layers,
            shape=[1, cfg.text_len, m.num_heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        self.cross_initialized = False
        self.kv_bytes = sum(c['k'].numel() * c['k'].element_size() +
                            c['v'].numel() * c['v'].element_size()
                            for c in self.kv_cache)

        # ---- decoder precision ----
        # The conditioning encode above ran in upstream FP32. Cast only now,
        # so the encode is untouched and the cast applies to decode alone.
        self.decode_dtype = {'fp32': torch.float32, 'fp16': torch.float16,
                             'bf16': torch.bfloat16}[args.vae_dtype]
        if self.decode_dtype is not torch.float32:
            pipe.vae.model = pipe.vae.model.to(self.decode_dtype)
        self.decoder = IncrementalDecoder(pipe.vae, self.decode_dtype)

        # ---- world pose state ----
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.lat_frames_done = 0
        self.chunks_done = 0
        self.history = []
        sync()

    # -- action -> per-latent-frame relative poses -------------------------
    def _relative_poses(self, action, amount):
        """One 4x4 relative pose per latent frame in this chunk.

        Upstream builds absolute poses for the whole trajectory, converts to
        framewise relatives, then divides every translation by the largest one
        in the sequence. That global normalisation cannot work interactively:
        the scale of "forward" would depend on moves the player has not made
        yet, and would change retroactively. Instead each step emits a
        framewise relative whose translation is already unit-scaled, which is
        what upstream's normalisation produces for a constant-speed segment.
        """
        poses = []
        for _ in range(self.chunk):
            dR = np.eye(3)
            dt = np.zeros(3)
            if action in TRANSLATIONS:
                dt = TRANSLATIONS[action] * amount
            elif action in ROTATIONS:
                ax, sign = ROTATIONS[action]
                dR = rot_matrix(ax, sign * amount)
            self.R = self.R @ dR
            self.t = self.t + self.R @ dt
            p = np.eye(4)
            p[:3, :3] = dR
            p[:3, 3] = dt
            poses.append(p)
        rel = torch.from_numpy(np.stack(poses)).float()
        if self.lat_frames_done == 0:
            rel[0] = torch.eye(4)          # upstream forces frame 0 to identity
        n = torch.norm(rel[:, :3, 3], dim=-1).max()
        if n > 0:
            rel[:, :3, 3] = rel[:, :3, 3] / n
        return rel.to(self.device)

    def _plucker(self, rel):
        from wan.utils.cam_utils import get_plucker_embeddings
        from einops import rearrange
        emb = get_plucker_embeddings(rel, self.K.repeat(len(rel), 1), self.h, self.w)
        emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                        c1=self.h // self.lat_h, c2=self.w // self.lat_w)[None]
        return rearrange(emb, 'b (f h w) c -> b c f h w',
                         f=len(rel), h=self.lat_h, w=self.lat_w).to(self.cfg.param_dtype)

    # -- one action = one causal chunk -------------------------------------
    @torch.no_grad()
    def step(self, action, amount=None):
        if self.lat_frames_done + self.chunk > self.max_lat_f:
            raise RuntimeError(
                f'session horizon reached ({self.max_lat_f} latent frames). '
                'Raise --max_lat_frames or use `reset`.')
        amount = amount if amount is not None else (
            self.args.turn_deg if action in ROTATIONS else self.args.move_amount)
        pipe = self.pipe
        torch.cuda.reset_peak_memory_stats()
        sync()
        t_start = time.perf_counter()

        i0 = self.lat_frames_done
        rel = self._relative_poses(action, amount)
        plucker = self._plucker(rel)
        cond = self.y[:, i0:i0 + self.chunk]
        latent = torch.randn(16, self.chunk, self.lat_h, self.lat_w,
                             dtype=torch.float32, generator=self.gen,
                             device=self.device)

        kwargs = dict(
            context=[self.context[0]],
            seq_len=self.chunk * self.frame_seqlen,
            y=[cond],
            dit_cond_dict={'c2ws_plucker_emb': plucker.chunk(1, dim=0)},
            kv_cache=self.kv_cache,
            crossattn_cache=self.cross_cache,
            current_start=i0 * self.frame_seqlen,      # KV write position
            max_attention_size=self.kv_size,
            frame_seqlen=self.frame_seqlen,
        )

        t_dit = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
            for ti in range(len(self.timesteps)):
                ts = torch.stack([self.timesteps[ti]]).to(self.device)
                noise_pred = pipe.model(x=[latent], t=ts,
                                        cross_attn_first_call=not self.cross_initialized,
                                        **kwargs)[0]
                self.cross_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred, xt=latent,
                    timestep=self.timesteps[ti], scheduler=pipe.scheduler)
                if ti < len(self.timesteps) - 1:
                    latent = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=self.gen,
                                        device=x0.device, dtype=x0.dtype),
                        self.timesteps[ti + 1])
            # fifth forward: writes the accepted x0 into the KV cache. This is
            # what makes the next action continue this world instead of a new one.
            pipe.model(x=[x0], t=torch.stack([self.timesteps[-1] * 0.0]).to(self.device),
                       cross_attn_first_call=False, **kwargs)
        sync()
        t_dit_done = time.perf_counter()

        self.lat_frames_done += self.chunk
        self.chunks_done += 1
        rec = dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action, amount=amount, chunk_index=self.chunks_done - 1,
            dit_seconds=t_dit_done - t_dit,
            setup_seconds=t_dit - t_start,
            t_action_start=t_start, t_dit_done=t_dit_done,
            latent_frames_done=self.lat_frames_done,
            kv_global_end=int(self.kv_cache[0]['global_end_index'].item()),
            kv_local_end=int(self.kv_cache[0]['local_end_index'].item()),
            kv_capacity_tokens=self.kv_size,
            kv_evicting=bool(self.kv_cache[0]['global_end_index'].item()
                             > self.kv_cache[0]['local_end_index'].item()),
            seed=self.seed,
            camera_position=self.t.tolist(),
            alloc_gib=gib(torch.cuda.memory_allocated()),
            peak_alloc_gib=gib(torch.cuda.max_memory_allocated()),
            reserved_gib=gib(torch.cuda.memory_reserved()),
        )
        self.history.append(rec)
        # x0 is returned undecoded on purpose: decoding is presentation work.
        # The next causal step reads the KV cache and its own noise, never the
        # RGB, so the caller is free to decode this on another stream while the
        # next action's DiT already runs.
        return x0, rec

    def reset(self):
        """Wipe the world back to frame 0 without reloading anything."""
        for c in self.kv_cache:
            c['k'].zero_(); c['v'].zero_()
            c['global_end_index'].zero_(); c['local_end_index'].zero_()
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.lat_frames_done = 0
        self.chunks_done = 0
        self.gen.manual_seed(self.seed)
        self.decoder = IncrementalDecoder(self.pipe.vae, self.decode_dtype)
        torch.cuda.empty_cache()


class Presenter:
    """Decodes chunks and emits frames the instant each one exists.

    Runs on its own HIP stream in its own thread so that the next action's DiT
    can start while the previous chunk is still being decoded. That is safe
    because the two touch disjoint mutable state: the DiT owns the KV and
    cross-attention caches, the decoder owns the VAE feature cache, and the
    next causal step never reads decoded RGB. The handoff is a CUDA event on
    the latent, plus `record_stream` so the caching allocator cannot reuse that
    memory before the decode stream is done with it.

    Whether the two *actually* overlap on the device rather than time-slicing
    is a separate question, and is measured rather than assumed -- see
    `--overlap 0` for the serial control.
    """

    def __init__(self, world, out_dir, args):
        self.world = world
        self.out = out_dir
        self.args = args
        self.frames_dir = os.path.join(out_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        self.q = queue.Queue()
        self.results = queue.Queue()
        self.inflight = 0
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream() if args.overlap else None
        self.all_frames = []
        self.frame_counter = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, x0, rec):
        ev = torch.cuda.Event()
        ev.record()
        if self.stream is not None:
            x0.record_stream(self.stream)
        with self.lock:
            self.inflight += 1
        self.q.put((x0, rec, ev))

    def _emit(self, tensor, rec):
        """Write one decoded group of pixel frames as PNGs, immediately."""
        from PIL import Image
        arr = ((tensor.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
        paths = []
        for i in range(arr.shape[1]):
            im = Image.fromarray(arr[:, i].transpose(1, 2, 0))
            path = os.path.join(self.frames_dir, f'f_{self.frame_counter:05d}.png')
            im.save(path, compress_level=1)
            self.frame_counter += 1
            paths.append(path)
        return paths

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            x0, rec, ev = item
            ctx = (torch.cuda.stream(self.stream) if self.stream is not None
                   else contextlib.nullcontext())
            groups = []
            t_first = None
            with ctx, torch.no_grad():
                if self.stream is not None:
                    self.stream.wait_event(ev)
                t_dec0 = time.perf_counter()
                for g in self.world.decoder.iter_decode(x0):
                    if self.stream is not None:
                        self.stream.synchronize()
                    else:
                        torch.cuda.synchronize()
                    if t_first is None:
                        self._emit(g, rec)
                        t_first = time.perf_counter()
                        rec['t_first_frame'] = t_first
                        rec['first_frame_latency'] = t_first - rec['t_action_start']
                    else:
                        self._emit(g, rec)
                    groups.append(g.cpu())
                t_dec1 = time.perf_counter()
            frames = torch.cat(groups, dim=1)
            rec['vae_decode_seconds'] = t_dec1 - t_dec0
            rec['latency_seconds'] = t_dec1 - rec['t_action_start']
            rec['frames'] = int(frames.shape[1])
            rec['fps_effective'] = frames.shape[1] / rec['latency_seconds']
            rec['peak_alloc_gib'] = gib(torch.cuda.max_memory_allocated())
            self.all_frames.append(frames)
            if not self.args.no_chunk_mp4:
                save_mp4(frames, os.path.join(
                    self.out, f'chunk_{rec["chunk_index"]:03d}.mp4'), self.args.fps)
                rec['file'] = f'chunk_{rec["chunk_index"]:03d}.mp4'
            with self.lock:
                self.inflight -= 1
            self.results.put(rec)

    def drain(self, log, quiet=False):
        """Collect every finished chunk record that is ready."""
        done = []
        while True:
            try:
                rec = self.results.get_nowait()
            except queue.Empty:
                break
            done.append(rec)
            log.write(json.dumps({k: v for k, v in rec.items()
                                  if not k.startswith('t_')}) + '\n')
            log.flush()
            if not quiet:
                report(rec)
        return done

    def wait_idle(self, log, quiet=False):
        out = []
        while True:
            out += self.drain(log, quiet)
            with self.lock:
                if self.inflight == 0 and self.results.empty():
                    break
            time.sleep(0.02)
        return out + self.drain(log, quiet)

    def busy(self):
        with self.lock:
            return self.inflight > 0

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=5)


def report(rec):
    ff = rec.get('first_frame_latency')
    print(f'  {rec["action"]:<10} first-frame {ff:6.2f}s  full {rec["latency_seconds"]:6.2f}s  '
          f'{rec["frames"]:2d} frames  | dit {rec["dit_seconds"]:5.2f}s '
          f'vae {rec["vae_decode_seconds"]:5.2f}s '
          f'| kv {rec["kv_local_end"]}/{rec["kv_capacity_tokens"]} '
          f'{"EVICTING" if rec["kv_evicting"] else "filling"} '
          f'| vram {rec["peak_alloc_gib"]:.2f} GiB', flush=True)


def save_mp4(frames, path, fps):
    from wan.utils.utils import save_video
    save_video(tensor=frames[None].cpu(), save_file=path, fps=fps, nrow=1,
               normalize=True, value_range=(-1, 1))


HELP = """
commands
  w / forward      move forward          s / back       move backward
  a / left         strafe left           d / right      strafe right
  q / turn_left    yaw left              e / turn_right yaw right
  r / up           rise                  f / down       descend
  t / tilt_up      pitch up              g / tilt_down  pitch down
  x / stay         hold position
  <action> <amt>   override step size (metres, or degrees for turns)
  script w w e w   run several actions in sequence
  reset            wipe world state, keep the model loaded
  stats            show session state
  help             this text
  quit             end session and write the combined video
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--ckpt_dir',
                    default='/ai/models/lingbot-world-v2-1.3b-causal-fast-assembled')
    ap.add_argument('--task', default='i2v-1.3B')
    ap.add_argument('--size', default='480*832')
    ap.add_argument('--chunk_size', type=int, default=1,
                    help='latent frames per action. 1 -> 4 video frames and the '
                         'lowest keypress-to-frame latency; 2 and 3 are also '
                         'valid and trade responsiveness for frames per action.')
    ap.add_argument('--local_attn_size', type=int, default=18)
    ap.add_argument('--sink_size', type=int, default=6)
    ap.add_argument('--max_lat_frames', type=int, default=90)
    ap.add_argument('--shift', type=float, default=10.0)
    ap.add_argument('--timesteps_index', type=int, nargs='+',
                    default=[0, 250, 500, 750])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--move_amount', type=float, default=1.0)
    ap.add_argument('--turn_deg', type=float, default=8.0)
    ap.add_argument('--fov_deg', type=float, default=60.0)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--vae_dtype', default='fp16', choices=['fp32', 'fp16', 'bf16'],
                    help='VAE decoder precision. fp16 measured 3.77x faster than '
                         'fp32 at 3.3e-03 relative error and half the peak memory; '
                         'bf16 is slightly slower AND 8x less accurate here.')
    ap.add_argument('--overlap', type=int, default=0,
                    help='decode on a second HIP stream so the next DiT can start. '
                         'Measured on gfx1201 as a net loss -- the VAE decode '
                         'saturates the device, so the two serialize anyway and '
                         'the contention inflates DiT latency ~5x. Off by default.')
    ap.add_argument('--queue_max', type=int, default=1,
                    help='bounded in-flight chunk queue. 1 keeps each action '
                         'uncontended, which is what minimises felt latency; '
                         'raise to 2-4 to let input queue during presentation.')
    ap.add_argument('--no_chunk_mp4', action='store_true',
                    help='skip per-chunk mp4s; PNGs and session.mp4 still written')
    ap.add_argument('--cond_cache', default=None,
                    help='where to cache the encoded conditioning horizon')
    ap.add_argument('--script', type=str, default=None,
                    help='space-separated actions to run non-interactively, then exit')
    args = ap.parse_args()

    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)

    print('loading world ...', flush=True)
    world = World(args)
    from PIL import Image
    Image.open(args.image).convert('RGB').resize((world.w, world.h)).save(
        os.path.join(out, 'initial.png'))

    meta = dict(
        created=datetime.now(timezone.utc).isoformat(),
        args=vars(args), resolution=f'{world.h}x{world.w}',
        latent=f'{world.lat_h}x{world.lat_w}',
        frame_seqlen=world.frame_seqlen,
        frames_per_chunk=(world.chunk - 1) * 4 + 1,
        kv_capacity_tokens=world.kv_size,
        kv_cache_gib=gib(world.kv_bytes),
        kv_window_frames=args.local_attn_size,
        sink_frames=args.sink_size,
        model_load_seconds=world.t_model_load,
        t5_seconds=world.t_t5,
        vae_encode_seconds=world.t_vae_encode,
        conditioning_cache_hit=world.cond_cached,
        conditioning_cache_file=world.cond_cache_file,
        torch=torch.__version__, hip=torch.version.hip,
        gpu=torch.cuda.get_device_properties(0).gcnArchName,
        vae_decode_dtype=args.vae_dtype,
        vae_conv3d_temporal_split=os.environ.get(
            'WAN_VAE_CONV3D_TEMPORAL_SPLIT', '(default 1)'),
    )
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    init_total = world.t_model_load + world.t_t5 + world.t_vae_encode
    print(f'\nworld ready in {init_total:.1f}s '
          f'(model {world.t_model_load:.1f}s, T5 {world.t_t5:.1f}s, '
          f'image {world.t_vae_encode:.1f}s'
          f'{" [cached]" if world.cond_cached else " [encoded, cached for next time]"})')
    print(f'  {world.h}x{world.w}  latent {world.lat_h}x{world.lat_w}  '
          f'{world.frame_seqlen} tokens/frame')
    print(f'  KV cache {gib(world.kv_bytes):.2f} GiB  window {args.local_attn_size} '
          f'frames  sink {args.sink_size}')
    print(f'  chunk {world.chunk} latent frames  seed {args.seed}')
    print(HELP)

    actions_log = open(os.path.join(out, 'actions.jsonl'), 'a')
    presenter = Presenter(world, out, args)

    def do(action, amount=None):
        t_entered = time.perf_counter()
        x0, rec = world.step(action, amount)
        rec['t_entered'] = t_entered
        rec['dit_complete_latency'] = rec['t_dit_done'] - rec['t_action_start']
        rec['queue_wait'] = rec['t_action_start'] - t_entered
        presenter.submit(x0, rec)
        # Bounded queue: never let more than --queue_max chunks be in flight,
        # so input cannot build up unboundedly behind a slow decoder.
        while presenter.inflight >= args.queue_max:
            presenter.drain(actions_log)
            time.sleep(0.02)
        presenter.drain(actions_log)

    def finish():
        presenter.wait_idle(actions_log)
        if presenter.all_frames:
            combined = torch.cat(presenter.all_frames, dim=1)
            p = os.path.join(out, 'session.mp4')
            save_mp4(combined, p, args.fps)
            print(f'\ncombined session video: {p}  ({combined.shape[1]} frames)')
            print(f'individual frames: {presenter.frames_dir}  '
                  f'({presenter.frame_counter} PNGs, written as decoded)')
        presenter.close()
        actions_log.close()

    if args.script:
        for tok in args.script.split():
            a = ALIASES.get(tok.lower(), tok.lower())
            if a not in ACTIONS:
                print(f'  unknown action {tok!r}, skipping')
                continue
            do(a)
        finish()
        return

    while True:
        try:
            line = input('world> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        presenter.drain(actions_log)
        if not line:
            continue
        parts = line.split()
        cmd = ALIASES.get(parts[0].lower(), parts[0].lower())
        if cmd in ('quit', 'exit'):
            break
        if cmd == 'help':
            print(HELP); continue
        if cmd == 'reset':
            presenter.wait_idle(actions_log)
            world.reset(); presenter.all_frames.clear()
            print('  world reset (model still loaded)'); continue
        if cmd == 'stats':
            print(f'  chunks {world.chunks_done}  latent frames '
                  f'{world.lat_frames_done}/{world.max_lat_f}  '
                  f'camera {np.round(world.t, 2).tolist()}  '
                  f'in flight {presenter.inflight}  '
                  f'alloc {gib(torch.cuda.memory_allocated()):.2f} GiB')
            continue
        if cmd == 'script':
            for tok in parts[1:]:
                a = ALIASES.get(tok.lower(), tok.lower())
                if a in ACTIONS:
                    do(a)
                else:
                    print(f'  unknown action {tok!r}')
            continue
        if cmd not in ACTIONS:
            print(f'  unknown command {parts[0]!r} — try `help`'); continue
        amt = float(parts[1]) if len(parts) > 1 else None
        try:
            do(cmd, amt)
        except RuntimeError as e:
            print(f'  {e}')

    finish()


if __name__ == '__main__':
    main()
