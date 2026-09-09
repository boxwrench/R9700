# gfx1201 Conv3d investigation — running findings

Chronological. Superseded conclusions are kept, marked, and not deleted.

## F1. No MIOpen system database ships for gfx1201 (Experiment A)

`/opt/rocm-7.2.1/share/miopen/db` carries gfx1030, gfx803, gfx900, gfx906,
gfx908, gfx90a, gfx942 and gfx950. There is **no gfx1201 entry**, so every
convolution shape must be resolved into the *user* FindDb.

That user DB does exist and does persist across processes:
`~/.config/miopen/gfx1201_32.HIP.3_5_1_dabb6df2b9.ufdb.txt` (191 entries)
plus `~/.cache/miopen/3.5.1.dabb6df2b9/gfx1201_32.ukdb`. `MIOPEN_USER_DB_PATH`
is honoured. So caching is *not* broken.

## F2. The VAE decode cost is not search, JIT or cache misses (Experiment B)

Three decodes in one process: 110.76 / 110.74 / 110.74 s. Flat.

17 unique Conv3d shapes. The dominant one costs **the same on call 1 as on
call 54** (5.068 s vs 5.061 s). If search or compilation were responsible the
first call would be an outlier. It is not.

One shape accounts for ~82% of the decode:

| Conv3d input (pre-pad) | weight | calls/3 decodes | first | repeat | total |
|---|---|---:|---:|---:|---:|
| `[1,96,4,480,832]` | `[96,96,3,3,3]` | 54 | 5.068 s | 5.061 s | 273.3 s |
| `[1,96,4,480,832]` | `[3,96,3,3,3]` | 9 | 3.086 s | 3.096 s | 27.9 s |
| `[1,192,4,240,416]` | `[192,192,3,3,3]` | 54 | 0.307 s | 0.286 s | 15.4 s |
| `[1,384,2,120,208]` | `[384,384,3,3,3]` | 45 | 0.143 s | 0.136 s | 6.1 s |

## F3. `MIOPEN_FIND_MODE=FAST` changes nothing (Experiment C)

110.65 / 110.68 / 110.74 s, byte-identical output
(`sha256[:32] = 9a45019d4b5646636ed72d67377ac013` in both lanes), and the
per-shape table matches the default lane to three decimals. Consistent with
F2: there is no search to avoid.

## F4. Bypassing MIOpen runs out of memory (Experiment F)

`torch.backends.cudnn.enabled = False` on a clean process OOMs inside the
decoder: *"Tried to allocate 15.43 GiB"*. The native ATen fallback appears to
materialise an im2col buffer. Not usable as-is at 480x832, and therefore not
yet a valid negative control — it needs re-running at a reduced shape.

## F5. Terminology correction

The `transient_gib` column in `logs/expB-default.json` is the **peak PyTorch
allocation delta across the call**, not a demonstrated MIOpen workspace. No
MIOpen-reported workspace figure has been obtained yet. The earlier session
note of a "7.46 GiB workspace" was an allocator figure, not a solver figure.

## F6. Correction: the first microbenchmark used the pre-padding shape

`CausalConv3d` sets `self.padding = (0,0,0)` and applies `F.pad` itself, so
the shape my instrumentation logged (`[1,96,4,480,832]`) is what
`CausalConv3d.forward` receives, **not** what `F.conv3d` receives. The real
convolution sees the padded tensor.

Benchmarked at the logged (pre-pad) shape, the convolution is fast — 0.146 s,
2.70 TFLOP/s — which briefly suggested the cost lay outside the convolution.
That was wrong. At the true padded shape it reproduces exactly:

| input to `F.conv3d` | weight | output | warm median | effective |
|---|---|---|---:|---:|
| `[1,96,4,480,832]` | `[96,96,3,3,3]` | `[1,96,2,478,830]` | **0.146 s** | 2.70 TFLOP/s |
| `[1,96,6,482,834]` | `[96,96,3,3,3]` | `[1,96,4,480,832]` | **5.045 s** | 0.157 TFLOP/s |
| `[1,96,6,482,834]` | `[3,96,3,3,3]` | `[1,3,4,480,832]` | **3.168 s** | 0.005 TFLOP/s |
| `[1,96,4,480,832]` `padding=1` | `[96,96,3,3,3]` | `[1,96,4,480,832]` | **5.549 s** | 0.143 TFLOP/s |

The slow case does **2x** the output work of the fast case and takes **34x**
the time — a ~17x collapse in efficiency. Passing `padding=1` to `F.conv3d`
rather than pre-padding reproduces it too, so it is not specific to `F.pad`.

Neither autocast (fp32 or bf16), `no_grad`, real weights, nor going through
the real `CausalConv3d` module changes the fast-case number (all 0.146-0.148 s),
so pipeline context is ruled out.

**Reproducer** — 5 lines, ~5 s, no VAE, no model:

```python
import torch, torch.nn.functional as F
w = torch.randn(96, 96, 3, 3, 3, device='cuda')
x = torch.randn(1, 96, 6, 482, 834, device='cuda')
F.conv3d(x, w); torch.cuda.synchronize()   # ~5.0 s on gfx1201 / ROCm 7.2.1
```

## Open question

Whether the trigger is the padded *shape* (odd spatial extents 482x834, or
the temporal extent), the tensor *layout*, or the padding content. Sweep in
progress.
