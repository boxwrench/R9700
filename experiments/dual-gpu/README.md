# MiniMax H3 dual-GPU experiment

Purpose: keep the MiniMax H3 FP8 diffusion model on the Radeon AI PRO R9700 while placing the Qwen3-VL AWQ text encoder on the Radeon RX 7900 XT.

This lane is isolated from the golden `/ai/comfyui` runtime. It uses logical device order:

```text
cuda:0 = AMD Radeon AI PRO R9700 / gfx1201
cuda:1 = Radeon RX 7900 XT / gfx1100
```

The experimental ComfyUI endpoint is <http://127.0.0.1:8191>. The golden endpoint remains <http://127.0.0.1:8190>, but the two services are intentionally prevented from running simultaneously.

The experimental launcher intentionally omits `--disable-smart-memory`. The pinned ComfyUI build defines that flag as aggressive model offload to system RAM, which defeats the encoder-residency objective. It continues to use `--disable-dynamic-vram`, BF16 VAE execution, and a 2 GiB VRAM reserve.

The first run is a functional five-second H3 Turbo shakedown, not a benchmark. No DisTorch, layer splitting, GGUF, model download, or ComfyUI update is used.

Pinned components:

- ComfyUI: `c2bcbecd82ec5ae66594340b395c24ef0217b238`
- ComfyUI-MultiGPU: `b51c99a525e9607e43545ee2a8b7694c74a4775a`
- MiniMax H3 Turbo node: `55fee864dd7b2976b1c4ce3c3d5f7968f181409f`

Rollback:

```bash
systemctl --user stop comfyui-h3-dualgpu.service
systemctl --user start comfyui-h3.service
```
