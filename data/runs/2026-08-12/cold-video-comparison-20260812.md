# Process-cold video comparison — 2026-08-12

## Method

- Restarted `comfyui-h3.service` before every lane, waited for the HTTP API, then submitted exactly one graph.
- Every run used a new ComfyUI PID and reported zero cached graph nodes.
- Linux filesystem caches and persistent compiled-kernel caches were preserved; this is process-cold/model-cold, not disk-cold.
- Same brass-robot prompt, numeric seed 8112026, 24 fps, synchronized audio, and approximately 5 seconds in every lane.
- Timing is `execution_success.timestamp - execution_start.timestamp` from ComfyUI history, excluding service startup.

## Results

| Lane | Process startup | Prompt-to-artifact | Restart-to-artifact | Native workload |
|---|---:|---:|---:|---|
| MiniMax H3 Standard FP8 | 5.175 s | 261.038 s | 266.213 s | 864x480, 124 frames, 20 steps |
| MiniMax H3 Turbo v4 FP8 | 3.873 s | 80.927 s | 84.800 s | 864x480, 124 frames, 4 Turbo steps |
| LTX-2.5 distilled INT8 | 3.889 s | 67.035 s | 70.924 s | 896x512, 121 frames, 8+3 steps |

- H3 Turbo is 3.23x faster than H3 Standard.
- LTX-2.5 is 1.21x faster than H3 Turbo and 3.89x faster than H3 Standard by prompt-to-artifact time.
- LTX generated about 8.0% more native pixel-frames than H3 despite its three fewer frames. Its geometry cannot natively match 864x480 because the two-stage half-resolution latent must align to 32 pixels.
- These are controlled practical latency numbers, not architecture-normalized throughput scores. Same numeric seeds are reproducible within a lane but do not map to the same latent across model families.

## Artifacts and validation

### H3 Standard

- Prompt ID: `5897f14a-c699-4b4f-bae7-b67362584320`
- Output: `/ai/artifacts/runs/minimax-h3/minimax-h3/cold-brass-robot-standard-fp8-20_00001_.mp4`
- SHA-256: `780f9b78401fc093269bf27e5364c259ba4ffe06ca70730c8070d5f5b58f95f3`
- H.264 864x480, exactly 124 frames at 24 fps; stereo AAC 32 kHz; 5.167 seconds.

### H3 Turbo v4

- Prompt ID: `c348e5e6-cd3f-44ca-974e-46090986abe8`
- Output: `/ai/artifacts/runs/minimax-h3/minimax-h3/cold-brass-robot-turbo-v4-4_00001_.mp4`
- SHA-256: `abc3e2680b3ca2b188ae9f60300017faa40e0cc349f4f11c60056671a276556e`
- H.264 864x480, exactly 124 frames at 24 fps; stereo AAC 32 kHz; 5.167 seconds.

### LTX-2.5 distilled

- Prompt ID: `bb51d0be-9065-42d9-9965-0e1aad5578bf`
- Output: `/ai/artifacts/runs/ltx-2.5/cold-brass-robot-LTX25-native-896x512.mp4`
- SHA-256: `2adeee6d0e47242b9a391e71c8e4012c6dfc1df4f1436f35c12a051e5dd10ec9`
- H.264 896x512, exactly 121 frames at 24 fps; stereo AAC 48 kHz; 5.042 seconds.

All three video streams decoded fully without error. The run window contained no GPU reset, VM fault, ring timeout, or OOM. Queue-eviction messages occurred only when intentionally restarting the ComfyUI process; one non-fatal AMDGPU SVM workqueue latency warning appeared during H3 Standard loading.

