# Workflow inventory

The JSON files in `workflows/` are copied byte-for-byte from the local ComfyUI
workflows and benchmark records. Do not reformat them when comparing hashes.

| Lane | File | SHA-256 |
|---|---|---|
| H3 Standard UI | `workflows/minimax-h3/MiniMax-H3-Native-FP8.json` | `23353acceadd769c352bc5a2fd367712ca448de1505c0946d570cd4d7d10b277` |
| H3 Turbo UI | `workflows/minimax-h3/MiniMax-H3-Turbo-v4-FP8.json` | `da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f` |
| LTX-2.5 comparison UI | `workflows/ltx-2.5/LTX-2.5-Brass-Robot-Comparison.json` | `db11db4591a280c5e668882084a3b40f62c0d1287fe841541deaae7e7736a4bf` |
| H3 Standard reproducible copy | `workflows/minimax-h3/minimax-h3-native-fp8.json` | `23353acceadd769c352bc5a2fd367712ca448de1505c0946d570cd4d7d10b277` |
| H3 Turbo reproducible copy | `workflows/minimax-h3/minimax-h3-turbo-v4-fp8.json` | `da892ad99a5491dd1e100a9428972b4215f75e5e7c894b10c9c42d4965a1d23f` |
| H3 quality API graph | `workflows/minimax-h3/minimax-h3-quality-native-fp8-20-api.json` | `66d8ea1c62d5d25b6bbdca2390fd1c3f393505af9f83ea29c7a127a216155423` |
| H3 dual-GPU UI | workflows/minimax-h3/MiniMax-H3-Turbo-v4-FP8-DualGPU-Qwen-on-7900XT.json | 70754239ca071fae7e3c2df4a07c6991c32dd3a451f10a41b9cee1ada4e64d06 |
| H3 dual-GPU API/shakedown | workflows/minimax-h3/h3-dualgpu-shakedown-5s-turbo-v4-api.json | 1dd5380b0a44c0d3a1067cda6905b4ef18972f72c7c58696d493d2c00675fedf |
| H3 Turbo 4-step single-R9700 control API | workflows/minimax-h3/minimax-h3-turbo-v4-4-control-api.json | 64b57c6b8322bd55c183543eb022895e55cdfebd89095b97556f046f319db262 |

The verifier parses every JSON file. It also checks for personal absolute paths
and credential-shaped strings before publication.
