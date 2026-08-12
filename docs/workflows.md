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

The verifier parses every JSON file. It also checks for personal absolute paths
and credential-shaped strings before publication.
