# Dependency locking

`baseline.in` is an intent record, not an installable lock. After the raw
ROCm/PyTorch gate passes:

1. record the Python interpreter and ROCm wheel source;
2. capture `uv pip freeze --python /ai/environments/comfyui-h3/bin/python` as a
   dated freeze record;
3. capture `python -m torch.utils.collect_env` as a dated benchmark artifact;
4. record the ComfyUI, patch, and custom-node Git SHAs in `STACK.md`;
5. rerun the smoke tests from a fresh environment before declaring the lock golden.

Never place a Hugging Face token or other credential in a requirements file.
