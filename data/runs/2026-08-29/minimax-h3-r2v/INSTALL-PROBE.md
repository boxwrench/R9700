# Sampler probe — experiment record

The passive probe was not installed into the production `/ai/comfyui/custom_nodes` tree. The read-only mount was left unchanged.

For this campaign it was loaded only by the isolated launchers through an extra custom-node path. It wraps `comfy.samplers.CFGGuider.inner_sample`, is inactive unless a run ID is set through `/h3mem/set_run_id`, and writes atomic `/tmp/h3-mem-<run_id>.json` records on success or exception.

Probe source: `instrumentation/h3-sampler-mem-probe.py`

SHA-256: `1c51477416f0b30e410b2c6330abc706d3788e57a734997441c474e1950fa8d0`

The production service does not require this probe for normal operation.
