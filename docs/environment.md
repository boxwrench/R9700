# Environment records

The `data/environment/` directory contains the public-safe parts of the local
stack record:

- AMD ROCm/PyTorch constraints and wheel hashes;
- the intent requirements record;
- installed ROCm package versions;
- PyTorch smoke-test and HMM transfer JSON results.

Model caches, Python environments, tokens, service files, and private
correspondence are not copied. The recorded paths under `/ai` in the dated
reports are workstation source references, not portable installation paths.
