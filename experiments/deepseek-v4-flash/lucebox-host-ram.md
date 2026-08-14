# Lucebox DeepSeek with R9700 plus system DDR5

## Objective

Determine whether Lucebox's DeepSeek V4 ROCmFPX path can use 192 GiB of system
RAM as an intentional cold-expert tier behind the standalone 32 GiB R9700 and
beat the saved 8.137 tok/s llama.cpp result at 32K or greater context.

## Hardware distinction

Lucebox's published 32 tok/s run used a Ryzen AI MAX+ 395. Its Radeon 8060S
directly accesses the same 128 GB, 256 GB/s LPDDR5X pool as the CPU. The R9700
has 32 GB of 640 GB/s local GDDR6, but reaches ordinary DDR5 over PCIe. The
capacity is sufficient; the memory topology is not equivalent.

A passing implementation must keep dense, attention, shared-expert tensors and
a bounded hot-expert cache in VRAM while cold routed experts remain in DDR5.
Cold misses must either execute on the CPU or transfer selectively and
asynchronously. HIP managed-memory oversubscription, whole-model page migration,
or rereading the full target across PCIe per token counts as a failed design.

## Ordered gates

1. Pin the Lucebox/ROCm versions and build the complete DeepSeek path for
   `gfx1201`, without downloading weights or loading the GPU.
2. Locate or implement an explicit host-resident expert tier and add telemetry
   for placement, cache hit rate, page faults, PCIe traffic, RAM/swap and VRAM.
3. Add scoped, resumable process recovery that can reset only the R9700 and only
   when no foreign render-node owner exists.
4. Load at 8K with six experts, exact prefill, target-only greedy decoding and
   at least 32 GiB of host RAM reserved for the OS and runtime.
5. Add fused DSpark q=4, then indexed sparse prefill, one variable at a time.
6. Test 16K and then 32K at 90% occupancy. Require exact retrieval, the saved
   deterministic quality checks, no material swap growth, and 1.5 GiB end-of-
   probe VRAM headroom.
7. Only after exact six-expert 32K passes, test sparse prefill and top-k four.
   Label both approximate; top-k four changes the model's default execution.
8. Continue exact six-expert context through 40K, 48K and 64K, stopping at the
   first unsafe result.

The published 32 tok/s result is a reference ceiling, not an expected result.
It combined a 102.3 GB mixed ROCmFPX target, 11.3 GB DSpark draft, q=4 fused
verification, sparse prefill, top-k four, and only 8K context. A generic R9700
benchmark in the Lucebox hardware table is not evidence that this complete 284B
standalone configuration already works.

ROCmFPX is allowed only as an inseparable kernel/format experiment. It does not
replace UD-Q4_K_XL as the primary quality baseline.

## Sources

- <https://www.lucebox.com/blog/deepseek-v4-strix-halo>
- <https://github.com/Luce-Org/lucebox>
- <https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html>
- <https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html>

Checked 2026-08-14. Recheck Lucebox flags and R9700 support before enabling any
row because upstream is changing rapidly.
