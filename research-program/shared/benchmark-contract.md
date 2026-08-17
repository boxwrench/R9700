# Research Benchmark Contract

To maintain absolute scientific rigor across Track A, Track B, and subsequent Integration work, all investigations must adhere to the following binding rules.

---

## 1. Hardware & Environment Invariance

* **Identical Host & Accelerator**: Matched A/B comparisons must run on the exact same physical workstation and GPU (`AMD Radeon AI PRO R9700 / gfx1201`).
* **Explicit Toolchain & Commit Freezing**: Every reported run must record the exact source commit SHA, compiler version, backend driver version (e.g., Mesa/RADV or ROCm/HIP), and OS kernel.
* **Exact Model & Weight Hashes**: Every benchmark must identify model filenames and SHA256 hashes. No implicit or unverified weight transformations are permitted.

---

## 2. Benchmark Execution Protocol

* **Fixed Test Vectors**: Matched runs must use identical prompts, seeds, token generation limits, context limits (`n_ctx`), and batch sizes (`n_ubatch`).
* **Mandatory Unrelated Warmup**: Every timed execution must be preceded by an unrelated warmup run to eliminate JIT shader compilation, memory allocation spikes, and initial driver initialization noise.
* **Sufficient Statistical Repetition**: Minimum 5 repetitions for end-to-end decode harnesses; minimum 100 iterations for microbenchmarks. Always report mean, standard deviation, median, p10, and p90.
* **Cache State Sensitivity**: Distinguish cold/uncached vs steady-state cached measurements. The R9700 Infinity Cache (64 MB) must be accounted for when benchmarking tensors near or under cache capacity.
* **Single-Variable Control**: Change exactly one variable at a time (e.g., kernel shader, draft head size, or speculative depth). Never combine multiple experimental changes in a single arm.
* **Preservation of Negative Results**: Negative and neutral findings must be logged, quantified, and formally closed to prevent cyclic reinvestigation.

---

## 3. Optimization Validation Chain

Isolated microbenchmarks and kernel diffs **do NOT count as production decode gains**. Every candidate optimization must pass the complete four-stage validation chain:

```text
1. Component / Microbenchmark
   (Measure isolated kernel execution time, bandwidth, and numerical correctness)
               │
               ▼
2. In-Graph Differenced Measurement
   (Profile node execution inside live model graph under real dispatch conditions)
               │
               ▼
3. End-to-End Decode Harness
   (Measure real tok/s, MTP round latency, and acceptance on representative prompt)
               │
               ▼
4. Repeated Production-Style Holdout
   (Validate across multiple unseen holdout workloads with full statistical spread)
```

---

## 4. Explicit Metric Classification

Every reported numeric value must be explicitly categorized as:

* **`MEASURED`**: Directly captured from physical hardware timers, profiler queries, or token/second counters.
* **`CALCULATED`**: Derived via exact mathematical formulas from measured quantities.
* **`INFERRED`**: Deduced through differencing or model subtraction.
* **`ESTIMATED`**: Projected or theoretical model values (never treated as measured proof).
* **`UNEXPLAINED`**: Time or behavior that is real and measured but not accounted for by any identified component. **An unexplained residual stays labelled `UNEXPLAINED`.** It is never redistributed across known components, absorbed into a rounding note, or attributed to a plausible-sounding cause that was not isolated.

### 4a. No Causal Claim Without Direct Evidence

Reporting *what* changed is always permitted. Reporting *why* requires evidence that isolates the mechanism.

* Correlation between a change and a result does **not** license naming a cause.
* A mechanism that is structurally present is **not** thereby shown to be limiting. Track A entry 15 identified redundant `IQ4_XS` dequantization from pipeline statistics and shader source, inferred it was the performance limiter, and was **refuted by entry 16** when removing it changed almost nothing. The statistics were correct; the causal claim was not.
* Where a cause is believed but unproven, state it as a hypothesis and name the measurement that would settle it.
* Retractions are made **in place**, at the original claim, pointing forward to the evidence that overturned it.

---

## 5. Primacy of Real Throughput

* **Primary Objective**: Real end-to-end committed decode throughput (`tokens/sec`).
* **Secondary / Diagnostic**: MTP round latency (`ms`), proposer latency (`ms`), positional acceptance rates ($p_0, p_1$), and draft lengths.
* **Decision Rule**: Proxy metrics (such as isolated kernel speedups or higher draft acceptance alone) cannot justify an optimization if real committed decode throughput regresses or remains neutral.
