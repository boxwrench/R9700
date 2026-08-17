# Standard Metrics & Terminology

To avoid ambiguity in reporting, all research tracks use the following unified definitions and formulas.

---

## Throughput & Latency Metrics

* **Base Decode Throughput (`tok/s`)**:
  Serial (non-speculative) generation speed in tokens per second, measured as committed tokens divided by generation wall time.
* **MTP Decode Throughput (`tok/s`)**:
  Speculative MTP generation speed in tokens per second, measured as total committed tokens divided by total speculative decode wall time.
* **MTP Multiplier**:
  $$\text{MTP Multiplier} = \frac{\text{MTP Decode (tok/s)}}{\text{Base Serial Decode (tok/s)}}$$
* **Prompt Processing (`PP`, tok/s)**:
  Prefill throughput evaluated over prompt tokens.
* **Time To First Token (`TTFT`, ms)**:
  Wall-clock elapsed time from request dispatch to emission of the first committed token.
* **MTP Round Latency (`ms`)**:
  Total wall-clock duration of a single speculative cycle, comprising target model verification, proposer forward pass(es), and sampling/synchronization.
* **Proposer Latency (`ms/round` or `ms/step`)**:
  Wall-clock duration spent evaluating the MTP draft proposer block to generate candidate speculative tokens.

---

## Speculative Acceptance Metrics (for $n_{\text{max}}=2$)

* **Drafted Tokens ($N_{\text{drafted}}$)**:
  Total candidate tokens emitted by the MTP proposer during generation.
* **Accepted Tokens ($N_{\text{accepted}}$)**:
  Total candidate tokens verified and accepted by the target verifier.
* **Aggregate Acceptance Rate ($\%$)**:
  $$\text{Aggregate Acceptance} = \frac{N_{\text{accepted}}}{N_{\text{drafted}}} \times 100\%$$
* **Positional Acceptance at Step 0 ($p_0$)**:
  Probability that the first drafted token ($k=1$) is accepted by the verifier:
  $$p_0 = P(\text{accept } k=1)$$
* **Joint Positional Acceptance at Step 1 ($\text{joint-}p_1$)**:
  Probability that *both* the first and second drafted tokens are accepted in a single round:
  $$\text{joint-}p_1 = P(\text{accept } k=1 \land \text{accept } k=2)$$
* **Conditional Positional Acceptance at Step 1 ($\text{conditional-}p_1$)**:
  Probability that the second drafted token is accepted *given* that the first token was accepted:
  $$\text{conditional-}p_1 = \frac{\text{joint-}p_1}{p_0}$$

---

## Draft Count vs Committed Token Accounting

> [!IMPORTANT]
> **Accepted Drafts / Round** and **Committed Tokens / Round** are distinct quantities differing by the target verifier's base token ($+1$). Do not interchange these metrics.

* **Accepted Drafts per Round**:
  The expected number of *speculative draft tokens* accepted per MTP round (excluding the base verified token):
  $$\text{Accepted Drafts / Round} = p_0 + \text{joint-}p_1$$
* **Committed Tokens per Round**:
  The total number of *output tokens committed to context* per MTP round (1 base verified token $+$ accepted draft tokens):
  $$\text{Committed Tokens / Round} = 1 + p_0 + \text{joint-}p_1$$

---

## Resource Metrics

* **VRAM Usage (`MiB` or `GB`)**:
  Total dedicated device memory consumed, partitioned by model weights, KV cache, recurrent states, compute scratch buffers, and driver overhead.
* **Effective Bandwidth (`GB/s`)**:
  $$\text{Effective Bandwidth} = \frac{\text{Tensor Bytes Loaded}}{\text{Kernel Execution Time (s)} \times 10^9}$$
