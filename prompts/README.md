# Prompts

The neutral brass-robot prompt is the shared T2V prompt used by the three
baseline lanes. Prompt enhancement was disabled and the numeric seed was
`8112026`. The LTX workflow also used the model-native negative prompt in
`neutral-brass-robot-negative.txt`; H3 did not use a negative-conditioning field
in the recorded graph.

`boxwrench-v1/` is a separate five-scene comparison suite. Its I2V and T2V lanes
must remain separate. Do not compare an I2V result against a T2V result as if
they were the same workload.
