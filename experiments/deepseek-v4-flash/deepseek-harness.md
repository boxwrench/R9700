# DeepSeek Harness integration

## Decision

DeepSeek Harness (`dsh`) is the primary agent and user-facing implementation
for local DeepSeek V4. The inference backend remains independently replaceable:
llama.cpp/Vulkan, llama.cpp/HIP, and Lucebox are eligible when they expose an
OpenAI-compatible Chat Completions endpoint. Hermes is a migration/A/B control.

`dsh` is an agent harness, not an inference engine or trained model. It does not
change GPU placement or raw decoding speed. It provides DeepSeek-oriented prompt
and tool assembly, an agent loop, persistent sessions, approval policy, and a
plugin architecture.

## Backend boundary

Each backend must expose on loopback:

- `/v1/chat/completions` with streaming and tool-call support.
- A stable, recorded model alias.
- DeepSeek reasoning content preservation when reasoning is enabled.
- The context and output ceilings recorded by its backend experiment.

Register it as a custom provider with `api: openai-completions`; do not use the
hosted DeepSeek provider for local experiments. Save the pinned Harness version
or commit and `dsh --profile <name> --dump-config` with every campaign.

## Two scorecards

Direct HTTP probes remain authoritative for prompt/decode throughput, TTFT,
DSpark acceptance, retrieval, deterministic quality, VRAM/RAM/swap, and crash
recovery. Neither `dsh` nor Hermes belongs in that measurement path.

Every backend finalist then runs through a pinned `dsh` headless profile in a
fresh workspace/session. Record completed tasks, artifact correctness, tool-call
validity, retries and corrective turns, token counts, wall-clock task time,
context growth/compaction, transcript, and server logs.

The initial task corpus covers repository orientation, read-only diagnosis, a
bounded code edit with tests, supplied-text rewriting, summarization, exact
long-context retrieval, and recovery after a deliberately failing command.
Mutation tests use a disposable workspace.

A regular-use winner must pass exact 32K backend gates, complete every required
`dsh` task without malformed tool calls or silent context loss, route only to
loopback, resume cleanly after a backend crash, and preserve or improve time per
successful task. Hermes is tested afterward with the identical backend/tasks;
differences are attributed to the harness, not the GPU.

DeepSeek Harness is in developer preview and warns of breaking changes. Pin it
for each campaign and recheck its schema before applying the example settings.

Sources:

- <https://github.com/deepseek-ai/deepseek-harness>
- <https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md>
- <https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md>

Checked 2026-08-14.
