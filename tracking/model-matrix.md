# Model coverage matrix

One row per model the lab has tried or intends to try on Wormhole.

- **Upstream evidence** is what Tenstorrent's own sources said on the date given. It is not a lab measurement.
- **Lab status** is one of: `not tried`, `runs` (Gate 1 passed), `served` (endpoint up, Gates 1 to 4 recorded), `adopted` (in nightly regression), `out of scope`.
- Every `runs` or better row links to its run note.

## Seed rows from upstream sources (2026-11-16)

None of these has been run on the lab's boxes yet. Phase 0 and Phase 1 of the [roadmap](../docs/03-roadmap.md) replace "not tried" with measurements.

| Model | Family | Hardware | Path | Upstream evidence | Lab status | Run note |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | Dense LLM | n150, n300, t3k | A | `COMPLETE` on all three in tt-inference-server; 128K context on n300 and t3k | not tried | |
| Llama-3.3-70B-Instruct | Dense LLM | t3k | A | Tenstorrent's recommended first model for QuietBox and LoudBox | not tried | |
| Qwen 2.5 7B | Dense LLM | n300 | A | tt-metal README: 22.1 tokens/s/user at batch 32, target 30, tensor-parallel 2 | not tried | |
| Qwen 2.5 72B | Dense LLM | t3k | A | tt-metal README: 15.4 tokens/s/user at batch 32, target 20, tensor-parallel 8 | not tried | |
| QwQ-32B | Reasoning LLM | t3k | A | Listed `Functional`, 128K context, 32 users, `tt_transformers` | not tried | |
| Mixtral 8x7B | MoE LLM | t3k | A | tt-metal README: 24.9 tokens/s/user at batch 32, target 33, tensor-parallel 8 | not tried | |
| Whisper distil-large-v3 | Speech | n150 | A | tt-metal README: 105 tokens/s on one n150 | not tried | |

## To establish in Phase 1

Families with no confirmed Wormhole status in the sources read so far:

| Family | Question to answer |
| --- | --- |
| Vision-language | Which VLMs does the catalog list for `t3k` or `n300`? |
| Embedding and reranking | Does the Forge vLLM plugin route work on Wormhole, or only on Blackhole? |
| Diffusion image | Which image models fit and run on Wormhole? |
| Vision (CNN, ViT) | Which tt-forge-models variants pass on `n300-llmbox` in tt-xla's test config? |

## Lab rows

Add rows here as bring-ups happen.

| Model | Family | Hardware | Path | Class | Lab status | Gate 1 PCC | Throughput | Pins | Run note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
