# Triage: <model name>

Filed by: <name> · Released: <date and time> · Filed: <date and time>

## Five facts

| | |
| --- | --- |
| Hugging Face repo | |
| Architecture and family | <closest supported model, and what differs> |
| Modality | <LLM / VLM / embedding / speech / image / video / other> |
| Size | <parameters>; <GB at bfloat16>; <GB at bfloat8_b>; fits on <n150 / n300 / t3k / does not fit> |
| License and gating | <license>; gated: <yes / no>; access requested: <date> |
| Unusual operations | <RoPE variant, attention variant, MoE, custom norm, vision tower, ...> |

## What already exists

- [ ] Serving catalog (`tt model list`, `models_by_hardware.md`): <entry and status, or none>
- [ ] Same architecture in `tt-metal/models`: <path, or none>
- [ ] Loader in `tt-forge-models`: <path and test status, or none>
- [ ] Upstream issue or bounty: <link, or none>

## Decision

- **Class:** <1 sibling / 2 compilable / 3 gap / 4 out of scope>
- **Path:** <A / B / C>
- **Owner:** <name>
- **Blockers:** <gated access, missing op, memory, ...>
- **If out of scope:** <reason, and the nearest variant that fits>
