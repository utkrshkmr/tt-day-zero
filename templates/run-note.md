# {model} on {box} ({date})

Owner: {name} · Triage note: {link} · Path: {A catalog / B compiler / C native}

## Pins

| | |
| --- | --- |
| tt-metal | {commit} |
| tt-inference-server | {tag} |
| vLLM | {commit} |
| Container image | {name and digest} |
| tt-xla / tt-mlir | {version or commit, if Path B} |
| Firmware / tt-kmd | {versions} |
| Device | {n150 / n300 / t3k}, chips used: {ids} |

## Launch

```bash
{the exact command}
```

Environment that differs from the default: {variables}

## Gates

| Gate | Result | Notes |
| --- | --- | --- |
| 1. Correctness | PCC {x.xxxx} against {reference} | {modules checked; prompts read} |
| 2. Throughput | {tokens/s/user} at batch {n}; {total tokens/s}; TTFT {ms} | {percent of reference, and which reference} |
| 3. Context and concurrency | context {k} tokens; {n} users | |
| 4. Stability | {minutes} under load | |

Dtype and fidelity: {weights dtype}, {math fidelity}

## What it took

- Time from release to Gate 1: {hours}
- Time from release to endpoint: {hours}
- Changes made: {files, with links to commits}
- Failures on the way, and the layer each was in:

## Limits and follow-ups

- {known limits: context cap, prefill chunk size, unsupported features}
- {what to send upstream}
