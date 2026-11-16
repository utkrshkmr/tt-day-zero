# tt-day-zero

Getting newly released models running on Tenstorrent Wormhole boxes (TT-QuietBox / TT-LoudBox, eight chips) on the day they are released.

This repository holds the lab's process and plan. It has three parts:

| Read this | To learn |
| --- | --- |
| [docs/01-stack-hierarchy.md](docs/01-stack-hierarchy.md) | The whole Tenstorrent software stack, from `tt serve` down to a Tensix core: every compiler stage, every module, which repository and directory it lives in, and where to make a given kind of change. |
| [docs/02-day0-workflow.md](docs/02-day0-workflow.md) | The runbook. A model was released an hour ago: triage, choose one of three paths, acceptance gates, performance, publishing. |
| [docs/03-roadmap.md](docs/03-roadmap.md) | The 26-week plan for two developers, 2026-11-16 to 2027-05-14: roles, phases, dated milestones, exit criteria, risks. |
| [docs/04-references.md](docs/04-references.md) | Links to the upstream documentation, repositories and tech reports the other three draw on. |

## A model was just released

1. Copy [`templates/triage.md`](templates/triage.md) and fill in the five facts. Target: within four hours.
2. Follow the decision tree in [the workflow](docs/02-day0-workflow.md#1-pick-a-path-in-the-first-hour):
   - in the serving catalog, or the same architecture as something that is: **Path A**, serve it;
   - an ordinary Hugging Face PyTorch or JAX model: **Path B**, compile it with tt-xla;
   - a new architecture the lab wants to use: **Path C**, native bring-up in `tt_transformers`.
3. Pass the four [acceptance gates](docs/02-day0-workflow.md#7-step-3-acceptance-gates).
4. Commit a run note from [`templates/run-note.md`](templates/run-note.md) under `runs/` and update [`tracking/model-matrix.md`](tracking/model-matrix.md).

## Layout

```
_config.yml  GitHub Pages settings: theme, sidebar order, diagram rendering
docs/        the three documents above, plus references
templates/   triage note and run note
tracking/    model coverage matrix (gap ledger and stack pins are added in Phases 0 and 2)
runs/        one note per bring-up, added as they happen
scripts/     preflight, d0-serve, d0-compile: deliverables of roadmap Phases 0 to 2, not yet written
```

## Status

Started 2026-11-16. The documents are written; nothing in them has yet been run on the lab's boxes. Phase 0 of the roadmap measures the starting point, and the matrix is filled in from there.

Facts about the upstream stack were read from Tenstorrent's documentation and repositories on 2026-11-16. That stack changes weekly: check the pinned versions before trusting a command, and correct this repository when it is wrong.
