# Script Library Guide

Use this file when changing shared script-library code, especially `scripts/lib/pipeline.py`.

## Pipeline Guard Library

`scripts/lib/pipeline.py` contains shared, tested guards for pipeline orchestration. Use these guards from bash wrappers instead of reimplementing decision logic in shell.

Current CLI entrypoint:

```bash
uv run python -m scripts.lib.pipeline --help
```

Common subcommands:

- `check-stage`
- `manifest-count`
- `gpu-preflight`
- `gpu-hardware-guard`
- `check-sentinel`
- `check-intervention-contract`
- `log-run`
- `active-run-status`
- `check-active-run-git-guard`
- `check-live-output-track-state`

## Adding Guards

- Only add a new guard when it traces to a real incident or removes logic duplicated across 5+ pipeline scripts.
- Each new guard needs a docstring or nearby comment with the incident or duplication rationale.
- Add focused coverage in `tests/test_pipeline.py`.
- Keep the CLI help and `scripts/infra/AGENTS.md` examples current.

Run guard-library tests with:

```bash
uv run pytest tests/test_pipeline.py
```
