# Tooling ROI Assessment — Mech-Interp Research Workflow

_Generated 2026-03-19. Refocused 2026-03-20 after splitting throughput/performance analysis into `docs/throughput-assessment.md`._

## Scope

This note is for tooling and workflow improvements that help the research loop.
## Current State

| Dimension | Current State |
|---|---|
| Workload | Local GPU inference with custom PyTorch hooks for neuron-level intervention |
| Shell/dev env | Bash, tmux 3.4, Ghostty, ripgrep, fzf, bat, eza, stow |
| Quality gates | `ruff`, `ty`, `prek`, `pytest`, CI audit |
| Agent tooling | Amp, Claude Code, Codex CLI |
| Long-job safety | `systemd-inhibit` already in use |
| Resume support | Already implemented via `load_existing_ids()` |

## Keep

### `wandb` opt-in tracking

Implemented in:

- `scripts/run_intervention.py`
- `scripts/run_negative_control.py`
- `scripts/collect_responses.py`
- `scripts/extract_activations.py`

Shared support in:

- `scripts/utils.py`
  - `sanitize_run_config()`
  - `get_git_sha()`

Tests:

- `tests/test_wandb_integration.py`

Why keep it:

- Good run metadata capture for long experiments
- Cheap provenance for benchmark, alpha sweep, and commit SHA
- Offline mode already supported for later sync

Current usage:

```bash
uv run python scripts/run_intervention.py --benchmark faitheval --wandb
WANDB_MODE=offline uv run python scripts/run_intervention.py --benchmark faitheval --wandb
wandb sync runs/<run-dir>
```

### Experiment-folder convention

Still worth doing. The repo has many lightweight experiment runs whose provenance would be easier to review if config and outputs were co-located.

Recommended lightweight shape:

```text
data/<model>/runs/
  YYYY-MM-DD_<description>/
    config.json
    alpha_0.0.jsonl
    alpha_1.0.jsonl
    results.json
```

Minimal config capture to add near run start:

```python
import datetime
import json
import subprocess
import sys

config = {
    "args": vars(args),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python_version": sys.version,
    "torch_version": torch.__version__,
}
```

Why this is still high ROI:

- Makes runs reproducible without asking “what exact command produced this?”
- Low engineering cost
- Helps later paper, site, and rebuttal cleanup

Suggested first targets:

- `scripts/run_intervention.py`
- `scripts/run_negative_control.py`

## Add

### Low-friction run provenance

Recommendation:

- Write `config.json` to each experiment output dir
- Record the launch command verbatim in a sibling text file when practical
- Keep `results.json` as the compact aggregate and JSONL files as row-level evidence

Confidence: High.

Why:

- This does not alter model behavior
- It directly reduces future ambiguity in analysis and reporting

## Defer

These remain reasonable ideas, but they should not displace throughput work or experiment-provenance fixes.

| Tool / Pattern | Current Verdict | Trigger to Reconsider |
|---|---|---|
| `vLLM` | Defer | Reconsider only if the workload becomes batchable without custom intervention hooks |
| `@file_cache` pattern | Defer | Reconsider only if repeated pure-function recomputation becomes a measured bottleneck |

## Notable Boundaries

- Do not treat profiling-only outputs as canonical experiment artifacts.
- Do not mix exploratory measurement scaffolding with committed experiment outputs.
