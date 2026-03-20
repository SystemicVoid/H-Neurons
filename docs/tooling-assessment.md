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

Status: adopted as the local standard for research-output scripts.

Standard:

- Keep semantic artifacts (`results.json`, `summary.json`, existing `metadata.json`) unchanged.
- Write one adjacent provenance sidecar per producing script.
- Use file targets for single-output scripts and directory targets for multi-artifact runs.
- Keep W&B opt-in; provenance must still be readable locally without W&B.

Canonical sidecar naming:

- File output: `<target>.provenance.json`
- Directory output: `<target>/<script_stem>.provenance.json`

Canonical payload:

- `schema_version`, `script`, `argv`, `command`
- `args` with secret-like fields stripped via `sanitize_run_config()`
- `cwd`, `hostname`, `python_version`
- `git_sha`, `git_dirty`
- `started_at_utc`, `completed_at_utc`, `status`
- `output_targets`
- `safe_env` limited to `CUDA_VISIBLE_DEVICES` and `WANDB_MODE`
- optional `wandb` block when enabled

Examples:

```text
data/gemma3_4b/pipeline/answer_tokens.jsonl
data/gemma3_4b/pipeline/answer_tokens.jsonl.provenance.json
```

```text
data/gemma3_4b/intervention/faitheval/experiment/
data/gemma3_4b/intervention/faitheval/experiment/run_intervention.provenance.json
```

Local launch shape:

```bash
uv run python scripts/run_intervention.py --benchmark faitheval
systemd-inhibit --what=sleep:idle --who="run-intervention" --why="FaithEval sweep" \
  uv run python scripts/run_intervention.py --benchmark faitheval
```

Why this is the right level:

- Boring: sidecars live next to the artifacts researchers already inspect
- Low-friction: no new service, DB, or run registry
- Useful: answers “what produced this?” even when shell history and W&B are unavailable

### Migration Tracker

| Script | Scope Status | Anchor | Notes |
|---|---|---|---|
| `collect_responses.py` | done | `output_path` file | Single JSONL output |
| `extract_answer_tokens.py` | done | `output_path` file | Single JSONL output |
| `sample_balanced_ids.py` | done | `output_path` file | Single JSON output |
| `extract_activations.py` | done | `output_root` dir | Covers location subdirs |
| `classifier.py` | done | `metrics_out` else `save_model` | Lists both outputs when both are written |
| `run_intervention.py` | done | `output_dir` dir | Sidecar lives beside `results.json` and alpha JSONL files |
| `evaluate_intervention.py` | done | `input_dir` dir | Post-hoc judge writes `results.json` in-place |
| `run_negative_control.py` | done | `output_base` dir | Covers seed subdirs plus comparison artifacts |
| `prepare_bioasq_eval.py` | done | `output_parquet` file | Lists parquet + summary |
| `extract_sae_activations.py` | done | `output_root` dir | Keeps existing `metadata.json` untouched |
| `classifier_sae.py` | done | `metrics_out` else `save_model` | Mirrors CETT classifier behavior |
| `characterize_swing.py` | done | `output_dir` dir | Covers summary + feature table + figures dir |
| `export_site_data.py` | deferred | site JSON outputs | Report/export helper, not primary experiment generation |
| `plot_intervention.py` | deferred | `output` file | Visualization helper |
| `investigate_neuron_4288.py` | deferred | `output_dir` dir | One-off analysis notebook-style script |
| `review_batch3500.py` | deferred | `output_path` file | Review/report helper |
| `remap_faitheval_standard_parse_failures.py` | deferred | output summary/jsonl files | Targeted remediation script |
| `regenerate_long_false.py` | deferred | `verbosity-test-data-fixed.json` file | Dataset repair utility |

## Defer

These remain reasonable ideas, but they should not displace throughput work or experiment-provenance fixes.

| Tool / Pattern | Current Verdict | Trigger to Reconsider |
|---|---|---|
| `vLLM` | Defer | Reconsider only if the workload becomes batchable without custom intervention hooks |
| `@file_cache` pattern | Defer | Reconsider only if repeated pure-function recomputation becomes a measured bottleneck |

## Long-Running Jobs & System Suspend

On Pop!_OS / COSMIC DE, system auto-suspend will kill tmux jobs mid-run. Always hold a `systemd-inhibit` lock for jobs longer than ~20 minutes.

**At launch:**
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    uv run python scripts/run_intervention.py ...
```

**For already-running jobs** — watcher in a tmux window named `inhibit`:
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    bash -c 'while tmux list-windows | grep -q "<job-window-name>"; do sleep 30; done'
```

**Post-suspend recovery:** if a job is running slowly after wake, `kill -9` the python PID (`pgrep -a python | grep <script>`). The script's resume logic picks up from the last written line.

## GPU Monitoring

`nvitop` is available on PATH (installed via `uv tool`). Use `nvitop -1` for a one-shot status check before/during long GPU jobs to verify utilization, VRAM headroom, and process state. On this host's `nvitop 1.6.2`, the old `--no-header -o compact` form is invalid; for logging, prefer plain `nvitop -1` or check `nvitop --help` for the current CLI.

## Notable Boundaries

- Do not treat profiling-only outputs as canonical experiment artifacts.
- Do not mix exploratory measurement scaffolding with committed experiment outputs.
