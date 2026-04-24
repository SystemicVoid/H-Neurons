# Active-Run Git Guard Plan

## Implementation Status

Implemented on 2026-04-24:

- Active-run registry, liveness checks, staged-path guard, status CLI, and
  tracked-output launch check live in `scripts/lib/pipeline.py`.
- `start_run_provenance()` writes active-run locks and
  `finish_run_provenance()` removes them.
- `.pre-commit-config.yaml` runs `check-active-run-git-guard` before slower
  hooks.
- Operational docs are updated in `scripts/AGENTS.md`.
- Regression coverage is in `tests/test_pipeline.py` and
  `tests/test_wandb_integration.py`.

## Context

On 2026-04-24, a sibling agent committed live selector artifacts while
`scripts/select_faitheval_sae_utility_features.py` was still writing. No rows
were lost because the JSONL writer reopens, flushes, and fsyncs per record, but
the incident exposed a broader gap: the repo currently relies on instructions
that say not to touch output directories during GPU runs. That does not protect
future runs with different names or output locations.

This plan deliberately avoids worktree requirements for now. The first durable
step is to make active run ownership structural inside the existing worktree.

## Goal

Prevent Git operations during normal agent workflows from staging or committing
files owned by any live run, independent of run name, benchmark, model, or data
directory layout.

The guard must be driven by runtime ownership metadata, not path-specific
`.gitignore` entries.

## Non-Goals

- Do not add per-run `.gitignore` pins.
- Do not require separate worktrees for routine agent work.
- Do not block commits merely because stale provenance files still say
  `"status": "running"`.
- Do not redesign all pipelines to write outside the worktree in this first
  pass, although that remains the strongest long-term option.

## Design

### 1. Add a repo-wide active-run registry

Extend `scripts/utils.py:start_run_provenance()` so every run that creates
provenance also writes an active-run lock record.

Use the Git common directory rather than the working tree:

```text
$(git rev-parse --git-common-dir)/h-neurons-active-runs/<run_id>.json
```

This keeps locks out of commits and still works if we later use multiple
worktrees.

Each lock record should include:

- `schema_version`
- `run_id`
- `hostname`
- `pid`
- process start identity from `/proc/<pid>/stat`
- `cwd`
- `script`
- `argv`
- `started_at_utc`
- `provenance_path`
- `output_targets`
- `protected_paths`

`protected_paths` should be normalized absolute paths. For file targets, protect
the file. For directory targets, protect the full directory subtree. Do not
derive protection from semantic folder names such as `experiment/` or
`selector/`.

Update `finish_run_provenance()` to remove the lock record after writing the
final provenance status. If removal fails, leave a warning only; stale locks
must be handled by liveness checks.

### 2. Implement liveness-aware guard logic in `scripts.lib.pipeline`

Add reusable functions to `scripts/lib/pipeline.py`, with unit tests:

- `active_run_registry_dir()`
- `load_active_run_locks()`
- `is_lock_live(lock)`
- `path_intersects_live_run(path, lock)`
- `check_staged_paths_against_live_runs(paths)`

Liveness rules:

- The lock is live only if `hostname` matches the current host and `pid` exists.
- Verify the process start identity from `/proc/<pid>/stat` so a reused PID does
  not keep a stale lock alive.
- If the lock belongs to another host, treat it as not live for local pre-commit
  purposes and report it as stale/remote in diagnostics.
- Malformed lock records should not crash commits. Warn and ignore them unless
  they can be safely interpreted as live.

Intersection rules:

- A staged file path is blocked if it is equal to a protected file target.
- A staged file path is blocked if it is below a protected directory target.
- Deletions and renames count as writes and must be blocked the same way.

Add CLI subcommands:

```bash
uv run python -m scripts.lib.pipeline active-run-status
uv run python -m scripts.lib.pipeline check-active-run-git-guard
```

`active-run-status` should print live, stale, and malformed locks for operators.

`check-active-run-git-guard` should read staged paths from
`git diff --cached --name-status -z`, check them against live locks, and exit
non-zero with a concrete error message if any staged path intersects live output.

### 3. Add a pre-commit hook through `.pre-commit-config.yaml`

Add a local hook before slower lint/type/audit hooks:

```yaml
- id: active-run-git-guard
  name: active run Git guard
  entry: uv run python -m scripts.lib.pipeline check-active-run-git-guard
  language: system
  pass_filenames: false
```

Expected failure message shape:

```text
Active run Git guard blocked this commit.

The following staged paths are owned by live run <run_id>:
  data/.../utility_scores.jsonl
  data/.../answer_span_scores.jsonl

Run:
  uv run python -m scripts.lib.pipeline active-run-status

Wait for the run to finish, or explicitly stop/finalize it before committing
these paths.
```

This catches the common agent path: `git add data/... && git commit`.

### 4. Add launch-time protection for tracked live outputs

A pre-commit hook does not prevent `git restore`, checkout, reset, merge, or an
agent cleanup flow from rewriting tracked files before a commit. To reduce that
risk without worktrees, runs should refuse to start when their live output
targets are tracked by Git.

Add a helper in `scripts/lib/pipeline.py`:

```bash
uv run python -m scripts.lib.pipeline check-live-output-track-state \
  --output-target <path> [--output-target <path> ...]
```

Behavior:

- Resolve each target to a path relative to the repo root.
- Use `git ls-files` to detect tracked files under protected targets.
- Exit non-zero if any protected output file is already tracked.
- Print the tracked files and recommend archiving the old run or choosing a new
  semantic run directory.

Then wire this check into `start_run_provenance()` before writing the active-run
lock. Default behavior should fail closed for long-running pipeline scripts.

If a short report-generation script intentionally rewrites a tracked summary,
allow an explicit opt-out parameter to `start_run_provenance()`, for example:

```python
allow_tracked_live_outputs=True
```

Use the opt-out sparingly and document each use. Long GPU runs should not use
it.

### 5. Update operational docs

After implementation, update `scripts/AGENTS.md`:

- Move "Never touch output directories during a GPU run" from advisory guidance
  to structural guard behavior.
- Document `active-run-status`.
- State that long runs should write to untracked active output paths and publish
  final artifacts only after completion.

## Test Plan

Add tests in `tests/test_pipeline.py` or a new focused test file.

Required cases:

- live lock blocks a staged file target
- live lock blocks a staged file below a protected directory
- unrelated staged path passes
- stale lock with missing PID passes
- reused PID with mismatched process start identity passes
- malformed lock warns but passes
- tracked output target detection blocks a tracked file
- tracked output target detection passes for untracked output files

Manual smoke test:

1. Create a temporary lock record for the current shell PID with a protected
   path under `data/tmp-active-run-guard/`.
2. Stage a file under that path.
3. Confirm `uv run python -m scripts.lib.pipeline check-active-run-git-guard`
   exits non-zero.
4. Stage an unrelated docs file.
5. Confirm the guard exits zero.

## Acceptance Criteria

- A commit cannot proceed when staged paths intersect output targets of a live
  run.
- Stale `"status": "running"` provenance files do not block commits by
  themselves.
- The mechanism works for arbitrary output directories because protection is
  derived from each run's declared `output_targets`.
- Long runs fail before launch if they would write to tracked live output files,
  unless an explicit, documented opt-out is used.
- Existing `ty check` and pipeline tests pass.

## Later Hardening

If this class of incident recurs despite the guard, the next escalation is to
move active run writes outside the Git worktree and copy final artifacts into
`data/...` only after completion. That would make Git physically unable to
rewrite live outputs, but it is intentionally deferred to avoid adding workflow
overhead now.
