# Data and Run Outputs Guide

Use this file for committed outputs, experiment run directories, provenance sidecars, and output-path restructuring.

## Active Run Safety

- Before any `git add`, `git rm`, `mv`, `rm`, or restructuring near output paths, inspect live run locks:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

- The pre-commit hook runs `uv run python -m scripts.lib.pipeline check-active-run-git-guard` and blocks commits when staged paths intersect output targets owned by a live process.
- Long runs should write to untracked active output paths and publish final artifacts only after completion.
- Never delete, bypass, or overwrite `*.provenance.json`.

## Run Directory Conventions

Keep the semantic layout:

```text
data/<model>/intervention/<benchmark>/experiment/
```

The provenance sidecars already carry the "when" and "how."

When a re-run would overwrite an existing `experiment/` directory that contains committed or analysed data, archive it first:

```text
data/<model>/intervention/<benchmark>/experiment_YYYY-MM-DD_<reason>/
```

For genuinely new experiments, create a new semantic directory rather than just timestamping the old one. Prefer names that describe what varies.

## Run Lifecycle

After a successful claim-relevant run, append to `notes/runs_to_analyse.md`:

```markdown
## <ISO timestamp> | <run_dir relative path>
What: <one-line: benchmark + method + alpha grid>
Key files: results.json, *.provenance.json, activations/responses.jsonl
Status: awaiting analysis
```

Remove the entry once analysed.
