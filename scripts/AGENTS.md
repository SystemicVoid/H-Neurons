# Scripts — Evaluation Notes

## Jailbreak response truncation bias (max_new_tokens=256)

`run_intervention.py` generates jailbreak responses with `max_new_tokens=256`. This is
too low for the compliant response pattern and introduces a systematic classification
error.
**Fix for future runs:** Use `max_new_tokens=512` (minimum) for jailbreak, preferably
1024. Greedy decoding (`do_sample=False`) is also preferable for gold-label
generation so responses are reproducible.

## JSONL writes: reopen file per record

<important>
All JSONL writers in eval scripts **must** open the file per record, not hold one
file descriptor for the entire sample loop.

```python
# CORRECT — resilient to path unlinking between writes
for sample in samples:
    ...
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# WRONG — if anything unlinks the path during the run, all subsequent
# writes land on an orphaned inode and are silently lost
with open(out_path, "a") as f:
    for sample in samples:
        ...
        f.write(...)
        f.flush()
```

**Why:** On Linux, unlinking a file while a process holds an open fd (e.g. via
`git rm`, `rm`, or directory restructuring) keeps the inode alive but removes the
directory entry. The process writes succeed silently, but the data is irrecoverably
lost when the fd closes. Reopening per record means the next `open()` creates a
fresh file if the path was removed — you lose at most one record instead of the
entire run. The syscall overhead is negligible vs. 20–30s/sample GPU inference.

**Incident:** On 2025-03-23 a concurrent `git commit` unlinked a jailbreak JSONL
mid-run, destroying 500 samples (~5 hours of GPU time) with zero errors reported.
</important>

## Never touch output directories while a GPU run is writing

<important>
The per-record reopen pattern above limits blast radius, but **do not rely on it as
the sole safeguard.** Unlinking a path mid-run still loses the in-flight record and
forces a resume. The rules below remain in effect:

1. Before any `git add`, `git rm`, `git checkout`, `mv`, `rm`, or directory
   restructuring, check whether a GPU job is running that writes to the affected
   path. Use `ps aux | grep run_intervention` or check tmux panes.
2. If a run is active, do **not** touch its `output_dir` or any parent directory
   in the path. Wait for the run to finish.
3. If you need to archive or restructure data, do it **before** launching the run
   or **after** the run completes and the process has exited.
4. The provenance sidecar (`*.provenance.json`) in the output directory has
   `"status": "running"` while the job is active. Check it.
5. This applies to **all** benchmark output directories, not just jailbreak.
</important>

## OpenAI evaluation: batch mode is mandatory, fast mode is forbidden

<important>
All OpenAI API evaluation scripts (`evaluate_intervention.py`, `evaluate_csv2.py`)
**must** use `--api-mode batch` (or `--api_mode batch`). Never use `--api-mode fast`
(synchronous per-request mode).

**Why batch mode is mandatory:**
1. **Crash-safe.** The batch path saves a `.eval_batch_state.json` /
   `.csv2_batch_state.json` state file per chunk. If the process dies, hits a quota
   limit, or the machine reboots, re-running the same command resumes from where it
   left off via `resume_or_submit()`. Zero work is lost.
2. **50% cheaper.** OpenAI batch API is half-price vs synchronous.
3. **No rate-limit cascade.** Batch requests are queued server-side and execute
   within the quota window. Synchronous mode hammers the API and triggers 429s
   that compound with exponential backoff, wasting wall-clock time.

**Why fast mode is forbidden:**
1. **All-or-nothing writes.** `evaluate_alpha_file()` reads the entire JSONL,
   judges every record in memory, and writes the file only on completion. If the
   process dies at record 190/500 (quota, OOM, Ctrl-C), all 190 judged results are
   discarded. There is no incremental persistence.
2. **Non-resumable.** There is no state file. A retry re-judges every record from
   scratch, doubling the API cost.

**Incident (2026-03-27):** Alpha=1.0 binary judge evaluation was run in fast mode.
OpenAI quota was exhausted at ~190/500 records. All in-memory results were lost.
The batch that was submitted earlier (31 requests) was also stuck at 0/31 completed
due to the same quota issue. Total waste: ~190 judge calls ($) and ~30 minutes of
wall-clock time, plus the need to re-run the entire 500-record evaluation.

**In pipeline scripts**, always use batch mode:
```bash
# evaluate_intervention.py uses a hyphen:
uv run python scripts/evaluate_intervention.py --api-mode batch ...

# evaluate_csv2.py uses an underscore:
uv run python scripts/evaluate_csv2.py --api_mode batch ...
```

Note the flag inconsistency between the two scripts (`--api-mode` vs `--api_mode`).
Both default to batch, so omitting the flag is also acceptable. But if you spell it
out, get the punctuation right — a wrong flag silently fails with an argparse error.

**Before queuing batch-eval jobs, verify Tier-2 limits via Codex CLI:**
1. The pipeline scripts `scripts/infra/jailbreak_alpha1_pipeline.sh`,
   `scripts/infra/jailbreak_alpha1_eval_only.sh`, and
   `scripts/infra/csv2_pipeline.sh` now run
   `scripts/infra/check_openai_batch_limits_via_codex.sh` by default.
2. That helper launches `codex exec --search`, checks the current official OpenAI
   model pages, compares them against the hardcoded Tier-2 queue table in
   `scripts/openai_batch.py`, and only patches the local table if the docs moved.
3. The intent is "trust local constants for speed, but verify them before long,
   quota-sensitive jobs." Think of it like checking the tide chart before pushing
   a boat off the dock: the chart is local, but you still confirm the water level
   before committing to the trip.
4. Logs are written to `logs/openai_batch_limit_check_<timestamp>.log` and the
   final Codex verdict is written to
   `logs/openai_batch_limit_check_<timestamp>.summary.txt`.
5. Set `CODEX_VERIFY_OPENAI_LIMITS=0` only when you intentionally want to skip the
   preflight. Tune the helper with `OPENAI_LIMIT_CHECK_TIMEOUT`,
   `OPENAI_LIMIT_CHECK_MODELS`, or `CODEX_LIMIT_CHECK_CODEX_MODEL` if needed.
</important>

## Future: isolate runs by directory

The per-record reopen and the operational rules above are mitigations, not a root
fix. Every run still writes to a shared mutable path (`experiment/alpha_0.0.jsonl`)
which remains a collision surface for concurrent agents, interrupted reruns, and
accidental overwrites. The structurally robust solution:

- **Write into a run-scoped directory** (e.g. `experiment/runs/<run_ts>/alpha_0.0.jsonl`)
  so no two runs ever share a file path.
- **Maintain a `latest` symlink or manifest** pointing at the most recent completed
  run, so downstream scripts (evaluate, export) don't need path changes.
- **Optionally `fsync` every N records** for crash/power-loss durability (distinct
  from unlink safety — `close()` flushes userspace buffers but doesn't guarantee
  on-disk persistence).

This would eliminate the entire class of "something touched my output file" failures
rather than limiting their blast radius. Not urgent — the current mitigations are
adequate — but worth doing if the run matrix grows or multi-agent concurrency
increases.
