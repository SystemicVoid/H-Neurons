# Scripts — Evaluation Notes

## Jailbreak generation: use `max_new_tokens=512` minimum

`run_intervention.py` defaults to 256, which truncates compliant responses and biases classification. Prefer 1024 and `do_sample=False` for reproducibility.

## JSONL writes: reopen file per record

Open the output file **inside** the sample loop, not outside it. Holding one fd across the loop silently loses all writes if the path is unlinked mid-run (e.g. by a concurrent `git commit`). Incident: 2025-03-23, 500 samples destroyed.

## Never touch output directories during a GPU run

Before any `git add/rm`, `mv`, `rm`, or restructuring, check for active runs:
```bash
ps aux | grep run_intervention   # or check tmux panes
```
A running job has `"status": "running"` in its `*.provenance.json`. Wait for it to finish.

## OpenAI evaluation: batch mode only, never fast

Always use `--api-mode batch` / `--api_mode batch`. Fast mode is all-or-nothing in memory — a crash or quota hit loses everything. Batch is crash-safe (`.eval_batch_state.json`) and 50% cheaper.

**Flag names:** Both `evaluate_intervention.py` and `evaluate_csv2.py` accept `--api-mode` (hyphen). `evaluate_intervention.py` also accepts `--api_mode` (underscore alias). Both default to batch — omitting the flag is fine.

Pipeline scripts run `scripts/infra/check_openai_batch_limits_via_codex.sh` automatically before submitting. Set `CODEX_VERIFY_OPENAI_LIMITS=0` to skip. Incident: 2026-03-27, 190 judge calls lost.
