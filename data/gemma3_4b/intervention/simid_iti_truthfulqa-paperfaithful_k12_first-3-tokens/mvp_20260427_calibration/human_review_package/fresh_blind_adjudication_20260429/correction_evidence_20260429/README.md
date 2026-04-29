# Correction Evidence 2026-04-29

Append-only duplicate-collapsed correction-evidence table derived from the fresh
targeted blind adjudication pass. This is measurement cleanup only; it does not
make SIMID open correctness claim-bearing.

Files:

- `duplicate_collapsed_correction_evidence.jsonl` - machine-readable local
  correction evidence plus diagnostic exact-answer propagation guardrails.
- `duplicate_collapsed_correction_evidence.md` - compact human-readable view of
  the same rows.
- `validation_summary.json` - schema, coverage, comparison, and propagation
  checks.
- `correction_evidence.provenance.20260429_*.json` - source and output hashes.

Regenerate from the repository root with:

```bash
uv run python scripts/build_simid_boundary_correction_evidence.py --overwrite
```
