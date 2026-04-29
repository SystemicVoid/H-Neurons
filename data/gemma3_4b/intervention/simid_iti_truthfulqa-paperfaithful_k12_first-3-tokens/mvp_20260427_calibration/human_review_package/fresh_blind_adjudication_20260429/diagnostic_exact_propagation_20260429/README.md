# Diagnostic Exact Propagation 2026-04-29

Append-only SIMID measurement-cleanup sensitivity outputs. These artifacts apply
fresh labels only to full MVP rows with the same question and the same normalized
model response as an eligible duplicate-collapsed correction-evidence row.

Files:

- `row_corrections.jsonl` - one row per 4,800-row MVP open response, including
  original label, diagnostic exact-propagation label, evidence family, status,
  and notes.
- `diagnostic_sensitivity_summary.json` - paired base-item open-correctness
  rates and deltas under primary and diagnostic exact-propagation labels, plus
  coverage and validation checks.
- `diagnostic_sensitivity.md` - compact human-readable sensitivity report.
- `diagnostic_exact_propagation.provenance.*.json` - command, inputs, outputs,
  active-run guard metadata, and hashes.

Regenerate from the repository root with:

```bash
uv run python scripts/build_simid_boundary_propagation_sensitivity.py --overwrite
```
