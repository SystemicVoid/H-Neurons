# Prospective SIMID Effect-Run Gate r2

This append-only r2 package supersedes
`prospective_effect_run_gate_20260429/`. The original package remains
diagnostic-only and is not rewritten.

This package freezes the next SIMID effect-run protocol after the positive
prospective open-grading agreement result, with an additional requirement that
claim-bearing open correctness must use complete external blind labels.

Files:

- `protocol.md`: reviewer-facing frozen protocol and commands.
- `effect_run_manifest.json`: machine-readable protocol, path/hash bindings,
  metrics, controls, and claimability gates.
- `excluded_effect_sample_ids.jsonl`: historical MVP and prospective
  calibration sample/base IDs that the future manifest must exclude.

`gpt-4o` adjudication is diagnostic-only in this protocol. This package is a
planning gate only. It does not run SIMID and does not make historical MVP
open-correctness metrics claim-bearing.
