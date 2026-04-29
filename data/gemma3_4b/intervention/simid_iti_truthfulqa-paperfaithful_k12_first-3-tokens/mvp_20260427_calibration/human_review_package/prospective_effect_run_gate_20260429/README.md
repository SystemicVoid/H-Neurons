# Prospective SIMID Effect-Run Gate

This append-only package freezes the next SIMID effect-run protocol after the
positive prospective open-grading agreement result.

Files:

- `protocol.md`: reviewer-facing frozen protocol and commands.
- `effect_run_manifest.json`: machine-readable protocol, path/hash bindings,
  metrics, controls, and claimability gates.
- `excluded_effect_sample_ids.jsonl`: historical MVP and prospective
  calibration sample/base IDs that the future manifest must exclude.

This package is a planning gate only. It does not run SIMID and does not make
historical MVP open-correctness metrics claim-bearing.
