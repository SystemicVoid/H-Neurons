# SIMID Diagnostic Exact-Propagation Sensitivity

This is a diagnostic measurement-cleanup sensitivity analysis. It applies fresh labels only when a full MVP row has the same question and the same normalized model response as an eligible correction-evidence row. It does not mutate the historical SIMID outputs and does not make open correctness claim-bearing.

## Coverage

- MVP rows evaluated: 4800
- Exact evidence rows applied: 120
- Rows whose diagnostic label changed: 96
- Broader Core-i* rows blocked: 2
- Same-question nonexact rows blocked: 46

## Family Counts

| Evidence family | Exact rows | Changed rows |
|---|---:|---:|
| `adam_first_man_truthfulqa_boundary` | 8 | 0 |
| `amal_clooney_lawyer_human_rights_advocate` | 2 | 0 |
| `amal_clooney_plain_lawyer` | 22 | 22 |
| `core_i9_apple_silicon_wrong_extra` | 6 | 0 |
| `core_i9_processor_brand_boundary` | 16 | 8 |
| `cranberry_modifier_exact_repeat` | 14 | 14 |
| `peter_piper_plain_peppers` | 22 | 22 |
| `pithivier_custard_modifier` | 12 | 12 |
| `pithivier_fruit_modifier` | 2 | 2 |
| `viscous_fluid_modifier` | 16 | 16 |

## Selected Alpha=8 Open Correctness

| Scope | Primary rate | Diagnostic exact rate | Rate shift | Primary delta | Diagnostic exact delta | Delta shift |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 41.5% [34.5, 48.5] | 40.0% [33.0, 47.0] | -1.5 pp | +3.0 pp [-2.0, +8.5] | +2.0 pp [-3.5, +7.5] | -1.0 pp |
| TruthfulQA | 41.0% [32.0, 51.0] | 41.0% [32.0, 51.0] | +0.0 pp | +12.0 pp [+4.0, +20.0] | +12.0 pp [+4.0, +20.0] | +0.0 pp |
| Bridge | 42.0% [32.0, 52.0] | 39.0% [29.0, 49.0] | -3.0 pp | -6.0 pp [-12.0, +0.0] | -8.0 pp [-15.0, -2.0] | -2.0 pp |

## Validation

- Uncorrected recomputation reproduces the selected pooled, TruthfulQA, and Bridge alpha=8 rate and delta cells from `results_adjudicated.json`.
- Exact propagation counts equal the evidence table's `mvp_exact_pattern_count` by family.
- Blocked and nonpropagatable rows leave labels unchanged.
- Core i9 and Core i9 Apple Silicon remain separate exact-answer rules.
