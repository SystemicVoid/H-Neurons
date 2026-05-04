# Validation Summary

Status: **PASS**

- Scope rows: 908
- New labels: 876
- Reused canary labels: 32
- Paired sample_id coverage: 454 / 454
- Distinct base_sample_id clusters covered: 227 / 227
- Rule-gap rows: 0

All hard-gate checks passed:

- row count matches scope after exclusions and canary reuse
- every sample_id has both alpha=0 and alpha=8 labels
- no duplicate blind_case_id and every full-scope blind row has exactly one label
- no private-field leakage in merged labels
- every label is bound to the package blind-cases hash and frozen rubric hash
- label / confidence / rule_gap / flags all satisfy the closed schema
