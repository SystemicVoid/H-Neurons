# Rule-Gap Triage — `simid_bridge_odql_6602`

Single-row triage for the one `rule_gap: true` label in the Opus 4.7 canary
(merged labels file
`prospective_open_labels_opus47_canary.jsonl`,
sha256 `9a6406dafdf68a0e605edb6ccf78130a5fd4c0c2a9bb6fbb4a5af3ec9413b588`).

## Identifiers

- `blind_case_id`: `simid_prosp_open_canary_blind_113213d946502f4e`
- `review_order`: 51
- `sample_id` / `base_sample_id`: `simid_bridge_odql_6602`
- `dataset`: `triviaqa_bridge`
- `condition`: `random_direction_seed4`
- `alpha`: 8.0
- `source_id`: `odql_6602` (TriviaQA `rc.nocontext` validation QID)
- `bridge_primary_gold_alias` (from manifest): `Rhopalocystis nigra`
- Manifest entry:
  `data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json`,
  line ≈ 55336

## What the rater observed

```json
{
  "label": "NOT_ATTEMPTED",
  "confidence": 3,
  "rule_gap": true,
  "flags": ["malformed_case"],
  "notes": "Gold aliases describe Aspergillus niger fungus while the question asks about a West African river; predicted Niger River neither contains nor contradicts the alias set."
}
```

## Diagnosis

The row is genuinely malformed at the **upstream TriviaQA layer**, faithfully
forwarded by the SIMID Bridge generator.

- The question text on Bridge `simid_bridge_odql_6602` is the West Africa
  Niger-River question:
  > Which river of West Africa rises in the Fouta Djallan Highlands only 150
  > miles from the Atlantic, but runs for over 4000 kilometres in length before
  > flowing into the Atlantic?
- The `gold_aliases` are aliases for the fungus *Aspergillus niger*:
  `Rhopalocystis nigra`, `A niger`, `Aspergillopsis nigra`, `Black mould`,
  `Sterigmatocystis nigra`, `Aspergillus niger var. niger`,
  `Aspergillus niger`.
- The Bridge manifest's `distractor_provenance` records that both the question
  and the gold come from `source_question_id = odql_6602`, i.e. the SIMID
  Bridge generator joined what it expected to be a self-consistent TriviaQA
  question/answer pair. The mismatch is therefore in TriviaQA's
  `rc.nocontext` validation parquet (or the answer-alias mapping for
  `odql_6602`), not in the SIMID generator's join logic.
- The same `(question, gold_aliases)` pair appears on every condition × alpha
  copy of `simid_bridge_odql_6602`, so this is a **single, persistent**
  malformed row — not a one-off corruption, not condition-specific.

## Rubric assessment

The frozen rubric
(`prospective_open_calibration_gate_20260429/rubric.md`,
sha256 `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`)
explicitly authorizes `rule_gap: true` in this exact case. From rule 8:

> Set `rule_gap: true` only for malformed rows or genuine rubric conflicts not
> resolved here. Ordinary hard calls, unfavorable labels, or low confidence are
> not rule gaps.

A row whose question and gold target describe entirely different entities (a
West African river vs. a fungus species) is the canonical example of a
"malformed row." The rater's `rule_gap: true` and `flags: ["malformed_case"]`
calls are rubric-correct.

The label `NOT_ATTEMPTED` is also defensible under the rubric: the predicted
"The Niger River" neither contains nor contradicts an *Aspergillus niger*
alias. Once the row is treated as malformed, however, the label itself is
moot — the row should be excluded from the analysis.

## Best-practice fix

The rubric is **frozen** under
`simid_open_grading_rubric_prospective_20260429/v1` and the manifest-bound
exclusions file at
`prospective_effect_run_gate_20260429_r2_external_labels/excluded_effect_sample_ids.jsonl`
is hash-bound to the authority manifest
(`content_sha256: ed59129d99f3c90d36aa41412270e20d0d54807c5a3663b8164391b3c32f21be`).

Do **NOT**:

- modify `rubric.md`,
- modify the manifest-bound `excluded_effect_sample_ids.jsonl`,
- mutate the canary's merged labels file to drop or relabel this row,
- re-rate this row (the rule_gap call is correct).

**Do**:

1. Keep this canary label as-is. The `rule_gap` flag and the
   `malformed_case` tag are the correct outputs.
2. Record `simid_bridge_odql_6602` in
   `excluded_sample_ids_recommended_for_full_batch.jsonl` (sibling file in
   this triage directory) so that the full-batch labeling pipeline can pick
   it up under a **fresh** effect-run-gate manifest binding.
3. Before launching the full-batch labeling pass, run an LLM-based
   question/gold-consistency auditor over all 400 unique Bridge rows
   (the heuristic token-overlap audit in
   `bridge_question_gold_overlap_audit.{jsonl,md}` is too loose to screen
   on its own — 342/400 = 85.5% of rows have zero overlap by chance).
   Promote any confirmed malformations to the recommended-exclusion sidecar
   alongside this row.
4. When the full-batch open label package is built, bind the new exclusions
   file by sha256 in the new effect-run-gate manifest. The new manifest
   supersedes the existing `r2_external_labels` manifest for the
   external-label package; it does not retroactively modify the existing one.

## Why this is canary-internal and not authority-bearing

The canary is explicitly not claim-bearing. Its purpose is to catch exactly
the class of issue it caught: a single pre-existing data-layer malformation
that would have polluted ~1/854 of every per-condition analysis. The fix
posture is to leave the frozen artifacts alone, persist the finding here,
and let the full-batch run consume the recommendation under a fresh binding.
