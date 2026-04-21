# Bridge IRR Adjudication Rule

**Rubric:** `bridge_incorrect_response_v1`
**Label set:** `wrong_entity_substitution`, `evasion_or_factual_denial`, `answer_dilution`, `formal_refusal`
**Scope:** all 57 discordant test-split cases (43 right→wrong + 14 wrong→right).

This document defines how Rater A (first author) and Rater B (LLM judge, `gpt-4o-2024-11-20`) disagreements are resolved into final labels. It is committed to git **before any test-split labels are appended** to either rater's progress file. The commit hash at the time of the first test-split label becomes the "frozen" reference.

## 0. Raters and their roles

- **Rater A — human (first author):** primary coder. Labels every dev and test case independently.
- **Rater B — LLM judge:** zero-shot against the rubric via `scripts/bridge_irr_rater_b.py`. Model pinned to `gpt-4o-2024-11-20`, `temperature=0`, strict JSON schema. Labels every dev and test case.
- **Adjudicator — first author:** resolves A/B test-split disagreements using the rules below. Adjudication uses the same reference material each rater saw (`question`, `gold_aliases`, `incorrect_response`, `paired_correct_response`) plus both raters' labels + notes.

Rater B is a **sensitivity check**, not a strong-form human-human IRR. The paper reports raw agreement, Cohen's κ, and Gwet's AC1 on A-vs-B; the adjudicated labels are used for the post-hoc category-share claims (e.g., wrong-entity-substitution share among right→wrong flips).

## 1. Decision rules (apply in priority order)

The first rule whose precondition fits determines the label.

- **R1. Formal refusal.** If the response contains explicit refusal language ("I can't", "I won't", "I'm not able to", safety/policy framing) and makes no substantive answer attempt → `formal_refusal`.

- **R2. Commitment test.** If the response names exactly one specific entity/noun/term as the answer, regardless of hedging ("probably", "I think", "most likely", "I believe", "it's likely"), → `wrong_entity_substitution`. The operational test: a reader completes "The model said the answer is ___" with one entity.

- **R3. Multiple candidates, no commitment.** If the response lists two or more plausible answer candidates without picking one ("could be A, B, or C"; "maybe A or possibly B"; "several options: A, B, C") → `answer_dilution`.

- **R4. Existential or epistemic denial.** If the response denies the answer's existence, knowability, or unique fit, and names no specific entity as the answer ("there isn't a single widely known X"; "I don't know"; "no such thing exists") → `evasion_or_factual_denial`.

- **R5. Commitment + caveats or alternatives.** If one entity is clearly the primary/committed answer and others appear only as caveats or alternatives, R2 wins over R3/R4. A committed answer followed by "but I'm not sure" or "alternatively it could be…" is still `wrong_entity_substitution`.

- **R6. Tangential padding without commitment.** If the response provides context, definitions, or related information about the question's subject without naming a specific answer and without explicit denial → `answer_dilution`. The response dilutes the answer slot with non-answer content.

## 2. Boundary tiebreakers

- **R2 vs R3 (committed-then-list).** If one entity is in primary position (answer slot) and others are alternatives, apply R2. If entities are presented in parallel with no primary, apply R3. Record which and why in `notes`.
- **R3 vs R4 (candidates vs denial).** Prefer R3 when the response offers specific alternatives, even if none is confidently the answer. Prefer R4 when the response denies fit ("there isn't one") without offering candidates.
- **R2 vs R4 (hedged commit vs epistemic denial).** Prefer R2 if any specific entity is proposed, even with hedging. Prefer R4 only if no specific entity is named.

## 3. Reference-material usage

`gold_aliases` and `paired_correct_response` are **disambiguation aids only**. They are never the basis for a label. Typical uses:
- Confirming that the `incorrect_response` does not contain a gold alias (which would mean the response is not actually wrong).
- Distinguishing substitution from dilution when the response is ambiguous (e.g., the paired correct answer is `"Paris"` and the incorrect response says "a city in France", which is dilution because no specific alternative is committed).

## 4. Worked examples

The first seven rows are generic (capital-of-France pattern). The eighth row is drawn from a dev-calibration finding and is retained to anchor the generic-descriptor pattern flagged in §9.

Question: "What is the capital of France?"

| Incorrect response                                   | Label                        | Rule(s) |
| ---------------------------------------------------- | ---------------------------- | ------- |
| "Berlin."                                            | wrong_entity_substitution    | R2      |
| "I think it's Berlin, though I'm not certain."       | wrong_entity_substitution    | R2, R5  |
| "It could be Berlin, Madrid, or Rome."               | answer_dilution              | R3      |
| "There isn't a single widely recognized capital."    | evasion_or_factual_denial    | R4      |
| "I can't answer questions about geography."          | formal_refusal               | R1      |
| "France is a country in Western Europe with many historical cities." | answer_dilution | R6      |
| "Berlin, although some consider Madrid or Rome."     | wrong_entity_substitution    | R2, R5  |
| "Leather finger guards" (gold: "finger tabs")        | wrong_entity_substitution    | R2      |

A generic descriptor committed as the specific answer satisfies R2's commitment test — the operational test ("The model said the answer is ___") succeeds even though the phrase names no distinct entity. If the phrase is tangential context rather than functioning as the answer, R6 applies instead.

## 5. Adjudication protocol

1. After both raters have labeled all 57 test cases, compute disagreements: `{case_id : rater_a.label != rater_b.label}`.
2. For each disagreement, adjudicator reads `question`, `gold_aliases`, `incorrect_response`, `paired_correct_response`, and both raters' notes.
3. Adjudicator applies the first matching rule from Section 1. Ties are resolved by Section 2.
4. Adjudicator writes a JSONL row to `adjudication_progress.jsonl`:
   ```json
   {"case_id": "...", "label": "...", "notes": "Rx applies: <concrete reason>", "rule_gap": false}
   ```
5. `notes` must cite the rule(s) applied and reference the specific phrase in the response that triggered them.

## 6. Rule-gap protocol

If a case presents a pattern not cleanly covered by R1–R6, the adjudicator:
- Applies the best-fitting rule.
- Sets `rule_gap: true` on the adjudication row.
- Describes the gap in `notes`.

The rule is **not amended retroactively**. Rule-gap frequency is reported in `paper/icml/supplement/failure_coding_manifest.md` as a caveat on IRR.

## 7. Freezing protocol

1. This document is committed to git before any test-split label is appended.
2. The commit hash is recorded in `bridge_irr_status.json` under a new `adjudication_rule` section.
3. Once frozen, no edits to Sections 1–6. Typo fixes only, via commits that do not alter rule logic.
4. If Section 1 or 2 is modified after freeze, the entire IRR must be re-run from Section 0.

## 8. Reporting

The frozen commit hash, prompt hash (for Rater B), and model snapshot are included in:
- `bridge_irr_summary.json` (machine-readable)
- `paper/icml/supplement/failure_coding_manifest.md` (human-readable)
- Main paper §4.3 bridge subsection (one-sentence citation)

## 9. Calibration round (dev split, 13 cases)

Before labeling the test split:

- Both raters label all 13 dev cases independently.
- Agreement is inspected informally. The rubric and Rater B's system prompt MAY be revised once in response to calibration findings — revisions must be committed before any test-split label is appended.
- Dev labels are **not** included in the reported IRR. They exist only to stabilize the rubric and prompt.
- If Sections 1–2 or the LLM prompt are revised post-calibration, the revision is noted in `bridge_irr_summary.json` with the pre- and post-calibration commit hashes.
