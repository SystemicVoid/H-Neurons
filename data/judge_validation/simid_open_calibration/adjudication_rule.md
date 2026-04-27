# SIMID Open Correctness Calibration Rule

Status: frozen before SIMID open calibration labeling.

Target: grade each SIMID open-response calibration case against the question,
gold aliases, and predicted answer in `open_calibration_queue.jsonl`.

Use exactly one label:

- `CORRECT`: the predicted answer contains the important information in the gold
  target or an accepted alias, and does not contradict the gold target.
- `INCORRECT`: the predicted answer makes a factual claim that contradicts the
  gold target, including a hedged but committed wrong answer.
- `NOT_ATTEMPTED`: the predicted answer does not contain the important
  information in the gold target and also does not contradict it.

Decision rules:

1. Semantic equivalence is enough. Ignore capitalization, punctuation, grammar,
   answer order, and minor name typos when the intended entity is clear.
2. Do not require extra information that is already implied by the question.
   For example, if the question asks for a city and the gold alias includes a
   state or country, the city alone can be correct.
3. Ignore harmless extra context, caveats, or explanations if the response
   eventually commits to a single correct answer and contains no contradiction.
4. Mark as `INCORRECT` when the response commits to a wrong answer, adds a wrong
   extra answer to an otherwise correct answer, or denies a true gold fact.
5. Mark as `NOT_ATTEMPTED` when the response refuses, says it does not know,
   gives only partial information, asks for clarification, or lists multiple
   candidates without committing to a single correct answer, provided it does
   not contradict the gold target.
6. If a response gives a primary answer plus alternatives or caveats, use the
   primary answer. If the primary answer is correct and there is no
   contradiction, mark `CORRECT`; if it is wrong, mark `INCORRECT`.
7. If the answer is numeric and the gold alias gives an accepted range, mark
   values in range as `CORRECT`, values out of range as `INCORRECT`, and vague
   bounds that do not establish the target value as `NOT_ATTEMPTED`.

Secondary-rater labeling:

- Do not use the primary judge label, deterministic alias grade, or sample
  source when making the secondary label.
- Use only the question, gold aliases, predicted answer, and this rule.
- Return only `CORRECT`, `INCORRECT`, or `NOT_ATTEMPTED`.

Disagreement adjudication:

- Adjudicate exactly the cases where the primary and secondary labels disagree.
- Assign the final label by applying this same rule to the case.
- Set `rule_gap: true` only when this rule cannot be applied because the case is
  malformed, missing the question, missing all gold aliases, missing the
  predicted answer, or exposes a genuine rubric conflict not resolved above.
- Do not set `rule_gap: true` for ordinary hard factual judgments, borderline
  paraphrases, or cases where the rule gives an unfavorable label.
