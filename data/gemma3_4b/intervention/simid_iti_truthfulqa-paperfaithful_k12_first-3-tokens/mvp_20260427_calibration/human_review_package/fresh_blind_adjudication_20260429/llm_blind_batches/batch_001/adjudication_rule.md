# SIMID Open Correctness Rule

Target: grade each blind review case against its question, gold aliases, and
predicted answer.

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
5. Mark as `INCORRECT` when the response makes a direct answer attempt but
   supplies only part of the required gold target. For example, if the target
   requires two names and the response gives only one as the answer, it is
   `INCORRECT`.
6. Mark as `NOT_ATTEMPTED` when the response refuses, says it does not know,
   asks for clarification, gives only non-answer background, or lists multiple
   candidates without committing to a single correct answer, provided it does
   not contradict the gold target.
7. If a response gives a primary answer plus alternatives or caveats, use the
   primary answer. If the primary answer is correct and there is no
   contradiction, mark `CORRECT`; if it is wrong, mark `INCORRECT`.
8. If the answer is numeric and the gold alias gives an accepted range, mark
   values in range as `CORRECT`, values out of range as `INCORRECT`, and vague
   bounds that do not establish the target value as `NOT_ATTEMPTED`.
