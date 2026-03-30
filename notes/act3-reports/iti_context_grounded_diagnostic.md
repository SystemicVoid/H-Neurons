# Diagnostic: `iti_context_grounded` is not a truthfulness artifact

## Construction

The `iti_context_grounded` artifact is extracted from SQuAD-v2 with this labeling:

| Question type | "Truthful" example | "Untruthful" example |
|---|---|---|
| Answerable (50%) | Gold answer (present in context) | Wrong answer (from another Q's gold, not in this context) |
| Impossible (50%) | Abstention string: "The context does not contain the answer." | Wrong answer (from another Q's gold) |

## The signal is passage-answer overlap, not truthfulness

For answerable questions, the gold answer is a **substring of the context** by construction (SQuAD is extractive QA). The wrong answer is sampled from a different question and almost certainly does not appear in the context. So the probe faces:

- Truthful: answer appears verbatim in context → high lexical/semantic overlap in activations
- Untruthful: answer is unrelated to context → low overlap

This is a trivial context-answer matching task, not truthfulness discrimination.

For impossible questions, the probe distinguishes the fixed abstention string from a random factual answer — also trivial and unrelated to truthfulness.

## Empirical confirmation

**Near-perfect probe metrics** (from `extraction_metadata.json`):

| Rank | Layer | Head | AUROC | σ |
|------|-------|------|-------|------|
| 1 | 14 | 7 | 0.9999 | 2.43 |
| 2 | 5 | 1 | 0.9998 | 7.47 |
| 3 | 13 | 1 | 0.9991 | 2.64 |

AUROC ≈ 1.0 across dozens of heads. This is consistent with a shortcut signal (context substring matching) rather than a deep truthfulness representation.

**Flat FaithEval calibration**: 85% compliance at all alphas (0.0–2.0, n=20). Despite large σ values (up to 7.47), the intervention produces zero behavioral change. The steering direction pushes toward "answer matches context" — which is orthogonal to FaithEval's counterfactual compliance task.

## Conclusion

`iti_context_grounded` is a **passage-relevance / local-fit detector**, not a truthfulness artifact. It should not be used for:
- Out-of-distribution truthfulness claims
- FaithEval compliance experiments
- Any generalization argument about ITI and truthfulness

Retired from primary ITI claims as of this note.
