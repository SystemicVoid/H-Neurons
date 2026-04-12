# 1. Introduction

A predictive internal signal — a neuron, feature, or direction that reliably discriminates between behavioral categories on held-out data — is a tempting steering target. If a feature predicts whether a language model will produce a hallucination, a harmful response, or a factually incorrect answer, it is natural to expect that amplifying or suppressing that feature will steer the model toward the desired behavior. This heuristic underlies much recent work in activation steering: identify a strong readout, then intervene through it (Li et al., 2023; Gao et al., 2025; Arditi et al., 2024).

The heuristic sometimes works. Refusal-mediating directions identified by simple difference-in-means produce reliable refusal modulation (Arditi et al., 2024). Hallucination-associated neurons selected by classification performance can shift over-compliance behavior (Gao et al., 2025). But the heuristic also sometimes fails, and the failure modes have received less systematic attention than the successes. When a strong readout fails as a steering target, is the problem in the readout, the intervention operator, the evaluation, or the generalization?

This paper tests the readout-to-steering heuristic empirically in Gemma-3-4B-IT. We compare multiple intervention families — neuron scaling, sparse autoencoder feature steering, inference-time intervention via attention heads, and gradient-based causal head selection — under a shared evaluation contract across contextual faithfulness, answer selection, open-ended factual generation, and jailbreak settings. We find repeated dissociations between four stages that are routinely conflated:

1. **Measurement** — Can we trust the evaluation? Truncation artifacts, binary-versus-graded scoring, and evaluator construct mismatch each changed the scientific conclusion about whether an intervention worked (§6).

2. **Localization** — Does the readout identify causally relevant components? SAE features matched H-neurons on detection quality (AUROC 0.848 vs. 0.843) yet produced null steering on the same benchmark. Probe-ranked heads achieved perfect discrimination (AUROC 1.0) yet null intervention (§4).

3. **Control** — Does intervention produce the intended behavioral change? When it did, the effect was narrow: H-neurons improved compliance on FaithEval but not on BioASQ; ITI improved answer selection but not open-ended generation (§5).

4. **Externality** — Does the effect transfer without causing harm? The TriviaQA bridge benchmark revealed that ITI does not merely degrade generation — it redistributes probability mass toward semantically nearby but wrong entities, producing confident factual substitutions rather than refusals (§5.3).

These results suggest that strong readouts are insufficient evidence for good steering targets and that credible mechanistic intervention claims require stage-specific validation. We organize these observations into a four-stage audit framework — measurement, localization, control, externality — and distill five concrete recommendations for researchers evaluating intervention claims (§7).

Figure 1 previews this four-stage scaffold and situates the paper's three anchor case studies within it.

**Contributions.** (1) A cross-method empirical dissociation between readout quality and steering utility, with matched-detection comparisons that control for evaluation surface and intervention operator. (2) An externality analysis showing that steering gains on answer-selection benchmarks do not transfer to open-ended generation, with a specific failure-mode diagnosis (confident wrong-entity substitution). (3) A staged audit framework and checklist for evaluating the credibility of mechanistic intervention claims.
