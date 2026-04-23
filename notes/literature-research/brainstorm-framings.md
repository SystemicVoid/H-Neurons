## What would most likely change the conclusion with additional evidence

A matched causal decomposition across all tasks would matter most: same intervention family, same alpha schedule, same model, and direct logit/token-margin traces for FaithEval, BioASQ, FalseQA, Jailbreak, TruthfulQA, SimpleQA, and TriviaQA.

The conclusion would also change if additional evidence showed that the same intervention robustly improves open-answer factual accuracy without increased non-attempts, substitutions, or evaluator-specific artifacts.

A stronger mechanistic identification would change the framing if it showed that the relevant internal direction causally tracks latent truth independent of answer length, prompt format, first-token priors, and refusal/attempt policy.








## Framings

### Truth Readouts Expose Answer-Margin Actuators, Not Portable Truth Variables

Abstract-style thesis

The evidence supports a narrower and more interesting claim than “truth readouts fail as steering targets.” Sparse internal readouts do identify behaviorally relevant directions, and some interventions have real causal force. But their causal action is not best understood as manipulating truth itself: the strongest unifying account is that these handles perturb answer-token margins and response policy, producing target-answer compliance, factual improvements, substitutions, non-attempts, or apparent evaluator-dependent changes depending on the task interface.

Core empirical spine
Start with real readability. A sparse readout of 38 units predicts answer correctness with readout.disjoint.auroc = 0.8429, and SAE readouts are similarly predictive with readout.sae.auroc = 0.8477.
Show real causal force in a controlled answer-selection setting. FaithEval h-neuron steering has intervention.faitheval.anti.delta_0_to_max_pp = 6.3 and beats random controls by intervention.faitheval.mean_slope_difference_pp_per_alpha = 2.0143.
Break the naive truth-control story. BioASQ changes 1339 / 1600 responses but gives intervention.bioasq.delta_0_to_max_pp = -0.0625; SAE readouts remain predictive but produce intervention.faitheval_sae.h_slope_pp_per_alpha = 0.16; utility-selected SAE features do not improve held-out compliance and reduce margins.
Show that gains and harms concentrate at the answer-policy layer. ITI improves TruthfulQA MC by 6.2595 to 7.4939 pp, but on open-answer transfer it drops TriviaQA bridge accuracy by -5.8 pp and SimpleQA attempt rate by -32.7 pp.
Diagnose the failure mode mechanistically. Right-to-wrong bridge failures are mostly wrong-entity substitutions, 0.7209, not formal refusals, 0.0, and first-token margins move in the expected direction: substitutions shift by -10.1552 nats while rescues shift by +4.7285 nats.
Use measurement as a boundary condition, not the headline. FaithEval standard raw scoring gives a negative slope, -1.4071, while target-answer scoring gives a positive slope, 2.0929; Jailbreak v2/v3 scorers change effect size. The measurement layer matters because the intervention changes response policy, not because the paper is “about evaluators.”


It explains both positive and negative results. The obvious “readouts are unreliable steering targets” framing explains BioASQ and SAE failure but has to treat FaithEval, FalseQA, TruthfulQA MC, and D7 as exceptions. The answer-margin framing predicts exactly this patchwork.
It turns transfer failures into mechanism evidence. TriviaQA wrong-entity substitutions, SimpleQA non-attempts, and first-token margin shifts are not just side effects; they reveal the layer where the intervention acts.
It avoids a generic measurement paper. Evaluator sensitivity is real, but it is downstream of the intervention changing output form and answer policy. That is a mechanistic interpretability story, not merely a benchmark audit.