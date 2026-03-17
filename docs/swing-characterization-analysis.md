# Swing Characterization: Analysis and Safety Interpretation

## Executive Summary

Of 1,000 FaithEval evaluation samples subjected to H-neuron scaling across seven alpha values (0.0–3.0), 600 always comply with misleading context, 262 never comply, and **138 (13.8%, 95% CI [11.8, 16.1]) are swing samples** whose compliance state changes with scaling strength. Profiling this sensitive subpopulation reveals that **94 of 138 swing samples (68.1%, 95% CI [59.9, 75.3]) follow R→C trajectories**: the model initially gives the correct answer but switches to following misleading context as H-neuron scaling increases. This is a genuine knowledge-override effect, not a formatting artifact — surface features (word overlap p=0.78, response length p=0.93) do not predict swing status. The safety concern is real but bounded: knowledge override affects 9.4% of the total population, not the full benchmark, and 23% of swing samples show the beneficial C→R pattern where scaling helps the model recover correct knowledge.

## 1. The Population Landscape

The tri-modal split:

| Population | Count | Share | 95% CI |
|---|---|---|---|
| Always compliant | 600 | 60.0% | [56.9, 63.0] |
| Never compliant | 262 | 26.2% | [23.6, 29.0] |
| **Swing** | **138** | **13.8%** | **[11.8, 16.1]** |

**What "swing" means mechanistically:** a sample is swing if its compliance state changes across any alpha value. Think of the 1,000 samples as an electorate. 600 always vote one way (always compliant — they follow misleading context regardless of H-neuron scaling), 262 always vote the other (never compliant — they resist context pressure at every alpha), and 138 are the undecided voters whose behavior the intervention can actually move. The entire 6.3pp headline compliance shift from α=0.0 to α=3.0 is concentrated in this 14% swing bloc.

## 2. Swing Subtypes: The Core Safety Finding

Each swing sample follows one of three trajectory shapes across the alpha sweep:

### R→C: Knowledge Override (68.1%, 95% CI [59.9, 75.3])

- **Count:** 94 of 138 swing samples
- **Pattern:** Resistant at low α → Compliant at high α
- **Interpretation:** The model initially knew the correct answer (gave it at α=0.0) but was overridden by H-neuron scaling into following the misleading context.
- **Safety relevance:** This is the concerning subtype. Scaling H-neurons causes the model to abandon genuine knowledge in favor of contextual misinformation.

### C→R: Uncertainty Resolution (23.2%, 95% CI [16.9, 30.9])

- **Count:** 32 of 138 swing samples
- **Pattern:** Compliant at low α → Resistant at high α
- **Interpretation:** The model initially followed the misleading context but recovered its correct knowledge as scaling increased.
- **Safety relevance:** This is the beneficial subtype. Here, H-neuron scaling helps resolve genuine uncertainty in the right direction. This counterpoint prevents characterizing all H-neuron scaling as harmful.

### Non-monotonic (8.7%, 95% CI [5.0, 14.6])

- **Count:** 12 of 138 swing samples
- **Pattern:** Oscillates between compliant and resistant across alpha values
- **Interpretation:** Near the decision boundary, small changes in scaling produce inconsistent outcomes. These samples likely sit on a knife-edge where the model's knowledge and the contextual pressure are approximately balanced.

## 3. Structural Proxies: What They Tell Us (and Don't)

We tested whether swing status can be predicted from surface-level features of the evaluation items.

### Informative proxies

| Feature | Test | p-value | Effect | Interpretation |
|---|---|---|---|---|
| Context length | Kruskal-Wallis | **0.0001** | H=18.1 | Always-compliant items have longer contexts (mean 2,251 chars) vs. swing (2,111) and never-compliant (2,099). Longer contexts may provide more persuasive material. |
| Source dataset | Chi-squared | **3.0e-5** | V=0.20 | Some source datasets contribute disproportionately to swing vs. fixed populations. Mercury and Mercury_SC dominate all groups, but the ratios differ. |

### Null proxies

| Feature | Test | p-value | Interpretation |
|---|---|---|---|
| Word overlap | Kruskal-Wallis | 0.78 | Context–question lexical similarity does not predict swing status. |
| Response length (standard) | Kruskal-Wallis | 0.93 | Output verbosity is identical across populations. |
| Topic distribution | Chi-squared | 0.06 (V=0.09) | Topic categories do not significantly separate populations. |
| Number of options | — | Constant (4) | All swing samples have exactly 4 options. |

### Interpretation

The null results are the important finding. If swing behavior were driven by input structure — longer questions, more persuasive language overlap, or specific topic domains — then the safety concern would reduce to "fix the evaluation items." Instead, surface features explain very little of who swings and who doesn't. This points to the mechanism being about **internal model states**: how the model weighs its prior knowledge against contextual evidence, which is exactly the kind of computation H-neurons might modulate.

The one informative proxy (context length) has a straightforward explanation: longer contexts provide more material for the model to anchor on, so always-compliant items tend to have longer contexts. But this is a population-level tendency, not a predictive tool for individual swing status.

## 4. Transition Dynamics

The transition alpha is the scaling factor at which a swing sample first changes its compliance state.

| Subtype | Mean α | Median α | n |
|---|---|---|---|
| R→C | 1.40 | 1.25 | 94 |
| C→R | 1.19 | 1.00 | 32 |
| Non-monotonic | 0.79 | 0.50 | 12 |

**R→C vs. C→R comparison:** Mann-Whitney U=1731, p=0.19, r=-0.15. The two subtypes do not differ significantly in when they transition. Both tend to flip early (α ≤ 1.5). The difference between R→C and C→R is directional (which way the model moves), not temporal (how much scaling is needed to trigger the move).

**R→C resistance strength distribution:**

| First compliant α | Count |
|---|---|
| 0.5 | 32 (34%) |
| 1.0 | 15 (16%) |
| 1.5 | 13 (14%) |
| 2.0 | 13 (14%) |
| 2.5 | 15 (16%) |
| 3.0 | 6 (6%) |

About a third of R→C samples flip at the lowest non-zero scaling (α=0.5), suggesting weak knowledge that is easily overridden. The remaining two-thirds require stronger scaling, with some (6 samples) holding out until the maximum α=3.0.

## 5. LLM Enrichment Results

*This section will be populated after the GPT-4o-mini enrichment completes.*

The enrichment pipeline classifies each sample along three dimensions:
1. **Knowledge classification:** Does the question test well-known facts, common knowledge, specialized knowledge, or obscure trivia?
2. **Independent answer verification:** Does GPT-4o-mini agree with the model's α=0.0 answer?
3. **Context persuasiveness:** How convincing is the misleading context on a 1–5 scale?

Expected analyses:
- Knowledge distribution by population (swing vs. always-compliant vs. never-compliant)
- Knowledge distribution by swing subtype (R→C vs. C→R)
- Persuasiveness ratings by population
- Verification agreement rates

## 6. Safety Implications

### Knowledge override is real but bounded

The 68% R→C rate among swing samples is a genuine safety signal. These are cases where the model demonstrably knew the correct answer and was steered away from it by H-neuron scaling. However, this affects 94 out of 1,000 total samples (9.4%), not the whole population. The safety concern is localized to a specific subpopulation.

### Not all scaling is harmful

The 23% C→R counterpoint is important for framing. In these cases, H-neuron scaling helped the model recover correct knowledge it had initially lost to context pressure. A blanket claim that "H-neuron scaling causes hallucination" would be inaccurate — the mechanism works in both directions, with the knowledge-override direction dominating.

### The mechanism is internal, not input-driven

The null structural proxies (word overlap, response length, topic) suggest that what determines whether a sample swings is not about how the question is worded or structured. It appears to be about the model's internal balance between prior knowledge and contextual evidence. This makes the safety concern harder to address with input-level mitigations (e.g., rephrasing prompts) and more likely to require model-level interventions.

## 7. What We Cannot Know

- **Single model:** All results are from Gemma 3 4B-IT. The R→C/C→R ratio could differ substantially on other architectures.
- **Single benchmark:** FaithEval tests a specific kind of misleading-context multiple choice. The swing population on open-ended generation tasks may behave differently.
- **Prompt sensitivity:** The anti-compliance prompt frame influences the baseline compliance rate. Different framing could shift the boundary between always-compliant and swing.
- **GPT-4o-mini as judge:** The LLM enrichment uses a different model to classify knowledge and verify answers. Agreement rates reflect cross-model consistency, not ground truth.
- **Causal direction:** We observe that H-neuron scaling correlates with compliance transitions, but the analysis does not isolate the causal pathway — the transition could be mediated by intermediate representations not captured by the 38-neuron set.

## 8. What Would Make This More Robust

1. **Cross-model replication:** Run the same swing characterization on Mistral-Small-24B or Llama-3.3-70B. If the 68/23 split holds across architectures, the finding is substantially stronger.
2. **Per-sample activation analysis:** Examine the 38 H-neuron activations at each alpha for swing samples to identify which neurons drive individual transitions.
3. **Larger swing population:** More FaithEval items or additional benchmarks would increase the swing sample count and tighten the confidence intervals.
4. **Adversarial contexts:** Test whether specifically crafted high-persuasiveness contexts increase the R→C rate, confirming the knowledge-override mechanism.

## Appendix: Statistical Details

### Population-level tests

| Feature | Test | Statistic | p-value | Effect size |
|---|---|---|---|---|
| Context length | Kruskal-Wallis | H=18.09 | 1.2e-4 | — |
| Question length | Kruskal-Wallis | H=7.96 | 0.019 | — |
| Response length (standard) | Kruskal-Wallis | H=0.14 | 0.93 | — |
| Word overlap | Kruskal-Wallis | H=0.49 | 0.78 | — |
| Source dataset | Chi-squared (df=36) | χ²=80.5 | 3.0e-5 | V=0.20 |
| Topic distribution | Chi-squared (df=10) | χ²=17.7 | 0.060 | V=0.09 |
| R→C vs. C→R transition α | Mann-Whitney U | U=1731 | 0.19 | r=-0.15 |

### Post-hoc comparisons (Bonferroni-corrected)

**Context length:**
- Always-compliant vs. never-compliant: p=0.0004
- Always-compliant vs. swing: p=0.022
- Never-compliant vs. swing: p=1.0

**Word overlap:**
- All pairwise: p=1.0 (fully null)

**Response length (standard):**
- All pairwise: p=1.0 (fully null)
