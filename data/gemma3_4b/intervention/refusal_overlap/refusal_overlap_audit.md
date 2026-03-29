# D3.5 Refusal-Overlap Audit: Gemma-3-4B H-Neuron Scaling

**Date:** 2026-03-29  
**Model:** `google/gemma-3-4b-it`  
**Question:** Does Baseline A act through refusal geometry strongly enough to change D4, or is the overlap too fragile to steer the next branch of the sprint?  
**Primary artifacts:** [analysis/summary.json](analysis/summary.json), [analysis/prompt_scores.csv](analysis/prompt_scores.csv), [analysis/layer_scores.csv](analysis/layer_scores.csv), [analysis/null_distribution.json](analysis/null_distribution.json), [analysis/closeout_note.md](analysis/closeout_note.md)

## Bottom Line

The D3.5 audit finds a real non-random overlap between the projected 38-neuron intervention and the D2 refusal geometry, but the mediation story does **not** survive a basic robustness check. The apparent prompt-level FaithEval and jailbreak correlations are dominated by **layer 33**. When that single dominant layer is removed, the correlations collapse or flip sign. The corrected D4 gate is therefore:

- `proceed_with_d4_unchanged`

This is the difference between “the whole mechanism leans on refusal geometry” and “one loud instrument can make the song sound refusal-like.” The current evidence supports the second reading, not the first.

## Data

### 1. Headline geometry against matched null

D2 stores the canonical direction as `mean(harmful) - mean(harmless)`. Under that convention:

- Negative signed cosine means **anti-refusal / harmless-ward** alignment.
- Positive signed cosine means **harmful/refusal-direction** alignment.

Headline geometry for the actual H-neuron residual update:

| Metric | Estimate | 95% CI | Null reference | Readout |
|---|---:|---:|---:|---|
| Canonical signed cosine mean | -0.0173 | [-0.0177, -0.0169] | — | Small but stable anti-refusal alignment |
| Canonical gap vs matched null | -0.0183 | [-0.0310, -0.0126] | null mean = +0.0010 | Real departure from random neuron sets |
| Refusal-subspace fraction mean | 0.0388 | [0.0373, 0.0404] | — | About 3.9% of update energy lies in top-3 refusal subspaces |
| Refusal-subspace gap vs matched null | +0.0361 | [+0.0251, +0.0390] | null mean = 0.0026 | Real subspace enrichment over random sets |

So the overlap is not imaginary. The actual intervention touches refusal-related residual geometry more than a matched random-neuron null does.

### 2. Prompt-level association with Baseline A outcomes

Primary prompt-level mediation results:

| Benchmark | Metric | Spearman ρ | 95% CI | Direction under D2 sign convention |
|---|---|---:|---:|---|
| FaithEval | canonical overlap vs compliance slope | -0.0869 | [-0.1604, -0.0124] | more anti-refusal alignment predicts larger compliance lift |
| FaithEval | subspace overlap vs compliance slope | +0.0857 | [+0.0110, +0.1586] | more refusal-subspace energy predicts larger compliance lift |
| Jailbreak | canonical overlap vs `csv2_yes` slope | -0.1164 | [-0.2060, -0.0238] | more anti-refusal alignment predicts larger harmfulness drift |
| Jailbreak | subspace overlap vs `csv2_yes` slope | +0.1120 | [+0.0219, +0.2021] | more refusal-subspace energy predicts larger harmfulness drift |

These are statistically non-zero but small effects. The useful reading is “suggestive correlation,” not “mechanism solved.”

Additional jailbreak diagnostics before the fragility check:

- Canonical overlap vs harmful payload share slope: `ρ = -0.1661`, 95% CI `[-0.2506, -0.0771]`
- Canonical overlap vs `csv2_yes@α=3` conditioned on `α=0`: `ρ = -0.2356`, 95% CI `[-0.3233, -0.1454]`

That pattern argues the signal is not only “hard prompts are bad at every alpha.” It also tracks amplification sensitivity. But that inference only matters if the geometry signal is robust, which it currently is not.

### 3. Dominant-layer fragility check

The cheap falsification test was: remove the most dominant overlap layer and recompute the prompt-level mediation summaries without recollecting activations.

Dominant layer under the subspace-gap criterion:

| Layer | H-neurons in layer | Signed gap vs null | Subspace gap vs null |
|---|---:|---:|---:|
| 33 | 1 | -0.5187 | +0.6647 |

Next-largest subspace gap:

| Layer | Subspace gap vs null |
|---|---:|
| 7 | +0.0155 |

Layer 33 is not just the largest contributor. It is larger than the next subspace-gap layer by roughly **43×**.

After excluding layer 33:

| Benchmark | Metric | Spearman ρ after exclusion | 95% CI |
|---|---|---:|---:|
| FaithEval | canonical overlap vs compliance slope | -0.0048 | [-0.0675, +0.0597] |
| FaithEval | subspace overlap vs compliance slope | -0.0218 | [-0.0859, +0.0437] |
| Jailbreak | canonical overlap vs `csv2_yes` slope | +0.0303 | [-0.0493, +0.1099] |
| Jailbreak | subspace overlap vs `csv2_yes` slope | -0.1587 | [-0.2373, -0.0780] |

This is the decision-relevant result. The full-model signal does not merely weaken; parts of it reverse.

## Interpretation

### What withstands scrutiny

- The H-neuron intervention is **not geometrically random** with respect to the refusal basis. Both canonical and subspace overlap exceed the layer-matched null in the expected directions.
- The full-model prompt-level correlations point in the direction you would expect if refusal-related geometry partly explained both FaithEval target gain and jailbreak safety cost.

### What does not withstand scrutiny

- The stronger claim, “Baseline A is refusal-mediated,” does **not** survive a dominant-layer exclusion test.
- The stronger gate, “orthogonalize D4 immediately,” would be premature. Right now it would amount to redesigning D4 around a signal that may mostly be a single-layer or single-neuron bridge.

### Best current reading

Baseline A likely **touches** refusal-related geometry, but the evidence is too concentrated and too fragile to say refusal geometry is the main explanatory channel. In practical terms: there is smoke, but it is coming from one vent.

## Decision

### D4 gate

- `proceed_with_d4_unchanged`

### Why this way, not the stronger alternative

Orthogonalizing D4 now would treat the D3.5 result like a distributed, stable mechanism. The leave-one-layer-out check shows it is not yet that. Good safety research should not let a single brittle layer reroute the whole experimental program.

## Remaining uncertainties

- **Layer 33 may be real or artifactual.** A strong single-layer bridge is possible. So is a detector-selection artifact or a model-specific quirk. D3.5 does not distinguish those yet.
- **The signed-cosine story depends on direction orientation.** Because D2 uses `harmful - harmless`, negative canonical overlap is the “more anti-refusal” direction. If a reader misses that sign convention, they will read the result backwards.
- **Jailbreak still lacks a benchmark-specific random-neuron negative control.** D3.5 helps explain geometry, not specificity of the harmfulness drift.
- **The prompt-level effect sizes are small.** Statistical significance here comes from `n=1000` and `n=500`, not from large correlations.

## Most valuable next steps

1. Run a **targeted layer-33 / top-neuron robustness pass**. The cheapest decisive test is to repeat D3.5 while excluding layer 33, then excluding its top neuron only, and compare how much signal remains. This tells us whether the fragility is a one-neuron story or a whole-layer story.
2. Finish the **jailbreak negative control** for Baseline A. The best geometric story in the world does not replace a specificity control.
3. Proceed to **D4 as currently planned**, not orthogonalized. Treat refusal-orthogonalization as a follow-up branch only if the overlap result survives the robustness passes above.

## Reporting hygiene

This file is the narrative source of truth for D3.5. Other documents should link here rather than restating the detailed numbers unless they are updating a higher-level decision.
