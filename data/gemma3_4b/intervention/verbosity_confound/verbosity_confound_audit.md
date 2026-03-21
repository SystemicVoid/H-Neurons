# Verbosity Confound Audit: H-Neuron CETT Activations

**Date:** 2026-03-21
**Model:** `google/gemma-3-4b-it`
**Script:** `scripts/run_verbosity_confound.py`
**Related reports:** [intervention_findings.md](../../intervention_findings.md)

---

## Bottom Line

- **Under full-response aggregation, H-neuron CETT activations are dominated by response length rather than truthfulness.** Under both mean and max aggregation, the length effect dominates the truth effect by 3.7:1 (mean agg, Cohen's d) and 16:1 (max agg). Per-neuron analysis confirms this: 36 of 38 neurons are length-dominant under mean aggregation.
- **A real but smaller truth signal exists.** 17/38 neurons show significant truth effects (p<0.05; 13 survive Bonferroni correction), but the direction splits evenly (18 positive, 19 negative), so the population-level truth signal is incoherent. Two individual neurons (L14:N8547, L10:N2536) show roughly balanced truth and length effects.
- **The truth signal concentrates in short responses.** A significant interaction (p=0.0002) shows the truth effect is present at short response lengths (d≈-0.23, ~10 tokens) but vanishes in long responses (d≈-0.001, ~112 tokens). This dilution effect means the classifier, trained on answer-token spans (short), may have validly captured truth information that this full-response readout dilutes.
- **Two pipeline fixes were applied before this run.** (P1) `get_response_end` now computes response span without trailer tokens. (P2) OOD classifier scoring was removed — the classifier was trained on answer-token activations, so scoring full-response means would be domain-shifted and misleading.
- **The intervention causal claims are unaffected.** This test measures activation readout, not causal intervention. The intervention results are robust against verbosity-mediation for three reasons: (1) FaithEval anti-compliance uses letter extraction, making it immune to response-length confounds; (2) negative controls show random neurons (which also encode length) produce zero compliance shift; (3) the original paper replicates across 6 models × 4 compliance dimensions. The confound finding is scoped to the detection side only.

---

## 1. Experimental Design

**Question:** Under full-response aggregation, do the 38 positive-weight H-neurons show stronger sensitivity to truth status or response verbosity?

**Design:** 2×2 factorial (truth × length), within-subject (paired across 100 items). Each item has a factual question and 4 pre-written responses:
- **short_true:** Correct answer in ~1 sentence (~10.7 tokens mean)
- **long_true:** Correct answer with elaboration (~112.3 tokens mean)
- **short_false:** Incorrect answer in ~1 sentence (~10.6 tokens mean)
- **long_false:** Incorrect answer with elaboration (~110.7 tokens mean)

400 total forward passes (100 items × 4 conditions). No generation — responses are pre-written and fed as assistant content. CETT activations extracted from the response token span only (excluding user prompt and trailer tokens).

**Key design properties:**
- Length ratio: 10.5× (short ~10.6 vs long ~111.5 tokens). This is extreme by design — maximises power to detect length effects.
- Token count balance across truth axis: short_true and short_false are near-identical (diff=0.14 tokens), long_true and long_false differ by 1.6 tokens (1.4%). This asymmetry is negligible for interpretation.
- Responses cover 20 domains (geography, biology, astronomy, history, etc.) with 5 items each.

---

## 2. Pipeline Fixes Reviewed

### P1: Response end boundary (resp_end tokenization)

**Problem:** The prior version computed `resp_end` from the full chat-template token count, which included trailer tokens (e.g. `<end_of_turn>\n`). Trailer tokens are not part of the model's response content and could carry different activation patterns.

**Fix:** `get_response_end` now (a) tokenizes user+generation_prompt to find `resp_start`, then (b) encodes only the response text with `tokenizer.encode(response_text, add_special_tokens=False)` to compute exact response length. `resp_end = resp_start + len(resp_ids)`.

**Assessment:** Correct. Minor caveat: BPE tokenization is context-dependent at token boundaries — the first token of the response may tokenise differently in isolation versus after the generation prompt header. Expected magnitude: 0–2 tokens. For long responses (~112 tokens) this is <2% error. For short responses (~10 tokens) this is up to ~20% of the span, but since the mean aggregation result is dominated by length effects anyway, this does not change the verdict.

### P2: OOD classifier scoring removal

**Problem:** The verbosity confound script originally ran `classifier.predict_proba` on full-response mean activations. The classifier was trained on answer-token CETT activations (short spans, typically 1–5 tokens). Feeding it 100-token response means is out-of-distribution and would systematically overstate length-correlated effects in the classifier output.

**Fix:** All classifier scoring was removed from the script. The analysis now operates directly on raw CETT activation values, which is the appropriate level for a confound test.

**Assessment:** Correct. The classifier scoring was not serving the experiment's purpose (testing what the neurons encode) and was actively misleading.

---

## 3. Results

### 3.1 Aggregate Effects

| Aggregation | Truth effect d | Truth p | Length effect d | Length p | |d_length|/|d_truth| | Verdict |
|-------------|---------------|---------|-----------------|---------|--------------------:|---------|
| Mean | -0.502 | 1.4e-05 | -1.864 | <1e-16 | 3.71 | A_verbosity |
| Max | -0.265 | 0.018 | +4.277 | <1e-16 | 16.14 | A_verbosity |

**Direction of effects (mean aggregation):**
- Truth: d=-0.50 means false conditions have *lower* mean CETT than true. This is the opposite of what a hallucination detector might naively predict — the H-neurons activate slightly more on true responses.
- Length: d=-1.86 means long responses have *lower* mean CETT than short. This is mechanical: averaging over more tokens dilutes peak activations.

**Direction of effects (max aggregation):**
- Truth: d=-0.27 (same direction as mean, attenuated).
- Length: d=+4.28 means long responses have *higher* max CETT than short. This is also mechanical: more tokens = more chances for a high peak.

The mean and max aggregations show **opposite signs** for the length effect (-1.86 vs +4.28). Any downstream analysis using CETT activations must specify its aggregation method, because the length confound operates in opposite directions.

### 3.2 Interaction: Truth × Length

| Length class | Truth effect (false−true) | Interpretation |
|-------------|--------------------------|----------------|
| Short (~10 tokens) | -0.002342 | Small but detectable |
| Long (~112 tokens) | -0.000100 | Effectively zero |
| Interaction p-value | 0.0002 | Significant |

The truth signal concentrates in short responses and is diluted away in long ones. This is consistent with a small number of truth-sensitive token positions whose signal gets averaged out over a 100+ token window.

**Implication for the classifier:** The classifier was trained on answer-token CETT activations (typically 1–5 tokens — comparable to the short condition). In that regime, the truth signal is detectable (d≈0.23 at the neuron-population level). The confound test's full-response readout dilutes this signal, so the A_verbosity verdict applies to full-response readout, not necessarily to the classifier's training domain.

### 3.3 Per-Neuron Breakdown

**Dominance (mean agg):** 36/38 neurons length-dominant, 2/38 truth-dominant.
**Dominance (max agg):** 34/38 neurons length-dominant, 4/38 truth-dominant.

**Significant truth effects (mean agg, p<0.05):** 17/38 neurons. After Bonferroni correction (p<0.00132): 13/38.

**Truth effect direction (mean agg):** 18 neurons show false > true, 19 show true > false. No consistent directional signal across the population.

**Notable individual neurons:**

| Neuron | Truth d | Length d | |Truth|/|Length| | Character |
|--------|---------|---------|----------------|-----------|
| L14:N8547 | +1.098 | -1.314 | 0.84 | Near-balanced; strongest truth signal |
| L10:N2536 | +0.619 | -0.662 | 0.94 | Near-balanced |
| L16:N4146 | +0.410 | -0.245 | 1.67 | Truth-dominant (one of only 2) |
| L0:N3088 | -0.185 | -0.047 | 3.97 | Truth-dominant (small effects) |
| L12:N5105 | +0.325 | +3.603 | 0.09 | Strongly length-dominated |
| L13:N9099 | -0.556 | +3.646 | 0.15 | Strongly length-dominated |
| L25:N3877 | -0.634 | -3.198 | 0.20 | Strongly length-dominated |

L14:N8547 is the most informative neuron: it has the strongest truth effect in the population (d=1.10) and roughly equal truth and length sensitivity. In this full-response readout, L14:N8547 shows relatively balanced truth and length sensitivity, making it less length-dominated than most neurons in the set. Whether it genuinely encodes truth (vs answer-form characteristics correlated with truth) requires further testing.

### 3.4 Condition Means and Token Counts

| Condition | Mean CETT (mean agg) | 95% CI | Mean tokens |
|-----------|---------------------|--------|-------------|
| short_true | 0.10328 | [0.10245, 0.10407] | 10.7 ± 2.4 |
| long_true | 0.09516 | [0.09460, 0.09573] | 112.3 ± 16.0 |
| short_false | 0.10093 | [0.09989, 0.10194] | 10.6 ± 2.4 |
| long_false | 0.09506 | [0.09450, 0.09559] | 110.7 ± 14.5 |

---

## 4. Implications

### 4.1 Evidence-pyramid interpretation

The intervention experiments (FaithEval, FalseQA, Jailbreak) operate causally: they multiply neuron activations by α during generation and measure downstream behavioral changes. The confound test is a correlational readout test. A neuron that encodes length in its activations may still causally affect compliance when amplified — these are different questions.

The evidence for H-neuron claims falls into a natural hierarchy:

- **Top tier (robust) — Causal intervention findings.** FaithEval letter-extraction scoring is immune to raw length confounding; negative controls rule out indirect response-style channels.
  1. **FaithEval anti-compliance uses letter extraction (A/B/C/D).** The metric only cares which letter was chosen, so response length is invisible to the scorer. However, verbosity/style could still mediate the causal path indirectly — if scaling H-neurons changes hesitation, hedging, or decisiveness, that could in turn change the chosen letter. The negative controls (point 2) are what rule out this indirect channel.
  2. **Negative controls.** Random neurons also encode verbosity (most neurons do), but produce zero compliance shift (slope 0.02 pp/α, interval [-0.106, 0.164]). If the effect were mediated through any response-characteristic channel (including verbosity-driven decisiveness changes), random-neuron scaling should show some compliance change.
- **Middle tier (robust) — Cross-model replication.** Gao et al.'s 6-model × 4-task replication remains intact. Verbosity mediation would require a chain of coincidences across all of these (Mistral-7B, Mistral-Small-24B, Gemma-3-4B, Gemma-3-27B, Llama-3.1-8B, Llama-3.3-70B).
- **Bottom tier (partially confounded) — Local classifier/detection claims.** The classifier's AUROC of 0.843 should now be interpreted as partly entangled with response-form/length correlations. The interaction result (truth signal present at short spans, absent at long spans) provides partial mitigation in the classifier's training domain, but the entanglement cannot be fully ruled out without an answer-token confound test.
- **Foundation tier (length-dominated in this test) — Full-response readout.** This audit's finding: under full-response aggregation, H-neuron CETT activations are dominated by length. This is the weakest evidential layer — it measures correlational readout, not causal influence.

The FalseQA response shortening (-9%, §1.7 of intervention_findings.md) is a *consequence* of the compliance change (confident false-premise acceptance = less hedging), not evidence of a verbosity-mediated cause.

### 4.2 Detection-side implications

1. **Classifier interpretation.** The classifier's AUROC of 0.843 on held-out answer-token classification should not be presented as pure hallucination-detection performance without noting the verbosity confound. The interaction result (truth signal present at short spans, absent at long spans) provides partial mitigation: in the classifier's training domain (answer-token spans), truth information is available.

2. **Aggregation sensitivity.** Any future analysis using CETT activations must specify and justify its aggregation method. Mean and max aggregations show opposite-sign length effects, so conclusions about "what H-neurons encode" depend critically on how activations are summarised.

---

## 5. Methodological Notes

- **Statistical tests:** Wilcoxon signed-rank (nonparametric, paired). Appropriate for non-normal CETT distributions.
- **Effect size:** Cohen's d for paired samples. The 2:1 dominance threshold for the verdict is a reasonable heuristic.
- **Bootstrap CIs:** 10,000 resamples, seed=42. Standard and adequate.
- **Multiple testing:** Per-neuron p-values are reported both raw and Bonferroni-corrected. 13/38 truth effects survive Bonferroni; 29/38 length effects survive.
- **Sample size:** n=100 items provides >80% power for d≥0.3 (paired Wilcoxon), adequate for detecting the effects observed.

---

## 6. Open Questions

1. **Answer-token confound test.** Running this same 2×2 factorial but extracting activations only from answer tokens (matching the classifier's training domain) would directly test whether the classifier's detection is truth- or length-driven in its native regime. This is the highest-value follow-up for understanding the detection mechanism.
2. **Individual neuron steering.** L14:N8547 and L10:N2536 show balanced truth/length effects. Do they drive a disproportionate share of the intervention's compliance effect?
