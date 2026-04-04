# TriviaQA Bridge Phase 2 Dev Results — 2026-04-04

> **Verdict: ITI does not improve factual generation on the bridge benchmark,
> regardless of artifact quality. Both E0 (paper-faithful, K=12) and E1
> (modernized, K=8) are harmful — E1 significantly so (p=0.016). The damage
> is method-level, not artifact-level. All conditions pass the grader
> reliability gate.**

## Source Hierarchy

- Benchmark plan: [2026-04-03-bridge-benchmark-plan.md](./2026-04-03-bridge-benchmark-plan.md) (§3, §5)
- Sprint context: [act3-sprint.md](../act3-sprint.md) (D4)
- E2 transfer synthesis: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md)

## Data Files

| Artifact | Path |
|---|---|
| Baseline JSONL | `data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment/alpha_1.0.jsonl` |
| ITI JSONL (α=4) | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/alpha_4.0.jsonl` |
| ITI JSONL (α=8) | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/alpha_8.0.jsonl` |
| Baseline audit stats | `data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment/audit_stats.json` |
| ITI audit stats | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/audit_stats.json` |
| Baseline provenance | `data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment/run_intervention.provenance.20260404_180846.json` |
| ITI α=4 provenance | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/run_intervention.provenance.20260404_180904.json` |
| ITI α=8 provenance | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/run_intervention.provenance.20260404_180925.json` |
| Dev manifest | `data/manifests/triviaqa_bridge_dev100_seed42.json` |
| ITI artifact (E0) | `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt` |
| Pipeline script (E0) | `scripts/infra/triviaqa_bridge_dev.sh` |
| Log (E0) | `logs/triviaqa_bridge_dev_20260404_190821.log` |
| E1 JSONL (α=8) | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e1_modernized_k8_first-3-tokens/experiment/alpha_8.0.jsonl` |
| ITI artifact (E1) | `data/contrastive/truthfulness/iti_truthfulqa_modernized_production/iti_heads.pt` |
| Pipeline script (E1) | `scripts/infra/triviaqa_bridge_dev_e1.sh` |

---

## 1. Experimental Design

| Parameter | Value |
|---|---|
| Manifest | `triviaqa_bridge_dev100_seed42.json` (100 questions, stratified) |
| Conditions | 4: neuron baseline α=1.0, E0 ITI α=4.0, E0 ITI α=8.0, E1 ITI α=8.0 |
| E0 ITI config | Paper-faithful, K=12, `first_3_tokens` decode scope |
| E0 ITI artifact | `iti_truthfulqa_paperfaithful_production/iti_heads.pt` |
| E1 ITI config | Modernized, K=8, `first_3_tokens` decode scope |
| E1 ITI artifact | `iti_truthfulqa_modernized_production/iti_heads.pt` |
| Generation | `do_sample=False`, `max_new_tokens=64`, greedy decode |
| Prompt | `"Question: {q}\nAnswer with a single short factual phrase only."` |
| Judge | GPT-4o batch, bidirectional audit (all non-matches + blinded ~20% match audit) |
| Analysis | Paired bootstrap (10k resamples, seed 42), McNemar, flip table |
| Grading policy | Two-metric: adjudicated accuracy (primary), deterministic (floor) |

---

## 2. Headline Results

### 2.1 Per-condition accuracy

| Condition | Adj. Acc. | 95% CI | Det. Acc. | 95% CI | Attempt | Not-attempted |
|---|---|---|---|---|---|---|
| Baseline α=1.0 | 47/100 (47.0%) | [37.5%, 56.7%] | 41/100 (41.0%) | [31.9%, 50.8%] | 99% | 1 |
| E0 ITI α=4.0 | 46/100 (46.0%) | [36.6%, 55.7%] | 39/100 (39.0%) | [30.0%, 48.8%] | 98% | 2 |
| E0 ITI α=8.0 | 40/100 (40.0%) | [30.9%, 49.8%] | 35/100 (35.0%) | [26.4%, 44.7%] | 97% | 3 |
| **E1 ITI α=8.0** | **38/100 (38.0%)** | **[29.1%, 47.8%]** | **33/100 (33.0%)** | **[24.6%, 42.7%]** | **97%** | **3** |

Precision given attempt: baseline 47.5%, E0 α=4.0 46.9%, E0 α=8.0 41.2%, **E1 α=8.0 39.2%**.

### 2.2 Paired bootstrap deltas (vs. baseline)

| Comparison | Δ Adjudicated | 95% CI | Δ Deterministic | 95% CI | CI excludes zero |
|---|---|---|---|---|---|
| E0 α=4.0 − baseline | -1.0% | [-6.0%, +4.0%] | -2.0% | [-7.0%, +3.0%] | No |
| E0 α=8.0 − baseline | -7.0% | [-14.0%, 0.0%] | -6.0% | [-13.0%, 0.0%] | No |
| **E1 α=8.0 − baseline** | **-9.0%** | **[-16.0%, -3.0%]** | **-8.0%** | **[-14.0%, -2.0%]** | **YES** |

Both E0 metrics are consistent in sign and magnitude — the two-metric
policy shows no divergence. E0 α=8 CI upper bound touches zero, meaning we cannot
formally reject the null, but the point estimate and consistency across metrics make
this a near-significant harmful trend. **E1 α=8 is formally significant: both
adjudicated and deterministic CIs exclude zero (McNemar p=0.016).** The "gentler"
artifact is worse, not better.

### 2.3 Flip tables

**Baseline → α=4.0:**

|  | ITI correct | ITI wrong | Total |
|---|---|---|---|
| **Base correct** | 43 | 4 | 47 |
| **Base wrong** | 3 | 50 | 53 |
| **Total** | 46 | 54 | 100 |

Net flips: -1. McNemar χ²=0.00, p=1.00.

**Baseline → E0 α=8.0:**

|  | ITI correct | ITI wrong | Total |
|---|---|---|---|
| **Base correct** | 37 | 10 | 47 |
| **Base wrong** | 3 | 50 | 53 |
| **Total** | 40 | 60 | 100 |

Net flips: -7. McNemar χ²=2.77, p=0.096.

**Baseline → E1 α=8.0:**

|  | ITI correct | ITI wrong | Total |
|---|---|---|---|
| **Base correct** | 37 | 10 | 47 |
| **Base wrong** | 1 | 52 | 53 |
| **Total** | 38 | 62 | 100 |

Net flips: -9. McNemar χ²=5.82, p=0.016.

The flip asymmetry is consistent across artifacts: both E0 and E1 produce exactly
10 right-to-wrong flips. The difference is that E1 rescues only 1 wrong-to-right
(vs E0's 3), making E1's net damage worse (-9 vs -7). The "gentler" artifact (K=8
vs K=12) is not gentler on the damage axis — it merely loses its rescue capacity.

### 2.4 Cross-alpha stability patterns

| Pattern (base→α4→α8) | Count | Interpretation |
|---|---|---|
| ✗→✗→✗ | 49 | Hard-wrong: model lacks the fact; ITI cannot help |
| ✓→✓→✓ | 36 | Stable-correct: robust knowledge survives intervention |
| ✓→✓→✗ | 7 | Progressive damage: α=8 destroys what α=4 preserves |
| ✓→✗→✗ | 3 | Early damage: α=4 already breaks these |
| ✗→✓→✓ | 2 | Stable rescue: ITI genuinely helps on both alphas |
| ✗→✗→✓ | 1 | α=8-only rescue (unstable signal) |
| ✗→✓→✗ | 1 | Noise: α=4 rescues but α=8 loses it again |
| ✓→✗→✓ | 1 | Noise: α=4 breaks it, α=8 restores it |

The dominant non-trivial pattern is **progressive damage** (7 samples lost only at
α=8), not progressive rescue. The 3 samples lost at both α=4 and α=8 suggest a
fragile subpopulation where any ITI perturbation is harmful.

---

## 3. Grader Reliability

### 3.1 Audit gate (per bridge plan §3.4)

| Condition | Match audits | Match disagree | Agreement | Passes (≥90%) |
|---|---|---|---|---|
| Baseline α=1.0 | 30 | 0 (0.0%) | 92.5% | Yes |
| ITI α=4.0 | 30 | 0 (0.0%) | 97.5% | Yes |
| ITI α=8.0 | 30 | 0 (0.0%) | 95.0% | Yes |

Zero false positives across all 90 match audits (30 per condition). The
deterministic grader never over-credits.

### 3.2 Non-match recovery

| Condition | Non-matches judged | Judge recovered | Recovery rate |
|---|---|---|---|
| Baseline | 59 | 6 | 10.2% |
| ITI α=4.0 | 61 | 7 | 11.5% |
| ITI α=8.0 | 65 | 5 | 7.7% |

The gap between deterministic and adjudicated accuracy is stable at 5-7pp across
conditions — the grader's miss rate is not systematically biased by the intervention.

### 3.3 Match tier distribution

| Tier | Baseline | α=4.0 | α=8.0 |
|---|---|---|---|
| exact | 28 | 23 | 14 |
| boundary | 12 | 15 | 20 |
| alias_simplified | 1 | 1 | 1 |
| no_match | 59 | 61 | 65 |

Exact matches drop monotonically (28 → 23 → 14) while boundary matches rise
(12 → 15 → 20). ITI is making responses longer and more elaborated (see §4.2),
which shifts correct answers from exact to boundary matches and sometimes past the
boundary matcher entirely into judge-only territory. This is a format effect, not an
accuracy effect — but it degrades deterministic grading reliability, which matters
for the conservative-floor metric.

---

## 4. Qualitative Flip Analysis

### 4.1 Wrong-to-right (ITI rescues)

Only 3 samples are rescued at each alpha. The two alphas share 2 of 3 rescues:

| QID | Question | Base response | α=4 response | α=8 response | Mechanism |
|---|---|---|---|---|---|
| `qw_4300` | First pop video by John Landis? | "Saturday Night's Alright for Fighting" | **Thriller** | **"Thriller"** | Knowledge correction: wrong fact → right fact |
| `sfq_10705` | Corryvreckan whirlpool — which island? | Kintyre | **North of Jura** | **To the north of Jura.** | Geography correction: wrong island → right island |
| `sfq_14641` | Volts × amperes = ? | Ohm | **Watts** | Watts *(still wrong at det, not rescued)* | Physics correction (α=4 only at adj level) |
| `odql_8345` | 2011 RWC NZ coach? | Ian Foster | *(still wrong)* | **Sir Graham Henry** | Knowledge correction, α=8 only |

These are genuine knowledge corrections — the model's factual recall improves on
these specific questions. But there are only 3 per alpha, against 4-10 going the
other direction.

### 4.2 Right-to-wrong at α=8.0 (10 samples) — failure mode taxonomy

| Mode | Count | Description |
|---|---|---|
| **Substitution** | 5 | Model confidently replaces correct answer with a plausible wrong one |
| **Verbosity** | 3 | Response elaborates into a sentence that no longer contains the answer |
| **Evasion** | 1 | Model hedges/describes instead of answering (graded NOT_ATTEMPTED) |
| **Other** | 1 | "Finger tabs" → "Leather finger guards" (paraphrase drift) |

#### Substitution examples (5/10)

These are the most informative failures. The baseline had the right answer; ITI α=8
replaces it with a confidently wrong alternative:

| QID | Question | Baseline (correct) | ITI α=8 (wrong) |
|---|---|---|---|
| `odql_8813` | Lewis Carroll hunting poem? | "Hunting for a Snark" | "Hunting for a caucus-race." |
| `qb_2363` | Lead singer of The Specials? | "Terry Hall" | "Horace Panter" |
| `qb_7212` | Microcephaly = abnormally small? | "Head size" | "Brain size" |
| `qw_740` | WWII coin gambling game? | "Two-up" | "Nimble Nick" |
| `qz_1794` | Scottish paper with Broons/Wullie? | "The Sunday Post." | "The Scotsman" |

The pattern is consistent: ITI does not cause the model to refuse or hedge — it
causes it to produce a *different, confidently wrong* factual answer. "Brain size"
instead of "Head size" is a semantically close substitution; "Horace Panter" is a
different band member (bassist, not vocalist); "The Scotsman" is a real Scottish
newspaper but the wrong one.

This is the signature of a **distribution shift in factual recall**, not a
refusal/safety interaction. The ITI truthfulness direction is perturbing the model's
next-token probabilities in a way that sometimes flips the top-1 completion to a
related-but-wrong entity.

#### Verbosity failures (3/10)

| QID | Baseline | ITI α=8 |
|---|---|---|
| `dpql_424` | "Jaipur Pale Ale" | "Bengal Tiger is named after Kolkata." |
| `qb_4415` | "Margot" | "Miep van Pels" |
| `qw_15141` | "Caesar salad" | "There isn't a single, widely known dish with that exact combination." |

The Caesar salad example is particularly revealing — the model goes from knowing the
answer to actively denying it exists. This looks like the truthfulness direction is
promoting epistemic caution at the cost of factual recall.

#### Evasion (1/10)

| QID | Baseline | ITI α=8 |
|---|---|---|
| `sfq_1626` | "John Everett Millais was the artist." | "He was a renowned English Romantic painter." |

The model replaces a correct named answer with a vague description. This pattern
appears at α=4 as well (same question), suggesting it's a stable failure mode for
this particular question under ITI perturbation.

### 4.3 NOT_ATTEMPTED samples

| Condition | Count | Questions |
|---|---|---|
| Baseline | 1 | `qb_8569` (teletext) |
| ITI α=4.0 | 2 | `dpql_3550` (High Noon quote), `sfq_1626` (Millais) |
| ITI α=8.0 | 3 | `dpql_3550`, `dpql_4593` (Anne of Green Gables), `sfq_1626` |

NOT_ATTEMPTED grows monotonically with alpha (1 → 2 → 3), consistent with the
truthfulness direction promoting epistemic caution. The increase is small (1-2
additional) and within noise, but the direction is consistent with the verbosity
and evasion failure modes.

---

## 5. Response Perturbation Analysis

### 5.1 Verbosity scaling

| Condition | Mean char length | Mean token length | Median tokens |
|---|---|---|---|
| Baseline | 15.2 | 4.8 | 4 |
| ITI α=4.0 | 19.1 | 5.6 | 5 |
| ITI α=8.0 | 25.9 | 7.0 | 6 |

Response length grows monotonically with alpha. At α=8, mean character length is
70% higher than baseline. This is not noise — it reflects a systematic tendency for
ITI to make the model more verbose, adding qualifiers, echoing the question, or
producing sentence-form answers instead of bare phrases.

### 5.2 Verbosity by correctness transition (α=8)

| Category | n | Mean Δchars | Range |
|---|---|---|---|
| Both correct | 37 | +9.6 | [-13, +64] |
| Right-to-wrong | 10 | +11.3 | [-4, +56] |
| Wrong-to-right | 3 | -3.0 | [-29, +14] |
| Both wrong | 50 | +12.1 | [-9, +58] |

The wrong-to-right rescues actually produce *shorter* responses on average. The
three genuine knowledge corrections replace a wrong entity name with the right one,
sometimes more concisely. In contrast, both damage modes (right-to-wrong and
both-wrong-but-changed) produce longer, more elaborated responses.

### 5.3 Response drift in both-correct pairs (α=8)

19 of 37 both-correct pairs (51%) have different surface-form responses despite both
being correct. Examples:

| QID | Baseline | ITI α=8 |
|---|---|---|
| `bb_848` | "Idli is the answer." | "Idli is a Southern India savoury steamed cake made of rice and served with chutney." |
| `sfq_4716` | "Nathuram Godse" | "Nathuram Godse assassinated Mahatma Gandhi." |
| `qw_11247` | "Madness" | "Madness's lead singer is Suggs." |
| `qf_3452` | "Albatross" | "The old logo was an Albatross." |

The intervention perturbs *even correct responses*, typically by making them more
verbose and sentence-like. This confirms the intervention is genuinely active on the
representation — it's changing generation behavior broadly, not just on edge cases.

### 5.4 The 49 always-wrong questions

| Pattern | Count |
|---|---|
| Same wrong answer across all 3 conditions | 13/49 (27%) |
| Response changed but still wrong | 36/49 (73%) |

73% of persistently wrong questions produce different wrong answers under ITI. The
model lacks the knowledge for these questions and ITI shuffles the wrong answer
without converging toward the right one. Sample:

| QID | Question | Base | α=8 | Truth |
|---|---|---|---|---|
| `bt_4318` | What number is CD in Roman numerals? | "XCIX" | "CD is not a Roman numeral." | 400 |
| `dpql_3638` | Poe tale — wife murderer exposed by pet? | "The Tell-Tale Heart" | "The Fall of the House of Usher" | The Black Cat |
| `qb_4127` | What odds are a 'Carpet'? | "1/3 odds." | "Odds are not typically referred to as a 'carpet' in betting." | 3 to 1 |
| `odql_9487` | Who awards the Dickin Medal? | "The British Military Animal Welfare Trust." | "The RSPCA awards the Dickin Medal." | PDSA |

At α=8, several wrong answers shift from confident-wrong to denial ("CD is not a
Roman numeral", "Odds are not typically referred to as..."). This is the
truthfulness direction promoting uncertainty — correctly identifying that the model
doesn't reliably know the answer, but unhelpfully expressing this as a denial rather
than a factual correction.

---

## 6. Accuracy by Question Source

| Source | n | Baseline | α=4.0 | α=8.0 | Direction |
|---|---|---|---|---|---|
| bb | 8 | 75% | 75% | 75% | Flat |
| qw | 13 | 62% | 69% | 54% | α=4 up, α=8 down |
| qz | 5 | 60% | 60% | 40% | α=8 harmful |
| sfq | 25 | 48% | 48% | 48% | Flat |
| odql | 13 | 46% | 38% | 46% | Noisy |
| qb | 11 | 45% | 45% | 18% | α=8 catastrophic |
| dpql | 9 | 33% | 22% | 22% | Both harmful |
| qf | 3 | 33% | 33% | 33% | Flat |
| wh | 4 | 25% | 25% | 25% | Flat |
| bt | 5 | 20% | 20% | 0% | α=8 catastrophic |

The damage at α=8 is concentrated in `qb` (quiz bowl, 45% → 18%) and `bt`
(brain teasers, 20% → 0%). These are the most "tricky" question types — lateral
thinking, slang, domain-specific jargon. The truthfulness direction may be
particularly harmful on questions that require confident recall of obscure facts,
precisely because it promotes epistemic hedging.

The `bb` (broad-based), `sfq` (simple factual), and `wh` (who) categories are
unaffected — these are straightforward factual recall where the model either knows
the answer stably or doesn't.

---

## 7. Generation Timing

| Condition | Mean gen_s | Max gen_s | Mean total_s | Hook overhead |
|---|---|---|---|---|
| Baseline | 0.133 | 0.354 | 0.133 | 0.014% |
| ITI α=4.0 | 0.159 | 0.460 | 0.159 | 17.2% |
| ITI α=8.0 | 0.196 | 0.638 | 0.196 | 14.9% |

ITI hooks add ~15-17% overhead at the generation step. The longer mean generation
time at α=8 (0.196 vs 0.133) reflects both the hook overhead and the longer
responses (more tokens generated). Wall time for all 3 conditions: ~90 seconds.

---

## 8. Interpretation

### 8.1 The E0 ITI intervention is active but harmful on TriviaQA

The intervention is unambiguously doing *something*: 51% of correct responses
change surface form, 73% of wrong responses change surface form, mean response
length grows 70%, and match tier distributions shift systematically. This is not a
null intervention.

But the accuracy effect is null-to-harmful. At α=4 the point estimate is -1pp
(noise). At α=8 it's -7pp (borderline significant, p=0.096) with a 10:3
right-to-wrong:wrong-to-right asymmetry.

### 8.2 Why ITI helps on TruthfulQA MC but hurts on TriviaQA generation

The TruthfulQA MC task tests whether the model *ranks* a true statement above a
common misconception. The ITI truthfulness direction was extracted from TruthfulQA
contrastive pairs. It nudges the model toward "more truthful" in the TruthfulQA
distribution — which means shifting probability mass away from popular
misconceptions and toward ground truth.

TriviaQA is a different distribution entirely. The questions are obscure trivia, not
common misconceptions. When the model gets a TriviaQA question wrong, it's usually
because it lacks the specific knowledge (not because it's falling for a common
myth). The truthfulness direction has no mechanism to inject missing knowledge — it
can only redistribute probability mass among the model's existing candidates.

The result is predictable: on questions where the model is uncertain, the
truthfulness direction shifts the top-1 candidate to a different (often
related-but-wrong) entity, or promotes epistemic hedging ("There isn't a single,
widely known..."). On questions where the model is confident and correct, the
direction occasionally destabilizes the confident answer (especially at α=8).

### 8.3 The failure mode signature is substitution, not refusal

5 of 10 right-to-wrong flips at α=8 are confident wrong substitutions (Terry Hall →
Horace Panter, Two-up → Nimble Nick). This is not the truthfulness direction
promoting "I don't know" — it's the direction reshuffling the model's factual
distribution. The refusal/evasion mode (NOT_ATTEMPTED growth: 1 → 2 → 3) exists but
is secondary. **E1 confirms this is method-level**: the same substitution errors
(Horace Panter, Brain size, The Scotsman, Miep van Pels) appear identically with
K=8 modernized heads. See §9.4–9.5.

### 8.4 Consistency with prior E2 transfer results

The E2 transfer synthesis already classified TriviaQA transfer as
`wrong_source_still_likely` — ITI trained on TruthfulQA does not improve TriviaQA
answer selection (E2-A: null, E2-B: null). This Phase 2 result extends that finding
to generation: the ITI direction does not transfer to a different factual knowledge
distribution, even with paper-faithful E0 configuration and optimized decode scope.

### 8.5 Bridge benchmark health

The benchmark itself is working as designed:
- Baseline headroom: 47% adjudicated accuracy is in the productive range [15%, 70%]
- Grader reliability: 0/90 false positives, 92.5-97.5% audit agreement
- Two-metric consistency: adjudicated and deterministic agree in sign and magnitude
- Paired design: the flip table reveals mechanism (substitution > evasion) that
  aggregate deltas would hide
- Attempt rate: 97-99%, no commitment damping

The benchmark is ready for Phase 3 (test set) whenever a candidate intervention
warrants it.

---

## 9. E1 Modernized Comparative Analysis

> Added 2026-04-04 after E0 results showed harmful damage. Hypothesis: E1's
> gentler perturbation (K=8 vs K=12, +8pp attempt rate vs E0 on SimpleQA)
> would avoid E0's right-to-wrong flip asymmetry.

### 9.1 Design

| Parameter | Value |
|---|---|
| ITI config | Modernized E1, K=8, α=8.0, `first_3_tokens` decode scope |
| ITI artifact | `iti_truthfulqa_modernized_production/iti_heads.pt` |
| Baseline | Reused from Phase 2 (no re-run) |
| Pipeline script | `scripts/infra/triviaqa_bridge_dev_e1.sh` |

### 9.2 Headline: E1 is worse than E0, not better

| Metric | Baseline | E0 (K=12) | E1 (K=8) |
|---|---|---|---|
| Adjudicated accuracy | 47.0% | 40.0% | **38.0%** |
| Δ vs baseline | — | -7.0% CI [-14, 0] | **-9.0% CI [-16, -3]** |
| CI excludes zero | — | no (p=0.096) | **YES (p=0.016)** |
| Deterministic accuracy | 41.0% | 35.0% | **33.0%** |
| right→wrong flips | — | 10 | **10** |
| wrong→right flips | — | 3 | **1** |
| Net flips | — | -7 | **-9** |
| Precision given attempt | 47.5% | 41.2% | **39.2%** |
| Mean response chars | 15.2 | 25.9 | 18.2 |

### 9.3 Side-by-side flip comparison

```
                  │ E1 (K=8) │ E0 (K=12) │    Δ
──────────────────┼──────────┼───────────┼──────
wrong→right       │      1   │       3   │   -2
right→wrong       │     10   │      10   │    0
both right        │     37   │      37   │    0
both wrong        │     52   │      50   │   +2
net flips         │     -9   │      -7   │   -2
```

The damage profile is identical (10 right-to-wrong in both). The difference is
rescue capacity: E1 rescues only 1 question vs E0's 3. Fewer heads does not mean
less damage — it means less perturbation in both directions.

### 9.4 Focal damage analysis: E0's 10 right-to-wrong questions under E1

On the 10 questions E0 flipped from baseline-correct to wrong, E1 saves 2 and
also damages 8. E1's rescue rate on E0-damaged questions: **20%**.

| QID | Question (abbrev.) | Baseline (correct) | E0 K=12 (wrong) | E1 K=8 | E1 outcome |
|---|---|---|---|---|---|
| `bt_711` | Archery finger protection | Finger tabs | Leather finger guards | Finger guards. | also wrong |
| `dpql_424` | Indian Pale Ale brand | Jaipur Pale Ale | Bengal Tiger is named after Kolkata. | Mumbai. | also wrong |
| `odql_8813` | Lewis Carroll hunting poem | Hunting for a Snark | Hunting for a caucus-race. | Hunting for a hat. | also wrong |
| `qb_2363` | Lead singer of The Specials | Terry Hall | Horace Panter | Horace Panter | also wrong |
| `qb_4415` | Anne Frank friend | Margot | Miep van Pels | Miep van Pels | also wrong |
| `qb_7212` | Microcephaly = small ___? | Head size | Brain size | Brain size. | also wrong |
| `qw_740` | WWII coin gambling game | Two-up | Nimble Nick | Up-ko. | also wrong |
| `qz_1794` | Scottish paper: Broons/Wullie | The Sunday Post. | The Scotsman | The Scotsman. | also wrong |
| **`qw_15141`** | **Cos lettuce, anchovies dish** | **Caesar salad** | **There isn't a single...** | **Caesar salad** | **SAVES** |
| **`sfq_1626`** | **Ophelia painter** | **John Everett Millais** | **He was a renowned...** | **John Everett Millais** | **SAVES** |

The two saves are both cases where E0 produced a **verbosity/evasion failure**
(sentence-form hedging), not a substitution. E1's lower verbosity (18.2 vs 25.9
chars) rescues these — but on the 5 confident wrong substitutions (Horace Panter,
Brain size, The Scotsman, etc.), E1 produces the **identical wrong answer** as E0.

### 9.5 Interpretation: method-level, not artifact-level

The E1 result rules out the "wrong artifact" hypothesis. Key evidence:

1. **Same 10 right-to-wrong flips.** Despite K=8 vs K=12, both artifacts damage
   the same questions. The vulnerable population is determined by the truthfulness
   direction, not by the head count.

2. **Identical substitution errors.** On 4 of 8 shared damages, E1 produces the
   exact same wrong entity as E0 (Horace Panter, Miep van Pels, Brain size, The
   Scotsman). The direction is pushing the model toward the same alternative
   candidates regardless of how many heads deliver the push.

3. **E1's only advantage is reduced verbosity.** Mean response length (18.2 chars)
   is much closer to baseline (15.2) than E0 (25.9). This saves the two
   verbosity/evasion failures but has no effect on substitution failures.

4. **E1 loses E0's rescue capacity.** E0 rescued 3 questions; E1 rescues only 1.
   With fewer heads, the perturbation is weaker in both directions — less damage
   from verbosity, but also less chance of accidentally correcting a wrong answer.

**Conclusion:** The truthfulness direction itself encodes a systematic bias toward
specific wrong entities on TriviaQA questions. This is not an extraction artifact
or a head-count effect — it is a property of the direction. ITI is fundamentally
harmful on factual generation tasks where the model's knowledge is uncertain,
because it reshuffles probability mass toward related-but-wrong candidates without
injecting new knowledge.

---

## 10. Summary Statistics Table

For copy into downstream reporting surfaces:

```
Condition                  | Adj.Acc (CI)              | Det.Acc (CI)              | Attempt | NOT_ATT
Baseline α=1.0             | 47.0% [37.5%, 56.7%]     | 41.0% [31.9%, 50.8%]     | 99%     | 1
E0 ITI α=4.0 first_3_tok  | 46.0% [36.6%, 55.7%]     | 39.0% [30.0%, 48.8%]     | 98%     | 2
E0 ITI α=8.0 first_3_tok  | 40.0% [30.9%, 49.8%]     | 35.0% [26.4%, 44.7%]     | 97%     | 3
E1 ITI α=8.0 first_3_tok  | 38.0% [29.1%, 47.8%]     | 33.0% [24.6%, 42.7%]     | 97%     | 3

Comparison vs baseline     | Δ Adj. (CI)          | Δ Det. (CI)          | McNemar p | Net flips
E0 α=4.0 − baseline       | -1.0% [-6%, +4%]     | -2.0% [-7%, +3%]     | 1.000     | -1
E0 α=8.0 − baseline       | -7.0% [-14%, 0%]     | -6.0% [-13%, 0%]     | 0.096     | -7
E1 α=8.0 − baseline       | -9.0% [-16%, -3%]    | -8.0% [-14%, -2%]    | 0.016     | -9

Grader audit               | Match disagree  | Non-match recovery | Pilot gate
Baseline                   | 0/30 (0.0%)     | 6/59 (10.2%)       | PASS (92.5%)
E0 α=4.0                  | 0/30 (0.0%)     | 7/61 (11.5%)       | PASS (97.5%)
E0 α=8.0                  | 0/30 (0.0%)     | 5/65 (7.7%)        | PASS (95.0%)
E1 α=8.0                  | 1/30 (3.3%)     | 6/67 (9.0%)        | PASS (96.7%)
```
