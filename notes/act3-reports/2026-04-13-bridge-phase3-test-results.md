# TriviaQA Bridge Phase 3 Test Results — 2026-04-13

> **Verdict: The externality break replicates on the held-out test set and is now
> statistically significant. E0 ITI (paper-faithful, K=12, α=8.0) reduces
> adjudicated accuracy by −5.8 pp [−8.8, −3.0] (CI excludes zero, McNemar
> p=0.0002). The dominant failure mode is wrong-entity substitution
> (70% of right-to-wrong flips), not refusal. Dev-to-test replication is
> tight (Δ adj −7.0pp dev → −5.8pp test, within dev CI). This upgrades the
> externality-break claim from directional-but-underpowered (L5) to
> publication-grade evidence.**

## Source Hierarchy

- Phase 2 dev results: [2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md)
- Benchmark plan: [2026-04-03-bridge-benchmark-plan.md](./2026-04-03-bridge-benchmark-plan.md)
- E2 transfer synthesis: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md)
- Sprint context: [act3-sprint.md](../act3-sprint.md) (D4, D5)
- Strategic assessment: [2026-04-11-strategic-assessment.md](../2026-04-11-strategic-assessment.md) (§4 Anchor 2)
- Paper Section 5: `notes/paper/draft/section_5_case_study_II.md`

## Data Files

| Artifact | Path |
|---|---|
| Baseline JSONL | `data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl` |
| ITI JSONL (α=8.0) | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/alpha_8.0.jsonl` |
| Baseline results | `data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json` |
| ITI results | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/results.json` |
| Baseline audit stats | `data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/audit_stats.json` |
| ITI audit stats | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/audit_stats.json` |
| Baseline provenance | `data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/run_intervention.provenance.20260413_093651.json` |
| ITI provenance | `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/run_intervention.provenance.20260413_093805.json` |
| Test manifest | `data/manifests/triviaqa_bridge_test500_seed42.json` (500 IDs) |
| ITI artifact (E0) | `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt` |
| Pipeline script | `scripts/infra/triviaqa_bridge_test.sh` |
| Log | `logs/triviaqa_bridge_test_20260413_103648.log` |

---

## 1. Experimental Design

| Parameter | Value | Locked from |
|---|---|---|
| Manifest | `triviaqa_bridge_test500_seed42.json` (500 questions, held-out) | Phase 1 split |
| Conditions | 2: neuron baseline α=1.0, E0 ITI α=8.0 | Phase 2 dev |
| ITI config | Paper-faithful, K=12, `first_3_tokens` decode scope | Phase 2 dev |
| ITI artifact | `iti_truthfulqa_paperfaithful_production/iti_heads.pt` | E0 production |
| Generation | `do_sample=False`, `max_new_tokens=64`, greedy decode | Phase 1 plan |
| Prompt | `"Question: {q}\nAnswer with a single short factual phrase only."` | Phase 1 plan |
| Judge | GPT-4o batch, bidirectional audit (all non-matches + ~20% match audit) | Phase 1 plan |
| Analysis | Paired bootstrap (10k resamples, seed 42), McNemar, flip table | Phase 2 dev |
| Grading policy | Two-metric: adjudicated accuracy (primary), deterministic (floor) | Phase 1 plan |
| Dev/test overlap | 0 IDs (verified — clean held-out) | Phase 1 split |

**One-shot protocol:** This test set was run exactly once. No parameters were tuned on test data. The script includes a FATAL guard that prevents re-running if both conditions are complete.

---

## 2. Headline Results

### 2.1 Per-condition accuracy

| Condition | Adj. Acc. | 95% CI | Det. Acc. | 95% CI | Attempt | Not-attempted |
|---|---|---|---|---|---|---|
| Baseline α=1.0 | 225/500 (45.0%) | [40.7%, 49.4%] | 199/500 (39.8%) | [35.6%, 44.2%] | 99.6% (498) | 2 |
| E0 ITI α=8.0 | 196/500 (39.2%) | [35.0%, 43.5%] | 176/500 (35.2%) | [31.1%, 39.5%] | 98.4% (492) | 8 |

Precision given attempt: baseline 45.2% (225/498), ITI 39.8% (196/492), **Δ = −5.3pp**.

### 2.2 Paired bootstrap deltas (vs. baseline)

| Metric | Δ (ITI − baseline) | 95% CI | CI excludes zero |
|---|---|---|---|
| Adjudicated accuracy | −5.8% | [−8.8%, −3.0%] | **YES** |
| Deterministic accuracy | −4.6% | [−7.6%, −1.6%] | **YES** |
| Attempt rate | −1.2% | [−2.4%, −0.2%] | **YES** |
| Precision given attempt | −5.3% | (see §2.1) | See note |

Both primary and floor metrics exclude zero, with entirely negative CIs. The two-metric policy shows no divergence — the signal is consistent across measurement approaches.

**McNemar test on adjudicated correctness:** χ² = 13.75, p = 0.0002. This is well below any conventional significance threshold (p < 0.001).

### 2.3 Flip table

|  | ITI correct | ITI wrong | Total |
|---|---|---|---|
| **Base correct** | 182 | 43 | 225 |
| **Base wrong** | 14 | 261 | 275 |
| **Total** | 196 | 304 | 500 |

- Net flips: **−29** (14 wrong→right, 43 right→wrong)
- Flip ratio: 3.1:1 (right→wrong : wrong→right)
- 81.0% of baseline-correct items remain correct under ITI (182/225)
- Only 5.1% of baseline-wrong items are rescued by ITI (14/275)

---

## 3. Dev-to-Test Replication

### 3.1 Point estimate comparison

| Metric | Dev (n=100) | Test (n=500) | Δ test−dev | Within dev CI? |
|---|---|---|---|---|
| Baseline adj acc | 47.0% | 45.0% | −2.0pp | Yes |
| ITI adj acc | 40.0% | 39.2% | −0.8pp | Yes |
| **Adj delta (ITI − base)** | **−7.0pp** | **−5.8pp** | **+1.2pp** | **Yes** |
| Baseline det acc | 41.0% | 39.8% | −1.2pp | Yes |
| ITI det acc | 35.0% | 35.2% | +0.2pp | Yes |
| **Det delta (ITI − base)** | **−6.0pp** | **−4.6pp** | **+1.4pp** | **Yes** |

### 3.2 Flip structure comparison

| Metric | Dev (n=100) | Test (n=500) |
|---|---|---|
| Wrong→right | 3 | 14 |
| Right→wrong | 10 | 43 |
| Net flips | −7 | −29 |
| R2W:W2R ratio | 3.3:1 | 3.1:1 |

The flip ratio is remarkably stable (3.3:1 dev → 3.1:1 test). Both the direction and the magnitude of the effect replicate cleanly.

### 3.3 Statistical upgrade

| Property | Dev (Phase 2) | Test (Phase 3) |
|---|---|---|
| Adj delta CI | [−14.0%, 0.0%] | [−8.8%, −3.0%] |
| CI excludes zero | No (touches zero) | **Yes** |
| McNemar p | 0.096 | **0.0002** |
| Interpretation | Directional, underpowered | **Significant, confirmed** |

The dev set correctly identified the direction and approximate magnitude of the effect. The 5× sample-size increase tightened the CI from a 14pp span to a 5.8pp span, resolving the ambiguity at the upper bound.

### 3.4 Baseline stability

The baseline accuracy is stable across dev and test (47.0% vs 45.0%, Δ = −2.0pp), indicating comparable difficulty distributions despite disjoint question sets. This rules out a selection-bias explanation for the observed delta.

---

## 4. Grader Reliability

### 4.1 Audit gate

| Condition | Match audits | Match disagree | Pilot agreement | Passes (≥90%) |
|---|---|---|---|---|
| Baseline α=1.0 | 37 | 0 (0.0%) | 97.5% (39/40) | Yes |
| ITI α=8.0 | 33 | 0 (0.0%) | 97.5% (39/40) | Yes |

Zero false positives across all 70 match audits. The deterministic grader never over-credits.

### 4.2 Non-match recovery

| Condition | Non-matches judged | Judge recovered | Recovery rate |
|---|---|---|---|
| Baseline | 301 | 26 | 8.6% |
| ITI α=8.0 | 324 | 20 | 6.2% |

The judge recovery rate is slightly lower for ITI than baseline (6.2% vs 8.6%). This is consistent with ITI producing more genuinely wrong responses (rather than paraphrases the deterministic grader misses), but the difference is modest.

### 4.3 Match tier distribution

| Tier | Baseline | ITI α=8.0 | Δ |
|---|---|---|---|
| exact | 140 | 70 | −70 |
| boundary | 49 | 97 | +48 |
| no_match | 301 | 324 | +23 |
| numeric | 4 | 6 | +2 |
| reverse_contain | 5 | 3 | −2 |
| alias_simplified | 1 | 0 | −1 |

The exact→boundary shift is dramatic: 53 responses migrated from exact to boundary match, meaning ITI preserved the correct answer but elaborated it into a longer response. Another 25 migrated from exact to no_match — these are cases where elaboration pushed the response past the deterministic grader's boundary detection.

### 4.4 Response length

| | Baseline | ITI α=8.0 |
|---|---|---|
| Mean tokens | 4.7 | 6.6 |
| Median tokens | 4 | 6 |
| Std | 2.1 | 3.2 |

Paired Δ: +1.90 tokens mean, +0 median. 244/500 responses (49%) got longer under ITI; only 63 (13%) got shorter. The ITI intervention systematically increases verbosity — generating phrased sentences ("X is the answer") rather than bare noun phrases.

---

## 5. Failure Mode Taxonomy

### 5.1 Mechanism breakdown (43 right-to-wrong flips)

| Failure mode | Count | % | Description |
|---|---|---|---|
| **Wrong-entity substitution** | 30 | 70% | Model replaces correct entity with a different, confidently stated entity |
| Evasion / factual denial | 8 | 19% | Model denies the premise or claims the answer is unknown |
| Formal refusal (NOT_ATTEMPTED) | 2 | 5% | Model explicitly refuses or evades |
| Answer dilution / verbosity | 3 | 7% | Correct answer included but hedged or diluted below grading threshold |

### 5.2 Wrong-entity substitution: same-neighborhood selection

The dominant pattern at scale — consistent with the dev-set finding — is that the ITI intervention redistributes probability mass to semantically nearby candidates within the same factual category. Examples that demonstrate the pattern:

| Question domain | Baseline (correct) | ITI (wrong) | Relationship |
|---|---|---|---|
| Danny Boyle 1996 film | "Trainspotting" | "Slumdog Millionaire" | Same director, wrong film |
| Third musician in 1959 crash | "Ritchie Valens" | "J.P. Richardson" | Same crash, wrong victim |
| Family Guy spin-off character | "Cleveland Brown" | "Peter Griffin" | Same show, wrong character |
| Ford model named after Italian resort | "Ford Cortina" | "Ford Escort" | Same brand, wrong model |
| Dickens novel with Merdle & Sparkler | "Little Dorrit" | "Bleak House" | Same author, wrong novel |
| DC comic introducing Superman | "Action Comics" | "Detective Comics" | Same publisher, wrong title |
| Absolutely Fabulous actress | "Julia Sawalha" | "Julia Faulkner" | Same first name, wrong person |
| Farthest point from Earth's center | "Mount Chimborazo" | "Mount Everest" | Common misconception |
| 1993 boy-writes-to-Annie film | "Sleepless in Seattle" | "Scent of a Woman" | Same year, wrong film |
| Stereophonics 2005 UK #1 | "Dakota" | "Little Shot of Me" | Same band, wrong song |

This is the sharpest mechanistic finding from the bridge experiment. The intervention does not suppress generation or cause generic degradation — it actively selects from the correct knowledge neighborhood but picks the wrong member. The "probability mass redistribution over nearby factual candidates" diagnosis from the dev set (5/10 = 50%) now holds with much stronger statistics (30/43 = 70%) and richer examples.

### 5.3 Evasion / factual denial (19% of R2W flips)

A second failure mode, less visible at dev scale (1/10), becomes apparent at test scale: the model denies the factual premise of the question. Examples:

- "Turandot" → "He did not complete a Puccini opera."
- "The Magnificent Seven" → "It was not based on 'Seven Samurai.'"
- "Caviar" → "Roe is not typically used as a substitute for any delicacy."
- "No specific minimum age was established" (1833 Factory Act — there was: nine years)

These are qualitatively different from wrong-entity substitution. The model is not retrieving a wrong entity — it is asserting that the correct answer does not exist. This could reflect the truthfulness direction amplifying a "skepticism" component, leading the model to reject well-established facts as uncertain.

### 5.4 Comparison with dev failure taxonomy

| Failure mode | Dev (10 flips) | Test (43 flips) |
|---|---|---|
| Wrong-entity substitution | 5 (50%) | 30 (70%) |
| Evasion / denial | 1 (10%) | 8 (19%) |
| Verbosity-induced loss | 3 (30%) | 3 (7%) |
| Formal refusal | 1 (10%) | 2 (5%) |

The dominant mode (wrong-entity) remains dominant but its share increased from 50% to 70%. Verbosity loss dropped from 30% to 7%. At test scale, the cleaner picture is: the intervention actively corrupts factual retrieval (89% of damage = wrong-entity + evasion), rather than passively degrading response formatting.

---

## 6. What the Wrong-to-Right Rescues Look Like (14 cases)

The 14 rescues are genuine and informative. The model replaces a confidently wrong baseline answer with the correct entity:

| Baseline (wrong) | ITI (correct) | Domain |
|---|---|---|
| "21" | "25" (Adele album) | Pop music |
| "Manchester United" | "Blackburn Rovers" (only 1× PL winner) | Football trivia |
| "Pinglish" | "Pidgin English" | Linguistics |
| "Ana Lins" | "Maria Bueno" (1959 Wimbledon) | Sports history |
| "Rocky Marciano" | "Henry Cooper" (1959-69 HW champ) | Boxing |
| "A nuclear power plant" | "Electric chair" (Ole Sparky) | Terminology |
| "English Setter" | "Irish Setter" (Red Setter) | Dog breeds |
| "Dunlin" | "European lapwing" (peewit) | Ornithology |

The rescue mechanism mirrors the damage mechanism: the ITI intervention shifts probability mass, and in these 14 cases the shift happens to land on the correct entity. This is consistent with the "indiscriminate redistribution" interpretation — the direction does not encode knowledge of which answer is correct; it merely perturbs the distribution, sometimes helpfully but more often harmfully.

---

## 7. Attempt Rate and Not-Attempted Analysis

| | Baseline | ITI α=8.0 | Δ |
|---|---|---|---|
| Not-attempted | 2 | 8 | +6 |
| NA rate | 0.4% | 1.6% | +1.2pp |
| Paired Δ CI | | | [−2.4%, −0.2%] (excludes zero) |
| Overlap (both NA) | 1 | | |
| ITI-only new NAs | 7 | | |

The attempt rate drop is statistically significant but small in absolute terms (6 new refusals out of 500). NOT_ATTEMPTED is not the dominant failure mode — it accounts for only 2/43 (5%) of the R2W flips. The ITI intervention predominantly corrupts answers, not suppresses them.

---

## 8. Interpretation and Claim Status

### 8.1 What the data says (established facts)

1. E0 ITI (paper-faithful, K=12, α=8.0) reduces TriviaQA bridge accuracy by −5.8pp [−8.8, −3.0] on the held-out 500-question test set. Both metrics exclude zero. McNemar p=0.0002.

2. The effect replicates from dev to test: point estimate within the dev CI, flip ratio stable at ~3:1, baseline accuracy stable (47% dev → 45% test).

3. The dominant failure mode (70% of 43 R2W flips) is wrong-entity substitution — the model selects from the correct factual neighborhood but picks the wrong member.

4. A secondary failure mode (19%) is factual denial — the model asserts that well-established facts are uncertain or false.

5. The grader is reliable: 0% match audit disagreement, 97.5% pilot agreement in both conditions.

### 8.2 What the data supports (interpretations that withstand scrutiny)

**The externality break is confirmed.** The same ITI direction that improves TruthfulQA MC1 by +6.3pp harms open-ended TriviaQA generation by −5.8pp, with CIs excluding zero in both directions. This is the cleanest evidence in the project that constrained answer-selection gains do not transfer to open-ended factual generation.

**The failure is mechanistically informative.** The wrong-entity substitution pattern — same domain, wrong member — is consistent with the mass-mean ITI direction redistributing probability mass across nearby factual candidates rather than injecting or extracting knowledge. The direction shifts answer likelihoods within a semantic neighborhood but is indiscriminate about which candidate it promotes.

**The evasion/denial pattern adds a second mechanism.** The 8 cases where ITI causes the model to deny factual premises suggest the truthfulness direction has a "skepticism" component that can inappropriately suppress confident factual assertions.

### 8.3 What the data does not establish (boundaries of the claim)

1. **Not a population-rate claim on failure modes.** The 70/19/7/5 taxonomy is based on manual classification of 43 flips, not a formal coding scheme with inter-rater reliability. The qualitative pattern is robust; the exact percentages are approximate.

2. **Not a mechanism claim about internal circuits.** "Probability mass redistribution" describes the behavioral pattern. We have not demonstrated that the attention heads are literally redistributing probability mass in the residual stream. This is a behavioral diagnosis, not a mechanistic one.

3. **Not evidence that all ITI configurations fail on generation.** We tested E0 (paper-faithful, K=12, α=8.0) only. The dev set showed E1 (modernized, K=8) is even worse, but the test set was not run on E1. The claim is specific to the mass-mean ITI family with TruthfulQA-sourced directions.

4. **Single-model, single-benchmark.** Gemma-3-4B-IT on TriviaQA. Generalization to other models and factual QA benchmarks is not established.

### 8.4 Claim upgrade status

| Claim | Before Phase 3 | After Phase 3 | Evidence |
|---|---|---|---|
| ITI harms TriviaQA generation | Directional, CI touches zero (L5) | **Confirmed, CI excludes zero** | Δ −5.8pp [−8.8, −3.0], p=0.0002 |
| Dominant failure is wrong-entity substitution | Qualitative (5/10 flips) | **Replicated at scale (30/43)** | 70% of R2W flips, stable ratio |
| Externality break: MC gains ≠ generation gains | "Partially earned" | **Earned** | MC +6.3pp vs generation −5.8pp, both significant |
| Failure-mode taxonomy | Dev-set qualitative diagnosis | **Test-scale diagnostic** (still not formal coding) | 43 flips classified, two dominant modes |

---

## 9. Provenance Audit

| Check | Result |
|---|---|
| Git SHA | `0052c07` (both runs, git clean) |
| Comparability class | `claimable` (both runs) |
| Run profile | `canonical` (both runs) |
| Baseline start/end | 09:36:51 → 09:38:03 UTC |
| ITI start/end | 09:38:05 → 09:39:40 UTC |
| Records per file | 500 / 500 |
| Manifest count | 500 |
| Dev/test ID overlap | 0 |

---

## 10. Implications for Paper

### 10.1 What changes in the paper draft

Section 5.3 currently hedges with "dev-set point estimates" and cites Limitation L5. These qualifications should be removed or replaced:

- **Replace** "dev set, n=100, baseline adjudicated accuracy 47.0%" with "test set, n=500, baseline adjudicated accuracy 45.0%"
- **Correct** the stale "held-out test split ($n = 400$)" to "$n = 500$" (the paper had a wrong projection)
- **Upgrade** the CI from [−14.0, 0.0] to [−8.8, −3.0] and add "CI excludes zero, p=0.0002"
- **Remove** the L5 hedge ("the main ITI configuration has not yet been run on the held-out test split")
- **Remove or revise** L5 row in §8 limitations table — the limitation is now resolved
- **Update** the failure-mode taxonomy from 5/10 flips to 30/43 (70%), noting the evasion mode as a secondary finding
- **Update** the flip table from dev (3 w→r, 10 r→w) to test (14 w→r, 43 r→w)
- **Keep** Box B (Thriller rescue) as qualitative illustration — it's from dev but still valid as a worked example
- **Update** `notes/paper/draft/number_provenance.md` Bridge (§5.3) table: all values now source from this report instead of the Phase 2 dev report

### 10.2 What changes in the strategic assessment

- §3 externality-break claim: upgrade from "partially earned" to **"earned"**
- §4 Anchor 2: update the headline number from "−7pp/−9pp" to "−5.8pp [−8.8, −3.0]"
- §10: add a pointer to this report as the terminal test-set result
- Remove "bridge elevation" as a pending priority — it is now complete

### 10.3 Remaining uncertainties

1. **E1 on test set:** Dev showed E1 is worse than E0 (−9pp, p=0.016). The test set was not run on E1. If the "gentler artifact is worse" claim needs test-set evidence, another run would be needed. However, the core externality-break claim does not depend on the E0-vs-E1 comparison.

2. **Failure mode coding reliability:** A formal inter-rater coding exercise (blind taxonomy, two raters, Cohen's κ) would strengthen the 70% wrong-entity claim. The current single-rater classification is sufficient for the behavioral diagnosis but would not survive aggressive review as a quantitative claim.

3. **Wrong-entity circuit mechanism:** What circuit produces the same-neighborhood wrong-entity substitutions? This is the natural follow-up question but is outside the scope of the current paper.

---

## 11. Summary Numbers for Paper

```
Bridge Phase 3 (test set, n=500, held-out, one-shot)
  Baseline:     45.0% [40.7, 49.4] adjudicated, 39.8% [35.6, 44.2] deterministic
  E0 ITI α=8:   39.2% [35.0, 43.5] adjudicated, 35.2% [31.1, 39.5] deterministic
  Δ adj:        −5.8pp [−8.8, −3.0], CI excludes zero
  Δ det:        −4.6pp [−7.6, −1.6], CI excludes zero
  McNemar:      χ²=13.75, p=0.0002
  Net flips:    −29 (14 w→r, 43 r→w), ratio 3.1:1
  Failure mode: 70% wrong-entity substitution, 19% factual denial, 5% formal refusal, 7% dilution
  Grader:       0/70 match audit disagreements, 97.5% pilot agreement both conditions
```
