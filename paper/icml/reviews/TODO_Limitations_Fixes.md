## Verdict

**The highest-value new experiment is not L4 or L5.** It is a targeted attack on **L2/L3**: test whether the SAE null survives a utility-aware / output-effect feature-selection ablation. That is the experiment most likely to improve the paper’s contribution in April 2026, because current SAE-steering work has moved exactly toward “SAEs can steer if you select the right features.” Arad et al. distinguish input-active vs output-effective SAE features and report 2–3× steering improvements after output-score filtering; Wang et al. train 90 SAEs and find interpretability is only weakly associated with steering utility, while utility-aware feature selection improves steering substantially. Your current SAE result is strongest if it is explicitly about **readout-selected** SAE features, not “SAEs” in general. ([ACL Anthology][1])

The workshop’s current CFP explicitly rewards rigorous negative results, clear falsifiable hypotheses, reproducibility, and honest limitation reporting; it also says long papers are held to a higher standard of rigor and depth. So the right strategy is not “run more things.” It is: close cheap credibility holes, then add one experiment that sharpens the central methodological lesson. ([MechInt Workshop][2])

---

# 1. Locked policy for L5

## Q1: Scoring scope

Choose **Both Rulers**, but with a sharper policy than the handoff currently states.

Use:

1. **Legacy CSV-v2** for historical continuity with the existing `+2.30 pp/α` claim.
2. **CSV-v3** for the current measurement thesis and the richer outcome taxonomy.
3. Do **not** mix v2 and v3 in one statistical comparison.
4. Do **not** make v2 “primary” just because it preserves the older significant result.
5. In the main paper, frame the result as **rubric-sensitive**, not as a robust jailbreak-steering success.

The main-text phrasing should become something like:

> Under the legacy graded rubric, the H-neuron jailbreak effect remains positive relative to random-neuron controls. Under the current v3 harmful-binary rubric, the binary effect is weak or null, while the richer severity taxonomy preserves evidence of a partial-to-substantive compliance shift. We therefore treat the jailbreak case as measurement-sensitive rather than as a clean steering-success claim.

That is much more tasteful than trying to defend one ruler.

One detail: your handoff says Option C requires `6 × 4 × 500 = 12,000` calls, but the table as written appears to list only four missing scoring blocks: seed-1 v2, seed-2 v2, seed-0 v3, seed-2 v3. That is `4 × 4 × 500 = 8,000` judge calls unless there are two additional unlisted cells. Add a pre-run completeness check before launching.

## Q2: Historical seed-0 p-value mismatch

Choose **Recompute Canonically**.

The `p = 0.013` number should be superseded if the current canonical paired utility gives `p ≈ 0.00066` on the same legacy data. Keeping the old number after discovering a non-reproducible analysis path is worse than changing it. It looks like provenance debt.

Policy:

> All p-values in the paper use the current canonical paired trajectory analysis. Historical p-values from pre-utility analysis are preserved only in the audit/provenance note and are not used as paper claims.

In the main paper, I would avoid emphasizing the smaller p-value. Say `p < 0.001` or report the effect size and CI. The scientific point is not that the old effect is “more significant than we thought.” The point is that analysis-policy drift was detected and resolved.

## Q3: v3 slope estimator

Use **paired complete-case trajectories as the primary estimator**, and report the per-alpha valid-rate fit as a sensitivity analysis.

So:

* Primary v3 H-neuron slope: `+0.60 pp/α`, complete-case paired trajectory estimator.
* Sensitivity: `+0.46 pp/α`, per-alpha valid-rate fit from `analyze_csv2_control.py`.
* Conclusion: unchanged, because both are weak/null on harmful-binary scoring.

Reason: the paper’s causal comparisons are paired by prompt. The paired complete-case estimator better matches the intervention design and the permutation machinery. The per-alpha valid-rate estimator is useful as an attrition sensitivity check, not the main estimand.

Also: do **not** report a pooled 3-seed p-value that treats `seed × prompt` rows as independent. The H-neuron trajectory is reused across controls. Report:

* H slope.
* Each random seed slope.
* H-minus-random slope difference per seed.
* H-minus-mean-random slope difference with prompt bootstrap.
* Optional jackknife over the three random seeds, clearly labeled as rough because `n_seed = 3`.

---

# 2. Should you address the five limitations?

## L1: Single model

**Do not spend the next 3 weeks trying to fix this.**

A real second-model replication would be valuable, but it is not the best deadline move. A rushed second-model pilot risks becoming noisy, partial, and space-consuming. The workshop accepts rigorous case studies and negative results, provided the scope is honest. Your model name is already in the title, and the thesis is methodological: “here is a disciplined audit showing where the readout-to-steering inference breaks in one model.”

Only attempt a second-model appendix pilot after everything else is stable. Do not let it touch the main narrative unless it cleanly replicates the FaithEval localization→control break.

## L2: Matched-readout confound

**This is the most important remaining limitation.**

Current critique a reviewer can make:

> “You matched AUROC, but SAE features and neurons differ in operator form, layer coverage, selection method, and feature family. Maybe the SAE null is not about readout quality; maybe it is about bad SAE target selection.”

That critique is fair. It does not kill the paper, but it is the most central weakness.

The best response is not a broad SAE sweep. The best response is a **target-selection ablation inside the SAE family**:

* readout-selected SAE features: current null;
* output-effect / utility-selected SAE features: new ablation;
* random layer/frequency-matched SAE features: negative control.

This directly engages the current field: AxBench reports that SAEs were not competitive on its steering benchmark, while later SAE-steering work argues that feature selection is the critical variable. ([OpenReview][3])

## L3: SAE layer coverage

**Address only through the L2 ablation.**

Do not run a blind “more layers, more widths” sweep. That creates a fishing expedition and bloats the paper.

Instead, expand SAE candidates only enough to answer a precise question:

> Does an intervention-aware SAE selector find steerable FaithEval features where the readout selector failed?

That partially addresses layer coverage without turning the paper into an SAE benchmark.

## L4: Bridge inter-rater reliability

**Do it immediately.**

But improve the plan:

1. Code **all discordant bridge cases**, not only the 43 right-to-wrong flips. Include the 14 wrong-to-right rescues. This lets you compare damage modes vs rescue modes.
2. Use a real human second rater if at all possible. An LLM second rater is acceptable as a sensitivity check, but weaker as IRR evidence.
3. Report raw agreement, Cohen’s κ, and preferably Gwet’s AC1 as a robustness statistic because κ can behave badly under skewed category prevalence.
4. Predefine an adjudication rule before seeing disagreements.
5. Keep the main claim qualitative unless agreement is strong: “wrong-entity substitution is the dominant coded mode,” not “exactly 70%.”

The current L4 plan is good but too minimal. The added value is not just κ. The value is making the bridge result look like a serious behavioral mechanism analysis rather than an anecdotal taxonomy.

see [L4_PLAN](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/icml/TODO_L4_interrater.md)

## L5: Jailbreak H-neuron multi-seed specificity

**Do it, but demote its role.**

L5 should become a measurement robustness appendix/main-text paragraph. It should not become a fourth anchor. The main paper already has enough anchors: FaithEval localization→control, ITI bridge externality, and jailbreak measurement sensitivity.

Given LLM-judge robustness concerns, the measurement story is timely. Recent work shows safety judges can be sensitive to prompt/style shifts and artifacts; that supports your decision to treat evaluator disagreement as part of the phenomenon rather than as mere noise. ([OpenReview][4])

see [L5_PLAN](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/icml/TODO Limitation_5_multi-seed.md)

---

# 3. Highest-value experiment stack

## Priority 0 — Finish the writing-feedback implementation - DONE

This gates everything. Do not add experiments until the current draft has the tightened thesis, updated related work, explicit hypotheses, revised title/framing, and cleaned limitation language. The workshop explicitly values clarity about what evidence does and does not support. ([MechInt Workshop][2])

See [P0](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/icml/TODO_Pro_icml_mech_interp_review_report.md)

## Priority 1 — L4 bridge IRR + expanded discordant-case coding - DONE (2026-04-21)

**Value:** Very high.
**Cost:** Low.
**Risk:** Low.
**Where it appears:** Main text, §4.3 bridge subsection.

**Closure:** raw agreement 96.5% on 57 discordant cases (Cohen's κ = 0.90, Gwet's AC1 = 0.96); R→W wrong-entity share 72.1% [57.3, 83.3]; W→R rescues 14/14 wrong-entity. Two disagreements resolved under pre-frozen rule, zero rule-gaps. Full analysis: [`../reports/2026-04-21-bridge-irr-review.md`](../reports/2026-04-21-bridge-irr-review.md).

Minimum deliverable:

> A second blinded coder labeled the 57 discordant bridge cases. Agreement was X%, Cohen’s κ = Y, Gwet’s AC1 = Z. The wrong-entity-substitution conclusion remained stable after adjudication: N/M right-to-wrong flips were coded as substitutions.

Better deliverable:

Add a small table:

| Transition  |  n | Substitution | Evasion/denial | Dilution | Refusal |
| ----------- | -: | -----------: | -------------: | -------: | ------: |
| right→wrong | 43 |            … |              … |        … |       … |
| wrong→right | 14 |            … |              … |        … |       … |

This would make the externality section materially more convincing.

## Priority 2 — SAE utility-selector ablation on FaithEval

**Value:** Highest scientific upside.
**Cost:** Medium.
**Risk:** Medium.
**Where it appears:** Main text if clean; appendix if noisy.

This is the most important optional experiment.

### Question

> Did SAE steering fail because readout quality is insufficient, or because the chosen SAE features were not output-effective steering handles?

### Design

Use the same model, same FaithEval surface, same held-out evaluation protocol.

Compare three SAE target sets:

1. **Readout-selected SAE features** — current 266-feature null.
2. **Utility/output-selected SAE features** — selected by an intervention-aware proxy, not by held-out AUROC.
3. **Random matched SAE features** — matched by layer, activation frequency, and preferably norm.

Possible utility selectors:

* Arad-style output score: does activating the feature shift model outputs toward the desired token/output region?
* Wang-style token-confidence / next-token distribution perturbation: how much does activating the feature change relevant next-token confidence?
* Small validation-set causal screen: on a frozen `n≈100–200` FaithEval validation subset, measure endpoint effect at one or two α values, then run one held-out test.

Do not overbuild this. The cleanest version is:

1. Freeze candidate pool.
2. Freeze selector.
3. Select top-k features on validation only.
4. Run exactly one held-out FaithEval steering test.
5. Compare to current readout-selected SAE null and random matched features.

### Interpret outcomes

If utility-selected SAE features still fail:

> Stronger paper. You now show that even a more intervention-aware SAE selection pass did not recover FaithEval control under this operator.

If utility-selected SAE features work:

> Also stronger paper. The conclusion becomes more nuanced and more valuable: readout-selected SAE features failed, but intervention-aware selection can rescue control, exactly motivating the audit framework.

This second outcome is not a threat. It would make the paper more useful to the field.

The main-text framing would be:

> The SAE null should not be read as “SAEs cannot steer.” Rather, it shows that held-out behavioral readout is not enough to choose SAE steering targets. A utility-aware selector is required to test whether the representation basis contains a usable handle.

That is a better contribution than “SAEs failed.”

## Priority 3 — Bridge logprob-margin mechanism check — DONE (2026-04-21, directional outcome differs from prediction)

**Value:** High.
**Cost:** Low/medium.
**Risk:** Low.
**Where it appears:** Supplement paragraph with one-line footnote in §4; not main-text headline.

**Closure:** executed on n=257 cases (A=31 substitution, B=12 non-substitution, C=14 rescue, D=200 random-wrong controls) at ITI α=8.0, K=12, `first_3_tokens` scope. The broad claim survives — ITI compresses the gold-vs-wrong logprob margin on R→W (A first3 = −10.16 nats [−12.81, −7.60]) and expands it on W→R rescues (C = +4.73 [+2.27, +7.25]). **The TODO-predicted "clean result" shape (substitution > non-substitution > controls in margin-shift magnitude) did not obtain:** non-substitution cases show ~4× the margin compression of substitution cases (B = −40.06 [−50.09, −30.37]; A−B = +29.90, one-sided p = 1.0), primarily — but not exclusively — driven by the first-token answer frame. A vs. D (random controls) survives in the predicted direction but modestly (A−D = −5.10 [−8.22, −2.16], p = 0.0081). Precommit decision tree branch (iii) fires: drop the A<B headline; the behavioral substitution taxonomy does not index a distinct margin-shift signature, though it remains a reliable behavioral description (κ = 0.90 per [L4 closure](../reports/2026-04-21-bridge-irr-review.md)). Full analysis: [`../reports/2026-04-21-bridge-margin-precommit.md`](../reports/2026-04-21-bridge-margin-precommit.md) (post-run, with sealed precommit as Appendix A).

This deepens the most tasteful part of the paper: wrong-entity substitution.

### Question

> Does ITI actually shift probability mass from the gold answer toward the wrong generated entity, or is the substitution taxonomy merely post-hoc?

### Design

For each bridge discordant case, collect:

* gold answer / alias;
* baseline generated answer;
* ITI generated answer;
* coded failure mode.

For right→wrong substitution cases, compute under both baseline and ITI intervention:

[
\Delta_{\text{margin}} =
\log p(\text{gold answer} \mid \text{prompt})
---------------------------------------------
Δmargin​=logp(gold answer∣prompt)−logp(ITI wrong entity∣prompt)


\log p(\text{ITI wrong entity} \mid \text{prompt})
]

Then test whether ITI reduces this margin specifically on substitution cases.

Useful comparisons:

* right→wrong substitution cases;
* right→wrong non-substitution cases;
* wrong→right rescue cases;
* random wrong-entity controls matched for answer length/entity type if feasible.

A clean result would look like:

> On substitution-coded R→W flips, ITI reduced the gold-vs-wrong-entity log-likelihood margin by X nats/token, whereas non-substitution flips and random wrong-entity controls showed smaller or no margin shifts.

This would upgrade the bridge result from behavioral taxonomy to a mechanistic/probabilistic diagnosis.

## Priority 4 — L5 both-ruler multi-seed closure

**Value:** Medium/high for credibility; medium for core contribution.
**Cost:** Low API.
**Risk:** Low if framed correctly.
**Where it appears:** Measurement section + appendix/provenance.

Do it in parallel with L4.

The right outcome is not “v2 wins” or “v3 wins.” The right outcome is a table that makes the measurement thesis undeniable:

| Ruler  |         Outcome metric | H slope | Random seed slopes | H-minus-control | Verdict                  |
| ------ | ---------------------: | ------: | -----------------: | --------------: | ------------------------ |
| CSV-v2 |     strict harmfulness |       … |                  … |               … | positive / specific      |
| CSV-v3 |         harmful_binary |       … |                  … |               … | weak/null                |
| CSV-v3 | substantive_compliance |       … |                  … |               … | severity shift / partial |

That table is worth more than another paragraph.

## Priority 5 — Minimal externality/capability battery

**Value:** Medium.
**Cost:** Low/medium.
**Risk:** Medium because it can sprawl.

Only do this if the previous items are done.

The most useful version is small and targeted:

* H-neuron α=3 vs no-op on 200–500 items from a general QA or instruction-following panel.
* ITI α=8 vs baseline already has SimpleQA/Bridge; no need to add much.
* Report as “externality smoke test,” not as comprehensive capability evaluation.

This can support the audit checklist, but it is not essential.

## Priority 6 — Second-model pilot

**Value:** Potentially high, expected value lower than it looks.
**Cost:** High.
**Risk:** High.
**Recommendation:** Mostly avoid.

A clean second-model replication would be excellent, but a rushed one is likely to be underpowered, partial, and distracting. The current submission should stay a disciplined Gemma-3-4B-IT case study.

A tiny appendix pilot is acceptable only if your codebase makes it almost automatic and it does not compete with the SAE selector ablation.

---

# 6. Revised limitation strategy

After these experiments, your limitation table should not pretend everything is solved. It should become more mature:

| Limitation                           | Current status      | After recommended work                                            |
| ------------------------------------ | ------------------- | ----------------------------------------------------------------- |
| L1 single model                      | Real and unresolved | Still unresolved; honestly scoped                                 |
| L2 matched-readout confound          | Central weakness    | Partially addressed by SAE readout-vs-utility selector ablation   |
| L3 SAE layer coverage                | Real but narrower   | Partially addressed if candidate pool expands; otherwise explicit |
| L4 bridge single-rater               | Easy objection      | Closed with IRR + adjudication                                    |
| L5 jailbreak single-seed specificity | Credibility issue   | Closed as multi-seed, multi-ruler measurement-sensitivity result  |

That is the right shape. You do not need to eliminate all limitations. You need to show excellent judgment about which limitations constrain which claims.

---

# 7. The paper’s strongest final contribution

The best final version is not:

> “SAEs fail and H-neurons work.”

It is:

> “Readout quality is a weak target-selection criterion. In a matched cross-family case, readout-selected SAE features and H-neurons had similar detection quality but diverged under intervention. A bridge externality test showed that even successful steering can corrupt open-ended factual generation through wrong-entity substitution. A jailbreak case study showed that measurement choices can change the verdict. Together these motivate an audit discipline: validate measurement, target selection, control, and externality as separate empirical gates.”

That lands well in the current field. AxBench already made broad steering benchmarks more competitive; Arad and Wang already moved SAE steering toward feature-selection quality; Opiełka et al. show causal vectors can be format-specific rather than invariant. Your contribution should therefore be a **target-selection and audit-methodology paper**, not another “steering failed” paper. ([OpenReview][3])


[1]: https://aclanthology.org/2025.emnlp-main.519/ "SAEs Are Good for Steering – If You Select the Right Features - ACL Anthology"
[2]: https://mechinterpworkshop.com/cfp/ "Call for Papers | Mechanistic Interpretability Workshop at ICML 2026"
[3]: https://openreview.net/forum?id=K2CckZjNy0 "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | OpenReview"
[4]: https://openreview.net/forum?id=kPMfYS2ugs "Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges | OpenReview"
