# BioASQ OOD Probe-Transfer Audit: Gemma-3-4B

**Related reports:** [pipeline_report.md](../../pipeline/pipeline_report.md), [intervention_findings.md](../../intervention_findings.md)

## Bottom line

- **This BioASQ result is trustworthy enough to cite internally as an approximate OOD transfer result, not as an exact paper-faithful replication.** The recovery run on `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json` gives **accuracy 0.6982** and **AUROC 0.8219** over **1,090** scored examples, with bootstrap 95% CIs **[0.6688, 0.7257]** and **[0.7976, 0.8459]** from the saved local activations and `models/gemma3_4b_classifier_disjoint.pkl`.
- **The main conclusion is stable under the recovery patch.** Coverage improved materially while metrics stayed nearly flat: answer-token rows **1542 -> 1561**, activation coverage **1060/1078 -> 1090/1092**, accuracy **0.7019 -> 0.6982**, AUROC **0.8294 -> 0.8219** (`data/gemma3_4b/probing/bioasq13b_factoid/report.json`, `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`).
- **BioASQ is materially harder than local TriviaQA disjoint eval for this detector, but the gap is modest and mostly accuracy-side.** Reproducing the local TriviaQA disjoint eval from `data/gemma3_4b/test_qids_disjoint.json` and `data/gemma3_4b/activations/answer_tokens/` gives **accuracy 0.7654** and **AUROC 0.8429**, versus BioASQ recovery **0.6982** and **0.8219**.
- **The detector’s clearest BioASQ weakness is overcalling verbose faithful biomedical answers as hallucination-positive, consistent with the detector-side verbosity confound documented in the verbosity confound audit.** It is much better at catching long/list-like false answers than at sparing long/list-like true answers. This is a true detector weakness, distinct from the earlier extraction/span data-loss issue.

## What was run

- Primary committed evidence:
  `data/gemma3_4b/probing/bioasq13b_factoid/samples.jsonl`,
  `data/gemma3_4b/probing/bioasq13b_factoid/answer_tokens.jsonl`,
  `data/gemma3_4b/probing/bioasq13b_factoid/eval_qids.json`,
  `data/gemma3_4b/probing/bioasq13b_factoid/report.json`,
  `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`,
  `data/gemma3_4b/probing/bioasq13b_factoid/classifier_metrics.json`,
  `data/gemma3_4b/probing/bioasq13b_factoid/classifier_metrics_recovery.json`.
- Code / provenance anchors:
  commit `c41ee28` (`fix(bioasq): harden answer token span recovery`),
  commit `f07d167` (`data(bioasq): add Gemma-3-4B OOD transfer artifacts`),
  `scripts/collect_responses.py`,
  `scripts/extract_answer_tokens.py`,
  `scripts/extract_activations.py`,
  `scripts/sample_balanced_ids.py`,
  `scripts/classifier.py`.
- Paper framing used:
  `original-paper-markdown-converted.md` on BioASQ as cross-domain OOD eval, the paper’s single-response OOD setup, and Table 1 values for Gemma-3-4B.
- Dataset and model:
  official BioASQ Task B factoid subset in `data/benchmarks/bioasq13b_factoid.parquet`,
  summary in `data/benchmarks/bioasq13b_factoid.summary.json`,
  generator `google/gemma-3-4b-it`,
  reused detector `models/gemma3_4b_classifier_disjoint.pkl`.
- Local-only evidence used for deeper diagnostics:
  `data/gemma3_4b/bioasq13b_factoid_extract_tokens.log`,
  `data/gemma3_4b/bioasq13b_factoid_extract_tokens_recovery.log`,
  `data/gemma3_4b/bioasq13b_factoid_extract_activations.log`,
  `data/gemma3_4b/bioasq13b_factoid_extract_activations_recovery.log`,
  `data/gemma3_4b/bioasq13b_factoid_activations/answer_tokens/`,
  `data/gemma3_4b/bioasq13b_factoid_classifier.log`,
  `data/gemma3_4b/bioasq13b_factoid_classifier_recovery.log`.

### Mirror provenance note

- The mirrored activation directory at `/home/hugo/Documents/Learning/Bluedot Project Technical AI safety/h-neurons-extension/data/gemma3_4b/bioasq13b_factoid_activations/answer_tokens/` is present as a local data mirror but is intentionally not git-tracked.
- As of 2026-03-17, that mirror matches the source directory at `/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/bioasq13b_factoid_activations/answer_tokens/` under `diff -qr`, with **1179** `.npy` files in each location.
- In the Bluedot mirror repo, the directory is excluded via `.git/info/exclude`. That is deliberate: the activation binaries are heavy local byproducts, while the committed provenance lives in the reports and JSON metrics above.
- Analyses in this memo that depend on the mirrored activation directory are explicitly labeled local-only: rescoring against `models/gemma3_4b_classifier_disjoint.pkl`, the mtime-based overlap reconstruction, and inspection of the remaining 2 activation misses.

## Is the BioASQ result trustworthy?

- **OOD setup: mostly paper-faithful on the important axes.** The paper’s OOD setup is “single response” on BioASQ after training the detector on TriviaQA (`original-paper-markdown-converted.md`). This run matches that shape: one sampled BioASQ response per question in `data/gemma3_4b/probing/bioasq13b_factoid/samples.jsonl`, official BioASQ factoid questions from `data/benchmarks/bioasq13b_factoid.parquet`, and a reused TriviaQA-trained detector in `models/gemma3_4b_classifier_disjoint.pkl`.
- **Not an exact replication.** The repo’s detector selection is the simpler baseline described in `data/gemma3_4b/pipeline/pipeline_report.md`, not the paper’s full suppression-aware selection rule. That means “close to the paper’s BioASQ number” is a fair internal claim; “paper-faithful replication succeeded” is not.
- **Label path: sound enough, but the main residual fragility.** `scripts/collect_responses.py` uses an LLM judge (`gpt-4o`) with the prompt “Return 't' if correct, 'f' if incorrect” against BioASQ alias lists. Operationally it was stable on this run: `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json` shows **560 true, 1040 false, 0 uncertain, 0 error**. The remaining risk is not pipeline crashes; it is semantic leniency on biomedical paraphrases. The later representative audit in `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` and `data/gemma3_4b/intervention/bioasq/representative_audit_summary.json` narrows that risk: judge-plus-benchmark issues explain about **14%** of reported detector errors, not most of them.
- **Inference:** several high-confidence false positives look semantically close to the alias list rather than plainly wrong. Examples from `data/gemma3_4b/probing/bioasq13b_factoid/samples.jsonl` and local scoring:
  `Mtm1` vs ground truth `MTM1 gene test`,
  `Yellow-orange` vs `yellow`,
  `TRK (Tropomyosin receptor kinase)` vs `tropomyosin receptor kinases`.
  The representative audit confirms this is a real but minority issue rather than the dominant explanation for detector error.
- **Balanced eval is appropriate for accuracy/AUROC, but not for natural-prevalence precision.** `scripts/sample_balanced_ids.py` samples equal true/false IDs after judging. That is aligned with the paper’s detector-evaluation framing, but it means the reported precision is not the operating precision at BioASQ’s raw prevalence.
- **Recovery patch improves fidelity more than it changes the target.** The patch in `c41ee28` does two conservative things: parse more extractor outputs, and map ordered subsequences to minimal enclosing contiguous spans. That can slightly widen a selected answer region, but it stays inside the answer-focused target rather than switching to “whole response.” The near-flat metrics after recovery are evidence against a large target shift.

## Initial vs recovery comparison

| Run | Answer-token rows | Dropped | Label counts | Balanced eval | Activation coverage | Accuracy | AUROC |
|---|---:|---:|---|---:|---:|---:|---:|
| Initial | 1542 | 58 | 539 true / 1003 false | 1078 | 1060/1078 | 0.7019 | 0.8294 |
| Recovery | 1561 | 39 | 546 true / 1015 false | 1092 | 1090/1092 | 0.6982 | 0.8219 |

- Recovery gained **19** answer-token rows: **+7 true** and **+12 false** (`data/gemma3_4b/probing/bioasq13b_factoid/report.json`, `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`).
- Coverage improved from **96.38%** to **97.56%** at answer-token extraction, Wilson 95% CIs **[95.34, 97.19]** and **[96.69, 98.21]**. Activation coverage improved from **98.33%** to **99.82%**, Wilson 95% CIs **[97.38, 98.94]** and **[99.33, 99.95]**.
- Point-metric movement was small:
  accuracy **-0.0037**,
  precision **-0.0030**,
  recall **-0.0024**,
  F1 **-0.0029**,
  AUROC **-0.0076**.
- The **19 recovered rows were not random**. Looking at the appended rows in `data/gemma3_4b/probing/bioasq13b_factoid/answer_tokens.jsonl`:
  mean response length rose from **3.79** words in the initial 1542 rows to **9.05** words in the 19 recovered rows,
  mean token count from **7.03** to **16.05**,
  mean answer-token count from **6.13** to **13.26**.
- Recovered rows were dominated by formatting-heavy biomedical strings:
  **9/19** had apostrophes,
  **5/19** hyphens,
  **5/19** list-like punctuation,
  **4/19** parentheses,
  **4/19** all-caps abbreviations,
  **0/19** slashes,
  **0/19** actual Greek characters.
  This matches the syntax-failure pattern in `data/gemma3_4b/bioasq13b_factoid_extract_tokens.log`, which repeatedly shows `Expecting ',' delimiter`, and the clean recovery log in `data/gemma3_4b/bioasq13b_factoid_extract_tokens_recovery.log`.
- **Inference, with an evidence gap:** the exact initial `data/gemma3_4b/probing/bioasq13b_factoid/eval_qids.json` was overwritten by the recovery run, so an exact saved overlap comparison of initial-vs-recovery eval IDs is unavailable.
- To compensate, I used local-only activation file mtimes in `data/gemma3_4b/bioasq13b_factoid_activations/answer_tokens/`. On the **current recovery eval sample**, restricting evaluation to rows that were already present before the recovery activation run gives **accuracy 0.6993** and **AUROC 0.8232** over **1074** scored examples, almost identical to the recovery headline **0.6982 / 0.8219**. That strongly suggests the patch changed coverage more than it changed the measured result.
- The **16 recovered rows that made it into the current balanced eval** scored worse (**accuracy 0.625**, **AUROC 0.7143**, local-only, no CI, very small `n`), so the recovered cases are not obviously “easy wins.” They look more like hard or noisy edge cases than metric-padding.

## BioASQ vs TriviaQA comparison

- Recomputing the local disjoint TriviaQA eval from `data/gemma3_4b/test_qids_disjoint.json`, `data/gemma3_4b/activations/answer_tokens/`, and `models/gemma3_4b_classifier_disjoint.pkl` gives:
  **accuracy 0.7654** with bootstrap 95% CI **[0.7346, 0.7949]**,
  **AUROC 0.8429** with bootstrap 95% CI **[0.8143, 0.8705]**.
- BioASQ recovery gives:
  **accuracy 0.6982** with bootstrap 95% CI **[0.6688, 0.7257]**,
  **AUROC 0.8219** with bootstrap 95% CI **[0.7976, 0.8459]**.
- So the local OOD gap is:
  **-6.72 percentage points** in accuracy,
  **-2.10 points** in AUROC.
  This is directionally consistent with the paper’s Gemma-3-4B row in `original-paper-markdown-converted.md`:
  **TriviaQA 76.9**, **BioASQ 71.0**.
- BioASQ raw faithfulness on the single-response collection is poor:
  **560/1600 = 35.0% faithful** with Wilson 95% CI **[32.7%, 37.4%]**,
  **1040/1600 = 65.0% false** with Wilson 95% CI **[62.6%, 67.3%]**
  from `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`.
- I do **not** think we can make a like-for-like raw faithfulness comparison against the repo’s TriviaQA artifacts. The local TriviaQA pipeline in `data/gemma3_4b/pipeline/pipeline_report.md` is consistency-filtered over multiple samples, not this single-response OOD setup.
- Within BioASQ’s raw 1600-question sample, longer and list-like outputs are more failure-prone:
  **5-7 word responses:** **72.4% false** (exploratory, no CI),
  **list-like responses:** **75.7% false** vs **63.9%** for non-list-like (exploratory, no CI).
- Alias count is not a strong difficulty driver here:
  raw false rate is **65.0%** for 1-alias questions, **66.5%** for 2-alias, **62.2%** for 3+ alias questions (exploratory, no CI).
- Year effects are present but modest:
  raw false rate **62.0%** for 2013-2016,
  **66.8%** for 2017-2020,
  **65.4%** for 2021-2024
  from `data/benchmarks/bioasq13b_factoid.parquet` joined to `data/gemma3_4b/probing/bioasq13b_factoid/samples.jsonl` (exploratory, no CI).

## Failure mode analysis

- **Pipeline/data-loss issue 1: extractor parse brittleness.** The initial answer-token failures were mostly formatting failures, not semantic failures. `data/gemma3_4b/bioasq13b_factoid_extract_tokens.log` is full of repeated JSON parse errors such as `Expecting ',' delimiter`; `data/gemma3_4b/bioasq13b_factoid_extract_tokens_recovery.log` has no such errors.
- **Pipeline/data-loss issue 2: activation span matching brittleness.** The initial run missed **18** activation files because extracted answer tokens were semantically right but not an exact contiguous string match under tokenizer decoding. Recovery reduces that to **2** misses.
- **The recovery patch did not quietly change the task in a major way.** Its main effect was to stop dropping valid rows for punctuation/list formatting and ordered-subsequence spans. The stability of the headline metric after recovery argues that the patch removed nuisance loss rather than shifting the detector to an easier target.
- **True detector weakness is elsewhere:** even after data loss is fixed, the detector still has **267 false positives** and **62 false negatives** on the recovery eval. Those are model/detector behavior issues, not pipeline bookkeeping.
- Remaining 2 activation misses, both from local-only inspection with tokenizer decoding:
  `530cf4e0c8a0b4a00c000002` is a long false answer where the extracted span ends with `)` but the actual tokenizer boundary is `),`.
  `66156200fdcbea915f00004d` is a long false answer where the extracted span ends with `)` but the actual tokenizer boundary is `).`
- **Recommendation on whole-output fallback:** not justified as the default.
  For `66156200fdcbea915f00004d`, whole-output would be close to the intended answer-only region.
  For `530cf4e0c8a0b4a00c000002`, whole-output would swallow the later hedged explanation (`though it’s often... There’s a potential...`), which would blur the evaluation target from “answer tokens” toward “rambling false output.”
- There is also a second reason to avoid naive whole-output fallback: the current output-region logic in `scripts/extract_activations.py` still includes assistant-header tokens under the Gemma chat template. A future fallback should be **punctuation-tolerant boundary matching**, not “take the whole output region.”

## Detector error analysis

- Overall confusion on the recovery eval, from local-only rescoring of `data/gemma3_4b/probing/bioasq13b_factoid/eval_qids.json` against `data/gemma3_4b/bioasq13b_factoid_activations/answer_tokens/`:
  **267 false positives**,
  **62 false negatives**.
- **False positives cluster on faithful but broad / verbose / list-like biomedical answers.**
  Examples:
  `Phospholipid hydrolysis` for a lipid inositol phosphatase activity question,
  `Mtm1` for `MTM1 gene test`,
  `Yellow-orange` for `yellow`,
  `TRK (Tropomyosin receptor kinase)` for `tropomyosin receptor kinases`,
  `Human epithelial cells from larynx carcinoma.` for a HEp-2 origin question.
- The strongest FP pattern is **response form, not domain symbol soup alone**:
  among judge-true answers in the balanced eval, list-like answers have **90.6% FP rate** versus **46.3%** for non-list-like, and answers longer than 2 words have **66.4% FP rate** versus **36.8%** for 1-2 word true answers (exploratory, no CI).
- **False negatives cluster on crisp wrong entity substitutions.**
  Examples:
  `Bardet-Biedl syndrome` instead of `Oguchi disease`,
  `MicroRNA` instead of the BioASQ answer describing miRs,
  `ApoE4` instead of `ApoE2`,
  `IL-5` instead of `IL-6`,
  `Naloxone` instead of `flumazenil`,
  `1955` instead of `1954`.
- These false negatives are usually **short and fluent**, not rambling:
  mean FN length is **2.69 words** versus **5.15** for false positives.
  That means the detector is **not** just picking up “uncertain” or “rambling” style.
- But style still matters a lot on the positive side:
  among judge-false answers, longer false answers are actually **easier** for the detector, not harder.
  Detection rate is **92.9%** for false answers longer than 2 words versus **85.2%** for 1-2 word false answers (exploratory, no CI).
- So the most defensible reading is:
  the classifier tracks an activation signal correlated with false-vs-true answer-token labels,
  but on BioASQ it also overweights verbosity / explanatory form enough to punish faithful long answers,
  and it undercatches compact wrong biomedical substitutions.
  This pattern is consistent with the verbosity confound finding (intervention_findings.md Finding 7).

## Recommended headline result

- **Primary headline:** the **recovery run** in `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`.
- **Sensitivity analysis:** report the initial run from `data/gemma3_4b/probing/bioasq13b_factoid/report.json` beside it.
- Why:
  the recovery run fixes obvious pipeline loss,
  raises coverage on both extraction and activations,
  and leaves the detector result essentially unchanged.
- The initial number is slightly higher, but the evidence points to that being a **mild dropout artifact**, not a more faithful estimate. If you force the story onto one number, recovery is the less biased choice.

## Claims we can make

### High-confidence

- The Gemma-3-4B BioASQ OOD answer-token classifier result is **about 70% accuracy** and **0.82 AUROC** on the recovery run, specifically **0.6982** accuracy and **0.8219** AUROC on **1090** scored examples from `data/gemma3_4b/probing/bioasq13b_factoid/report_recovery.json`, close to the paper’s BioASQ accuracy of **71.0** in `original-paper-markdown-converted.md`. Note: the detection interpretation is partially confounded by response-form/length correlations (see `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`).
- The recovery patch in `c41ee28` materially improved pipeline fidelity without materially changing the conclusion: answer-token rows **+19**, activation misses **18 -> 2**, accuracy **0.7019 -> 0.6982**, AUROC **0.8294 -> 0.8219**.
- BioASQ transfer is worse than the local TriviaQA disjoint eval for this detector: **0.6982 vs 0.7654** accuracy and **0.8219 vs 0.8429** AUROC, using `data/gemma3_4b/probing/bioasq13b_factoid/eval_qids.json` and `data/gemma3_4b/test_qids_disjoint.json`.

### Medium-confidence

- BioASQ appears harder mainly because the model hallucinates a lot in this single-response biomedical setting and because the detector overcalls verbose faithful answers — an effect consistent with the detector-side verbosity confound (see `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`). Evidence: raw BioASQ false rate **65.0%**, plus very high FP rates on long/list-like true answers.
- The recovery patch mostly restored formatting-heavy biomedical answers rather than “easy” generic cases. Evidence: recovered rows are longer and enriched for apostrophes, hyphens, parentheses, abbreviations, and list-like formatting.
- Some of the measured detector error on BioASQ is label-path-sensitive, because the GPT-4o judge accepts semantically broad faithful paraphrases that strict alias matching would miss. But the later representative audit indicates this is a minority effect, not the main driver of the detector’s BioASQ error profile.

## Claims we should avoid

- Do **not** say this is an exact paper-faithful replication of the Gemma-3-4B BioASQ row. The OOD shape is close, but the detector selection rule and label path are repo-specific.
- Do **not** say the detector is robust to BioASQ answer form. The strongest error pattern in this audit is the opposite: verbose/list-like faithful biomedical answers are overcalled as hallucinations.
- Do **not** say the recovery patch proved the detector got better or worse. The metric movement is too small, and the exact initial eval ID list was overwritten, so the cleanest supported statement is “coverage improved; headline performance stayed roughly flat.”

## Highest-value next analyses

- **Treat the representative BioASQ ground-truth audit as the current judge-quality reference.** The follow-on report in `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` replaces the earlier proposed top-error spot check with a representative weighted audit. Future work should extend that estimator if needed, not restart from a convenience sample.
- **Micro-sensitivity on the remaining 2 activation misses.** Implement only a punctuation-tolerant boundary matcher for `),` / `).` style cases, then rescore just those two examples. This is cheap and would close the last obvious span-recovery gap without blurring the target the way whole-output fallback would.
- **Preserve eval IDs for future OOD audits.** The exact initial-vs-recovery overlap analysis is limited because the initial `data/gemma3_4b/probing/bioasq13b_factoid/eval_qids.json` was overwritten. Future OOD runs should commit both initial and sensitivity eval ID lists so “same IDs only” comparisons are first-class rather than reconstructive.
