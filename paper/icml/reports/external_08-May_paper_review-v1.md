# External Scientific Review — ICML 2026 Mechanistic Interpretability Workshop

**Paper:** “Strong Readouts, Local Levers: A Gate-Based Steering Audit of Gemma-3-4B-IT”
**Review date:** May 8, 2026
**Package snapshot:** May 8, 2026
**Reviewer stance:** skeptical but constructive submission-readiness review, not an official ICML review.

## 0. Resolution Tracking

| Issue | Status | Commit | Concise state |
|---:|---|---|---|
| 1 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | `supplement/reference/main.tex` now matches the live manuscript, and the supplement package manifest builds `reference/main.tex` from `paper/icml/main.tex`; `tests/test_build_icml_supplement_package.py` guards both source and built-package equality. |
| 2 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | Figure 2, §3.2, Appendix Table 5, the cited SAE audit, and localization/provenance ledgers now separate the full-replacement SAE path movement from the delta-only feature-specific check; `make -C paper/icml`, figure regeneration, rendered-page inspection, and supplement-package tests guard the fix. |
| 3 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | §5 now separates the claim-bearing CSV-v2 strict-harmfulness slope from the CSV-v3 same-output sensitivity re-score and reports both slope/gap rows; `make -C paper/icml`, targeted stale-wording greps, and supplement reference equality tests guard the fix. |
| 5 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | The manuscript title/running title now foreground the four-gate audit, and the cross-family comparison is explicitly limited to a Gemma-local tested-operator result; `make -C paper/icml`, rendered-page inspection, stale-title greps, and supplement-package tests guard the fix. |

## 1. Verdict And Scorecard

**Submission-readiness verdict: _Major revision before submission_.**

The empirical core is workshop-relevant and close to defensible, but I would not submit this exact frozen package. The main blockers are: (i) the supplement contains a stale `reference/main.tex` that materially conflicts with the current `paper/main.tex`; (ii) Figure 2 makes the SAE null visually fragile unless the full-replacement/path-effect issue is made explicit; (iii) the jailbreak measurement section uses CSV-v2 for the claim-bearing graded effect while saying CSV-v3 is retained, but the included ledger shows CSV-v3 gives a much weaker analogous signal; and (iv) reproducibility/resources are documented in the supplement but not clearly surfaced in the main paper.

| Dimension | Rating | Rationale |
|---|---:|---|
| Workshop fit | 4 | Strong fit as a mechanistic-interpretability audit/negative-result case study: it uses internal states to test readout→control, control→externality, and measurement gates. It should not be framed as a practical intervention paper. |
| Soundness | 3 | Most headline numbers are traceable and the limitations are unusually honest. The remaining concerns are not fatal experiments, but they affect inference: operator/basis confounding, SAE full-replacement artifacts, rubric dependence, and stale supplement anchoring. |
| Presentation | 3 | The paper is readable and compact, with strong figures conceptually. It is over-compressed in places, the title under-sells/misframes the actual contribution, and the appendix/supplement boundary needs cleanup. |
| Significance | 3 | The contribution is useful for the workshop because it turns a common steering inference into auditable gates. The significance is empirical and methodological rather than a new method or broad mechanistic law. |
| Originality | 3 | Detector/control divergence, probe cautions, SAE-steering caveats, and evaluator fragility are not new. The distinctive part is the matched Gemma FaithEval readout/control comparison, bridge failure taxonomy, and gate-based reporting discipline. |
| Reproducibility/resources | 2 | Number provenance and reviewer-facing summaries are strong, and the supplement includes curated code/manifests. However, raw outputs are intentionally omitted, the main paper lacks a reproducibility paragraph, and the supplement’s manuscript anchor is stale. |
| Limitations honesty | 4 | Main text explicitly notes single-model scope, failed/incomplete Mistral gates, SIMID diagnostic-only status, operator-form confounds, LLM second-rater limits, and single-seed jailbreak specificity. This is one of the paper’s strongest features. |
| Overall recommendation | 3 | Weak reject / borderline as-is. With the fixes below, I would expect a workshop-appropriate weak accept because rigorous negative/audit results are in scope. |
| Confidence | 4 | I checked the rendered PDF, page images, `paper/main.tex`, `paper/number_provenance.md`, ground-truth ledgers, supplement manifests/support files, selected current reports, and references. I did not rerun code or inspect omitted raw JSONLs, so my judgment relies on included provenance and summaries. |

**Recommended title:** **“When Detectors Don’t Steer: A Four-Gate Audit of Activation Interventions in Gemma-3-4B-IT.”**
This is clearer than “Strong Readouts, Local Levers” because it names the actual falsifiable inference under audit. Alternatives, in descending order: “Prediction Is Not Control: A Four-Gate Audit of Activation Steering in Gemma-3-4B-IT”; “From Readout to Risk: Auditing Activation Steering in Gemma-3-4B-IT”; “Audit Before Steering: Measurement, Control, and Externality in Gemma-3-4B-IT.”

## 2. Blocking And Major Issues

### 1. Stale supplement manuscript anchor conflicts with the frozen current paper

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); the supplement reference TeX is synchronized with the current manuscript, the package manifest sources the built reference from `paper/icml/main.tex`, and the supplement-package tests guard against drift.

- **Severity:** Blocker if the supplement is uploaded as-is; otherwise Major.
- **Location:** `supplement/reference/main.tex:51-74`, `supplement/reference/main.tex:115-120`, `supplement/reference/main.tex:293-309`; current source is `paper/main.tex:51-76`, `paper/main.tex:116-125`, `paper/main.tex:311-338`, `paper/main.tex:711-759`. Supplement front matter also points reviewers to the stale anchor: `supplement/README.md:3-4,14-25`, `supplement/artifact_manifest.md:31-46`, `supplement/number_provenance.md:3`.
- **Problem:** The supplement claims `reference/main.tex` is the anonymized manuscript/section-numbering anchor, but that TeX file is not the current manuscript. It has the older title “A Steering Audit” rather than “A Gate-Based Steering Audit,” an older abstract, stronger/staler truncation language, weaker study-orientation caveats, and it lacks current appendix material such as the Mistral/SIMID stress-test ledger and revised limitations.
- **Evidence from package:** The package manifest defines `paper/main.pdf`, `paper/main.tex`, provenance, supplement, and ground-truth ledgers as the claim surface (`PACKAGE_MANIFEST.md:11-23,33-41`). The current paper says Mistral and SIMID are failed/incomplete appendix-only gates (`paper/main.tex:124-125`, `paper/main.tex:430-438`, `paper/main.tex:711-759`), while stale `supplement/reference/main.tex` lacks those current lines and still uses older measurement wording around truncation (`supplement/reference/main.tex:293-309`). The supplement README says the supplement is tied to `reference/main.tex` (`supplement/README.md:3-4,14-25`).
- **Recommended fix:** Replace `supplement/reference/main.tex` with the exact current anonymized `paper/main.tex`, then regenerate the supplement zip, `supplement/number_provenance.md`, and any section anchors. If the reference copy is not necessary, delete it and make all supplement documents refer directly to `paper/main.pdf` / `paper/main.tex` section names.
- **Fix type:** Reproducibility; structural; formatting/anonymity.

### 2. Figure 2 makes the central SAE null look under-explained

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); Figure 2 now labels the full-replacement SAE path and reports the delta-only near-zero slopes, while §3.2, Appendix Table 5, the cited SAE audit, and the localization/provenance ledgers state that the delta-only slope CI is not bundled.

- **Severity:** Major.
- **Location:** PDF p. 5, Figure 2; `paper/main.tex:172-178,201-208`; Appendix Table 5 at `paper/main.tex:512-531`; support summary `supplement/support/localization_summary.md:33-55`; provenance `paper/number_provenance.md:24-40`.
- **Problem:** The text says the tested SAE intervention is null, but Figure 2 and Appendix Table 5 show SAE H-feature and SAE-random compliance rates that are high at non-no-op alphas and reset at the no-op alpha. A skeptical reviewer can reasonably ask whether the SAE hook/reconstruction path is causing generic output perturbation and whether the plotted full-replacement curve is the right visual for a “no control” claim.
- **Evidence from package:** Main text reports full-replacement SAE slope `+0.16` pp/α with CI crossing zero and no monotonic trend, then says delta-only H-feature and random-feature slopes are near zero (`paper/main.tex:172-175`). Appendix rates show SAE H-feature: 72.3%, 74.7%, 66.0%, 75.0%, 75.1%, 74.9%, 69.9% across α, and SAE random: 74.9%, 74.8%, 66.0%, 75.0%, 74.9%, 74.9%, 74.6% (`paper/main.tex:522-528`). The support file explicitly reports a delta-only null but no CI (`supplement/support/localization_summary.md:44-48`).
- **Recommended fix:** Do not let the figure carry an ambiguous inference. Replot Figure 2 with separate panels for full-replacement and delta-only SAE intervention, or annotate the current plot as “full-replacement SAE path; non-monotone generic path effect; feature-specific delta-only controls remain null.” Add one sentence in §3.2: “The full-replacement SAE curve shows non-monotone path/reconstruction movement shared by target and random SAE sets; the claim-bearing feature-specific check is the delta-only contrast, where H-feature and random-feature slopes remain near zero.” Also add CIs for the delta-only slopes if available; if not, say “audit summary; CI not in package.”
- **Fix type:** Figure/table; wording-only if no replot is possible.

### 3. CSV-v2 versus CSV-v3 is not disciplined enough for the jailbreak measurement claim

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); §5 now states that CSV-v2 is the claim-bearing same-output strict-harmfulness endpoint, retains CSV-v3 for taxonomy/evaluator-construct reporting, reports the smaller CSV-v3 slope/gap point estimates, and weakens the supported conclusion to rubric sensitivity.

- **Severity:** Major.
- **Location:** `paper/main.tex:318-338`, Table 2 at `paper/main.tex:340-357`; `supplement/support/measurement_summary.md:11-23,35-53`; `evidence/ground-truth/metric_tables_only.md:156-185`; `paper/number_provenance.md:101-118`.
- **Problem:** The main text’s claim-bearing graded result is CSV-v2: H-neuron slope `+2.30` pp/α, random `-0.47`, gap `+2.77`, `p=0.013`. Immediately afterward, the paper says CSV-v3 is retained for richer taxonomy. The included ground-truth ledger shows an analogous CSV-v3 H slope of only `+0.46` pp/α and H-minus-random gap `+0.80` pp/α. Without explanation, this looks like rubric cherry-picking or a stale analysis boundary.
- **Evidence from package:** Main text uses CSV-v2 for the positive graded effect (`paper/main.tex:323-325`) and CSV-v3 for retained evaluator taxonomy (`paper/main.tex:330-338`). The support summary says the graded CSV2 surface detects a trend but that CSV-v3 and StrongREJECT tie on holdout accuracy (`supplement/support/measurement_summary.md:15-22,35-53`). The ground-truth table records the weaker CSV-v3 slope/gap (`evidence/ground-truth/metric_tables_only.md:178-185`).
- **Recommended fix:** Add a clarifying sentence before Table 2: “The claim-bearing same-output granularity audit used the pre-existing CSV-v2 strict-harmfulness endpoint; CSV-v3 is retained only for taxonomy/evaluator-construct reporting. Re-scoring the same intervention with CSV-v3 yields a smaller, non-headline signal (careful with model counfound), so our supported claim is rubric sensitivity, not CSV-v3-confirmed control.” Better still, report both CSV-v2 and CSV-v3 slopes in a tiny table and weaken the conclusion to: “graded/rubric choices materially change whether the effect gate appears to pass.”
- **Fix type:** Wording-only plus optional figure/table.
IMPORTANT: pull from ground truth data in the repo for this, we graded alot on V3 but that may not be reflected in paper, also Mistral had different profile than Gemma.

### 4. Main paper does not surface reproducibility/access clearly enough

- **Severity:** Major.
- **Location:** Main body has no explicit reproducibility paragraph before references (`paper/main.tex:116-125`, `paper/main.tex:428-462`); appendix claim-defense index mentions support files only at `paper/main.tex:653-658`; supplement details are in `supplement/README.md:19-25`, `supplement/artifact_manifest.md:35-78`, `supplement/reproduction_manifest.md:84-145`, and `supplement/evaluation_manifest.md:146-220`.
- **Problem:** Workshop reviewers are specifically asked to consider reproducibility and code/data/prompt/model access. The supplement is reasonably rich, but the main paper does not tell reviewers what exists. If a reviewer does not read the appendix or supplement deeply, reproducibility may be under-credited or misunderstood.
- **Evidence from package:** The supplement is a curated code/data bundle, not a full raw-output release (`supplement/README.md:19-25`). It lists included code, manifests, judge summaries, and intentional exclusions (`supplement/artifact_manifest.md:35-78`) and gives a minimal rerun map (`supplement/reproduction_manifest.md:84-145`). Raw response/scored JSONLs and harmful prompt gold labels are omitted for safety/anonymization (`supplement/artifact_manifest.md:65-78`; `supplement/reproduction_manifest.md:141-145`).
- **Recommended fix:** Add a short main-paper paragraph after “Study orientation” or before limitations:

  > **Reproducibility resources.** The anonymous supplement contains a curated code/test slice, safe manifests, judge/rubric manifests, compact metric ledgers, bridge IRR summaries, and rerun maps for the paper-critical analyses. It is not a full raw-output release: raw response/scored JSONLs, harmful prompt gold labels, and provenance sidecars are omitted for safety/anonymization; derived summaries and safe manifests are provided for traceability.

- **Fix type:** Reproducibility; wording-only.

### 5. Title and framing still imply a narrower/readout-centric contribution than the paper actually makes

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); the title/running title now use “When Detectors Don’t Steer,” and the related-work framing states that the cross-family comparison is Gemma-local and operator-specific, with no operator-matched causal-basis proof.

- **Severity:** Major for reviewer first impression; Minor for scientific validity.
- **Location:** Title/running title at `paper/main.tex:47-53`; abstract at `paper/main.tex:69-76`; framework at `paper/main.tex:362-426`; limitations at `paper/main.tex:430-439`.
- **Problem:** “Strong Readouts, Local Levers” is memorable but too narrow and too close to a truism. It foregrounds one localization/control result, while the paper’s stronger contribution is a four-gate audit in which measurement and externality are equally important. It may also invite reviewers to treat the paper as claiming “readouts are strong but levers are local” rather than “specific steering claims require separate gate evidence.”
- **Evidence from package:** The contribution list has five pieces, only one of which is the readout/control comparison (`paper/main.tex:110-114`). The framework and recommendations are explicitly about measurement, localization, control, and externality (`paper/main.tex:362-426`). The limitations explicitly demote broad detector/control novelty (`paper/main.tex:439`).
- **Recommended fix:** Retitle to **“When Detectors Don’t Steer: A Four-Gate Audit of Activation Interventions in Gemma-3-4B-IT.”** Add one footnote-level clause: “The cross-family comparison is not an operator-matched proof that neurons are better causal bases than SAE features; it is a Gemma-local audit showing that comparable detector AUROC failed to predict control under the tested operators.”
- **Fix type:** Wording-only; structural framing.

### 6. Appendix is too long and inefficient for a workshop submission even if the formal page limit is met

- **Severity:** Major for reviewer legibility; Minor for formal compliance.
- **Location:** Appendix begins at `paper/main.tex:473-475`; rendered pages `rendered-pages/page-11.png` through `rendered-pages/page-16.png`; claim-defense index `paper/main.tex:653-708`; Mistral/SIMID stress table `paper/main.tex:711-764`; limitation inventory `paper/main.tex:767-789`.
- **Problem:** The main paper carries the core argument, but the appendix is six rendered pages and includes material that is closer to supplement navigation than paper evidence. Workshop reviewers are unlikely to read unlimited appendices, and mentor feedback says 2–3 appendix pages are the practical upper bound.
- **Evidence from package:** The appendix contains detailed construct map, FaithEval rates, SAE selector table, power summary, bridge margin analysis, claim-defense index, Mistral/SIMID stress-test ledger, and limitation inventory (`paper/main.tex:486-789`). The claim-defense index largely duplicates what the supplement is for (`paper/main.tex:653-708`; `supplement/artifact_manifest.md:35-64`).
- **Recommended fix:** Keep only appendix blocks that materially protect claims: construct map, FaithEval rates with SAE-path annotation, SAE selector table, bridge margin table. Move the claim-defense index and full Mistral/SIMID stress ledger to the supplement. In the main limitations, keep the one-paragraph Mistral/SIMID summary already present (`paper/main.tex:430-438`).
- **Fix type:** Structural.

### 7. Bridge failure coding is defensible but should avoid “first author” and human-IRR overtones

- **Severity:** Minor to Major depending on reviewer sensitivity.
- **Location:** `paper/main.tex:253-262`, limitations `paper/main.tex:435-436`, supplement `supplement/failure_coding_manifest.md:40-50,65-69`, support summary `supplement/support/externality_summary.md:71-96`.
- **Problem:** The bridge taxonomy is a strong behavioral diagnosis, but one rater is the first author and the second rater is GPT-4o. The paper mostly states this honestly, but “first author” is awkward in double-blind review and “dual-rated” can sound stronger than author+LLM sensitivity checking.
- **Evidence from package:** Main text says the 57 discordant cases were coded by the first author and a blinded LLM judge (`paper/main.tex:256-257`) and limitations correctly state the second rater is an LLM with limited force (`paper/main.tex:435-436`). The failure manifest repeats that Rater A is the first author and Rater B is GPT-4o (`supplement/failure_coding_manifest.md:40-50`).
- **Recommended fix:** Replace “first author” with “one author.” Replace “dual-rated” with “author-coded with blinded LLM adjudication/sensitivity check” in at least one place, while retaining the numeric agreement. Suggested wording: “One author and a blinded GPT-4o judge independently coded all 57 discordant cases under a frozen four-category rule; because the second rater is not human, we treat agreement as a robustness check rather than human IRR.”
- **Fix type:** Wording-only; formatting/anonymity.

### 8. MDE inconsistency between construct map and power appendix

- **Severity:** Minor.
- **Location:** Construct map says JailbreakBench binary evaluation is underpowered with MDE `~6` pp (`paper/main.tex:489-506`, especially line 502). Power summary says JailbreakBench MDE `~5` pp (`paper/main.tex:593-606`).
- **Problem:** This is small, but it gives a reviewer an easy “sloppy numbers” complaint.
- **Evidence from package:** `paper/main.tex:502` versus `paper/main.tex:605`.
- **Recommended fix:** Use the same approximate value or distinguish binary endpoint MDE from graded slope MDE. Suggested fix: “Binary endpoint MDE `~6` pp; graded slope table MDE `~5` pp” if both are correct.
- **Fix type:** Wording-only.

### 9. Supplement directory/source-vs-built package can confuse reviewers about where code lives

- **Severity:** Minor.
- **Location:** `supplement/README.md:14-17`, `supplement/artifact_manifest.md:45-50`; actual source tree shows code under `supplement/build/icml_supplement_package/code/`, while the top-level `supplement/` source directory does not have a `code/` directory.
- **Problem:** The README says `code/` is generated by the supplement builder, but if a reviewer opens the source supplement directory rather than the built upload package, they will not see top-level `code/`.
- **Evidence from package:** The built package contains `supplement/build/icml_supplement_package/code/`, but the top-level source tree contains `supplement/build`, `supplement/data`, `supplement/reference`, and `supplement/support` only.
- **Recommended fix:** Upload the built supplement zip, not the source directory. Also edit `supplement/README.md` to say “In the built upload package, `code/` contains…” or include a top-level generated `code/` directory in the frozen supplement folder.
- **Fix type:** Reproducibility; structural.

## 3. Workshop-Fit Audit

**Does it further mechanistic interpretability?** Yes. The paper is squarely about whether internal model states used as readouts can support intervention claims. It does not merely evaluate benchmark performance; it asks when internal-state predictors become control handles and when that inference fails. The strongest workshop framing is: a rigorous empirical audit/negative result for activation steering claims, with a practical checklist for reporting.

**Best framing.** This is not a practical intervention paper and should not be sold as one. It is best framed as a **Gemma-local audit framework and empirical case study with rigorous negative/null results**. The positive H-neuron FaithEval result is a control-gate anchor, but the center of gravity is gate discipline: measurement, localization, control, externality.

**Falsifiable hypotheses and evidence boundaries.** The current introduction does this reasonably well. The paper asks bounded questions about readout quality, answer-selection transfer, and measurement stability (`paper/main.tex:120-124`) and states that Mistral/SIMID artifacts are failed or incomplete gates (`paper/main.tex:125`). The limitations further constrain single-model scope, Mistral stress tests, SAE operator confounds, LLM second-rater limits, single-seed jailbreak specificity, and SIMID diagnostic-only status (`paper/main.tex:430-439`). This is venue-appropriate.

**Limitations.** The limitations are unusually mature and should be preserved. Do not “clean up” the narrative by hiding failed gates. The workshop explicitly accepts negative results, critiques, tools, datasets, and position-style contributions; honesty is a strength here.

**Main-paper self-containment.** Mostly adequate, but not yet robust enough. The main paper contains the FaithEval, externality, measurement, limitation, and checklist core. However, Figure 2’s SAE interpretation and the CSV-v2/CSV-v3 measurement boundary need to be self-contained because reviewers may not read Appendix D or the supplement. The reproducibility resources should also be stated in the main body.

**Reproducibility/code/data/prompt access.** The supplement is credible as a curated reviewer-facing artifact: it includes manifests, support summaries, judge prompts, holdout comparison JSON, bridge IRR summary, a rerun map, and a code/test slice in the built package (`supplement/artifact_manifest.md:35-64`; `supplement/reproduction_manifest.md:84-145`; `supplement/evaluation_manifest.md:146-220`). It intentionally omits raw JSONLs, harmful prompts, and provenance sidecars (`supplement/artifact_manifest.md:65-78`). That is defensible for workshop review if the main paper says so explicitly and if the stale supplement manuscript anchor is fixed.

**Double-blind and page limit.** The rendered submission appears double-blind: `paper/main.pdf` metadata reports anonymous author fields, the TeX uses anonymous author/affiliation fields (`paper/main.tex:55-60`), and I did not find visible GitHub/HuggingFace usernames or acknowledgments in the paper/supplement source. “First author” in bridge coding should be changed to “one author” (`paper/main.tex:256`). The PDF has 16 pages total; the main body and impact statement end on page 8, references occupy pages 8–10, and the one-column appendix begins on page 11 (`paper/main.tex:464-475`; rendered pages `page-08.png` through `page-11.png`). This satisfies an 8-page long-paper limit excluding references/appendix, but it is not a short paper.

**Exact venue-fit edits.**

1. Retitle to “When Detectors Don’t Steer: A Four-Gate Audit of Activation Interventions in Gemma-3-4B-IT.”
2. Add the reproducibility paragraph proposed above.
3. Add one sentence in the abstract or Introduction: “We present an audit framework and negative-result case study, not a new steering method or a model-general claim.” The paper already says this in substance (`paper/main.tex:116-125`); foreground it.
4. Make the CSV-v2/CSV-v3 boundary explicit in §5.
5. Compress the appendix and move navigational support tables to the supplement.

## 4. Claim Support Audit

| Claim or passage | Verdict | Evidence checked | Risk | Recommended edit |
|---|---|---|---|---|
| “Strong readouts” versus control: H-neuron and SAE readouts comparable, but only H-neuron scaling controls FaithEval. | **Supported, but narrow.** The readout and intervention numbers are traceable. | `paper/main.tex:157-178`; `paper/number_provenance.md:13-40`; `supplement/support/localization_summary.md:18-55`; `evidence/ground-truth/metric_tables_only.md:7-38,56-68,86-94`. | Cross-family comparison confounds basis, operator form, layer coverage, and feature granularity. Figure 2 can be read as SAE generic path movement rather than clean null. | Keep claim Gemma-local and operator-specific. Add figure annotation and one sentence that the cross-family comparison is not an operator-matched causal-basis test. |
| H-neuron FaithEval result: 38 neurons, AUROC 0.843, slope +2.09 pp/α, random controls null. | **Well-supported.** | `paper/main.tex:157-170,219-222`; `paper/number_provenance.md:15-34`; `supplement/support/localization_summary.md:20-43`; `evidence/ground-truth/metric_tables_only.md:58-68`. | It is a context-faithfulness/anti-compliance surface, not general factuality or truthfulness. | Preserve current wording that FaithEval is surface-specific (`paper/main.tex:119`). Avoid “truthfulness improvement” language. |
| SAE readout/control and SAE target-selection evidence. | **Supported, but easy to overread.** SAE readout quality and selector/margin evidence are traceable; no SAE behavioral endpoint passes. | `paper/main.tex:158-159,172-187,196-198`; Appendix D `paper/main.tex:534-586`; `paper/number_provenance.md:19-22,35-51`; `evidence/ground-truth/metric_tables_only.md:90-129`; reports `2026-04-22-faitheval-sae-utility-selector-review.md` and `2026-04-25-faitheval-answer-span-extension.md`. | “SAEs don’t steer” would be overbroad; evidence covers specific Gemma Scope layers, candidate pool, and operators. Main paper relies on compressed Appendix D for selector nuance. | Say “under the tested Gemma Scope layers/operators.” In the main text, add that selector success is margin-level only and does not reach compliance. Current `paper/main.tex:180-187` mostly does this; keep it. |
| TruthfulQA MC versus open-generation transfer. | **Well-supported as a transfer failure/externality claim.** | `paper/main.tex:230-237,241-263`; `paper/number_provenance.md:69-80,85-99`; `supplement/support/externality_summary.md:29-80`; `evidence/ground-truth/metric_tables_only.md:291-305,349-381`. | TruthfulQA MC gain is not a general truthfulness gain. SimpleQA local prompt removes the explicit escape hatch, so it is a stress surface, not direct benchmark replication. | Keep “answer selection” language. Add “local copy/prompt stress test” wherever SimpleQA is summarized in figures/captions. |
| TriviaQA bridge wrong-entity substitution. | **Supported as behavioral diagnosis; not mechanistic circuit evidence.** | `paper/main.tex:239-263`; Appendix margin table `paper/main.tex:612-649`; `paper/number_provenance.md:81-99`; `supplement/support/externality_summary.md:49-96`; `evidence/paper-reports/2026-04-21-bridge-irr-review.md:3-18,49-58,106-145,185-200`; `evidence/paper-reports/2026-04-21-bridge-margin-report.md` via appendix summary. | Second rater is GPT-4o, not a second human. Margin analysis explicitly does not find a substitution-specific signature. | Replace “first author” with “one author.” Keep “behavioral diagnosis without claiming an internal substitution circuit” (`paper/main.tex:260-262`). |
| BioASQ and FalseQA effects. | **Supported.** H-neuron intervention is active on compliance-adjacent surfaces, but BioASQ factual accuracy is flat. | `paper/main.tex:219-226`; `paper/number_provenance.md:57-65`; `supplement/support/externality_summary.md:15-27`; `evidence/ground-truth/metric_tables_only.md:72-84`. | FalseQA should not be treated as open-ended factual improvement; BioASQ perturbation without accuracy gain weakens broad utility claims. | Keep as supporting surface-locality evidence, not a headline. Current wording is appropriate. |
| JailbreakBench measurement sensitivity. | **Supported, but currently under-disciplined.** | `paper/main.tex:301-338`; `paper/number_provenance.md:101-118`; `supplement/support/measurement_summary.md:11-58`; `evidence/ground-truth/metric_tables_only.md:156-185`; Mistral measurement stress `2026-05-06-mistral24b-anchor3-jailbreak-measurement-review.md:3-22,108-109,180-286`. | CSV-v2 is the positive graded result, while CSV-v3 is retained for taxonomy and gives weaker same-output slope. Single-seed p=0.013 specificity remains limited. | Report CSV-v2 and CSV-v3 boundary in main. Say rubric/granularity changed the verdict, not “the graded evaluator establishes the true effect.” |
| SIMID calibration and prospective labels. | **Correctly diagnostic-only.** | `paper/main.tex:438,755-759`; `PACKAGE_MANIFEST.md:39`; `evidence/ground-truth/metric_tables_only.md:229-245,323-343`; final strategy `evidence/paper-reports/final_paper_sprint_strategy.md:27`; SIMID reports `2026-04-28-simid-open-calibration-review.md` and `2026-05-03-simid-prospective-partial-external-label-review.md`. | Any use as truthfulness-improvement evidence would be unsupported. Historical MVP numbers and partial prospective selected-TruthfulQA numbers should not be upgraded. | Keep only in appendix/limitations. Do not mention in abstract. Current treatment is mostly correct. |
| Mistral anchor 1 stress tests: sparse readout and FaithEval null. | **Supported only as failed/incomplete gate evidence.** | `paper/main.tex:430-431,711-741`; `PACKAGE_MANIFEST.md:40`; final strategy `evidence/paper-reports/final_paper_sprint_strategy.md:28`; Mistral reports `2026-04-29-mistral24b-cp23-pipeline-review.md`, `2026-04-30-mistral24b-cp5-faitheval-review.md`, `2026-04-30-mistral24b-h1-c-sweep-review.md`. | Not a Mistral replication or explanation of why Gemma differs. | Keep in limitations/appendix only. Current wording is good. |
| Mistral anchor 2 TruthfulQA MC. | **Supported as failed source-surface gate.** | `paper/main.tex:743-747`; `evidence/paper-reports/2026-05-07-mistral24b-anchor2-truthfulqa-mc-review.md:3-14,143-166,217-240`; `evidence/ground-truth/metric_tables_only.md:311-318`. | Cannot support a Mistral bridge claim because wrapper stopped before bridge. | Keep “MC1 source gate failed; no bridge externality claim.” Do not hide this; it improves credibility. |
| Mistral anchor 3 JailbreakBench. | **Supported as measurement stress test, not specificity/control claim.** | `paper/main.tex:749-753`; `evidence/paper-reports/2026-05-06-mistral24b-anchor3-jailbreak-measurement-review.md:3-22,108-109,180-286`; `evidence/ground-truth/metric_tables_only.md:191-214`. | Large alpha curve lacks matched random/layer controls, so it cannot upgrade the Gemma control claim. | Keep as appendix/limitation only. Main text should say it constrains the Gemma truncation lesson rather than generalizing it. |
| Audit-framework contribution and novelty. | **Supported, but novelty should be modest.** | `paper/main.tex:95-114,133-147,362-426,439`; related work in `paper/references.bib`; strategy memo `evidence/paper-reports/final_paper_sprint_strategy.md:10-16,34-42`. | The general detector/control warning is already in probe/SAE/steering literature. The novelty is the integrated audit and matched behavioral case study. | Frame as “a stricter audit scaffold plus matched cross-representational Gemma case study,” not as discovery that readouts and control can diverge. |

## 5. PDF, Format, And Anonymity Audit

**Page count and boundary.** The PDF is 16 pages total (`PACKAGE_MANIFEST.md:13-16`). The main body runs through the impact statement on page 8; references start on page 8 and continue through page 10; the appendix begins after `\clearpage`, `\onecolumn`, `\appendix` at `paper/main.tex:473-475` and appears on rendered page 11. This appears compliant for an 8-page ICML long paper excluding references and appendices. It is not compliant as a 4-page short paper.

**Rendered layout.** I checked `rendered-pages/page-01.png` through `rendered-pages/page-16.png`.

- Pages 1–2: readable title/abstract/intro and Figure 1. Figure 1 succeeds as a scaffold overview, but its caption is long.
- Page 5: Figure 2 is visually readable but scientifically ambiguous because SAE curves show non-monotone path movement; this is the main figure-level issue.
- Page 6: Figure 3 is readable and carries the externality result well.
- Page 7: Table 1 examples and Table 3 are readable but dense; long question cues wrap heavily.
- Page 8: conclusion/impact/references boundary is acceptable.
- Pages 11–16: appendix is legible in one-column form, but long relative to workshop expectations. Pages 13/14/16 have noticeable whitespace/low information density.

**Anonymity.** The TeX uses anonymous author, affiliation, and email fields (`paper/main.tex:55-60`). The PDF metadata is anonymous at the author level. I did not find obvious GitHub usernames, HuggingFace usernames, acknowledgments, local `/home/...` paths, or author names in the paper/supplement source outside normal bibliography entries. The phrase “first author” should be changed to “one author” for double-blind cleanliness (`paper/main.tex:256`; `supplement/failure_coding_manifest.md:40`). If any cited prior work is by the submission authors, keep all self-citations in third person and avoid anonymous-breaking repository links.

**Bibliography and citation formatting.** The bibliography appears to render normally; no undefined-citation issue was apparent from the package logs. Related citations are broad and venue-appropriate. The references include several 2025/2026 related-work anchors; the paper’s novelty language correctly avoids claiming detector/control divergence as wholly new (`paper/main.tex:142-147,439`).

**Supplement discoverability.** The appendix claim-defense index maps claims to support files (`paper/main.tex:653-708`), but this is the wrong place to spend paper appendix pages. Add a main-body reproducibility paragraph and move claim-defense navigation to the supplement. Also fix the stale supplement reference TeX before upload.

## 6. Figure, Table, And Appendix Audit

### Figure 1 — Four-stage scaffold

- **What it tries to show:** The paper’s conceptual gate sequence: measurement, localization, control, externality.
- **Whether it succeeds:** Yes. It helps reviewers understand the contribution as a reporting/audit discipline rather than a bag of benchmark results.
- **Inconsistency/risk:** Caption is long and the figure could be interpreted as a universal framework rather than a case-study-derived checklist.
- **Exact improvement:** In the caption, add “case-study-derived” or “reporting scaffold” and remove one subordinate clause. Keep it in the main paper.

### Figure 2 — FaithEval readout/control anchor

- **What it tries to show:** Similar H-neuron/SAE readout AUROC but divergent behavioral control.
- **Whether it succeeds:** It succeeds for readout comparability and H-neuron monotonicity. It partially fails for SAE interpretation because the full-replacement SAE curves show visible non-monotone movement.
- **Inconsistency/risk:** Appendix Table 5 shows SAE H-feature and SAE-random rates far from the no-op baseline at several alphas (`paper/main.tex:522-528`), while the caption says SAE/random show no reliable monotonic dose-response (`paper/main.tex:204-207`). That is statistically compatible but visually fragile.
- **Exact improvement:** Replot with delta-only SAE as the claim-bearing SAE control panel, or add an inset/table: “Full-replacement SAE path movement is non-monotone and shared by random SAE controls; delta-only target-specific slopes are +0.12 and -0.09 pp/α.”

### Figure 3 — Surface-local control and bridge failure modes

- **What it tries to show:** TruthfulQA MC gain, generation-surface harm, and bridge failure taxonomy.
- **Whether it succeeds:** Yes. It is probably the strongest reviewer-facing figure.
- **Inconsistency/risk:** SimpleQA is a local copy with a modified prompt that removes the explicit escape hatch (`paper/main.tex:235-236`). The figure caption says “generation surfaces” but should remind readers that SimpleQA is a stress surface.
- **Exact improvement:** Add “SimpleQA stress prompt” or “local SimpleQA stress surface” to the caption. Keep the bridge taxonomy as behavioral, not mechanistic.

### Table 1 — Representative wrong-entity substitutions

- **What it tries to show:** Qualitative texture of the bridge failure mode.
- **Whether it succeeds:** Yes. The examples make the error mode concrete.
- **Inconsistency/risk:** The table is dense in the two-column layout and long question cues wrap. This is tolerable but not elegant.
- **Exact improvement:** Keep 3 examples in main and move the 4th to supplement if space is needed. Ensure no raw benchmark license/privacy issue from quoting questions.

### Table 2 — Evaluator holdout accuracy

- **What it tries to show:** CSV-v3 and StrongREJECT tie on a clean holdout; no evaluator superiority claim.
- **Whether it succeeds:** Mostly. It supports construct pluralism.
- **Inconsistency/risk:** It sits immediately after a CSV-v2 claim-bearing effect, which makes the v2/v3 boundary confusing.
- **Exact improvement:** Rename table caption to “Evaluator holdout accuracy; not the claim-bearing slope estimator” or add a pre-table sentence clarifying CSV-v2 versus CSV-v3 roles.

### Table 3 — Minimum audit checklist

- **What it tries to show:** Practical reporting requirements for activation-steering claims.
- **Whether it succeeds:** Yes. This is the clearest venue-fit artifact.
- **Inconsistency/risk:** It could be read as a prescriptive universal standard from one case study.
- **Exact improvement:** Add “minimum audit we recommend based on this case study and related failures” in caption or surrounding text.

### Appendix Table 4 — Construct map

- **What it tries to show:** Distinct measurement constructs across FaithEval, TruthfulQA MC, TriviaQA, JailbreakBench, BioASQ.
- **Whether it succeeds:** Yes. It protects against benchmark flattening.
- **Inconsistency/risk:** JailbreakBench MDE says `~6` pp here (`paper/main.tex:502`) but `~5` pp in the power table (`paper/main.tex:605`).
- **Exact improvement:** Align the MDE values or label them as binary-endpoint versus graded-slope MDE.

### Appendix Table 5 — FaithEval detailed rates

- **What it tries to show:** Raw rates behind Figure 2.
- **Whether it succeeds:** It is useful, but it exposes the SAE path-effect ambiguity.
- **Inconsistency/risk:** Without explanatory text, readers may think the SAE intervention has large generic effects despite “null” slope.
- **Exact improvement:** Add a note under the table: “SAE full-replacement rates include generic non-monotone path/reconstruction movement; slope and delta-only contrasts are the relevant target-specific tests.”

### Appendix Table 6 — SAE target-selection ablation

- **What it tries to show:** Readout, prompt-end utility, and answer-span selectors move margins differently but do not recover compliance.
- **Whether it succeeds:** Yes; this is important claim support.
- **Inconsistency/risk:** It is too important to be appendix-only if reviewers challenge SAE target selection.
- **Exact improvement:** Keep the main text’s compressed paragraph (`paper/main.tex:180-187`) and add one small parenthetical: “full details in Appendix D; all compliance CIs include zero.”

### Appendix Table 7 — Power summary

- **What it tries to show:** Which nulls are well-powered.
- **Whether it succeeds:** Yes, with the MDE caveat.
- **Inconsistency/risk:** MDE mismatch with construct map.
- **Exact improvement:** Align values and state whether MDE is rate endpoint or slope endpoint.

### Appendix Table 8 — Bridge margin analysis

- **What it tries to show:** Teacher-forced margin shifts refine the bridge behavioral taxonomy.
- **Whether it succeeds:** Yes, and it is scientifically mature because it explicitly says the substitution-specific prediction is reversed (`paper/main.tex:646-649`).
- **Inconsistency/risk:** None material if kept as appendix evidence.
- **Exact improvement:** Keep the “Unsupported reading” paragraph. It prevents mechanistic overclaiming.

### Appendix Table 9 — Claim-defense index

- **What it tries to show:** Where each claim is supported in the supplement.
- **Whether it succeeds:** Useful for authors/rebuttal, not for paper appendix.
- **Inconsistency/risk:** Consumes valuable appendix space and duplicates supplement function.
- **Exact improvement:** Move to `supplement/` and reference it in the main-paper reproducibility paragraph.

### Appendix Table 10 — Mistral and SIMID stress-test ledger

- **What it tries to show:** Generalization constraints and failed/incomplete gates.
- **Whether it succeeds:** Scientifically, yes; presentation-wise, it is dense.
- **Inconsistency/risk:** Readers may treat it as extra claims rather than limitation evidence.
- **Exact improvement:** Move the full table to supplement; keep only a one-sentence limitations summary in main and, if desired, a short appendix paragraph.

### Appendix Table 11 — Limitation inventory

- **What it tries to show:** Compact constraint list.
- **Whether it succeeds:** Yes, but it largely duplicates the main limitations paragraph.
- **Inconsistency/risk:** Low.
- **Exact improvement:** Keep if appendix space remains; otherwise move to supplement and preserve main limitations.

## 7. Related Work And Novelty

The related-work section is broadly fair and well-targeted. It covers probe critiques and control-task concerns (`paper/main.tex:133`), representation engineering/activation addition/ITI and activation steering (`paper/main.tex:135-137`), SAE steering and selection (`paper/main.tex:135-136`), evaluator/judge fragility (`paper/main.tex:139`), factuality spillover/tradeoff work (`paper/main.tex:140`), and concurrent SAE/synthetic-benchmark divergence work (`paper/main.tex:142-147`). The bibliography includes the relevant local keys in `paper/references.bib`, including FaithEval, TruthfulQA, TriviaQA, JailbreakBench, StrongREJECT, Gemma Scope, SAE steering, AxBench, and recent evaluator papers.

The novelty framing should remain modest. The paper is not the first to observe that probes/readouts do not imply causal control, that SAE feature salience is not steering utility, or that evaluator choice matters. The current manuscript mostly acknowledges this (`paper/main.tex:142-147,439`). The defensible novelty is narrower:

1. A matched Gemma FaithEval comparison where H-neuron and SAE readout AUROC are close but behavioral control diverges under tested operators.
2. A within-SAE target-selection audit showing metric-specific margin movement without endpoint recovery.
3. A TruthfulQA MC to TriviaQA bridge externality diagnosis with frozen failure coding.
4. A measurement audit showing that scoring granularity/rubric choices can alter whether a steering effect appears to clear the gate.
5. A practical four-gate checklist for reporting activation-steering claims.

I would not add many more citations before the deadline unless the authors know of a directly overlapping 2026 workshop/preprint. The higher-value related-work edit is framing: state that the paper is an empirical audit that integrates already-known cautions into one reviewer-legible protocol, not a discovery of detector/control divergence in general.

## Final pre-submission checklist

1. [x] Replace/regenerate `supplement/reference/main.tex` and the built supplement zip so the supplement matches the current paper ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD)).
2. [x] Change title to “When Detectors Don’t Steer: A Four-Gate Audit of Activation Interventions in Gemma-3-4B-IT.” ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD))
3. [ ] Add the main-body reproducibility paragraph.
4. [x] Fix Figure 2 or its caption/text to separate full-replacement SAE path movement from delta-only target-specific nulls ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD)).
5. [x] Clarify CSV-v2 versus CSV-v3 roles in §5, ideally with both slopes/gaps reported ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD)).
6. [ ] Change “first author” to “one author” in bridge coding language.
7. [ ] Align JailbreakBench MDE numbers.
8. [ ] Move claim-defense index and full Mistral/SIMID stress ledger to supplement if appendix length must be reduced.
9. [ ] Upload the built supplement package, not the source supplement directory, and verify the `code/` directory is visible in the uploaded artifact.
10. [ ] Re-render and re-check `paper/main.pdf`, especially pages 5, 8, and appendix boundaries.
