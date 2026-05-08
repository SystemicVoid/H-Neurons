# External Scientific Review: “Strong Readouts, Local Levers”

Review date: May 7, 2026. Package treated as a frozen snapshot. Reviewed `paper/main.pdf`, `rendered-pages/page-01.png` through `page-17.png`, `paper/main.tex`, `paper/main.log`, `paper/number_provenance.md`, `supplement/number_provenance.md`, supplement manifests/support files, `evidence/ground-truth/metric_tables_only.md`, recent paper reports, source-summary JSON where needed, local literature index, and `paper/references.bib`.

## 0. Resolution Tracking

| Issue | Status | Commit | Concise state |
|---:|---|---|---|
| 1 | Addressed | [`b6a40bd`](https://github.com/SystemicVoid/H-Neurons/commit/b6a40bd79b24e6991bb753adfb630a39d1631332) | Supplement number provenance now matches the bridge IRR source and support files: 31/43 wrong-entity, 9/43 evasion, 3/43 dilution, 0/43 formal refusal, plus 55/57 agreement, κ = 0.90, AC1 = 0.96. `tests/test_bridge_irr.py::test_paper_bridge_taxonomy_provenance_matches_irr_summary` guards the cross-file consistency. |
| 2 | Addressed | [`c89470d`](https://github.com/SystemicVoid/H-Neurons/commit/c89470d9dbf515eb0e64d4f68409c108b3607c99) | Appendix floats now render in source order via a one-column appendix, section barriers, and rebuilt `paper/icml/main.pdf`; `make -C paper/icml`, rendered-page inspection, and LaTeX log checks guard the fix. |
| 3 | Addressed | [`d4c69fa`](https://github.com/SystemicVoid/H-Neurons/commit/d4c69fa27623388a86a18d2f6f16bb07d65a65b3) | Figure generators now escape TeX percent labels, disable label clipping, add export padding, and regenerate Figures 2-3 plus `paper/icml/main.pdf`; script reruns, rendered-page inspection, and LaTeX log checks guard the fix. |
| 4 | Addressed | [`ff5faea`](https://github.com/SystemicVoid/H-Neurons/commit/ff5faea9ff1f9837ebe4a42355e94afe4fcc833c) | Measurement wording now says same-output choices change whether the intervention clears the claimed effect gate, mirrored in the supplement reference copy and rebuilt PDF; `rg` and `pdftotext` checks guard against stale reversal wording. |
| 5 | Addressed | [`8b34c63`](https://github.com/SystemicVoid/H-Neurons/commit/8b34c6315cd9beddbf10616a87dc7ac619cf9b1b) | Bridge rubric wording now uses pre-specified/frozen language across the manuscript and supplement reference; `scripts/check_icml_prose.py paper/icml/main.tex` and an Issue 5 target `rg` check guard against stale unsupported bridge wording. |
| 6 | Addressed | [`d8c9973`](https://github.com/SystemicVoid/H-Neurons/commit/d8c9973e9cd0ea38dada4075bfa3a5e699c682aa) | The supplement package now bundles the bridge IRR rule, machine-readable summary, and redacted adjudicated labels, replaces report links with bundled derivatives, and `build_icml_supplement_package.py` validates relative Markdown links. |
| 7 | Addressed | [`d642ba0`](https://github.com/SystemicVoid/H-Neurons/commit/d642ba002b2d079a450d1e9e88a52219931bcbea) | The full claim-defense ledger now lives in the supplement provenance ledger, with compact readable indexes in the manuscript and supplement reference appendices; `make -C paper/icml`, rendered page-14 inspection, and supplement package checks guard it. |
| 8 | Addressed | [`ad90c46`](https://github.com/SystemicVoid/H-Neurons/commit/ad90c46b2afafc23062e5cdc99cce76838ac4495) | Supplement front matter now has an explicit Artifact Scope paragraph covering raw JSONL/gold-label/provenance omissions and included/redacted bridge IRR derivatives; `tests/test_build_icml_supplement_package.py::TestBuildIcmlSupplementPackage::test_repo_manifest_builds_bundle` guards the built README wording. |
| 9 | Addressed | [`db3f21b`](https://github.com/SystemicVoid/H-Neurons/commit/db3f21b9a9586784ad31e35c92cb71f4ac7059bf) | Table 5 now explicitly presents rates only and points Wilson CIs to provenance/support files, mirrored in the supplement reference copy; `make -C paper/icml`, page-11 text extraction, and rendered-page inspection guard the caption-table match. |
| 10 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | Table 7 now defines MDE as an approximate paired-rate endpoint effect at 80% power with a two-sided 0.05 test level, mirrored in the supplement reference copy; `rg`, `pdftotext`, and `make -C paper/icml` guard the wording. |
| 11 | Addressed | [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD) | The abstract now foregrounds a single-model Gemma-3-4B-IT case-study scope, mirrored in the supplement reference copy; `make -C paper/icml`, `pdftotext` page-1 extraction, and rendered-page inspection guard the wording. |

## 1. Verdict

**Major revision before submission.** The core scientific story is mostly defensible: the Gemma FaithEval readout/control dissociation is supported by manuscript-level provenance; the H-neuron FaithEval, TruthfulQA MC, SimpleQA/TriviaQA bridge, BioASQ/FalseQA, JailbreakBench measurement, SIMID, and Mistral stress-test claims are generally presented with appropriate scope limits. The paper’s strongest contribution is not a new steering method, but a disciplined case-study audit that separates localization, control, externality, and measurement gates.

The submission is not yet ready because the package has reviewer-visible presentation problems. Issue 1’s stale bridge-taxonomy contradiction is addressed in the tracking table above. The remaining serious blocker is the rendered appendix: orphan section headings, tables detached from their sections, mostly blank pages, and an unreadable claim-defense ledger. Figures 2 and 3 also have visibly clipped labels.

Most important remaining edit: regenerate the rendered PDF after fixing figure scripts and appendix float placement. Then tighten a few overstatements: “pre-registered” should become “pre-specified/internal frozen” unless a public preregistration artifact is included, and “reverse the conclusion” should become “change the pass/fail verdict.”

## 2. Blocking And Major Issues

### Issue 1 — Bridge-failure provenance reconciliation

**Status:** Addressed in [`b6a40bd`](https://github.com/SystemicVoid/H-Neurons/commit/b6a40bd79b24e6991bb753adfb630a39d1631332); keep the consistency test in the submission branch.  
**Severity:** Resolved blocker  
**Location:** `paper/main.tex:247-258`; `paper/number_provenance.md:81-99`; `supplement/number_provenance.md:77-91`; `supplement/support/externality_summary.md:71-80`; `supplement/failure_coding_manifest.md:17-24`.  
**Original problem:** In the reviewed snapshot, the main text and support files reported the bridge right-to-wrong taxonomy as 31/43 wrong-entity substitutions, 9/43 evasion/factual denial, 3/43 answer dilution, and 0/43 formal refusals. The supplement’s headline number ledger reported stale counts: 30/43, 8/43, 3/43, and 2/43. A reviewer checking the supplement would see two incompatible versions of a central qualitative claim.  
**Evidence from original package:** `paper/main.tex:255-258` stated 31/43, 9/43, 3/43, 0/43. `paper/number_provenance.md:90-93`, `supplement/support/externality_summary.md:75-78`, and `supplement/failure_coding_manifest.md:21-24` agreed with the main text. `supplement/number_provenance.md:86-89` contradicted them.  
**Resolution:** `supplement/number_provenance.md` now reports the adjudicated bridge IRR/failure-coding source values and includes CIs plus raw agreement, κ, and AC1 rows. `tests/test_bridge_irr.py::test_paper_bridge_taxonomy_provenance_matches_irr_summary` checks the source JSON, main text, paper ledger, supplement ledger, support summary, and failure-coding manifest. The resolved paper-facing row is: “Wrong-entity substitution 31/43 (72.1%) [57.3, 83.3]; evasion/factual denial 9/43 (20.9%) [11.4, 35.2]; answer dilution 3/43 (7.0%) [2.4, 18.6]; formal refusal 0/43 (0.0%) [0.0, 8.2]; raw agreement 55/57; Cohen’s κ = 0.90; Gwet’s AC1 = 0.96.”  
**Fix type:** Analysis/provenance.

### Issue 2 — Appendix rendering is not reviewer-legible

**Status:** Addressed in [`c89470d`](https://github.com/SystemicVoid/H-Neurons/commit/c89470d9dbf515eb0e64d4f68409c108b3607c99).
**Severity:** Major, potentially blocking for submission polish  
**Location:** PDF pages 11-17; rendered images `rendered-pages/page-11.png` through `page-17.png`; appendix source `paper/main.tex:472-819`.  
**Problem:** Appendix section headings are orphaned or detached from their content. Page 11 shows Appendix B and C headings with little/no content before D; pages 12-13 contain tables whose sections appeared earlier; page 15 is mostly blank with Appendix H and I headings; Tables 10 and 11 appear on pages 16-17 after large whitespace. This makes the appendix hard to use as reviewer evidence.  
**Evidence from package:** The relevant floats are declared with `[t]` or `[ht]` in `paper/main.tex:486`, `508`, `544`, `585`, `622`, `750`, and `800`, and the visual defects are visible in `rendered-pages/page-11.png` to `rendered-pages/page-17.png`. `paper/main.log:600-618` and `paper/main.log:635-656` show repeated underfull vbox warnings consistent with poor float/page balancing.  
**Recommended fix:** Rebuild the appendix around explicit barriers. Use `\FloatBarrier` after each appendix section, or insert `\clearpage` between major appendix blocks. For small tables, use `[!htbp]` and keep them after the section heading. For the huge claim ledger and stress-test table, either move them to the supplement or place them as landscape/shortened tables with a clean page break. Acceptance check: every appendix heading must be followed immediately by its prose/table, and no page should be mostly empty unless intentionally blank.  
**Fix type:** Structural / figure-table.

### Issue 3 — Figure labels are visibly clipped or malformed

**Status:** Addressed in [`d4c69fa`](https://github.com/SystemicVoid/H-Neurons/commit/d4c69fa27623388a86a18d2f6f16bb07d65a65b3).
**Severity:** Major  
**Location:** PDF pages 5-6; `rendered-pages/page-05.png`, `rendered-pages/page-06.png`; `paper/figures/fig2_matched.pdf`; `paper/figures/fig3_bridge.pdf`; figure captions in `paper/main.tex:200-208` and `paper/main.tex:288-295`.  
**Problem:** Figure 2 panel (b) contains a malformed annotation: the second line appears as only “95” rather than a complete CI statement. Figure 3 panel (b) has truncated bar labels such as “31 (72”, “9 (21”, “3 (7”, and “0 (0”, missing percent signs/closing parentheses. These defects will be noticed immediately in review.  
**Evidence from package:** The clipping is visible in the rendered page images and standalone figure PDFs. The caption text in `paper/main.tex:203-206` and `paper/main.tex:291-293` depends on these figures to carry central quantitative claims, so malformed labels degrade the argument.  
**Recommended fix:** Regenerate `fig2_matched.pdf` and `fig3_bridge.pdf` with larger right margins, larger annotation boxes, `clip_on=False` for labels, and `bbox_inches="tight", pad_inches=...` if using Matplotlib. For Figure 3, prefer labels such as `31 (72%)`, `9 (21%)`, `3 (7%)`, `0 (0%)` inside or just outside bars with sufficient x-limits. For Figure 2, either print the full `95% CI [0.94, 2.92]` text or remove the partial CI annotation and keep the CI in the caption.  
**Fix type:** Figure/table.

### Issue 4 — Measurement-section wording overstates “reversal”

**Status:** Addressed in [`ff5faea`](https://github.com/SystemicVoid/H-Neurons/commit/ff5faea9ff1f9837ebe4a42355e94afe4fcc833c).

**Severity:** Major  
**Location:** `paper/main.tex:300-327`, especially line 306; PDF page 7.  
**Problem:** The text says “same-output measurement choices can reverse the conclusion.” The evidence shows binary scoring is weak/non-decisive while graded scoring is positive; that is a change in evidential verdict, not a reversal from positive to negative or harmful to beneficial. A skeptical reviewer may attack the wording as rhetorical overreach.  
**Evidence from package:** `paper/main.tex:319-324` states binary endpoint +3.0 pp with CI including zero, graded slope +2.30 pp/α with CI excluding zero, and H-minus-random slope gap +2.77 pp/α. `paper/number_provenance.md:105-110` and `supplement/support/measurement_summary.md` support those numbers. Line 327 itself correctly says “binary evaluation suggests a weak or null effect; graded evaluation shows a positive dose-response.”  
**Recommended fix:** Replace line 306 with: “The specific finding here is that same-output measurement choices can change whether a representation-engineering intervention clears the claimed effect gate, even when two held-out evaluators tie in aggregate accuracy.”  
**Fix type:** Wording-only.

### Issue 5 — “Pre-registered” is not supported by the package

**Status:** Addressed in [`8b34c63`](https://github.com/SystemicVoid/H-Neurons/commit/8b34c6315cd9beddbf10616a87dc7ac619cf9b1b).
**Severity:** Major  
**Location:** `paper/main.tex:255`; related bridge-margin phrasing at `paper/main.tex:612`; `supplement/failure_coding_manifest.md:17`, `40`, `61`; `supplement/support/externality_summary.md:80`, `95-96`.  
**Problem:** The paper says the bridge coding rubric was “pre-registered.” The package supports “pre-frozen,” “pre-committed,” or “internally frozen” via a git hash/commit and adjudication rule, but I did not find a public preregistration record or timestamped registry artifact. “Pre-registered” has a stronger methodological meaning and invites an avoidable challenge.  
**Evidence from package:** `supplement/failure_coding_manifest.md:17` says the rubric/adjudication rule was committed at git `0e965d5`; `supplement/failure_coding_manifest.md:40` says disagreements were resolved under the pre-frozen rule. `supplement/support/externality_summary.md:80` says “pre-frozen adjudication rule.”  
**Recommended fix:** Replace “pre-registered four-category rubric” with “pre-specified four-category rubric” or “internally frozen four-category rubric.” Suggested line: “Each of the 57 discordant test cases ... was coded ... against a pre-specified four-category rubric whose adjudication rule was frozen before final label commitment.” Keep “pre-registered” only if the supplement includes an anonymized preregistration artifact or immutable public record.  
**Fix type:** Wording/provenance.

### Issue 6 — Broken supplement cross-links and missing referenced IRR artifacts

**Status:** Addressed in [`d8c9973`](https://github.com/SystemicVoid/H-Neurons/commit/d8c9973e9cd0ea38dada4075bfa3a5e699c682aa).
**Severity:** Major  
**Location:** `supplement/support/externality_summary.md:80`; `supplement/failure_coding_manifest.md:3`, `52-63`; `supplement/artifact_manifest.md:5-31`, `32-44`; `supplement/README.md:19-24`.  
**Problem:** The supplement points readers to report and machine-readable artifacts that are not in the supplement tree. `supplement/support/externality_summary.md:80` links to `../../reports/2026-04-21-bridge-irr-review.md`; `supplement/failure_coding_manifest.md:3` links to `../reports/2026-04-21-bridge-irr-review.md`. No `supplement/reports/` directory exists in the package. The manifest also lists `data/judge_validation/bridge_irr/bridge_irr_summary.json` and `data/judge_validation/bridge_irr/adjudicated_labels.jsonl` in `supplement/failure_coding_manifest.md:61-63`, but these files are not included under `supplement/data/judge_validation/`.  
**Evidence from package:** `supplement/artifact_manifest.md:5-31` lists included files and does not include bridge IRR summary or adjudicated labels; `supplement/README.md:23` and `supplement/artifact_manifest.md:40-42` explain raw JSONL and sidecar exclusions. The review package includes report evidence under `evidence/paper-reports/`, but that is not the same as the submitted supplement.  
**Recommended fix:** Make all supplement-relative links resolvable. Either include a redacted `reports/2026-04-21-bridge-irr-review.md` and safe `bridge_irr_summary.json`, or remove those links and state that the included `support/externality_summary.md` plus `failure_coding_manifest.md` are the reviewer-facing derivatives. For `adjudicated_labels.jsonl`, either include a redacted case-label file or explicitly mark it as omitted. The included summaries are enough for paper-level review, but not enough for independent recomputation of the bridge taxonomy.  
**Fix type:** Supplement/provenance.

### Issue 7 — Claim-defense ledger is too compressed to be useful

**Status:** Addressed in [`d642ba0`](https://github.com/SystemicVoid/H-Neurons/commit/d642ba002b2d079a450d1e9e88a52219931bcbea).
**Severity:** Major  
**Location:** PDF page 14; `rendered-pages/page-14.png`; `paper/main.tex:642-741`.  
**Problem:** Table 9 is a dense, tiny claim-defense ledger. Long metric IDs and source paths wrap every few characters. It is technically present, but not reviewer-legible at normal zoom, and it consumes an appendix page while failing its purpose.  
**Evidence from package:** `paper/main.tex:648-740` defines a very wide five-column table with `\resizebox{\textwidth}{!}` and long `\path|...|` strings. The rendered result in `rendered-pages/page-14.png` is cramped and visually hard to parse.  
**Recommended fix:** Move the full ledger to the supplement and keep a short appendix table with one row per claim family, using short metric IDs and one-hop support files. Alternative: split Table 9 into three smaller tables: localization/control, externality, measurement. Acceptance check: each row is readable at 100% PDF zoom without horizontal scanning.  
**Fix type:** Figure/table / structural.

### Issue 8 — Raw-artifact boundary is reasonable but must be clearer in the manuscript package

**Status:** Addressed in [`ad90c46`](https://github.com/SystemicVoid/H-Neurons/commit/ad90c46b2afafc23062e5cdc99cce76838ac4495).
**Severity:** Major for reproducibility framing; not a scientific invalidation  
**Location:** `PACKAGE_MANIFEST.md:40-42`; `supplement/README.md:19-24`; `supplement/artifact_manifest.md:32-44`; `supplement/evaluation_manifest.md:56-80`; `supplement/reproduction_manifest.md:58-62`.  
**Problem:** The package intentionally excludes raw response JSONLs, raw scored JSONLs, raw provenance sidecars, harmful prompt gold labels, bridge IRR machine-readable summary, and adjudicated labels. This is a defensible safety/anonymization choice, but the paper/supplement should not imply full independent recomputation from the upload.  
**Evidence from package:** `PACKAGE_MANIFEST.md:42` states the package includes summary/provenance artifacts rather than every raw run output. `supplement/README.md:23`, `supplement/artifact_manifest.md:40-42`, `supplement/evaluation_manifest.md:62-63`, and `supplement/reproduction_manifest.md:58-62` enumerate omissions.  
**Recommended fix:** Add a short “Artifact scope” paragraph to the supplement front matter: “The supplement supports claim traceability and minimal rerun orientation, not full recomputation from raw generations. Raw response/scored JSONLs, harmful-prompt gold labels, and provenance sidecars are omitted for safety/anonymization; where omitted, reviewer-facing summaries and safe manifests are provided.” For bridge IRR, explicitly say whether `bridge_irr_summary.json` is included, redacted, or omitted.  
**Fix type:** Supplement wording/provenance.

### Issue 9 — Table 5 caption claims Wilson CIs that are not displayed

**Status:** Addressed in [`db3f21b`](https://github.com/SystemicVoid/H-Neurons/commit/db3f21b9a9586784ad31e35c92cb71f4ac7059bf); the Table 5 caption now states that the table reports rates only and directs Wilson CIs to provenance/support files.
**Severity:** Minor-to-Major because it is an exact table/caption mismatch  
**Location:** PDF page 12; `paper/main.tex:505-525`.  
**Problem:** The caption says “FaithEval compliance by method and scaling factor (`n=1,000`; Wilson 95% CIs),” but the table contains only rates and no CI columns.  
**Evidence from package:** `paper/main.tex:510` includes the CI claim; `paper/main.tex:514-524` defines only `α`, `Neurons`, `SAE H-feat.`, and `SAE rand.` columns.  
**Recommended fix:** Either add CI columns/parenthetical CIs or change the caption to “Rates only; Wilson CIs are in the provenance ledger/support files.”  
**Fix type:** Figure/table wording.

### Issue 10 — Benchmark power table defines MDE too tersely

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); Table 7 now states the endpoint, power level, test level, paired-bootstrap basis, baseline-rate assumption, and item-pair dependence assumption.
**Severity:** Minor  
**Location:** PDF page 13; `paper/main.tex:582-601`.  
**Problem:** Table 7 lists “MDE” values but does not define the estimator, power level, pairing assumption, or endpoint. Reviewers will not know whether these are approximate endpoint MDEs, slope MDEs, paired binary MDEs, or simulation-based values.  
**Evidence from package:** `paper/main.tex:587` caption says “minimum detectable effect,” and `paper/main.tex:593-598` reports approximate values only.  
**Recommended fix:** Add one sentence before the table: “MDEs are approximate endpoint paired-rate effects at 80% power under the observed baseline and paired-dependence assumptions,” or whatever the true calculation is. If the calculation is informal, label the column “Approx. detectable endpoint effect.”  
**Fix type:** Wording / methods.

### Issue 11 — Abstract should foreground single-model scope

**Status:** Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD); the live abstract and bundled supplement reference copy now explicitly describe the work as a single-model Gemma-3-4B-IT case study.
**Severity:** Minor  
**Location:** `paper/main.tex:68-75`; PDF page 1.  
**Problem:** The abstract names Gemma-3-4B-IT but does not explicitly say this is a single-model case study. The limitations later do, but reviewers form scope expectations from the abstract.  
**Evidence from package:** `paper/main.tex:70` says “We audit this inference in Gemma-3-4B-IT...”; `paper/main.tex:429-430` later states the primary claim-bearing experiments use a single model.  
**Recommended fix:** Replace line 70 with: “In a single-model Gemma-3-4B-IT case study, we audit this inference across contextual faithfulness, multiple-choice answer selection, open-ended QA, and jailbreak evaluation.”  
**Fix type:** Wording-only.

## 3. Claim Support Audit

| Claim or passage | Verdict | Evidence checked | Risk | Recommended edit |
|---|---|---|---|---|
| “Strong readouts” vs control: H-neuron and SAE readouts have similar AUROC, but only H-neuron scaling controls FaithEval. | Supported. | `paper/main.tex:156-177`; `paper/number_provenance.md:15-39`; `supplement/support/localization_summary.md`; `evidence/ground-truth/metric_tables_only.md` readout/control rows. | Comparable AUROC does not equal matched basis/operator/layer coverage; SAE CI details are clearer in provenance than in compact metric table. | Keep the claim narrow: “similar held-out readout quality under the tested setup did not predict intervention utility.” |
| H-neuron FaithEval results: +2.09 pp/α slope, +4.5 pp no-op-to-max, random controls flat. | Supported. | `paper/main.tex:167-170`, `218-220`; `paper/number_provenance.md:28-34`; `evidence/ground-truth/metric_tables_only.md` FaithEval rows. | Surface is contextual FaithEval compliance/anti-compliance, not general factuality. | No major change; retain “compliance-adjacent/context-faithfulness” language. |
| SAE readout/control comparison and target-selection ablation. | Supported with caveats. | `paper/main.tex:171-186`, `529-580`; `paper/number_provenance.md:35-51`; `evidence/paper-reports/2026-04-22-faitheval-sae-utility-selector-review.md`; `evidence/paper-reports/2026-04-25-faitheval-answer-span-extension.md`. | Within-SAE target-selection evidence is margin-level; no SAE selector reaches the accuracy endpoint. Layer/operator coverage remains limited. | Say “within the available Gemma Scope SAE pool/operators/layers” wherever generalizing. |
| TruthfulQA MC improves but open-generation transfer fails. | Supported. | `paper/main.tex:229-236`; `paper/number_provenance.md:71-79`; `evidence/ground-truth/metric_tables_only.md:291-305`; `supplement/support/externality_summary.md`. | TruthfulQA MC is answer selection; SimpleQA has low baseline and prompt/autorater dependencies. | Keep answer-selection/open-generation distinction explicit. Do not call the MC gain a factuality gain. |
| TriviaQA bridge wrong-entity substitution. | Supported; prior supplement-ledger blocker addressed in [`b6a40bd`](https://github.com/SystemicVoid/H-Neurons/commit/b6a40bd79b24e6991bb753adfb630a39d1631332). | `paper/main.tex:240-262`; `paper/number_provenance.md:85-99`; `supplement/number_provenance.md:81-94`; `supplement/support/externality_summary.md:71-80`; `supplement/failure_coding_manifest.md:17-24`. | Second rater is LLM; “pre-registered” unsupported; behavior-level diagnosis not internal mechanism. | Replace “pre-registered” with “pre-specified,” and keep “behavioral diagnosis without claiming an internal substitution circuit.” |
| BioASQ and FalseQA effects. | Supported. | `paper/main.tex:218-225`; `paper/number_provenance.md:57-65`; `evidence/ground-truth/metric_tables_only.md` FalseQA/BioASQ rows. | FalseQA no-op-to-max effect includes zero even though slope/full sweep are positive; BioASQ style perturbation without accuracy movement can be overread. | Say “dose-response on FalseQA; no robust BioASQ alias-accuracy effect despite output perturbation.” |
| JailbreakBench measurement sensitivity. | Supported, but wording should be toned down. | `paper/main.tex:317-337`; `paper/number_provenance.md:105-117`; `supplement/support/measurement_summary.md`; `evidence/ground-truth/metric_tables_only.md` jailbreak rows. | “Reverse” overstates null-vs-positive. Single-seed specificity remains limited. | Use “changes gate verdict” rather than “reverse conclusion.” |
| SIMID calibration and prospective labels. | Supported as diagnostic-only. | `paper/main.tex:437`; `paper/main.tex:787-791`; `evidence/source-json/early_look_paired_delta_analysis.json`; `evidence/ground-truth/metric_tables_only.md:319-343`; `evidence/paper-reports/2026-04-28-simid-open-calibration-review.md`; `evidence/paper-reports/2026-05-03-simid-prospective-partial-external-label-review.md`. | SIMID pre-specified gates failed: calibration κ < 0.8, prospective effect CI includes zero, MC/attempt rate degrade. | Keep out of main positive story. Current diagnostic-only framing is correct. |
| Mistral anchor 1/2/3 stress tests. | Supported as failed/incomplete gates, not replication. | `paper/main.tex:429-430`, `743-791`; `evidence/ground-truth/metric_tables_only.md` Mistral rows; `evidence/paper-reports/2026-05-06-mistral24b-anchor3-jailbreak-measurement-review.md`; `evidence/paper-reports/2026-05-07-mistral24b-anchor2-truthfulqa-mc-review.md`. | Mistral FaithEval null, TruthfulQA MC1 source gate fails, JailbreakBench large curve lacks specificity controls. | Preserve current caveat: these constrain generalization and do not upgrade Gemma-local claims. |
| Audit-framework contribution and novelty. | Mostly supported. | `paper/main.tex:132-146`, `361-425`, `438`; `literature/INDEX.md:7-10`; `paper/references.bib` cited keys present; `evidence/paper-reports/final_paper_sprint_strategy.md`. | Field already has probe/control critiques, SAE steering-utility divergence, and evaluation fragility work. Novelty should not be overstated. | Frame as a “case-study audit scaffold plus matched cross-representational comparison,” not first evidence of detector/control divergence. |

## 4. PDF And Format Audit

The rendered PDF is **17 pages** (`pdfinfo paper/main.pdf`; `PACKAGE_MANIFEST.md:13`). Main text and impact statement run through page 8; references begin on page 8 and continue to page 10; appendix begins on page 11. The boundary is discoverable through “Appendix A,” but it would be more reviewer-friendly if the appendix had a cleaner first page or if references ended before a clear appendix start. There is no PDF outline, which is not fatal for a workshop submission but reduces navigability.

Layout quality is mixed. Pages 1-8 are mostly legible and ICML-like. Figure 1 on page 2 is readable. Figure 2 on page 5 and Figure 3 on page 6 have the clipping defects described above. Table 1 on page 7 is cramped but useful. Tables 2 and 3 are legible.

The appendix is the main rendering failure. Pages 11-17 show orphan headings, detached floats, large whitespace, and one unreadable ledger table. This is not a TeX-log-only issue: it is visible in the submitted PDF page images. `paper/main.log` has no overfull boxes or undefined citations/references in the inspected grep, but it has repeated underfull vbox warnings (`paper/main.log:600-618`, `635-656`) and underfull hbox warnings around appendix prose (`paper/main.log:660-693`).

Anonymity appears acceptable. `pdfinfo paper/main.pdf` reports `Author: Anonymous Authors`, the title is correct, and page 1 uses “Anonymous Authors,” “Anonymous Institution,” and `anon.email@domain.com`. I did not see obvious absolute local paths or identity-revealing metadata in the rendered PDF. The citation to `nguyen2025matsteer` is in normal third-person related-work form; if this is a self-citation, it is not obviously deanonymizing by itself. Supplement sidecars with host/local paths are intentionally omitted (`supplement/artifact_manifest.md:40`, `supplement/reproduction_manifest.md:58-62`).

Bibliography formatting appears stable: `paper/main.log` did not show undefined citation/reference warnings in my grep, and `paper/references.bib` contains the cited local literature keys. The reference section is dense but acceptable. The appendix and supplement are not yet coherent enough for final upload because of broken supplement links and missing referenced artifacts.

## 5. Figure, Table, And Appendix Audit

**Figure 1, page 2; `paper/main.tex` Figure 1 block.** The figure is trying to show the four-stage audit scaffold. It succeeds visually and earns its role. The caption is slightly long but clear. No material inconsistency found.

**Figure 2, page 5; `paper/main.tex:200-208`; `paper/figures/fig2_matched.pdf`.** This is the anchor figure: comparable readouts, divergent FaithEval control. It succeeds conceptually, and the caption matches the main claim. The visible annotation defect in panel (b) must be fixed. I would also add the AUROC CIs in panel (a) or caption to make the overlap visually explicit: H AUROC 0.843 [0.815, 0.870], SAE AUROC 0.848 [0.820, 0.874] from `paper/number_provenance.md:15-20`.

**Figure 3, page 6; `paper/main.tex:288-295`; `paper/figures/fig3_bridge.pdf`.** This figure is central to the externality story. Panel (a) usefully contrasts TruthfulQA MC improvement with generation degradation. Panel (b) is undermined by clipped labels. Regenerate with complete category labels and consider adding `n=43 R→W flips` directly in the panel title.

**Table 1, page 7; `paper/main.tex:264-286`.** The examples are persuasive and help distinguish wrong-entity substitution from refusal. It is cramped but readable. Consider reducing from five to four examples or moving the fifth to appendix if space is needed.

**Table 2, page 7; `paper/main.tex:339-356`.** The evaluator holdout table is legible and useful. Caption should perhaps include “no pairwise McNemar difference significant at n=50,” since the text uses that fact and the table only shows accuracies/CIs.

**Table 3, page 8; `paper/main.tex:378-399`.** The audit checklist is clear and valuable. It is a good workshop contribution artifact. No major change needed.

**Appendix A, page 11; `paper/main.tex:475-481`.** The detector caveat is useful but compressed. Define or avoid any unexplained acronyms if present in the rendered text. This block is not the main appendix problem.

**Tables 4-5, pages 11-12; `paper/main.tex:483-525`.** Table 4 is a good construct map but floats away from its heading. Table 5 is useful for exact rates, but its caption promises Wilson CIs that are absent. Fix placement and caption/content consistency.

**Table 6, page 13; `paper/main.tex:529-580`.** The SAE selector appendix is important and scientifically helpful. The source text is clear, but rendered placement is poor. Keep this material, but force the section and table to appear together.

**Table 7, page 13; `paper/main.tex:582-601`.** The power/MDE table is useful only if MDE is defined. Add calculation assumptions or remove the column.

**Table 8, page 13; `paper/main.tex:603-640`.** The bridge margin analysis is valuable because it prevents overclaiming about a substitution-specific internal mechanism. It should be kept, but section/table placement needs repair.

**Table 9, page 14; `paper/main.tex:642-741`.** This is not legible enough. Move most of it to the supplement and keep only a short paper-facing ledger.

**Table 10, page 16; `paper/main.tex:743-795`.** The Mistral/SIMID stress-test ledger is valuable and appropriately cautious. It appears too late and after a mostly blank page. Keep the content but fix page placement.

**Table 11, page 17; `paper/main.tex:797-819`.** The limitation inventory is useful and aligns with the main text. It appears detached from the heading and after large whitespace. Force it to appear immediately after Appendix I or combine it with the limitations subsection.

## 6. Related Work And Novelty

The related-work section is broadly fair and unusually well scoped for a near-submission mechanistic interpretability paper. It cites probe/control critiques, amnesic probing, unreliable probes, localization-control gaps, RePE, activation addition, ITI, SAEs, Gemma Scope, SAE steering-utility divergence, TruthfulQA/SimpleQA/TriviaQA/BioASQ/FaithEval/JailbreakBench, evaluator artifacts, and recent/concurrent work. The local literature package supports this coverage: `literature/INDEX.md:7-10` maps intervention/steering, detection/probes, safety/refusal, and steering reliability/evaluation; `paper/references.bib` includes the keys used in `paper/main.tex:132-146`.

The novelty claim is mostly calibrated. `paper/main.tex:141-146` says concurrent SAE-centric and synthetic-benchmark evaluations already make detection/control divergence clear, and positions the contribution as a cross-representational FaithEval comparison plus bridge diagnosis and measurement audit. `paper/main.tex:438` further states the paper does not claim discovery of detector/control dissociation in general. That is the right posture.

Two improvements would make the related-work positioning more reviewer-proof. First, move the “we do not claim discovery of detector/control dissociation” caveat from the limitations section into the related-work contribution paragraph as well. Second, avoid phrases like “real behavioral surface” if they imply other benchmark surfaces are not behavioral; say “the same contextual-faithfulness surface” or “a shared behavioral endpoint.”

I did not perform an external literature freshness search. Given the user-supplied review scope and local literature corpus, the package is sufficient for assessing internal positioning. If the authors want maximal ICML defensibility, they should do one final literature pass for 2026 SAE steering/activation-steering evaluation papers, but I did not find a local citation omission that blocks submission.

## 7. Recommended Revision Plan

| Priority | Edit | Files/sections | Acceptance check |
|---:|---|---|---|
| 1 | Reconcile bridge taxonomy counts across all paper and supplement ledgers. | `supplement/number_provenance.md`, `paper/number_provenance.md`, `supplement/support/externality_summary.md`, `supplement/failure_coding_manifest.md`, Figure 3 data source. | Addressed in [`b6a40bd`](https://github.com/SystemicVoid/H-Neurons/commit/b6a40bd79b24e6991bb753adfb630a39d1631332): no stale 30/43, 8/43, or 2/43 formal-refusal live rows remain; all reviewer-facing files report 31/43, 9/43, 3/43, 0/43 and 55/57 agreement. |
| 2 | Regenerate Figures 2 and 3. | `paper/figures/fig2_matched.py`, `fig2_matched.pdf`, `fig3_bridge.py`, `fig3_bridge.pdf`, PDF pages 5-6. | No clipped text; all bar/annotation labels complete in standalone PDFs and rendered manuscript. |
| 3 | Fix appendix float placement and whitespace. | `paper/main.tex:472-819`. | Pages 11-17 have no orphan headings, detached tables, or mostly blank pages; appendix headings are followed by their content. |
| 4 | Reformat or move Table 9. | `paper/main.tex:642-741`; supplement ledger. | Addressed in [`d642ba0`](https://github.com/SystemicVoid/H-Neurons/commit/d642ba002b2d079a450d1e9e88a52219931bcbea): the detailed ledger moved to `supplement/number_provenance.md`, and the paper-facing index is readable at 100% zoom. |
| 5 | Replace “reverse the conclusion.” | `paper/main.tex:300-327`. | Measurement section says scoring granularity changes whether the effect clears the gate; it does not claim sign reversal. |
| 6 | Replace unsupported “pre-registered” phrasing. | `paper/main.tex:255`; supplement bridge wording. | Uses “pre-specified,” “pre-frozen,” or includes an actual preregistration artifact. |
| 7 | Fix supplement links and missing-artifact declarations. | `supplement/support/externality_summary.md`, `supplement/failure_coding_manifest.md`, `supplement/artifact_manifest.md`. | Addressed in [`d8c9973`](https://github.com/SystemicVoid/H-Neurons/commit/d8c9973e9cd0ea38dada4075bfa3a5e699c682aa): every relative supplement link resolves during package build, and `bridge_irr_summary.json` plus redacted `adjudicated_labels.jsonl` are bundled and declared. |
| 8 | Correct Table 5 caption and define Table 7 MDE. | `paper/main.tex:505-525`, `582-601`. | Addressed across [`db3f21b`](https://github.com/SystemicVoid/H-Neurons/commit/db3f21b9a9586784ad31e35c92cb71f4ac7059bf) and [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD): captions match displayed columns, and MDE assumptions are stated. |
| 9 | Add single-model scope to abstract. | `paper/main.tex:68-75`. | Addressed in [`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD): the abstract clearly includes “single-model Gemma-3-4B-IT case study,” and page-1 text extraction confirms the rendered wording. |
| 10 | Final build and log sweep. | `paper/main.pdf`, `paper/main.log`, rendered pages. | No undefined citations/references, no overfull boxes, no severe underfull layout defects, metadata anonymized. |

## 8. Final Checklist

- [x] Regenerate `supplement/number_provenance.md` and verify bridge counts match `paper/main.tex`, `paper/number_provenance.md`, `support/externality_summary.md`, and `failure_coding_manifest.md` ([`b6a40bd`](https://github.com/SystemicVoid/H-Neurons/commit/b6a40bd79b24e6991bb753adfb630a39d1631332)).
- [x] Rebuild Figures 2 and 3; inspect standalone PDFs and pages 5-6 ([`d4c69fa`](https://github.com/SystemicVoid/H-Neurons/commit/d4c69fa27623388a86a18d2f6f16bb07d65a65b3)).
- [x] Repair appendix float placement with barriers/clear pages; re-render and inspect pages 11-17 ([`c89470d`](https://github.com/SystemicVoid/H-Neurons/commit/c89470d9dbf515eb0e64d4f68409c108b3607c99)).
- [x] Move or split the claim-defense ledger so it is readable ([`d642ba0`](https://github.com/SystemicVoid/H-Neurons/commit/d642ba002b2d079a450d1e9e88a52219931bcbea)).
- [x] Replace “pre-registered” unless a public preregistration artifact is included ([`8b34c63`](https://github.com/SystemicVoid/H-Neurons/commit/8b34c6315cd9beddbf10616a87dc7ac619cf9b1b)).
- [x] Replace “reverse the conclusion” with “change the gate/pass-fail verdict” ([`ff5faea`](https://github.com/SystemicVoid/H-Neurons/commit/ff5faea9ff1f9837ebe4a42355e94afe4fcc833c)).
- [x] Fix Table 5 CI caption ([`db3f21b`](https://github.com/SystemicVoid/H-Neurons/commit/db3f21b9a9586784ad31e35c92cb71f4ac7059bf)).
- [x] Define Table 7 MDE assumptions ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD)).
- [x] Add single-model Gemma-3-4B-IT case-study scope to the abstract ([`this commit`](https://github.com/SystemicVoid/H-Neurons/commit/HEAD)).
- [x] Resolve or remove all supplement links to absent reports/artifacts ([`d8c9973`](https://github.com/SystemicVoid/H-Neurons/commit/d8c9973e9cd0ea38dada4075bfa3a5e699c682aa)).
- [x] Add an explicit artifact-scope note for omitted raw JSONLs, harmful prompt gold labels, provenance sidecars, and bridge IRR derivative status ([`ad90c46`](https://github.com/SystemicVoid/H-Neurons/commit/ad90c46b2afafc23062e5cdc99cce76838ac4495)).
- [ ] Run final LaTeX checks: no undefined refs/citations, no overfull boxes, acceptable underfull warnings only, anonymous metadata, correct title, and page count within the workshop limit.
