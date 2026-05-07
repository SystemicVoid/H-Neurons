# ICML Supplement Artifact Manifest

This package is a submission-safe, reviewer-facing supplement for the ICML manuscript in [`reference/main.tex`](reference/main.tex). It is centered on the anonymous TeX paper rather than the longer internal markdown draft, and it is assembled as a curated artifact bundle rather than a raw repository snapshot.

## Included Files

| File | Role | Supports manuscript sections |
|---|---|---|
| `README.md` | Front-door index for reviewers, plus public-release scope note. | All sections |
| `package_manifest.json` | Explicit allowlist plus a redacted public summary of the supplement builder's anonymization rules. | All sections |
| `reference/main.tex` | Anonymous TeX manuscript copy used as the section-numbering anchor for this supplement. | All sections |
| `number_provenance.md` | Canonical reviewer-facing ledger for the manuscript’s main-body quantitative claims, keyed to one-hop bundled support files. | Abstract, §§3-5 |
| `evaluation_manifest.md` | Judge prompts, rubric versions, judge models, holdout artifact, and scoring entrypoints. | §5 |
| `failure_coding_manifest.md` | Coding guide and provenance for the TriviaQA bridge wrong-entity substitution audit. | §4.3 |
| `reproduction_manifest.md` | Minimal rerun map for the paper’s anchor results, with expected outputs and omitted sidecars policy. | §§3-5 |
| `code/README.md` | Curated-code index with the supported verification flow. | §§3-5 |
| `code/pyproject.toml` | Minimal `uv` environment file for the bundled code and tests. | §§3-5 |
| `code/requirements.txt` | Pinned environment export from the main repository. | §§3-5 |
| `code/scripts/` | Curated paper-critical code bundle: analysis, evaluation, and intervention entrypoints plus local dependencies. | §§3-5 |
| `code/tests/` | Safe regression test slice for the bundled code paths. | §§3-5 |
| `support/localization_summary.md` | Derived summary of the FaithEval readout-quality and control comparison. | §3 |
| `support/externality_summary.md` | Derived summary of FalseQA, BioASQ, TruthfulQA, SimpleQA, and TriviaQA bridge results. | §4 |
| `support/measurement_summary.md` | Derived summary of the seed-0 jailbreak control analysis and evaluator holdout validation. | §5 |
| `support/judge_prompts.md` | Prompt and rubric summary for CSV2 v2, CSV2 v3, and StrongREJECT, with scoring-field definitions. | §5 |
| `data/judge_validation/holdout_comparison.json` | Safe machine-readable holdout comparison artifact for the four-way evaluator audit. | §5 |
| `data/judge_validation/bridge_irr/adjudication_rule.md` | Frozen bridge IRR adjudication rule, with anonymized rater roles and rule hash provenance. | §4.3 |
| `data/judge_validation/bridge_irr/bridge_irr_summary.json` | Safe machine-readable bridge IRR summary backing the paper and supplement taxonomy counts. | §4.3 |
| `data/judge_validation/bridge_irr/adjudicated_labels.jsonl` | Redacted case-label file for the 57 discordant bridge cases; includes case IDs, transitions, labels, rater labels, confidence, and notes without raw question/gold/response corpora. | §4.3 |
| `data/manifests/triviaqa_bridge_test500_seed42.json` | Safe 500-question held-out ID manifest for the bridge benchmark. | §4.3, rerun map |
| `data/manifests/truthfulqa_final_fold0_heldout_mc1_seed42.json` | Safe held-out TruthfulQA MC1 fold-0 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold1_heldout_mc1_seed42.json` | Safe held-out TruthfulQA MC1 fold-1 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold0_heldout_mc2_seed42.json` | Safe held-out TruthfulQA MC2 fold-0 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold1_heldout_mc2_seed42.json` | Safe held-out TruthfulQA MC2 fold-1 manifest. | §4.2, rerun map |

## Intentional Exclusions

| Excluded artifact class | Reason |
|---|---|
| Archived long-form markdown manuscript | Historical writing artifact; not manuscript-centered and contains working context outside the reviewer-facing submission. |
| Archived long-draft provenance ledger | Historical long-draft ledger; superseded here by `number_provenance.md` aligned to the TeX manuscript. |
| Full repository snapshot | Not reviewer-efficient and harder to anonymize; the supplement instead ships a curated manuscript-centered code and data slice. |
| `scripts/infra/` and deployment helpers | These wrappers contain local orchestration assumptions and are not needed to interpret or verify the paper claims. |
| Raw `*.provenance*.json` sidecars | Omitted for anonymization: these files expose local paths, host metadata, and command-line details. |
| Raw response/scored JSONL files | Omitted to keep the package compact and reviewer-facing, and to avoid bundling unnecessary harmful-content corpora. |
| Raw bridge IRR queues and rater progress files | Omitted for anonymization and compactness; the frozen rule, machine-readable summary, and redacted adjudicated labels are bundled instead. |
| `tests/gold_labels/jailbreak_cross_alpha_gold.jsonl` | Omitted because it contains harmful jailbreak prompts and full response text; the bundled holdout summary JSON is the safe reviewer-facing artifact. |
| `logs/` outputs | Internal process artifacts; not needed for claim traceability. |

## Historical Notes

- The older long-draft provenance ledger now lives in the external archive as a historical-only artifact, but it is not part of this supplement.
- This package includes only files that directly support manuscript traceability or minimal rerun orientation.
- The built package is intended for direct OpenReview upload as anonymized supplementary material.
