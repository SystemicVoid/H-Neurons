# ICML Supplement Artifact Manifest

This package is a submission-safe, reviewer-facing supplement for the ICML manuscript in [`reference/main.tex`](reference/main.tex). It is centered on the anonymous TeX paper rather than the longer internal markdown draft.

## Included Files

| File | Role | Supports manuscript sections |
|---|---|---|
| `reference/main.tex` | Anonymous TeX manuscript copy used as the section-numbering anchor for this supplement. | All sections |
| `number_provenance.md` | Canonical reviewer-facing ledger for the manuscript’s main-body quantitative claims, keyed to one-hop bundled support files. | Abstract, §§3-5 |
| `evaluation_manifest.md` | Judge prompts, rubric versions, judge models, holdout artifact, and scoring entrypoints. | §5 |
| `failure_coding_manifest.md` | Coding guide and provenance for the TriviaQA bridge wrong-entity substitution audit. | §4.3 |
| `reproduction_manifest.md` | Minimal rerun map for the paper’s anchor results, with expected outputs and omitted sidecars policy. | §§3-5 |
| `support/localization_summary.md` | Derived summary of the FaithEval readout-quality and control comparison. | §3 |
| `support/externality_summary.md` | Derived summary of FalseQA, BioASQ, TruthfulQA, SimpleQA, and TriviaQA bridge results. | §4 |
| `support/measurement_summary.md` | Derived summary of the seed-0 jailbreak control analysis and evaluator holdout validation. | §5 |
| `support/judge_prompts.md` | Prompt and rubric summary for CSV2 v2, CSV2 v3, and StrongREJECT, with scoring-field definitions. | §5 |
| `data/judge_validation/holdout_comparison.json` | Safe machine-readable holdout comparison artifact for the four-way evaluator audit. | §5 |
| `data/manifests/triviaqa_bridge_test500_seed42.json` | Safe 500-question held-out ID manifest for the bridge benchmark. | §4.3, rerun map |
| `data/manifests/truthfulqa_final_fold0_heldout_mc1_seed42.json` | Safe held-out TruthfulQA MC1 fold-0 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold1_heldout_mc1_seed42.json` | Safe held-out TruthfulQA MC1 fold-1 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold0_heldout_mc2_seed42.json` | Safe held-out TruthfulQA MC2 fold-0 manifest. | §4.2, rerun map |
| `data/manifests/truthfulqa_final_fold1_heldout_mc2_seed42.json` | Safe held-out TruthfulQA MC2 fold-1 manifest. | §4.2, rerun map |

## Intentional Exclusions

| Excluded artifact class | Reason |
|---|---|
| `paper/draft/full_paper.md` | Long internal draft; not manuscript-centered and contains historical working context. |
| `paper/draft/number_provenance.md` | Historical long-draft ledger; superseded here by `number_provenance.md` aligned to the TeX manuscript. |
| Raw `*.provenance*.json` sidecars | Omitted for anonymization: these files expose local paths, host metadata, and command-line details. |
| Raw response JSONL files | Omitted to keep the package compact and reviewer-facing, and to avoid bundling unnecessary harmful-content corpora. |
| `tests/gold_labels/jailbreak_cross_alpha_gold.jsonl` | Omitted because it contains harmful jailbreak prompts and full response text; the bundled holdout summary JSON is the safe reviewer-facing artifact. |
| `logs/` outputs | Internal process artifacts; not needed for claim traceability. |

## Historical Notes

- The older long-draft provenance ledger remains in the repository as a historical-only artifact, but it is not part of this supplement.
- This package includes only files that directly support manuscript traceability or minimal rerun orientation.
