# ICML 2026 Supplement Package

This directory is the source for the anonymized supplementary material that accompanies the ICML 2026 submission in [`reference/main.tex`](reference/main.tex).
It is a reviewer-facing reproducibility bundle, not a full repository snapshot.

## Open First

1. [`artifact_manifest.md`](artifact_manifest.md) for the package map
2. [`number_provenance.md`](number_provenance.md) for headline-number traceability and moved appendix ledgers
3. [`reproduction_manifest.md`](reproduction_manifest.md) for the minimal rerun map

## Package Structure

- `reference/` contains the anonymized TeX manuscript used as the section-numbering anchor for this supplement.
- `support/` contains reviewer-facing summaries for the paper's three anchor empirical sections.
- `data/` contains only safe manifests and compact summary JSON used by the supplement.
- In the built upload package, `code/` contains the curated code and test slice for the paper-critical analysis paths. The source directory materializes it under `build/icml_supplement_package/code/`.

## Reviewer Access Route

Use the built package, not this source directory, as the anonymized code/data supplement for OpenReview upload. It follows the ICML supplementary-code route and the Mechanistic Interpretability Workshop reproducibility-access expectation by bundling the curated code slice, safe data manifests, bridge IRR derivatives, and reviewer-facing support ledgers. If the same material is mirrored as an anonymous repository, use the built package contents on a frozen submission branch.

## Artifact Scope

This package is tied to the current submitted manuscript and supports claim traceability plus minimal rerun orientation. It does not enable full recomputation from raw generations. Raw response/scored JSONLs, harmful prompt gold labels, raw bridge IRR queues/progress files, and raw provenance sidecars are omitted for safety/anonymization; where omitted, reviewer-facing summaries and safe manifests are provided. The bridge IRR reviewer surface is bundled: `data/judge_validation/bridge_irr/bridge_irr_summary.json` is included, `data/judge_validation/bridge_irr/adjudication_rule.md` is included, and `data/judge_validation/bridge_irr/adjudicated_labels.jsonl` is redacted. Pending follow-up work is excluded unless it is completed and integrated into the manuscript-facing summaries before submission.

## Public-Release Reminder

ICML states that the submitted supplementary files for accepted papers become public on OpenReview.
Only submission-safe, anonymized, publicly releasable files should be included in the built package.
