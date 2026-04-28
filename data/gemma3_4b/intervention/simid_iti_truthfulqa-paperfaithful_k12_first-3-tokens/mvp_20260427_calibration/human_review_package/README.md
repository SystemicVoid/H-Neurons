# SIMID Open Calibration Independent Review Package

This package supports the human/rater-diverse follow-up recommended by the
2026-04-28 SIMID open calibration review. It is diagnostic until a fresh
calibration gate passes.

## Files

- `index.html` - static local grading UI.
- `review_cases_blind.jsonl` - blinded cases for the human/static UI path.
  Independent AI raters should use `llm_blind_batches/batch_*/` instead.
- `adjudication_rule.md` - frozen grading rule copied into the package.
- `label_schema.json` - machine-readable schema for exported labels.
- `opus_4_7_independent_rater_prompt.md` - batch controller prompt for a
  separate Opus rater.
- `llm_blind_batches/batch_*/` - rater-safe independent-AI batch folders.
- `llm_blind_case_map.jsonl` - private reconciliation map from synthetic blind
  IDs to calibration IDs. Do not provide this to an independent rater.
- `review_manifest.json` - selection policy, hashes, and schema.
- `export_simid_open_review_package.provenance.*.json` - append-only export
  sidecars. The manifest's `generator.provenance_sidecar` names the sidecar for
  the current export; older sidecars may describe superseded package builds.

## Selection

The package includes all primary/secondary disagreements plus a deterministic
priority-weighted stratified sample of agreement cases.

- Total review cases: 100
- Counts by review set: `{'stratified_agreement_sample': 66, 'primary_secondary_disagreement': 34}`
- Counts by dataset: `{'truthfulqa': 54, 'triviaqa_bridge': 46}`
- Blind cases file SHA-256: `d3832171541f5a50dbde6da53419065d0f56413fbe95a7409bd4ae03a5f3dd67`
- Blind cases canonical row SHA-256: `c9c773d3fcbc163b14d67b9e126badec0494809ff3a8579f2d9a816852e67008`

The UI can be opened directly from:

`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/index.html`

## How to Use

1. Open `index.html` locally in a browser.
2. Enter a rater ID before grading.
3. For each case, read only the question, gold aliases, predicted answer, and
   frozen rule.
4. Choose one label, set confidence from 1 to 5, add flags/notes where useful,
   and press Save.
5. Use the status filter to verify all 100 cases are graded.
6. Press Export JSONL and keep the exported file as the human independent
   labels.
7. In a separate clean agent session, give Opus
   `opus_4_7_independent_rater_prompt.md`. It should process only the
   `prompt.md` files under `llm_blind_batches/batch_*/`, write
   `opus_4_7_labels.jsonl` inside each batch directory, and never paste label
   JSONL into the chat.
8. After all Opus batch files are written, merge and validate them with:

   `uv run python scripts/validate_simid_open_review_labels.py --package-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package --output data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/opus_4_7_labels.jsonl`

9. Compare human and Opus labels against primary/secondary/adjudication labels
   only after both independent label files are finalized.

Exported labels use schema `simid_open_independent_rater_label/v1` and should be kept as
new evidence. Do not edit the production queue, secondary labels, adjudications,
or summary in place.

The LLM-facing batch directories intentionally omit prior machine labels,
adjudication notes, datasets, conditions, sample-source metadata, and original
calibration IDs. The package root retains manifest/provenance files needed for
orchestration and reconciliation; do not provide the package root as rater
context. Use the source run outputs for post-label comparison only after
independent labels are finalized.

Current export provenance sidecar:
`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/export_simid_open_review_package.provenance.20260428_182723.json`.
