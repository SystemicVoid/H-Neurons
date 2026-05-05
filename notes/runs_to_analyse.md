# Runs to Analyse

This file is a queue for runs that still need analysis. Any analysed runs must be fully removed.

## 2026-05-02T11:13:57+00:00 | data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429
What: SIMID prospective effect calibrated open Gemma 3 4B resume
Key files: results.json, *.provenance.json
Status: full parent/control analysis still pending. The unstaged partial
`selected` / `truthfulqa` / alpha 0/8 external-label package has been reviewed
in `paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md`;
do not remove this queue entry until the remaining parent run evidence is
reviewed or explicitly abandoned.

## 2026-05-05T13:28:44+00:00 | data/mistral24b/intervention/jailbreak_anchor3_full500/experiment
What: JailbreakBench Anchor 3 full-500 neuron intervention sweep on Mistral-Small-24B-Instruct-2501 (alphas 0.0, 1.0, 1.5, 3.0; canonical run profile; locked manifest jbb_d7_full_harmful500_seed42_mistral24b)
Key files: results.20260504_164232.json, alpha_{0.0,1.0,1.5,3.0}.jsonl, run_intervention.provenance.20260504_164232.json, alpha15_separate_run_metadata/{results.20260504_215118.json, run_intervention.provenance.20260504_215118.json, README.md}, artifact_manifest.sha256, artifact_validation.json
Status: awaiting analysis. Note alpha=1.5 was generated in a separate same-seed invocation (RNG stream position differs from a hypothetical single-process four-alpha sweep) — see alpha15_separate_run_metadata/README.md.
