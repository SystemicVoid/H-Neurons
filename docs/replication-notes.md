# Paper-Faithful Replication Notes

- `scripts/classifier.py` sweeps `C` on held-out probe metrics but does **not** implement the paper's full selection rule (which also scores TriviaQA behavior after suppression). Treat as a detector-selection baseline, not the final paper-equivalent criterion.
- `scripts/run_intervention.py` defaults to `--prompt_style anti_compliance`; use `--prompt_style standard` for paper-faithful replication (matches official Salesforce/FaithEval framing).
- Jailbreak eval uses JailbreakBench (Chao et al., NeurIPS 2024) by default (`--jailbreak_source jailbreakbench`). Use `--jailbreak_source forbidden` + `--jailbreak_path` for the legacy 390-question forbidden set. `--benchmark jailbreak_benign` runs the JBB benign split for over-refusal testing.
