# Appendix

## Appendix A. Detector Interpretation Audits

This appendix collects the two detector-interpretation cautions referenced in §4.1.

**N4288 / L1 instability.** The top-weight CETT neuron at layer 20, neuron 4288, is not a stable "most important neuron" finding across regularization settings. In the pipeline audit, it is absent at $C <= 0.3$, appears at $C = 1.0$, and drops to rank 5 at $C = 3.0$ while a wider detector reaches 80.5% test accuracy with 219 positive neurons. Across six follow-up analyses, the verdict is that the extreme weight is better explained by L1 concentration among correlated features than by unique causal importance.^[Source: `data/gemma3_4b/pipeline/pipeline_report.md`, detector audit section; `scripts/investigate_neuron_4288.py` generated the underlying analyses.]

**Verbosity confound.** The verbosity audit shows that under full-response aggregation, response length dominates truthfulness signal by roughly $3.7\times$ (mean aggregation) to $16\times$ (max aggregation), and 36 of 38 H-neurons are more length-dominant than truth-dominant in that setting. The mitigation is scope: the classifier result is still a held-out answer-token discrimination result, but it should not be interpreted as a pure hallucination detector without this qualification.^[Source: `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`.]

These appendix notes are not separate headline claims. Their role is to justify the narrower wording used in the main text: the readouts are real, but detector interpretation is fragile.
