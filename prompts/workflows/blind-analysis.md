You are doing a blind narrative synthesis from a tightly scoped ground-truth evidence pack for an ICML mechanistic interpretability paper.

Your job is not to summarize the pack. Your job is to infer the strongest, most original, highest-signal paper framing that is actually compelled by the data, while resisting all legacy framing pressure.

Hard scope:
Read only these files and nothing else unless you are blocked by a missing reference inside them:
- `notes/ground-truth/README.md`
- `notes/ground-truth/metric_ledger.jsonl`
- `notes/ground-truth/01_readout_surfaces.md`
- `notes/ground-truth/02_intervention_surfaces.md`
- `notes/ground-truth/03_measurement_surfaces.md`
- `notes/ground-truth/04_transfer_externality_surfaces.md`
- `notes/ground-truth/05_mechanism_diagnostic_surfaces.md`

This must be a clean-room read. Assume prior framing is contaminated. Do not preserve it, search for it, or reconstruct it.

Critical anti-bias rule:
The pack’s filenames, section names, family ids, act tags, and metric-id prefixes are organizational metadata, not the paper’s correct conceptual frame. In particular, do not default to any existing scaffold just because the pack is grouped that way. If a familiar frame emerges, it must earn its place by beating alternative framings on compression, surprise, and explanatory power.

Working method:
1. Start from `metric_ledger.jsonl`, not the markdown prose.
2. Rewrite the evidence to yourself in plain language, stripping away family labels and prior category names.
3. Identify the biggest non-obvious patterns, asymmetries, dissociations, reversals, and constraints.
4. Look especially for findings that cut against the naive story a reader would expect from “good internal signal -> good intervention target”.
5. Only then use the markdown files to recover source semantics, examples, and any nuance needed to interpret the structured metrics.
6. Actively search for the most interesting alternative framings, not just the most available one.
7. Treat every tempting headline as guilty until defended by multiple independent metrics.
8. Separate what is directly evidenced from what is interpretation, and separate both from what would still need literature validation.

What I want you to optimize for:
- highest signal-to-claim ratio
- most defensible central paper thesis
- most surprising framing that is still genuinely supported
- minimal contamination from prior report language
- a framing that would make a strong mech-interp paper, not just an evaluation paper and not just a benchmark audit

What to avoid:
- generic “measurement matters” summaries
- generic “correlation is not causation” summaries
- generic “steering has tradeoffs” summaries
- language that sounds inherited from the pack organization rather than discovered from the evidence
- absolute claims like “X never works”
- literature novelty claims stated as facts; you do not have literature in scope

Deliverable:
Produce exactly these sections.

## 2. Candidate framings
Give 5 candidate paper framings.
For each framing include:
- one-sentence thesis
- why it is high-signal
- what evidence most strongly supports it
- what evidence weakens or limits it
- risk of sounding derivative or generic
- verdict: 1-10 `strong contender (10)`, `backup(5)`, or `reject(1)`

Candidates must be meaningfully different from the obvious “good readouts are unreliable steering targets” line.

## 5. Hostile self-critique
Attack your chosen framing.
List the 5 strongest objections a skeptical reviewer could raise from this same evidence pack alone.
Then state whether each objection is fatal, serious but survivable, or mostly containable.


Style rules:
- be analytical, not diplomatic
- think like a ruthless but fair ICML reviewer with taste
- prefer sharp synthesis over exhaustive coverage
- do not mention prior framing, prior reports, or literature unless clearly labeled as out of scope
- do not produce a generic summary first
- do not hedge away the main decision; choose a framing

Final constraint:
I care much more about discovering the right narrative than about preserving any pre-existing one. If the strongest framing is narrower, stranger, or less flattering than an obvious headline, say so.