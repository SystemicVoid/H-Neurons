# Clean-Room ICML Evidence and Framing Synthesis Prompt

You are an external scientific strategy arbiter for mechanistic interpretability and AI safety research.

Arbitration date: **April 24, 2026**. Treat this as "today" for literature freshness and field-positioning judgments.

You are given:

1. A ground-truth metrics/evidence pack from the project.
2. Several AI-generated or human-generated reports interpreting those metrics.
3. Optional ICML manuscript text, notes, literature reviews, or prior strategy memos.
4. Optional project constraints such as time, compute, budget, model access, and deadline reality.

Your job is to produce a clean-room synthesis answering:

> What is the best experiment or evidence we can obtain next to strengthen, or better, enable a more fruitful framing of the current evidence into something more novel, higher-leverage, and higher-signal for the ICML mechanistic interpretability and AI safety community as of April 2026?

The goal is not to endorse the existing framing. The goal is to identify the next evidence artifact that most improves what the project can honestly and compellingly claim.

The final answer should be a decision-grade research memo, not a literature-review collage and not a summary of the reports.

---

## Inputs and Authority

Use this authority ordering:

1. **Ground-truth metrics/evidence pack**
   - Treat exact metrics, examples, provenance notes, and confidence intervals as the primary local evidence.
   - Do not assume the prose interpretation around those metrics is correct.

2. **Raw or near-raw project artifacts**
   - If provided, use tables, JSONL ledgers, reports with direct artifact links, and provenance sidecars to resolve ambiguity.

3. **Existing reports**
   - Treat reports as hypotheses, candidate framings, and search leads.
   - They are not evidence by themselves.
   - Do not average them.
   - Do not defer to a report because it is polished, confident, long, or produced by a stronger model.

4. **External literature**
   - Verify key literature claims directly where browsing or paper retrieval is available.
   - Cite primary papers, benchmark docs, repositories, model cards, or official documentation where possible.
   - Do not cite the input reports as literature authority.

If the evidence pack and a report disagree, prefer the evidence pack unless the report identifies a concrete provenance or measurement issue that you verify.

---

## Core Objective

Identify the next experiment, analysis, audit, or evidence package that best improves the research.

"Best" means the evidence has high expected value under these criteria:

- It resolves the most decision-relevant uncertainty in the current project.
- It discriminates between plausible scientific framings.
- It can strengthen the current story if positive, but is still informative if null or negative.
- It improves novelty and reviewer-legibility for the ICML mechanistic interpretability and AI safety audience.
- It increases causal, mechanistic, or evaluation clarity rather than merely adding another benchmark row.
- It has a realistic path to execution under the provided constraints.
- It directly affects what the paper can honestly claim.
- It is robust against foreseeable reviewer objections.

Do not optimize for:

- completing a symmetric table if the table does not answer the live scientific question;
- making the existing narrative look better;
- maximizing the number of tasks, metrics, or citations;
- producing an impressive-looking but ambiguous experiment;
- preserving any prior headline.

The highest-value answer may be:

- a matched causal decomposition;
- an ITI-focused margin decomposition;
- a same-basis selector bake-off;
- a same-item multiple-choice vs open-ended conversion;
- a targeted human audit of evaluator-disagreement cases;
- a negative-control or externality audit;
- a model generalization check;
- a reframing that makes no new run the immediate top priority;
- or a different option inferred from the provided evidence.

You must decide.

---

## Non-Negotiable Epistemic Rules

1. **Ground truth first.**
   Quantitative conclusions must trace back to the metrics/evidence pack or verified primary sources.

2. **Reports are not authorities.**
   Use them to find candidate ideas, not to settle claims.

3. **Do not inherit the original prompt's objective function.**
   The phrase "same intervention family, same alpha schedule, same model" is a possible design virtue, not automatically the most important virtue.

4. **Distinguish evidence from interpretation.**
   For every major claim, classify it as observed, inferred, speculative, or unsupported.

5. **Treat nulls and harms as valuable.**
   A result that rules out a tempting story can be more valuable than a small positive effect.

6. **Do not confuse benchmark movement with mechanism.**
   A behavioral improvement is not automatically evidence for the proposed causal mechanism.

7. **Do not confuse readout quality with intervention quality.**
   Predictive localization, interpretability, and causal actuation may be different objects.

8. **Do not flatten task interfaces.**
   Multiple-choice, open-ended generation, jailbreak refusal, factoid QA, and context-compliance tasks may expose different mechanisms.

9. **Do not hide measurement dependence.**
   If a conclusion depends on parsing, evaluator choice, grading rubric, answer aliases, refusal handling, or token-margin construction, say so.

10. **Use uncertainty.**
    If confidence intervals, bootstrap intervals, binomial intervals, seeds, or sample sizes are available, include them for quantitative claims.

11. **Prefer discriminating experiments.**
    A good next experiment should separate at least two plausible framings or mechanisms.

12. **Outcome-contingent value matters.**
    For the top candidates, explain what the project learns if the result is positive, null, or negative.

13. **Be ICML-realistic.**
    Novelty, rigor, baselines, ablations, reproducibility, and skeptical reviewer objections all matter.

14. **Be willing to recommend against the user's favorite framing.**
    If a proposed experiment is not highest-value, say so directly and explain why.

---

## Literature Handling

Use live browsing or paper retrieval where available. If browsing is unavailable, state that literature freshness is limited and distinguish verified from unverified literature claims.

For April 2026 field positioning, check the most relevant recent work around:

- activation steering and representation engineering;
- inference-time intervention and truthfulness steering;
- causal mediation, activation patching, and causal feature selection;
- sparse autoencoders as interpretability or steering substrates;
- causal benchmarks for interpretability;
- steering evaluation protocols and steering side effects;
- truthful QA, open-ended factual QA, and multiple-choice evaluation artifacts;
- jailbreak evaluation, refusal evaluation, and intervention-sensitive safety metrics;
- benchmark leakage, evaluator reliability, and human adjudication practices.

You do not need to write a broad survey. Use the literature to answer the decision question:

> What evidence would make this project more novel, rigorous, and useful to the ICML MI and AI safety community?

Citation discipline:

- Cite primary sources, not the input reports.
- Give title, authors or organization, venue or publisher, date, URL/arXiv/DOI, and retrieval date where possible.
- For fast-moving claims, state "verified as of April 24, 2026" or the actual retrieval date.
- If a report cites a source but the source does not support the claim, flag it.
- If a source is weak, proprietary, benchmark-narrow, unreproduced, or only partially applicable, say so.

---

## Candidate Evidence Classes To Consider

At minimum, consider whether any of these are the highest-value next move. Do not assume they are exhaustive.

### Matched Causal Decomposition Across Tasks

Same model, same intervention family, same alpha schedule, direct logit/token-margin traces, and behavioral endpoints across tasks such as FaithEval, BioASQ, FalseQA, Jailbreak, TruthfulQA, SimpleQA, and TriviaQA.

Key questions:

- Does matchedness answer a central uncertainty, or just create a tidy artifact?
- Which intervention family is the best candidate under this design?
- Are task-specific anchors and margins valid enough to support causal interpretation?
- What controls are required to make the decomposition meaningful?

### ITI Margin and Task-Interface Decomposition

A focused test of whether an ITI/truthfulness direction behaves like an answer-margin actuator whose behavioral effect depends on task interface.

Key questions:

- Does the same direction improve multiple-choice truthfulness while harming or suppressing open-ended factual QA?
- Do signed alpha sweeps reverse the relevant margin shifts?
- Are harms driven by choice-set reweighting, answer substitution, refusal, attempt suppression, or evaluator artifacts?
- Do random-head and random-direction controls separate head identity from direction geometry?

### Same-Basis Selector Bake-Off

Compare readout-selected, probe-selected, causal/gradient-selected, utility-selected, and random controls within the same representational basis.

Key questions:

- Is the core contribution "detector does not imply intervention target"?
- Does causal selection outperform predictive localization under matched conditions?
- Are failures due to the basis, selector, alpha schedule, evaluator, or task?

### Same-Item Multiple-Choice vs Open-Ended Conversion

Evaluate identical underlying questions in forced-choice and open-ended forms under the same intervention.

Key questions:

- Are multiple-choice improvements just answer-option reweighting?
- Does open-ended evaluation expose knowledge, calibration, refusal, or entity-binding failures hidden by multiple-choice metrics?
- Does human alias adjudication change the conclusion?

### Targeted Human Audit Of Measurement Frontiers

Human adjudication of the cases where evaluator choices change conclusions: parse failures, refusal-shell-plus-harm, answer alias errors, not-attempted cases, or right-to-wrong flips.

Key questions:

- Is the current story bottlenecked by measurement trust?
- Which conclusions survive targeted human review?
- Would a small stratified audit do more than another large automated run?

### Externality And Capability Audit

Measure whether an intervention's apparent benefit is offset by capability loss, refusal drift, verbosity, degeneration, over-refusal, or task-format collapse.

Key questions:

- Can the project claim safety or truthfulness gains without hiding capability cost?
- Which alpha range is usable rather than merely effective on one metric?
- Are negative side effects mechanistically informative?

---

## Analysis Protocol

Before writing the final recommendation, do the following.

### 1. Build A Clean Evidence Map

Extract the decision-relevant local facts from the evidence pack:

- intervention families;
- models;
- tasks and task interfaces;
- alpha schedules and no-op definitions;
- endpoints and metrics;
- sample sizes and uncertainty intervals;
- controls and missing controls;
- known measurement caveats;
- current positive, null, and harmful effects;
- provenance gaps.

Use report prose only to guide what to look for.

### 2. Extract Report Contributions

For each input report, identify:

- useful candidate experiments or framing ideas;
- claims that are directly supported by ground-truth metrics;
- claims that are plausible but not proven;
- overclaims or misleading interpretations;
- blind spots;
- unique ideas worth preserving.

Do not score reports based on style. Score them based on decision usefulness and evidence discipline.

### 3. Identify Plausible Framings

Generate the best possible framings of the project from the evidence. Examples might include:

- a rigorous evaluation protocol for activation steering;
- detection/readout does not imply causal steering target;
- truthfulness directions are task-interface-specific answer-margin actuators;
- intervention-sensitive evaluation reveals hidden side effects;
- negative results and limits of activation steering;
- compliance or context-following rather than truthfulness;
- another framing you infer.

For each framing, ask:

- What evidence already supports it?
- What evidence currently threatens it?
- What one new result would most strengthen or falsify it?
- Would ICML MI and AI safety reviewers find it novel and technically substantive?

### 4. Generate Candidate Evidence Actions

List candidate next experiments or analyses. Include both report-suggested and newly inferred options.

For each candidate, specify:

- research question;
- hypothesis or mechanism tested;
- intervention family;
- model(s);
- task(s);
- alpha schedule and no-op;
- primary behavioral endpoints;
- required logit/token-margin traces;
- controls and nulls;
- statistical analysis;
- implementation cost;
- expected ambiguity risk;
- reviewer objection addressed;
- what claims become possible if it succeeds;
- what is learned if it fails.

### 5. Score Candidates

Score each candidate from 1 to 5 on:

- information gain;
- framing leverage;
- novelty for ICML MI;
- AI safety relevance;
- causal or mechanistic directness;
- measurement credibility;
- robustness to null or negative outcomes;
- feasibility under constraints;
- cost-efficiency;
- reviewer-legibility;
- ability to reduce overclaim risk.

Then give an overall priority: Top, Strong, Conditional, Defer, or Drop.

Do not let a single high score dominate. A high-novelty experiment that is not feasible may not be top priority. A feasible experiment that does not change the framing may be low priority.

### 6. Stress-Test The Top Three

For the top three candidates, write an outcome-contingency table:

| Outcome | What It Would Mean | What It Would Not Mean | Paper Framing Consequence | Next Decision |

Include at least:

- strong positive result;
- clean null;
- harmful or reversed result;
- mixed/task-heterogeneous result;
- measurement-confounded result.

### 7. Choose One Primary Recommendation

Pick one highest-value next evidence artifact.

If two are close, explain the dependency relationship:

- "Do A first because it determines whether B is worth doing."
- "Do A as MVP and B as companion."
- "Do A for the paper, B for follow-up."
- "Do not do B until A resolves measurement validity."

Avoid vague portfolios. The final answer must name a primary next move.

---

## Output Format

Produce the final memo in these sections.

### 1. Executive Verdict

State:

- the single best next experiment/evidence artifact;
- why it is higher leverage than the alternatives;
- what framing it could enable;
- what current claim it would strengthen, revise, or falsify;
- the biggest risk in doing it.

Keep this section direct and decision-useful.

### 2. Current Evidence State

Summarize what the ground-truth metrics currently support.

Use a table:

| Finding | Evidence Source | Strength | Main Caveat | Framing Implication |

Separate:

- robust positives;
- robust nulls;
- robust harms or externalities;
- measurement caveats;
- missing controls.

Include uncertainty intervals for quantitative claims where available.

### 3. Report Arbitration

Compare the input reports only as sources of ideas.

Use a table:

| Report | Useful Contributions | Unsupported Or Overstated Claims | Blind Spots | Net Influence |

Net Influence must be High, Medium, Low, or Drop.

Do not cite reports as evidence in later sections.

### 4. Framing Options

List the strongest possible framings for the project.

Use a table:

| Framing | What It Claims | Evidence For | Evidence Against / Missing | Novelty | ICML/Safety Fit | One Best Test |

Then explain which framing currently looks most promising and which is most fragile.

### 5. Candidate Evidence Ranking

Rank the main candidate evidence actions.

Use a table:

| Rank | Candidate | Question Answered | Information Gain | Framing Leverage | Feasibility | Null-Result Value | Main Risk | Priority |

After the table, briefly explain why the top candidate beats the original matched-decomposition idea, or why the matched-decomposition idea genuinely wins.

### 6. Recommended Experiment Or Evidence Package

Give a concrete protocol for the primary recommendation.

Include:

- name of the evidence package;
- research question;
- hypotheses;
- intervention family and frozen configuration;
- model(s);
- tasks and splits;
- alpha schedule and no-op definition;
- token/logit margin definitions;
- behavioral endpoints;
- controls and nulls;
- required traces/artifacts to save;
- statistical plan and uncertainty reporting;
- measurement audit or human adjudication plan;
- expected runtime/cost if inferable;
- minimum viable version;
- full version;
- stop/kill criteria;
- risks and mitigations.

Also include this table:

| Result Pattern | Interpretation | Claim Enabled | Claim Ruled Out | Next Move |

### 7. What To Deprioritize

Name work that should be postponed or dropped.

For each item, explain whether the issue is:

- low information gain;
- weak framing leverage;
- too expensive now;
- measurement not ready;
- confounded design;
- only useful after another result;
- unlikely to survive reviewer scrutiny.

### 8. Literature And Reviewer Positioning

Explain how the recommended evidence would be positioned relative to the April 2026 literature.

Include:

- the closest related work;
- what gap this evidence addresses;
- what skeptical ICML reviewers will challenge;
- what controls, uncertainty, or caveats are needed to make the result credible;
- what not to claim.

### 9. Final Decision

End with:

- one recommended next action;
- one backup action if the primary is blocked;
- one sentence describing the strongest honest paper framing if the recommendation succeeds;
- one sentence describing the strongest honest paper framing if it fails.

---

## Standards For A Good Answer

Your answer succeeds if it helps a serious researcher decide what to run next.

It should be:

- skeptical;
- concrete;
- outcome-contingent;
- grounded in the metrics;
- aware of the literature;
- explicit about uncertainty;
- willing to reject attractive but low-yield work;
- clear about what claim each experiment buys.

Your answer fails if it:

- mainly summarizes the reports;
- assumes the original matched-decomposition prompt is correct;
- recommends "more experiments" without a discriminating design;
- ignores nulls, harms, or measurement artifacts;
- omits controls;
- reports quantitative claims without uncertainty when uncertainty exists;
- treats multiple-choice, open-ended generation, and jailbreak behavior as interchangeable;
- writes a persuasive story without saying what evidence would falsify it.

---

## Attachment Template

After this prompt, provide the materials in this order when possible.

### A. Project Goal And Constraints

Paste:

- current research goal;
- target venue or audience;
- deadline reality;
- compute and budget constraints;
- models available;
- experiments already running or completed;
- what can be run locally.

### B. Ground-Truth Evidence Pack

Paste or attach the ground-truth metrics report first. Include machine-stable ledgers if available.

Recommended local files:

- `notes/ground-truth/README.md`
- `notes/ground-truth/metric_tables_only.md`
- `notes/ground-truth/01_readout_surfaces.md`
- `notes/ground-truth/02_intervention_surfaces.md`
- `notes/ground-truth/03_measurement_surfaces.md`
- `notes/ground-truth/04_transfer_externality_surfaces.md`
- `notes/ground-truth/05_mechanism_diagnostic_surfaces.md`
- `notes/ground-truth/metric_ledger.jsonl` if the model can ingest JSONL
- `notes/ground-truth/example_ledger.jsonl` if examples matter
- `notes/ground-truth/surface_crosswalk.jsonl` if provenance resolution matters

### C. Existing Reports To Arbitrate

Paste or attach the existing reports. Recommended local files:

- `notes/causal-decomposition-dossier/2026-04-24-5.4codex1-matched-causal-decomposition-candidate-analysis.md`
- `notes/causal-decomposition-dossier/2026-04-24-codex5.5-v2-matched-causal-decomposition-candidate-analysis.md`
- `notes/causal-decomposition-dossier/2026-04-24-5.5xhigh-codex-h-neuron-matched-decomposition-candidate.md`
- `notes/causal-decomposition-dossier/2026-04-24-matched-causal-decomposition-candidate-opus2.md`
- `notes/causal-decomposition-dossier/5.4-pro-full-evidence-icml_framing_synthesis.md`
- `notes/causal-decomposition-dossier/5.4-pro-lite-clean_room_framing_report.md`
- any newer local synthesis you want the model to critique rather than trust

### D. Current Paper Or Framing Surface

Optionally attach:

- `paper/icml/main.tex`
- relevant files from `paper/icml/reports/`
- current outline, abstract, figures, or mentor feedback

### E. Literature Seeds

Optionally attach:

- bibliography notes;
- paper PDFs;
- literature-review reports;
- links to the most relevant papers.

If literature seeds are missing, the model should browse or explicitly mark literature coverage as incomplete.
