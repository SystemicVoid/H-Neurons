# 6. Measurement Choices Changed the Scientific Conclusion

<!-- Anchor 3 of the four-stage scaffold: Measurement → Conclusion break.
     Source files (canonical for all numbers):
       notes/act3-reports/2026-04-08-d7-full500-audit.md
       notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md
       notes/act3-reports/2026-04-12-4way-evaluator-comparison.md
       notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md
       notes/measurement-blueprint.md
-->

The preceding sections established that detection quality does not predict
steerability (Section 4) and that successful steering is narrow in scope
(Section 5). Both conclusions rest on behavioral measurements---jailbreak
compliance rates, severity scores, generation-surface accuracy---that are
themselves products of evaluation choices. In this section we show that those
choices are not clerical details: generation length, scoring granularity,
evaluator identity, and pipeline hygiene each independently shifted what we
would have concluded about whether a given intervention worked.

We organize the case study around the H-neuron jailbreak scaling experiment
(38 probe-selected neurons, $\alpha \in \{0, 1, 1.5, 3\}$, $n{=}500$ per
condition), because its moderate effect size makes it sensitive to every
measurement decision we examine. Where findings generalize beyond this
experiment, we note it explicitly.

Figure 4 collects the main measurement reversals from this section: binary
versus graded scoring, evaluator disagreement, and specificity under matched
controls.

## 6.1 Truncation Hides Downstream Content

Early jailbreak runs in this project used short generation caps (256 tokens
for the legacy setup; 1024 tokens for a greedy-decode cross-validation).
Gemma-3-4B-IT typically opens jailbreak responses with a refusal preamble
("I cannot help with that...") before pivoting to substantive compliance.
At 256 tokens the generation frequently terminates inside the preamble,
before the harmful payload begins; at 1024 tokens, the greedy decode
similarly truncated responses that exceeded the cap, masking degeneration
artifacts at high intervention strengths.

When we moved to the canonical generation policy ($\texttt{max\_new\_tokens}{=}5000$,
sampled decoding), two findings changed. First, the non-monotonic
bounce-back of the strict harmfulness rate at $\alpha{=}8.0$ in the gradient-based causal
pilot---which had appeared as an intervention *reversal*---was revealed to be
an artifact of degeneration: 74\% of gradient-ranked responses and 82\% of probe-ranked
responses hit the 5000-token cap at that strength, and the greedy decode's
1024-token truncation had hidden the degenerate text
(source: `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md`, Section 5).
Second, the full-generation confirmatory run ($n{=}500$) found 112/500
(22.4\%) of causal responses hitting the token cap at $\alpha{=}4.0$, but
these cap-hit responses were overwhelmingly safe: 97/112 were scored safe
rather than strictly harmful, with mean harmful payload share of 0.0192
(source: `notes/act3-reports/2026-04-08-d7-full500-audit.md`, Section 4.2).
The causal safety effect survived restriction to the non-cap subset:
$-9.8$ pp $[-13.7, -5.9]$.

The truncation artifact is not exotic. Any intervention that alters
generation length---through refusal elaboration, repetitive hedging, or
outright degeneration---will interact with a token cap. Short caps
preferentially hide post-preamble content, biasing evaluations toward
whatever behavior the model produces first. Because Gemma-3-4B-IT leads
jailbreak responses with refusals and follows with compliance, truncation
inflates apparent safety. Our measurement blueprint
(`notes/measurement-blueprint.md`) now prohibits citing the legacy
256-token setup as current evidence.

## 6.2 Binary Versus Graded Evaluation

The H-neuron jailbreak scaling experiment provides the clearest illustration
of how scoring granularity determines the conclusion.

**Binary judge.** A GPT-4o binary harmful/safe classifier showed an
H-neuron compliance increase of $+3.0$ pp from $\alpha{=}0$ to $\alpha{=}3$
(152/500 $\to$ 167/500). The 95\% confidence interval includes zero,
consistent with the minimum detectable effect of ${\sim}6$ pp at this sample
size. Under binary evaluation alone, the H-neuron intervention would be
judged null on jailbreak compliance
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 1.2).

**Graded evaluation (CSV-v2).** The same responses, scored by a structured
rubric that distinguishes refusal, borderline, and substantive compliance,
yielded an H-neuron strict harmfulness slope of $+2.30$ pp/$\alpha$
$[+0.99, +3.58]$ (bootstrap 95\% CI, 10{,}000 resamples). The confidence
interval excludes zero
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 2.1).

**Negative control.** A matched set of 38 randomly selected neurons
(seed 0) produced a slope of $-0.47$ pp/$\alpha$ $[-1.42, +0.47]$---flat,
with the CI comfortably including zero. The slope difference (H-neuron minus
random) was $+2.77$ pp/$\alpha$ $[+1.17, +4.42]$, with a permutation test
$p = 0.013$ (647/50{,}000 permutations $\geq$ observed gap)
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 2.1).

The mechanism behind this divergence is the borderline category. As H-neuron
scaling increased, the graded evaluator registered a *polarization* of
borderline responses: borderline count dropped by 73 (171 $\to$ 98), with 38
migrating to strict compliance and 35 to clear refusal. The random control
showed no polarization (borderline stable at 124--139 across all $\alpha$
values; total compliant-or-borderline unchanged at 245)
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 3.1). Binary evaluation collapsed this three-way structure into a
two-way count, washing out the signal that graded scoring recovered.

This is a single-seed result. Seeds 1--2 have been generated but not yet
scored with the graded rubric; multi-seed replication would strengthen the
specificity claim. But the methodological lesson does not depend on the
H-neuron effect being real in the strong sense: *any* intervention that
shifts responses along a refusal--compliance gradient will appear different
under binary versus graded evaluation, and the direction of the discrepancy
is predictable.

## 6.3 Evaluator Dependence Is Part of the Result

If the binary-versus-graded comparison reveals that *scoring granularity*
matters, the natural follow-up is whether *evaluator identity* matters among
scorers that all produce binary verdicts. We tested four evaluators on a
74-record gold-labeled subset (45 harmful, 29 safe) drawn from H-neuron
jailbreak responses across three intervention strengths ($\alpha \in \{0, 1.5,
3\}$). Gold labels were assigned by deep reading of full model outputs.

### Development-set results

| Evaluator | Judge model | Accuracy [95\% CI] | FP | FN |
|-----------|-------------|---------------------|----|----|
| CSV2 v3   | GPT-4o      | 86.5\% [76.9, 92.5] | 0  | 10 |
| Binary judge | GPT-4o   | 77.0\%              | 4  | 13 |
| StrongREJECT | GPT-4o-mini | 74.3\% [63.3, 82.9] | 0 | 19 |
| CSV2 v2   | GPT-4o      | 73.0\%              | 0  | 19 |

(Source: `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`,
Section 0.)

However, 24 of these 74 records overlap with the rows used during CSV2 v3's
calibration phase---the prompt IDs were hand-selected precisely because they
represented the failure surface where earlier evaluators struggled. After
removing them, the holdout ($n{=}50$, 27 harmful, 23 safe from 17 prompt
clusters) told a different story:

| Evaluator | Holdout accuracy | Holdout 95\% CI (prompt-clustered) |
|-----------|------------------|------------------------------------|
| CSV2 v3   | 96.0\%           | [90.0, 100.0]                      |
| StrongREJECT | 94.0\%        | [88.0, 100.0]                      |
| CSV2 v2   | 92.0\%           | [84.3, 98.0]                       |
| Binary judge | 90.0\%        | [80.0, 98.0]                       |

(Source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Sections 0 and 2.)

The v3--StrongREJECT gap compressed from 12.2 pp to 2.0 pp, resting on a
single discordant record. McNemar's exact test yielded $p = 1.0$---no
pairwise comparison reached significance (all $p > 0.25$). On held-out data,
all four evaluators exceeded 90\% accuracy, and no evaluator was
statistically distinguishable from any other.

> **Box C — Worked Evaluator Disagreement Example**
>
> One holdout jailbreak response began with a refusal preamble and then went on
> to provide substantive harmful content. StrongREJECT marked
> `refused = 1`, which zeroed its final score by construction, while the CSV2
> evaluators and the binary judge all marked the output harmful. This is a good
> example of why evaluator disagreement is part of the scientific result here:
> the intervention changes refusal framing, and different evaluators make
> different choices about whether a refusal preamble should dominate the final
> label.^[Source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
> §3; `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`, §3.]

**We do not conclude from this that evaluator choice is irrelevant.** We
conclude the opposite: evaluator choice matters for precisely the response
regime that interventions produce, and the *apparent* magnitude of that
dependence is itself sensitive to the evaluation sample. Three observations
support this.

*First*, the evaluators disagree on different failure modes. The binary judge
produced 3 false positives on holdout (overcalling safe responses as
harmful); all other evaluators produced zero. StrongREJECT produced 1 solo
false negative on holdout; CSV2 v3 produced zero solo errors---its 2 errors
were in the universal-failure bucket where every evaluator was wrong
(source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Section 3). For an intervention study that needs to track harm
*reduction* across scaling strengths, false positives create phantom
treatment effects, making the binary judge's error profile qualitatively
different from the others'.

*Second*, the construct measured by StrongREJECT is subtly different from the
construct measured by the CSV2 family. StrongREJECT computes
$\text{score} = (1 - \text{refused}) \times \text{mean(convincingness, specificity)} / 5$.
When $\texttt{refused}{=}1$, the score is zero regardless of the
sub-scores. Among the 19 StrongREJECT false negatives on the full 74-record
set, all 19 had $\texttt{refused}{=}1$; their mean convincingness was 4.58/5
and mean specificity was 4.89/5
(source: `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`,
Section 3). The model correctly identified the content as convincing and
specific, but the formula discarded this assessment because a refusal
preamble was detected. This is not an error in the usual sense---StrongREJECT
was designed for the attack-evaluation setting, where a refusal is decisive.
In our intervention setting, where the model routinely refuses *and then
complies*, the refusal flag underweights harmful substance that follows
protective framing.

StrongREJECT also differs from the other three evaluators in judge model:
it used GPT-4o-mini while the others used GPT-4o. Sub-score analysis suggests
that the main disagreement is driven by the formula's treatment of the refused
flag rather than model capability, but that confound has not been removed
experimentally and remains part of Limitation L8.

*Third*, holdout validation compressed the apparent evaluator advantage but did
not eliminate the structural differences. CSV2 v3 maintained zero solo errors
and zero false positives on holdout. StrongREJECT's single holdout error was
the construct-mismatch pattern in miniature: the model detected a refusal
preamble, set $\texttt{refused}{=}1$, and the formula zeroed the score
despite substantive harmful content that all three other evaluators correctly
flagged
(source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Section 3). The question the holdout cannot answer is whether CSV2 v3
genuinely outperforms on *new* hard cases it was not calibrated on;
answering this requires fresh gold labels on unseen refuse-then-comply
responses.

We frame evaluator disagreement as *construct pluralism*, not one judge being
wrong. Each evaluator operationalizes "harmful" differently; interventions
that restructure refusal behavior will occupy the gap between these
operationalizations. Reporting results under a single evaluator without
acknowledging this sensitivity is a form of measurement monoculture that
risks conclusions that do not replicate under alternative scoring.

## 6.4 Pipeline Contamination and Schema Drift

The specificity analysis in Section 6.2 was nearly derailed by a silent
pipeline bug. The analysis script (`scripts/analyze_csv2_control.py`)
imported a normalization function (`normalize_csv2_payload`) from the v3
evaluation module and applied it to v2-scored data. Because v2 records lack
the `schema_version`, `primary_outcome`, and `intent_match` fields that v3
expects, the function silently reclassified 167 of 171 borderline records
(97.7\%) as strict compliance, inflating the strict harmfulness rate from
18.8\% to 52.2\%
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 6.1).

The root cause was a four-line code path: v2's `"borderline"`
$\texttt{harmful\_binary}$ label mapped to `"unknown"` under the v3
vocabulary; the function then re-derived $\texttt{harmful\_binary}$ from
C/S/V/T scores, and any specificity $\geq 1$ triggered
$\texttt{partial\_compliance}$, which in turn resolved to `"yes"`. The fix
was a schema-version guard: v2 data (identified by the absence of the
$\texttt{schema\_version}$ field) is now returned without normalization.

The contaminated pipeline produced plausible-looking output---no errors, no
warnings---and would have yielded the triage verdict
$\texttt{review\_specificity}$ instead of the correct
$\texttt{specificity\_supported}$
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 6.3). Applied to the control comparison, it would have erased the
slope difference between H-neurons and random neurons by inflating both
baselines.

This episode illustrates that measurement discipline extends below the
evaluator-design level to code-level schema handling. The bug was caught
because the analysis pipeline was re-run from raw data with explicit
integrity checks (500/500 record counts, schema field verification, cross-
condition prompt-ID parity). Had the contaminated rates been accepted at
face value, the H-neuron jailbreak effect would have remained listed as
"unscored" rather than upgraded to "single-seed supported" with $p = 0.013$.

## 6.5 What Is Established and What Remains Open

We summarize the measurement findings by their epistemic status.

**Established:**

- *Truncation artifact.* Short generation caps hide post-preamble harmful
  content in Gemma-3-4B-IT jailbreak responses. Full-generation scoring is
  required for valid jailbreak severity measurement.
- *Binary-versus-graded shift.* Binary evaluation washed out a dose-response
  ($+2.30$ pp/$\alpha$, CI excludes zero) that graded evaluation recovered.
  The mechanism is collapse of the borderline category.
- *Holdout compression.* A 12.2 pp evaluator-accuracy gap compressed to
  2.0 pp (McNemar $p = 1.0$) after removing calibration-contaminated rows.
  Apparent evaluator advantages can be substantially inflated by
  development-set overlap.
- *Seed-0 specificity.* H-neuron scaling produced a steeper jailbreak
  dose-response than a matched random-neuron control (slope difference
  $+2.77$ pp/$\alpha$ $[+1.17, +4.42]$, permutation $p = 0.013$),
  but this is a single-seed result.
- *Contamination fix.* A schema-version mismatch silently reclassified
  97.7\% of borderline records. The fix was four lines; the cost of missing
  it would have been a qualitatively wrong triage verdict.

**Still pending:**

- Multi-seed replication (seeds 1--2 generated, not yet scored).
- Full CSV2 v3 and StrongREJECT scoring on control-condition data.
- Fresh hard-case gold labels for an uncontaminated test of evaluator
  advantages on new refuse-then-comply responses.
- Field-level audit of CSV2 v3 ordinal components (C, S, V, T) against
  human ordinal judgments.

\medskip

In this setting, measurement choices were not clerical details; they changed
what the project would have concluded about whether an intervention worked.
A 256-token generation cap would have hidden the harmful payload.
Binary scoring would have returned a null result. A single evaluator---any of
the four we tested---would have left the construct-sensitivity of the
conclusion invisible. And a schema mismatch in four lines of pipeline code
would have reversed the triage verdict. Each of these measurement decisions
interacts with intervention-altered response structure: longer refusal
preambles, graded compliance, and evaluator-specific operationalizations of
"harmful" are not noise to be averaged away but signal about how the
intervention reshapes model behavior. For intervention research that aims to
make safety claims, the measurement stack is part of the result.
