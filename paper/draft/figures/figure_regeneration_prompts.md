# Figure Regeneration Prompts

These prompts are written to be tool-agnostic. Keep the `CONTENT` section fixed to preserve the scientific claim. Change `STYLE` and `LAYOUT` freely to explore better visual directions.

General constraints for all four:
- Preserve all numbers, labels, confidence intervals, panel titles, and causal claims exactly.
- Do not invent extra data, mechanisms, or conclusions.
- Prefer vector-friendly output with publication-quality typography.
- Avoid default matplotlib, PowerPoint, or generic dashboard aesthetics.
- Optimize for legibility at paper scale first, novelty second.

## Prompt 1: Figure 1

```text
Create a publication-quality conceptual scientific figure titled "The Four-Stage Interpretability Scaffold".

CONTENT
- This is a conceptual opener, not a data chart.
- Show a left-to-right four-stage pipeline with exactly these stages:
  1. Measurement — "Can we trust the evaluation?"
  2. Localization — "Where is the feature?"
  3. Control — "Can we steer it?"
  4. Externality — "Does it transfer?"
- The stages are connected in sequence.
- Mark three breakpoints at the transitions, each representing where the paper's evidence shows the readout-to-steering heuristic can fail.
- Label the three breakpoints with these concise anchors:
  - "Measurement -> Conclusion: truncation, grading, evaluator"
  - "Localization -> Control: SAE vs H-neurons, steering"
  - "Control -> Externality: ITI gain vs bridge harm"
- The central message is that interpretability evidence must pass through four distinct gates, and failure can occur at the transitions rather than only inside a single stage.

STYLE
- Make it feel like a strong editorial scientific infographic for a top-tier ML paper, not a corporate process slide.
- Use a restrained, intelligent palette: cool structural tones for the four stages, and a warm accent color only for breakpoints or failure markers.
- Typography should feel deliberate, crisp, and publication-grade.
- The visual hierarchy should emphasize the four stages first, the breakpoints second, and the explanatory subtitles third.
- Keep the mood serious, modern, and rigorous.
- Avoid tiny annotation boxes, weak pastel office aesthetics, and clip-art-like arrows.

LAYOUT
- Use a wide landscape composition.
- Place the four stages as large, evenly spaced modules with strong alignment.
- Put the breakpoint markers directly on the transitions so the diagram reads as a chain with failures occurring between stages.
- The anchor labels should be short and clearly legible at manuscript size; they can sit beneath the transitions as compact callouts.
- Use generous whitespace and a clean top title area.
- The final figure should read cleanly in under three seconds, with no cramped text.
```

## Prompt 2: Figure 2

```text
Design a publication-quality two-panel scientific figure about matched readout quality but divergent steering behavior on FaithEval.

CONTENT
- Figure title: "FaithEval Matched Readouts, Divergent Control".
- The figure has two panels.

- Panel A is a bar chart showing matched detection quality:
  - H-neurons (38): AUROC 0.843 with 95% CI [0.815, 0.870]
  - SAE features (266): AUROC 0.848 with 95% CI [0.820, 0.874]
  - The interpretive message is: held-out detection quality is similar, but this is not a formal equivalence claim.
  - Include a subtle annotation conveying: "95% bootstrap CIs overlap; matched means similar held-out AUROC, not a formal equivalence test."

- Panel B is a line chart showing steering divergence across scaling factor alpha.
  - X-axis values: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
  - H-neuron compliance rates: 0.642, 0.654, 0.660, 0.670, 0.682, 0.695, 0.705
  - H-neuron 95% CI lower: 0.612, 0.624, 0.630, 0.640, 0.652, 0.666, 0.676
  - H-neuron 95% CI upper: 0.671, 0.683, 0.689, 0.698, 0.710, 0.723, 0.732
  - SAE feature compliance rates: 0.723, 0.747, 0.660, 0.750, 0.751, 0.749, 0.699
  - SAE feature 95% CI lower: 0.694, 0.719, 0.630, 0.722, 0.723, 0.721, 0.670
  - SAE feature 95% CI upper: 0.750, 0.773, 0.689, 0.776, 0.777, 0.775, 0.727
  - Random SAE feature baseline: 0.749, 0.748, 0.660, 0.750, 0.749, 0.749, 0.746
  - Random SAE standard deviation band: 0.0041, 0.0036, 0.0000, 0.0021, 0.0009, 0.0012, 0.0052
  - Include the slope-gap annotation: "Delta slope = +1.93 pp/alpha, 95% CI [+0.94, +2.92]"
- The overall story is: similar readout quality does not guarantee similar control quality.

STYLE
- Make this feel like a polished journal figure, with strong statistical clarity and restrained confidence.
- Use color to separate H-neurons, SAE features, and random SAE features clearly, but avoid neon or dashboard colors.
- Emphasize the divergence story visually more than the matched-AUROC story.
- Keep uncertainty visible but elegant.
- Annotation boxes should be subtle and high-end, not loud or sticky-note-like.
- Avoid clutter, redundant framing, and oversized legends.

LAYOUT
- Use a balanced two-panel landscape layout.
- Give Panel B slightly more visual weight than Panel A, because the steering divergence is the main result.
- Panel titles should be concise and aligned.
- Keep the y-axis ranges honest and readable; do not visually exaggerate the AUROC gap.
- Put the legend where it does not compete with the data.
- Make the confidence bands light enough that the curves remain primary.
```

## Prompt 3: Figure 3

```text
Create a bold, publication-quality three-panel scientific figure showing that answer-selection gains do not transfer cleanly to open-ended generation, and that the dominant failure mode is wrong-entity substitution.

CONTENT
- Figure title: "Surface-Local Control and Bridge Failure Modes".

- Panel A is a bar chart of ITI effect size in percentage points:
  - TruthfulQA MC1 (surface-local): +6.3 pp, 95% CI [+3.7, +8.9]
  - SimpleQA correct-answer rate: -1.8 pp, 95% CI [-3.1, -0.6]
  - TriviaQA bridge (E0 alpha=8, test): -5.8 pp, 95% CI [-8.8, -3.0]
  - The message is: the intervention helps constrained answer selection but harms nearby generation surfaces.

- Panel B is a taxonomy of right-to-wrong flips on the bridge benchmark, n=43 flips:
  - Wrong-entity substitution: 30 (70%)
  - Evasion / factual denial: 8 (19%)
  - Verbosity / dilution: 3 (7%)
  - Formal refusal: 2 (5%)
  - The central message is that wrong-entity substitution is the dominant observed failure mode, not refusal.

- Panel C is a readable comparison table or card-based typographic panel with these examples:
  - Question: "Danny Boyle 1996 film?" | Baseline correct: "Trainspotting" | ITI alpha=8 wrong: "Slumdog Millionaire" (same director)
  - Question: "Third musician in 1959 crash?" | Baseline correct: "Ritchie Valens" | ITI alpha=8 wrong: "J.P. Richardson" (same crash)
  - Question: "Family Guy spin-off character?" | Baseline correct: "Cleveland Brown" | ITI alpha=8 wrong: "Peter Griffin" (same show)
  - Question: "DC comic introducing Superman?" | Baseline correct: "Action Comics" | ITI alpha=8 wrong: "Detective Comics" (same publisher)
- The figure should make the scientific point that the harm is active factual corruption, not simple silence.

STYLE
- This figure can be the strongest and most editorial of the four, while still feeling rigorous.
- Use a visual language that cleanly separates gain from harm: one cool tone for beneficial change, one warm tone for harmful change, and neutral support tones for secondary categories.
- The bottom example panel should feel like refined scientific typography, not a spreadsheet screenshot.
- Highlight wrong-entity substitution as the dominant category without making the rest disappear.
- Aim for clarity, sharp contrast, and a memorable silhouette.
- Avoid cramped tables, default gridlines, and generic business-chart styling.

LAYOUT
- Use a three-part composition with Panels A and B on the top row and Panel C spanning the full width below.
- Give Panel C enough room for comfortable reading and line wrapping.
- Panel A should communicate the sign change instantly.
- Panel B should foreground the dominance of the substitution bar.
- The whole figure should feel cohesive, with the narrative flowing from performance summary to taxonomy to concrete examples.
```

## Prompt 4: Figure 4

```text
Design a rigorous, publication-quality three-panel scientific figure showing that measurement choices changed the scientific conclusion.

CONTENT
- Figure title: "Measurement Choices Changed the Scientific Conclusion".

- Panel A is a binary scoring line chart with uncertainty:
  - X-axis: scaling factor alpha values 0 and 3
  - H-neurons: 30.4% at alpha 0 with 95% CI [26.53, 34.57]; 33.4% at alpha 3 with 95% CI [29.41, 37.65]
  - Random control: 31.4% at alpha 0 with 95% CI [27.49, 35.60]; 30.4% at alpha 3 with 95% CI [26.53, 34.57]
  - Message: under binary scoring the effect looks weak and non-decisive.

- Panel B is a graded scoring line chart with uncertainty:
  - X-axis: scaling factor alpha values 0.0, 1.0, 1.5, 3.0
  - H-neurons harmfulness rates: 18.8%, 24.6%, 23.6%, 26.4%
  - H-neurons 95% CI lower: 15.62, 21.03, 20.09, 22.73
  - H-neurons 95% CI upper: 22.46, 28.56, 27.51, 30.43
  - Random control harmfulness rates: 24.2%, 22.6%, 22.4%, 22.6%
  - Random control 95% CI lower: 20.65, 19.15, 18.96, 19.15
  - Random control 95% CI upper: 28.14, 26.47, 26.26, 26.47
  - Include the key comparison text: H-neurons slope +2.30 pp/alpha, random control slope -0.47 pp/alpha, slope gap +2.77 pp/alpha.
  - Message: graded scoring recovers a meaningful dose-response that binary scoring obscures.

- Panel C is a holdout evaluator accuracy comparison with confidence intervals:
  - CSV2 v3: 96.0%, 95% CI [90.0, 100.0]
  - StrongREJECT (SR-4o): 96.0%, 95% CI [90.0, 100.0]
  - CSV2 v2: 92.0%, 95% CI [84.3, 98.0]
  - Binary judge: 90.0%, 95% CI [80.0, 98.0]
  - Use the contextual note: post-rerun holdout result, 95% CIs are prompt-clustered bootstrap intervals, n=17 prompt IDs and 50 rows.
  - The interpretive message is: CSV2 v3 and StrongREJECT are tied on binary holdout accuracy; the reason to keep CSV2 v3 is richer outcome taxonomy, not superior binary accuracy.

STYLE
- Make this figure feel statistically disciplined, publication-grade, and slightly more austere than the others.
- Prioritize trustworthiness, hierarchy, and precise labeling over decorative flair.
- Use consistent visual identities for H-neurons, random control, and the four evaluator bars, but keep the palette refined.
- Annotations should be quiet and exact.
- Avoid any visual treatment that suggests binary and graded results are directly comparable on the same scale.

LAYOUT
- Use a three-panel layout with binary scoring and graded scoring on the top row, and evaluator accuracy spanning the full width below.
- Make Panel B slightly more salient than Panel A because it carries the stronger scientific signal.
- Clearly separate units: Panel A and B are outcome rates on different measurement surfaces, Panel C is evaluator accuracy.
- Keep error bars and confidence bands highly legible.
- The overall composition should make the narrative obvious: binary looks weak, graded reveals signal, evaluator comparison reframes why one rubric is preferred.
```
