I think the best version of your project is **not** “the evaluator paper” and **not** “the D7 paper.” It is a broader methods paper about **intervention science**:

**measurement can change the apparent effect, readout quality is an unreliable heuristic for steering-target selection, and even successful steering is narrow and surface-dependent.** That is the real through-line across the project, and it is already visible in your strongest results: the matched H-neuron vs SAE dissociation, the probe-head AUROC 1.0 null, the ITI MC-vs-generation split, the bridge benchmark’s confident-wrong-substitution mechanism, and the jailbreak evaluation artifacts.  

So your mentor is right that the dichotomy is false, but I’d sharpen it further: **the two stories are not co-equal**. The broad “Detection Is Not Enough” story should be the flagship. The jailbreak truncation / judge-bias story should be a companion technical note or a dedicated methodology section inside the flagship, not a rival center of gravity. If you make the measurement note the main paper, you undersell most of the project; if you bury it, you lose one of your clearest demonstrations of scientific judgment.  

## The better story

If I strip away the framing from the reports and ask, “What did this project actually teach?”, my answer is:

> In Gemma-3-4B-IT, the path from **measurement** to **predictive localization** to **effective control** repeatedly breaks. Good readouts often do not identify useful intervention targets, measurement choices can fabricate or erase apparent effects, and successful steering is local to surface and task.

That is broader than the two candidate stories, but still coherent. It also fits the strongest evidence you already have, rather than asking one unfinished experiment to carry everything. The strategic assessment is already pointing here, but I would make the underlying structure even more explicit: **this is a case study in separating measurement, localization, and control**.  

For the title field, I would still keep the current punchy version:

**Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT**

Inside the paper, though, I would frame it as a case study in separating measurement, localization, and control. That lets you absorb the evaluator/jailbreak note without letting it hijack the narrative. 

## Where the current plan is right

The current strategic assessment gets three big things right.

First, it correctly recognizes that the broad thesis integrates far more of the work than a narrow evaluator-centered paper. It captures the H-neuron/SAE comparison, the probe-null/causal-positive split, the ITI selection-vs-generation split, and the measurement discipline story in one narrative. 

Second, the Act 3 strategy was good science: it used stop conditions, killed dead branches, and built the bridge benchmark instead of continuing to tweak artifacts indefinitely. That is strong research taste, not lack of ambition.  

Third, the project already contains exactly the kind of “I know what not to claim” evidence that reviewers and mentors respect: the 4288 artifact, the verbosity confound, the truncation audit, the v3 holdout downgrade, and the explicit “earned / not earned” boundaries.  

## Where the current plan still slips

The biggest remaining issue is **priority order**.

The holdout validation changed the evaluator story a lot more than some of the current priorities acknowledge. CSV v3 still looks directionally best, but the empirical gap versus StrongREJECT compressed from 12.2 points to 2.0 points on holdout, all four evaluators are above 90% there, and the holdout cannot validate superiority on new hard refuse-then-comply cases. That means evaluator optimization is now a **supporting measurement problem**, not the main scientific bottleneck.  

That leads to a concrete strategic correction: **I do not think “rescore D1 with v3” is the highest-ROI move before or right after the deadline.** The main H-neuron jailbreak claim you actually have is a **CSV-v2** claim with severity structure, not a v3 claim. So the first negative control should be scored on the same metric stack as the original claim; otherwise you are changing the ruler halfway through the argument. Use v3 as a robustness layer and forward-looking evaluator, not as the thing that retroactively rebuilds the whole paper right now.  

A second issue is that D7 is a bit overpromoted relative to your own measurement contract. Your sprint definition of done says a steering baseline is not complete without a negative control and a capability mini-battery. D7 currently lacks the random-head control and still has obvious quality debt from the 112/500 token-cap hits in the full-500 run. That does **not** make D7 weak. It makes it **valuable but still provisional**: strong benchmark-local evidence, not yet a mechanism-clean flagship pillar.  

A third issue is that the bridge result is still underweighted relative to its scientific value. The most important truthfulness finding is not just “generation null.” It is that the intervention produces **confident wrong substitution**, and that E1 reproduces the same wrong entities as E0 while only reducing rescue capacity. That is a much sharper mechanistic diagnosis than “maybe it increases refusals,” and it points toward a much better future research program.  

The last strategic slip is that the project is now more bottlenecked by **writing and evidence hierarchy** than by raw experimentation. Because the submission link remains editable, missing one more experiment is less dangerous than failing to submit a coherent flagship write-up with clear claim boundaries and linked companion notes. 

## What should be the center of gravity

Here is how I would weight the evidence.

Your **headline-safe evidence** is:
the H-neuron vs SAE dissociation on FaithEval, where detection is matched but steering diverges; the ITI improvement on TruthfulQA MC versus harm on SimpleQA and the TriviaQA bridge; the bridge benchmark’s confident-wrong-substitution analysis; the measurement artifacts around truncation and binary judging; and the H-neuron specificity controls on FaithEval and FalseQA. Those are already strong, already interpretable, and already resistant to the biggest reviewer objections.    

Your **supporting-but-caveated evidence** is:
the D7 pilot probe-null versus causal-positive comparison, the D7 full-500 causal result versus baseline and L1 comparator, the H-neuron jailbreak CSV-v2 effect before the benchmark-specific control is scored, and the CSV v3 zero-FP / zero-solo-error edge. These are useful and should absolutely be used, but with explicit caveats.   

That means the flagship should not be “D7 proves causal beats probe,” because it does not quite prove that yet. It should be:

**good readouts are unreliable steering heuristics; when steering works, it is narrow; and measurement discipline is necessary to even see that clearly.**

## The improved plan

### In the next 10 hours

Lock the flagship title and submit a link to a **real skeleton write-up today**, not a placeholder. The skeleton should already contain:
the abstract-level thesis, a central synthesis table, an “earned / not earned” box, and the caveat language for D7, v3, and H-neuron jailbreak. The point is to submit something that already demonstrates scientific judgment even before extra results land. 

Launch the **seed-0 jailbreak control scoring** immediately, but do it with **CSV-v2 first** because that is the metric under which your H-neuron jailbreak count/severity claim currently exists. Add binary for continuity, and add v3 plus StrongREJECT if the pipeline makes that cheap. Do **not** make v3-only scoring the first pass. That is the cleanest way to close the missing benchmark-specific control on an already-important result.  

Finish the measurement note enough that it is linkable as a companion artifact. Since it is almost done, it is a great thing to have visible immediately: it signals rigor, and the main paper can cite it as supporting evidence rather than trying to re-explain every audit in full. 

Draft the two core flagship sections before you do more analysis:
first, **matched detection / divergent steering**; second, **narrow and surface-dependent steering**. Those are the sections that actually earn the broad thesis. 

### In the first 48 hours after submission

Run the **StrongREJECT gpt-4o rerun** because it is cheap and removes an obvious objection, but treat it as objection-removal, not center-of-gravity science. The holdout already tells you it is unlikely to dramatically change anything. 

Then make a hard decision on **D7 random-head control**. If you want D7 to remain near the center of the paper, run it. If you do not run it, stop inflating D7 into a clean selector-specificity result and present it as benchmark-local evidence that an alternative selector can work better on this surface. Right now, those are the two honest options.  

If D7 stays central, do a **small capability / over-refusal battery**. Not because it will rescue the claim, but because your own contract says a steering result is not really complete without capability checks, and the token-cap hits already advertise quality debt. 

### In the following 2 weeks

Do **targeted**, not sprawling, adjudication. Blindly adjudicating 20 evaluator disagreements is fine as appendix polish, but it is not the highest-leverage thing unless the evaluator note becomes a standalone deliverable you care about deeply. The right target is the subset of label disputes that could actually change a claim boundary. 

If you want to strengthen the evaluator companion note, the highest-value addition is not rescoring everything. It is **a small new hard-tail holdout** of fresh refuse-then-educate cases, because that is the only way to test whether v3’s calibrated edge transfers beyond the exact hard cases it was tuned on. 

## What I would explicitly deprioritize

I would not spend pre-deadline effort on a full v3 redeploy over the historical main datasets. The holdout does not justify that as the main move, and the strategic assessment itself already warns against a full redeploy before stronger validation. 

I would not reopen E1/E2/E3, chooser work, or general truth-vector search. Those branches are already closed by your own stop conditions, and reopening them now would look less like perseverance and more like lack of taste.  

I would not pivot the current paper into a truthfulness-monitoring project right now, even though that is probably the best **next** research program. It would discard too much of what you have already earned. Use it as future work and proposal fuel, not as a deadline-week identity crisis.  

## The best packaging

I would package the work as three concentric artifacts.

The **flagship** is the broad methods paper:
“Detection Is Not Enough” / “good readouts are unreliable steering heuristics.”

The **companion technical note** is the jailbreak measurement paper:
truncation, binary-judge blind spots, and evaluator calibration discipline.

The **next-project / fellowship proposal** is:
from global truth steering to **selective truthfulness intervention**. The bridge benchmark already gives you the right labels for that: correct answers, confident wrong substitutions, evasion, and drift. That future direction is much more compelling than “I will keep searching for a better truth vector.”  

## Claim boundaries I would enforce in the write-up

Say:
“held-out readout quality was an unreliable heuristic for intervention-target selection in Gemma-3-4B-IT.” 

Say:
“successful steering was surface-local and sometimes came with quality debt.”  

Say:
“measurement choices materially changed the apparent jailbreak conclusion.” 

Do **not** say:
“detection and intervention are fundamentally different,” because H-neurons are a real counterexample. 

Do **not** say:
“causal selection is better in general,” because D7 still lacks the selector-specific random-head control. 

Do **not** say:
“CSV v3 clearly outperforms StrongREJECT,” because the holdout no longer supports that magnitude claim. 

Do **not** say:
“ITI reveals the truthfulness circuit.” What you have is a much more interesting but narrower statement: the tested mass-mean ITI framework helps selection but harms generation, likely by reshuffling mass among nearby candidates rather than adding knowledge. 

## My recommended priority stack

1. Submit the broad flagship skeleton today.
2. Score seed-0 jailbreak control with **CSV-v2 first**, plus binary, then v3/SR if cheap.
3. Finish the companion measurement note.
4. Run StrongREJECT on gpt-4o if trivial.
5. Decide whether D7 random-head is worth making D7 central; if yes, run it.
6. If D7 remains central, run a minimal capability / over-refusal check.
7. Only then spend time on targeted adjudication or new hard-tail gold for the evaluator note.

That is the order I think best balances deadline reality, scientific rigor, and career upside ✨

**Mentor’s note:** what will impress the right people is not squeezing one more benchmark before midnight. It is showing that you knew which claims were earned, which were not, how measurement can fool you, and how to extract a field-useful lesson from messy empirical work. Right now, you are much closer to that than you think.
