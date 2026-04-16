# 5. Case Study II — Control Is Surface-Local and Can Externalize

The previous section established that strong readouts do not reliably identify useful steering targets. This section asks a complementary question: when steering *does* work, how far does the effect transfer? We find that successful interventions are surface-local — gains on one evaluation construct do not port to nearby constructs — and that the relevant externality is not generic degradation but a specific behavioral failure mode.

Figure 3 shows the bridge result at a glance: answer-selection gains on TruthfulQA coexist with generation damage on nearby factual surfaces, and the largest single failure mode is wrong-entity substitution.

![Figure 3. Surface-local control and bridge failure modes.](figures/fig3_bridge_failure.png)
*Figure 3. TruthfulQA answer-selection gains do not transfer to generation: the bridge benchmark shows net damage, and the most frequent manually diagnosed failure mode is wrong-entity substitution rather than refusal.*

## 5.1 Positive Results Exist, but Are Task-Local

Before testing transfer, we establish that detector-selected interventions can in fact steer behavior on their primary surfaces. H-neuron scaling (magnitude-ranked selection, 38 neurons) produces clear, dose-dependent effects on compliance-adjacent surfaces. On FaithEval anti-compliance prompts, MCQ context-acceptance rate increases by +4.5 pp [2.9, 6.1] above the no-op baseline at $\alpha = 3.0$ (slope +2.09 pp/$\alpha$ [1.38, 2.83]), confirmed by 8-seed negative controls that show flat random-neuron baselines (slope +0.02 pp/$\alpha$).^[Source: `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`; H-neuron sweep in `data/gemma3_4b/intervention/faitheval/experiment/results.json`.] On FalseQA, the same neurons show a dose-response slope of +1.62 pp/$\alpha$ [0.52, 2.74].^[Sources: `data/gemma3_4b/intervention/falseqa/experiment/results.json`; `data/gemma3_4b/intervention/falseqa/control/comparison_summary.json`; `data/gemma3_4b/intervention/falseqa/falseqa_negative_control_audit.md` (three random-neuron control seeds across three alphas).] On jailbreak, single-seed controls provide initial specificity support (seed-0 only; seeds 1–2 pending): H-neuron slope +2.30 pp/$\alpha$ [0.99, 3.58] versus random-neuron slope $-0.47$ pp/$\alpha$ [$-1.42$, 0.47], difference +2.77 pp/$\alpha$ [1.17, 4.42], $p = 0.013$.^[Source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`.]

These results establish that detector-selected targets can produce real, specific steering effects. The issue is not that detection-based selection never works — it is that working effects are task-local. The same 38 neurons that shift compliance on FaithEval show no robust net alias-accuracy effect on BioASQ factoid QA: $-0.06$ pp [$-1.5$, 1.4] on a well-powered sample ($n = 1{,}600$, MDE ${\sim}2$ pp), despite substantially perturbing answer style in 1,339 of 1,600 responses.^[Source: `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`.] The endpoint is flat, but the intervention is behaviorally active — this is a portability limit on the metric, not behavioral inactivity. The intervention modulates answer style and compliance-adjacent behavior but does not improve domain-specific factual accuracy under the current alias metric.

## 5.2 ITI Improves Answer Selection but Not Open-Ended Generation

Inference-Time Intervention using TruthfulQA-sourced truthfulness directions (Li et al., 2023) produces a clear improvement on answer selection: +6.3 pp MC1 [3.7, 8.9] and +7.49 pp MC2 [5.28, 9.82] on held-out TruthfulQA folds, with 61 incorrect→correct flips against 20 correct→incorrect flips at $\alpha = 8.0$.^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §2.]

This gain does not transfer to open-ended factual generation. On SimpleQA ($n = 1{,}000$) with a prompt that removes the explicit escape hatch, ITI at $\alpha = 8.0$ reduces correct-answer rate from 4.6% [3.5, 6.1] to 2.8% [1.9, 4.0] (paired $\Delta = -1.8$ pp [$-3.1$, $-0.6$]).^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §3b.] The attempt rate collapses from 99.7% to 67.0%, indicating that the truthfulness direction promotes epistemic caution — the model hedges or refuses rather than generating factual answers. Precision among attempted answers remains stable (${\sim}4$%), suggesting the intervention does not improve factual accuracy; it merely suppresses generation.

The contrast is stark: on a constrained answer-selection task (TruthfulQA MC), the same direction helps the model pick correct options. On an open-ended generation task (SimpleQA), it suppresses attempts without improving accuracy. Answer-selection success is not evidence for generation-level truthfulness improvement.

## 5.3 Bridge: The Sharpest Behavioral Diagnosis

The TriviaQA bridge benchmark (held-out test set, $n = 500$, baseline adjudicated accuracy 45.0%) provides the most informative externality result because it reveals a specific behavioral failure mode, not just a score drop. The test set was evaluated under a one-shot frozen protocol: all parameters were locked from Phase 2 dev results, the test manifest was generated at split time, and a pipeline guard prevented re-running. No parameters were tuned on test data.^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §§1, 2.1.]

**The intervention is active, not simply suppressive.** At $\alpha = 8.0$, the baseline TruthfulQA-sourced ITI variant (E0, $K = 12$) reduces adjudicated accuracy by $-5.8$ pp [$-8.8$, $-3.0$] (CI excludes zero), with 43 right-to-wrong flips against 14 wrong-to-right rescues (net $-29$, McNemar $p = 0.0002$).^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §§2.2-2.3.] Both the primary (adjudicated) and floor (deterministic) accuracy deltas exclude zero ($-5.8$ pp [$-8.8$, $-3.0$] and $-4.6$ pp [$-7.6$, $-1.6$] respectively), ruling out a grading artifact as the explanation for the observed harm.

**The observed harm is not primarily explained by refusal or grading loss.** NOT\_ATTEMPTED counts increase from 2 to 8, a statistically significant but small effect (1.2 pp). Formal refusal accounts for only 2 of 43 right-to-wrong flips (5%). The dominant damage is factual corruption, not silence.^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §§2.1, 5.1.]

**The most frequent manually diagnosed failure mode is wrong-entity substitution.** Of the 43 right-to-wrong flips at E0 $\alpha = 8.0$, 30 were manually classified as substitutions where the model replaces a correct factual answer with a different entity from the same semantic neighborhood (about 70% of the coded flips):^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §5.2.]

| Question | Baseline (correct) | ITI $\alpha = 8$ (wrong) |
|---|---|---|
| Danny Boyle 1996 film? | "Trainspotting" | "Slumdog Millionaire" (same director) |
| Third musician in 1959 crash? | "Ritchie Valens" | "J.P. Richardson" (same crash) |
| Family Guy spin-off character? | "Cleveland Brown" | "Peter Griffin" (same show) |
| Dickens novel with Merdle & Sparkler? | "Little Dorrit" | "Bleak House" (same author) |
| DC comic introducing Superman? | "Action Comics" | "Detective Comics" (same publisher) |

This pattern is consistent with a coarse behavior-level reweighting hypothesis toward nearby but wrong candidates rather than simple suppression of generation. The failure-mode taxonomy is based on single-rater manual classification; we therefore treat the exact shares as approximate even though wrong-entity substitution is the most frequent diagnosed mode at both dev scale (5/10 flips) and test scale (30/43 flips).

**A secondary failure mode is factual denial.** Eight of 43 R2W flips (19%) involve the model asserting that well-established facts are uncertain or false (e.g., "He did not complete a Puccini opera" — the answer is Turandot). This is qualitatively different from wrong-entity substitution and is consistent with the intervention sometimes over-amplifying epistemic caution on factual questions, though that interpretation remains a hypothesis rather than an established mechanism.^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §5.3.]

**The 14 wrong-to-right rescues are informative.** These rescues show that the intervention is behaviorally active in both directions: on some questions it moves a wrong baseline answer to the correct entity, though on this surface the damage rate is larger than the rescue rate. This bidirectional pattern is consistent with a coarse redistribution hypothesis, but it does not by itself establish how the direction operates internally.^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §6.]

> **Box B — Worked Bridge Rescue Example**
>
> On `qw_4300` ("First pop video by John Landis?"), the baseline answer was
> "Saturday Night's Alright for Fighting," which is wrong. ITI corrected this
> to "Thriller" at both $\alpha = 4$ and $\alpha = 8$ on the bridge dev set.
> This matters because the bridge story is not "ITI only breaks generation."
> The intervention is behaviorally active in both directions: it can rescue a
> few questions, but the damage rate is larger than the rescue rate on this
> surface.^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`,
> §4.1.]

**The failure is reproducible across intervention variants.** On the dev split ($n = 100$), both E0 and E1 produced exactly 10 right-to-wrong flips at $\alpha = 8.0$, damaging the same set of questions (E0 rescues 3, E1 rescues 1). E1 ($K = 8$) showed a larger deficit ($-9.0$ pp [$-16.0$, $-3.0$], McNemar $p = 0.016$); E1 was not run on the test set.^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §2.3.] This suggests the observed harm is not obviously idiosyncratic to a single direction variant, while leaving the exact cause of the failure open.

**Three additional failure modes account for the remaining 13 R2W flips:** evasion / factual denial (8/43, 19%), answer dilution from verbosity (3/43, 7%), and formal refusal (2/43, 5%).^[Source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`, §5.1.]

## 5.4 Externality as First-Class Evidence

The same ITI direction that improves TruthfulQA multiple-choice performance degrades held-out open-ended factual generation, primarily through manually diagnosed right-to-wrong factual substitutions rather than refusal. This dissociation is between constrained answer selection and open-ended exact factual generation — not merely between MC and generation formats. The bridge result is also consistent with one behavior-level explanation for why constrained selection can improve while open-ended generation worsens: a coarse direction that shifts relative plausibility among candidate answers may help when the correct answer is already present in a small candidate set, but harm when the model must retrieve and emit the exact entity in an open-vocabulary setting.

The H-neuron scope boundary reinforces the same lesson from a different direction: an intervention that works on compliance-adjacent surfaces (FaithEval, FalseQA, jailbreak) shows no robust net alias-accuracy effect on BioASQ, indicating a narrow over-compliance lever rather than general truthfulness improvement.

We use "wrong-entity substitution" as a behavioral diagnosis, not as a claim about the internal circuit mechanism.
