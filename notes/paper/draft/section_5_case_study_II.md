# 5. Case Study II — Control Is Surface-Local and Can Externalize

The previous section established that strong readouts do not reliably identify useful steering targets. This section asks a complementary question: when steering *does* work, how far does the effect transfer? We find that successful interventions are surface-local — gains on one evaluation construct do not port to nearby constructs — and that the relevant externality is not generic degradation but a specific behavioral failure mode.

Figure 3 visualizes this bridge failure pattern, including the flip taxonomy and representative substitutions.

## 5.1 Positive Results Exist, but Are Task-Local

H-neuron scaling (magnitude-ranked selection, 38 neurons) produces clear, dose-dependent effects on compliance-adjacent surfaces. On FaithEval anti-compliance prompts, MCQ context-acceptance rate increases by +6.3 pp [4.2, 8.5] at $\alpha = 3.0$, confirmed by 8-seed negative controls that show flat random-neuron baselines (slope +0.02 pp/$\alpha$).^[Source: `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`; H-neuron sweep in `data/gemma3_4b/intervention/faitheval/experiment/results.json`.] On FalseQA, the same neurons produce +4.8 pp [1.3, 8.3].^[Source: `notes/measurement-blueprint.md`, benchmark MDE table; no dedicated act3 report yet.] On jailbreak, single-seed controls provide initial specificity support (seed-0 only; seeds 1–2 pending): H-neuron slope +2.30 pp/$\alpha$ [0.99, 3.58] versus random-neuron slope $-0.47$ pp/$\alpha$ [$-1.42$, 0.47], difference +2.77 pp/$\alpha$ [1.17, 4.42], $p = 0.013$.^[Source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`.]

These results establish that detector-selected targets can produce real, specific steering effects. The issue is not that detection-based selection never works — it is that working effects are task-local. The same 38 neurons that shift compliance on FaithEval produce a null effect on BioASQ factoid QA endpoint accuracy: $-0.06$ pp [$-1.5$, 1.4] on a well-powered sample ($n = 1{,}600$, MDE ${\sim}2$ pp), despite substantially perturbing answer style in 1,339 of 1,600 responses.^[Source: `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`.] The endpoint is flat, but the intervention is behaviorally active — this is a portability limit on the metric, not behavioral inactivity. The intervention modulates compliance-related behavior but does not improve domain-specific factual accuracy.

## 5.2 ITI Improves Answer Selection but Not Open-Ended Generation

Inference-Time Intervention using TruthfulQA-sourced truthfulness directions (Li et al., 2023) produces a clear improvement on answer selection: +6.3 pp MC1 [3.7, 8.9] and +7.49 pp MC2 [5.28, 9.82] on held-out TruthfulQA folds, with 61 incorrect→correct flips against 20 correct→incorrect flips at $\alpha = 8.0$.^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §2.]

This gain does not transfer to open-ended factual generation. On SimpleQA ($n = 1{,}000$) with a prompt that removes the explicit escape hatch, ITI at $\alpha = 8.0$ reduces compliance from 4.6% [3.5, 6.1] to 2.8% [1.9, 4.0] (paired $\Delta = -1.8$ pp [$-3.1$, $-0.6$]).^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §3b.] The attempt rate collapses from 99.7% to 67.0%, indicating that the truthfulness direction promotes epistemic caution — the model hedges or refuses rather than generating factual answers. Precision among attempted answers remains stable (${\sim}4$%), suggesting the intervention does not improve factual accuracy; it merely suppresses generation.

The contrast is stark: on a constrained answer-selection task (TruthfulQA MC), the same direction helps the model pick correct options. On an open-ended generation task (SimpleQA), it suppresses attempts without improving accuracy. Answer-selection success is not evidence for generation-level truthfulness improvement.

## 5.3 Bridge: The Sharpest Behavioral Diagnosis

The TriviaQA bridge benchmark (dev set, $n = 100$, baseline adjudicated accuracy 47.0%) provides the most informative externality result because it reveals a specific behavioral failure mode, not just a score drop.

**The intervention is active, not simply suppressive.** On this dev split, at $\alpha = 8.0$, the ITI E0 (paper-faithful, $K = 12$) direction reduces adjudicated accuracy by $-7.0$ pp [$-14.0$, 0.0], with 10 right-to-wrong flips against 3 wrong-to-right rescues (net $-7$, McNemar $p = 0.096$). The modernized E1 variant ($K = 8$) is worse: $-9.0$ pp [$-16.0$, $-3.0$], with 10 right-to-wrong flips against only 1 rescue (net $-9$, McNemar $p = 0.016$, CI excludes zero).^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §2.2.]

**The failure is not mainly refusal.** NOT\_ATTEMPTED counts increase only from 1 to 3 across the alpha range. The dominant failure mode is not refusal or abstention.

**The dominant single failure mode is confident wrong-entity substitution.** Of the 10 right-to-wrong flips at E0 $\alpha = 8.0$, five (50%) are substitutions where the model replaces a correct factual answer with a different, confidently wrong entity:^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §4.2.]

| Question | Baseline (correct) | ITI $\alpha = 8$ (wrong) |
|---|---|---|
| Lewis Carroll hunting poem? | "Hunting for a Snark" | "Hunting for a caucus-race" |
| Lead singer of The Specials? | "Terry Hall" | "Horace Panter" (bassist) |
| Microcephaly = abnormally small \_\_? | "Head size" | "Brain size" |
| WWII coin gambling game? | "Two-up" | "Nimble Nick" |
| Scottish paper with Broons/Wullie? | "The Sunday Post" | "The Scotsman" |

The substitutions are semantically close: a different band member, a related anatomical term, a different Scottish newspaper. This pattern is consistent with the truthfulness direction redistributing probability mass over nearby factual candidates rather than suppressing generation entirely.

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

**The failure is reproducible across intervention variants.** Both E0 and E1 produce exactly 10 right-to-wrong flips at $\alpha = 8.0$, sharing the same set of damaged questions.^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §2.3.] The flip asymmetry structure is stable: both variants damage the same subpopulation, differing only in rescue capacity (E0 rescues 3, E1 rescues 1). This suggests the failure is driven by the intervention family's interaction with the model's factual retrieval, not by idiosyncratic properties of a single direction variant.

**Three additional failure modes account for the remaining flips:** verbosity-induced answer loss (3/10; the model elaborates past the target answer), evasion (1/10; the model describes instead of answering), and paraphrase drift (1/10).

## 5.4 Externality as First-Class Evidence

The bridge result illustrates why externality should not be an appendix concern. The same ITI direction that improves TruthfulQA MC1 by +6.3 pp causes $-7$ pp to $-9$ pp damage on open-ended factual generation on the bridge dev set (Limitation L5) — and the dominant damage mechanism (5 of 10 flips) is not generic quality loss but confident substitution of nearby wrong entities, with verbosity-induced answer loss (3/10), evasion (1/10), and paraphrase drift (1/10) accounting for the remainder. This failure would be invisible if the evaluation contract stopped at answer-selection benchmarks.

The H-neuron scope boundary reinforces the same lesson from a different direction: an intervention that works on compliance-adjacent surfaces (FaithEval, FalseQA, jailbreak) produces zero effect on a domain-specific factual QA task (BioASQ). The mechanism appears to be over-compliance modulation, not general truthfulness improvement. Without cross-surface evaluation, this scope boundary would go undetected.

> Steering success on one evaluation surface does not establish usefulness on a nearby surface. In this program, the relevant externality was not only score degradation but a specific failure mode: confident substitution of semantically nearby false answers for correct ones.

These bridge estimates remain dev-set point estimates. The promoted candidate has
not yet been run on the held-out test split ($n = 400$), and the failure-mode
taxonomy is based on manual analysis of 10 flips, so we use it for qualitative
diagnosis rather than population-rate estimation (Limitation L5). We also use
"confident wrong-entity substitution" as a behavioral diagnosis, not as a claim
about the internal circuit mechanism.
