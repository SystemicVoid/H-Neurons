# 2026-04-20 — Paper framing synthesis, novelty correction, D7 rehabilitation, and split-paper reassessment

**Status.** Supersedes the earlier 2026-04-20 version of this memo. That version correctly flagged the SAE layer-coverage confound and the strength of the jailbreak-measurement evidence, but it still misidentified the paper's deepest structural problem. The main issue is now novelty framing: `paper/icml/main.tex` headlines a slogan-level thesis whose weak form is already in the literature, while the paper's actually novel contributions sit one layer below the headline.

**Question.** After the arbitration audit, the GPT and Opus deep literature reviews, the three outline revisions (`paper/paper-outline-v1.md`, `paper/revised_flagship_outline-v2.md`, `paper/final_flagship_outline_review.md`), the current `main.tex`, and the 2026-04-20 D7 adversarial audit are read together, what should the paper actually claim, and which earlier recommendations are now stale?

**Scope.** Reference document only. No edits to `main.tex` yet. This note updates the framing diagnosis, retires stale recommendations, and keeps paper-splitting as a secondary venue strategy rather than the primary conclusion.

---

## 0. Executive correction

1. The deepest problem is **not just** that the SAE-vs-H-neuron comparison is confounded or that the measurement story is cleaner. The deeper problem is that the abstract and introduction still sell a slogan-level claim that is no longer fresh: in weak form, "strong readouts are insufficient steering evidence" is already occupied by Arad et al. (within SAEs), AxBench (detection vs steering as separate axes), Bhalla et al. ("predict/control discrepancy"), and Wang et al. (weak association between SAE interpretability and steering utility).
2. The paper's genuinely differentiating evidence is more specific than the slogan. The strongest paper-specific novelty now sits in four places:
   - the **wrong-entity substitution** diagnosis on TriviaQA bridge generation;
   - the **same-output measurement reversal** inside a representation-engineering setting;
   - a **cross-representational matched comparison on a real behavioral surface** (with an explicit SAE-coverage caveat);
   - and the newly rehabilitated **D7 within-family selector comparison**, which is cleaner than the SAE comparison on operator and representational-basis grounds.
3. The SAE comparison still matters, but only under a narrowed claim. The safe version is not "SAEs fail as steering mediators." The safe version is: **readout-selected SAE features at 10 of 34 layers matched H-neurons on FaithEval AUROC but did not reproduce the H-neuron dose-response on that surface.**
4. D7 changed status materially. The 2026-04-20 adversarial audit shows that the two strongest earlier objections to promoting D7 — "mixed ruler" and "112 token-cap hits make the result too dirty" — were overstated for the causal-vs-baseline/random claim. D7 is no longer appendix-only evidence.
5. The split-paper option remains feasible, but it is no longer the main recommendation. The first decision is how to re-headline the paper around its actual novel kernel. Split is a packaging choice after that, not the answer to the core framing problem.

---

## 1. What the first pass got wrong

The earlier version of this memo was directionally right about the SAE confound and about the strength of the measurement evidence, but it still over-weighted the wrong level of the problem.

- It treated the main strategic question as **which anchor should lead**.
- It treated the blog-vs-paper mismatch as primarily a **measurement-versus-SAE** ordering issue.
- It implicitly assumed the main novelty risk was **external validity or confounding**.

That is too shallow.

The three outline revisions had already diagnosed the deeper issue before the ICML sprint compressed the paper: the slogan is not the contribution. The contribution is the lower-level empirical pattern and the scaffold that organizes it. The arbitration audit and the two strongest deep literature reviews agree on that point. The current `main.tex` related-work section also silently agrees on it: it already cites Arad, AxBench, Wang, Bhalla, ITI, and Opielka. The novelty problem is therefore not hidden from the draft; it is created by the draft's current emphasis.

---

## 2. Stale recommendations retired

These are the recommendations from the earlier version that should now be treated as stale.

- ~~The main strategic choice is whether to split the measurement story into a separate paper.~~  
  Replacement: the main strategic choice is how to move the paper's novelty claim away from the slogan and onto the paper-specific kernels.

- ~~Option 4 (split the paper) is the best expected-value path.~~  
  Replacement: stale. Splitting may still be worthwhile, but only after the unified-paper novelty hierarchy is corrected. Otherwise Paper B keeps the same headline problem.

- ~~D7 remains too mixed-ruler / token-cap dirty to promote.~~  
  Replacement: stale for the causal-vs-baseline/random claim after `paper/icml/reviews/2026-04-20-d7-quality-debt-adversarial-audit.md`. D7 is still benchmark-local and not a universal selector theorem, but it is stronger than the earlier memo allowed.

- ~~The measurement story should become the default flagship simply because it is the cleanest result.~~  
  Replacement: too blunt. Generic evaluator dependence is already known. The narrow, defensible measurement contribution is the **same-output reversal inside this representation-engineering setting**, plus the truncation artifact as adjacent measurement-contract evidence.

---

## 3. Why the current headline is vulnerable

The novelty mismatch is now visible inside `paper/icml/main.tex` itself.

- The abstract closes by arguing that "strong readouts are insufficient evidence for good steering targets."
- The introduction opens with the "readout-to-steering heuristic" and frames the paper as testing it.
- The section title "`Similar Readout Quality Does Not Guarantee Control`" keeps the same emphasis.

But the related-work section already acknowledges most of the slogan-level prior art:

- Arad et al. already show that high-scoring SAE features need not steer well.
- AxBench already separates detection and steering as distinct evaluation axes.
- Wang et al. already report only a weak association between SAE interpretability and steering utility across 90 SAEs.
- Bhalla et al. already use the language of a predict/control discrepancy.
- ITI and Opielka already occupy much of the MC-vs-generation / format-specific-control territory.

So the current draft does two inconsistent things at once:

1. It cites the papers that make the weak slogan non-novel.
2. It still markets the weak slogan as if it were the paper's own main discovery.

That is the structural vulnerability. A reviewer who knows the 2025-2026 literature will not object that the paper lacks interesting evidence. They will object that the paper is selling the wrong layer of that evidence as the headline.

The safe hierarchy now looks like this:

- **Background / not novel:** decodability does not imply causal use; evaluator choice matters; MC gains need not transfer cleanly to generation.
- **Near-direct prior art / weak novelty at best:** strong readouts are unreliable steering-target heuristics in the abstract.
- **Still plausibly novel:** this paper's exact empirical form, failure-mode diagnoses, matched-comparison geometry, and the four-stage scaffold as an organizing framework.

---

## 4. The actual novel kernel

### 4.1 Wrong-entity substitution is the sharpest externality result

This is the most underweighted result in the current paper.

What the literature already covers:

- ITI and follow-on work already establish that multiple-choice or constrained-surface improvements need not transfer cleanly to generation.
- Pres et al. and Opielka et al. already make format-locality and surface-transfer failure plausible.

What appears paper-specific here:

- On the locked 500-question TriviaQA bridge test set, E0 ITI at `alpha = 8.0` reduces adjudicated accuracy by **-5.8 pp [-8.8, -3.0]**, with **McNemar p = 0.0002**.
- Among the **43 right-to-wrong flips**, **30** are manually coded as wrong-entity substitutions, about **70%** of the damage.
- The resulting behavioral picture is sharper than "generation got worse." The intervention is active but indiscriminate: it often stays in the right semantic neighborhood and picks the wrong member.

Why this matters for framing:

- This is more specific and more defensible than the generic MC->generation slogan.
- No cited prior paper appears to document this exact failure mode for truthfulness steering.
- It is the paper's best concrete behavioral mechanism claim, subject to one explicit caveat: the coding is still **single-rater** and should remain framed as a behavioral diagnosis, not a circuit-level mechanism.

### 4.2 The cross-representational FaithEval comparison is novel only in a narrow form

What the literature already covers:

- Arad et al. already show that within SAEs, "looks relevant" and "steers well" are not the same thing.
- AxBench and Wang make detection-versus-steering separation no longer novel in the abstract.

What still looks novel here:

- On a **real behavioral surface** rather than synthetic concepts, the paper compares two representation families in the **same model** with closely matched readout quality:
  - H-neurons: **AUROC 0.843**
  - SAE features: **AUROC 0.848**
- Steering then diverges sharply:
  - H-neurons: **+2.09 pp/alpha [1.38, 2.83]**
  - SAE h-features: **+0.16 pp/alpha [-0.51, 0.84]**

What narrows the claim:

- SAE coverage is only **10 of 34 layers**.
- **47%** of H-neuron weight lives in SAE-uncovered layers.
- **5 of the top 10** H-neurons by weight are in uncovered layers.
- Delta-only closes the reconstruction-noise objection, but not the coverage objection.

So the safe claim is:

> Readout-selected SAE features at 10 of 34 layers matched H-neurons on FaithEval detection quality but did not reproduce the H-neuron control effect on that surface.

That remains interesting and likely novel as an exact cross-representational comparison. It is **not** strong enough, in its current form, to headline the paper as a general theorem about "strong readouts."

### 4.3 The measurement contribution is the same-output reversal, not generic evaluator dependence

The literature already spent most of the generic claim:

- StrongREJECT already established binary-versus-graded reversals in jailbreak evaluation.
- Know Thy Judge and adjacent work already established judge dependence and artifact sensitivity.

So the paper-safe measurement claim has to be narrower:

- On the **same 5,000-token H-neuron jailbreak outputs**, CSV-v2 gives a positive harmfulness slope of **+2.30 pp/alpha [0.99, 3.58]**.
- On those same outputs under CSV-v3, the harmful-binary slope is near-null: **+0.46 pp/alpha [-1.46, 2.41]** in the paper-facing comparison, and **+0.60 pp/alpha [-0.85, 2.02]** under the fixed-denominator recheck. Either way, the CI includes zero.
- Separately, the older **256-token binary** run produced an apparent **+6.2 pp** effect that disappeared once the full-generation contract was used.

The strong version of the paper's claim is therefore:

> In this representation-engineering setting, different defensible measurement contracts changed the scientific verdict about the same intervention family; the narrow same-output reversal is the v2-v3 comparison, while truncation provides an adjacent measurement-contract failure in the earlier binary pipeline.

What this does **not** support:

- It does not support "CSV-v3 is the objectively correct judge."
- It does not support "we discovered evaluator dependence."
- It does not support the older binary-superiority framing against StrongREJECT, which is now stale after the holdout tie.

### 4.4 D7 is now a real framing asset, not appendix debt

This is the biggest update relative to the previous memo.

What the older framing said:

- D7 was interesting but too mixed-ruler, too token-cap-heavy, and too bookkeeping-dirty to promote.

What the newer audits show:

- In the matched pilot, **probe-ranked heads** include extremely strong readouts (top heads at **AUROC 1.0**) but the probe intervention is null, while the **gradient-ranked causal selector** reduces harmfulness.
- In the adversarial reanalysis of the full-500 causal panel under a **single v3 ruler**, the causal branch remains strong:
  - baseline: **34.2%**
  - L1: **36.4%** (null against baseline)
  - causal: **20.0%**, paired delta **-14.2 pp [-17.8, -10.4]**
  - random seed 1: **37.2%**
  - random seed 2: **38.4%**
  - direct gaps: **+17.2 pp [13.2, 21.4]** and **+18.4 pp [14.6, 22.4]** for random-vs-causal
- The 112 token-cap hits do **not** drive the causal effect.
- The strongest mixed-ruler objection can be neutralized with files already on disk.

What remains caveated:

- The single-ruler adversarial panel rehabilitates **causal-vs-baseline/random**, not a fully unified five-branch selector theorem.
- The probe branch is still not part of that exact v3-only adversarial panel, so the cleanest statement remains a two-part one:
  - probe-ranked AUROC-1 heads were inert in the matched pilot under the same operator;
  - causal heads beat baseline and random layer-matched controls on the full-500 v3 panel.
- This is still benchmark-local, single-judge evidence. It does **not** yet prove that causal selectors beat correlational selectors in general.

Net framing consequence:

- D7 should no longer be appendix-demoted on "too dirty" grounds.
- It is now a credible main-text pillar because it gives the paper a **second matched-detection story in a cleaner experimental geometry** than SAE-vs-H-neurons: same intervention family, same representational basis, no layer-coverage confound, no operator-form confound.

---

## 5. Revised strategic recommendation

### Recommended path: reframe the unified paper around the actual novel kernel

The unified paper is still viable, but the center of gravity has to change.

What should move up:

- **Bridge wrong-entity substitution** as the sharpest externality result.
- **D7** as the cleaner within-family selector comparison.
- **Same-output measurement reversal** as the narrow measurement contribution.

What should stay, but move down one level:

- The **SAE-vs-H-neuron FaithEval comparison**, with the 10/34 coverage asymmetry made explicit in the main text.

What should move out of headline status:

- The slogan-level claim that strong readouts are insufficient steering evidence.

The paper should instead present something closer to this structure:

1. **The paper's scientific object:** a staged audit of where representation-engineering claims break.
2. **Empirical kernel A:** matched readouts can diverge in control across representation families, but the SAE comparison is coverage-caveated.
3. **Empirical kernel B:** within one representational basis, selector choice matters; D7 provides the cleaner matched-selector geometry.
4. **Empirical kernel C:** even successful control can fail via a concrete externality mechanism, here wrong-entity substitution.
5. **Empirical kernel D:** measurement choices can reverse the verdict inside this intervention setting.

Under that reframe:

- the four-stage scaffold becomes the paper's methodological packaging;
- the slogan becomes a conclusion, not a discovery claim;
- and the draft stops asking the reviewer to credit novelty at the wrong level.

### What this implies for the abstract and introduction

The abstract and intro should stop implying:

> "We show that strong readouts are insufficient steering evidence."

and start implying:

> "We provide a staged case study showing four distinct ways a representation-engineering claim can break, including a matched cross-representational dissociation, a within-family selector dissociation, a concrete externality mechanism, and a same-output measurement reversal."

That is the claim hierarchy the evidence now supports.

---

## 6. Split-paper reassessment

Splitting is still viable, but it is no longer the answer to the core issue.

### What still survives from the split idea

- A narrow **measurement paper** is still real:
  - same-output v2-v3 reversal,
  - truncation artifact,
  - holdout evaluator audit,
  - richer taxonomy rather than judge-superiority framing.

- A narrower **localization/control/externality paper** is also still real:
  - caveated FaithEval matched comparison,
  - D7 selector comparison,
  - bridge wrong-entity substitution,
  - MC-vs-generation scope breaks.

### What changed

If the paper splits **without** first fixing the novelty hierarchy, Paper B still inherits the same problem:

- its headline sounds like a rediscovery of a known slogan,
- while its most novel claims remain subordinate.

So split should now be treated as:

- a **venue and length** decision;
- not a substitute for re-framing the core claim.

### Updated ranking

1. **Best path:** reframe the unified paper around the actual novel kernel.
2. **Second-best path:** split only if a narrow methods paper and a narrower control/externality paper fit the calendar and venue plan.
3. **Worst path:** split first while leaving the slogan-level novelty mismatch intact.

---

## 7. Promotion / demotion map

### Promote

- **Bridge wrong-entity substitution**
- **D7 within-family selector comparison**
- **Same-output measurement reversal**

### Keep, but caveat hard

- **FaithEval H-neuron positive control**
- **FaithEval SAE-vs-H-neuron matched comparison**

### Demote from headline status

- **"Strong readouts are insufficient steering evidence"** as a novel claim
- **Generic evaluator-dependence framing**
- **The old "D7 is too dirty" objection**

### Drop or avoid

- Claims that sound like:
  - "we are first to show good detectors fail as steering targets"
  - "SAEs are bad steering mediators"
  - "CSV-v3 is better because it wins holdout accuracy"
  - "D7 proves causal selection beats correlational selection in general"

---

## 8. File pointers

Primary sources for this revised synthesis:

- `paper/icml/main.tex`
- `paper/literature-research/research_arbitration_audit_detection_is_not_enough.md`
- `paper/literature-research/gpt-deep-literature-review.md`
- `paper/literature-research/opus-deep-literature-review.md`
- `paper/paper-outline-v1.md`
- `paper/revised_flagship_outline-v2.md`
- `paper/final_flagship_outline_review.md`
- `paper/icml/reviews/2026-04-20-d7-quality-debt-adversarial-audit.md`
- `notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md`
- `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`
- `notes/act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md`
- `notes/act3-reports/2026-04-13-phase3-jailbreak-pipeline-audit.md`
- `notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md`
- `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md`

---

## 9. Bottom line

The paper does not have a weak evidence problem. It has a **headline-selection problem**.

The current draft fronts a claim that reviewers can reasonably read as already made. The evidence that is actually distinctive is more concrete:

- a wrong-entity externality mechanism,
- a same-output measurement reversal,
- a narrow but real cross-representational matched comparison,
- and a newly rehabilitated within-family selector comparison in D7.

That is the level at which the paper now has to sell itself.
