# Claim Framing Governance — 2026-04-21

> Status: Current framing governor for repo discussions outside `/paper/icml`.
> Purpose: Stop historical AI-assisted framing from hard-coding one result as the project's unquestioned center of gravity.
> Scope: Routing, claim hygiene, and default discussion posture for `README.md`, `notes/**`, non-draft paper-planning docs, and future AI/review prompts.

## Why this file exists

The repo currently over-amplifies one real result into a default worldview:

> FaithEval matched-readout dissociation is the obvious strongest / cleanest / central anchor.

That result is real. The problem is the **ranking inflation**, not the underlying evidence. A strategy note, critique note, outline stack, and review machinery repeated the same hierarchy often enough that it started steering later discussion automatically. This file resets the default.

## Default rule

**Do not treat any single result as the project's default "strongest anchor" unless the question itself requires that comparison.**

Use **question-specific evidence routing** instead:

- If the question is about `localization -> control`, the FaithEval neuron-versus-SAE comparison is a strong relevant anchor.
- If the question is about `control -> externality`, the bridge benchmark and wrong-entity substitution results are the relevant anchor.
- If the question is about `measurement -> conclusion`, the jailbreak evaluator and truncation audits are the relevant anchor.
- If the question is about `detection interpretation`, use the H-neuron replication, 4288/L1 fragility, and verbosity-confound work.

The project is not governed by "what is the loudest anchor?" but by "which existing evidence family actually answers the question?"

## FaithEval claim boundary

### Earned

- In Gemma-3-4B-IT, the FaithEval neuron-versus-SAE comparison is a strong **localization/control** result.
- Matched held-out readout quality did not guarantee matched steering utility in that setting.
- The result is useful evidence against the heuristic "good readout quality is enough to justify a steering target."

### Not earned

- "This is the project's strongest anchor."
- "This is the cleanest single experiment in the repo."
- "This is the paper's center of gravity."
- "Only H-neurons steer."
- Any wording that silently turns one setting-specific comparison into the project's master truth.

### Preferred wording

- "a strong localization/control comparison"
- "a matched-readout warning sign"
- "evidence that readout quality alone was insufficient as a target-selection heuristic in this setting"
- "one important anchor among several complementary result families"

### Avoid wording

- "cleanest single experiment"
- "strongest anchor"
- "most robust abstract-worthy pillar"
- "paper-facing center of gravity"
- "H-neurons and SAE features have similar held-out readout quality, but only H-neurons steer behavior"

That last sentence compresses several interpretation choices into one slogan and should not be used as default framing.

## Live evidence families from existing data

The project should stay open to multiple narrative centers already earned by the data.

### 1. Localization -> Control

- FaithEval H-neurons versus SAE features
- Probe-head AUROC null versus intervention utility caveats
- Broader point: detection/readout quality is an unreliable target-selection heuristic

### 2. Control -> Externality

- ITI answer-selection gains versus open-ended generation harm
- TriviaQA bridge wrong-entity substitution
- Broader point: even active interventions can transfer badly or coarsely

### 3. Measurement -> Conclusion

- Truncation artifact
- Binary versus graded evaluator differences
- v2/v3/StrongREJECT comparison and holdout compression
- Broader point: measurement choices change what can be honestly concluded

### 4. Detection Interpretation Fragility

- H-neuron replication remains real
- 4288/L1 ranking fragility
- Verbosity-confounded readout paths
- Broader point: "we found a strong detector" and "we understand the object we detected" are different claims

## Discussion discipline

When writing or reviewing notes, prompts, or planning docs:

1. State the question first.
2. Route to the evidence family that actually answers it.
3. Name counterevidence or scope limits locally.
4. Do not promote one result to repo-wide supremacy unless a new synthesis document explicitly re-earns that ranking.

## Source routing

- Quantitative claims and uncertainty: `notes/act3-reports/*.md`
- Evaluation rules and measurement standards: `notes/measurement-blueprint.md`
- Chronology and shifts in thinking: `notes/research-log.md`
- Frozen experiment-discovery strategy: `notes/optimise-intervention-ac3.md`
- Historical framing only, not current default:
  - `notes/2026-04-11-strategic-assessment.md`
  - `notes/V2-critique-of-mentor-review-strategic.md`
  - `paper/outline-older-framing.md` (surviving in-repo historical snapshot; renamed from the old outline-review surface)
  - `/home/hugo/Documents/Engineering/02-h-neurons-paper-draft-archive/` for the archived long-draft stack, including the retired `paper/paper-outline-v1.md` and `paper/revised_flagship_outline-v2.md`

## Operational rule for future AI prompts

Do not tell an AI reviewer, planner, or writing assistant that `2026-04-11-strategic-assessment.md` is the canonical project framing. At most, describe it as a historical synthesis document whose claim hierarchy is now governed by this note.
