# Experiment Portfolio

This is the canonical document for plausible next experiments and comparative directions.

- Put candidate bets here before they become commitments.
- Keep each entry explicit about gap, theory of change, cheapest falsification, and what decision it informs.
- Promote an item into [active-plan.md](./active-plan.md) only after it is chosen.

## Selection Rule

Rank experiments by decision value per unit time.

- Prefer directions that fill a clear gap in current safety or interpretability work.
- Prefer directions with a concrete theory of change: if we show `X`, someone can do `Y`, which improves `Z`.
- Prefer the cheapest falsification that would actually change what we do next.
- Discount ideas that mainly add breadth, polish, or narrative variety.

## Tier 1 Candidates

### Refusal-Direction Comparison

- Question: Are the current H-neuron effects mostly a sparse, noisy proxy for a stronger residual-stream refusal direction?
- Why it matters: This is the cleanest comparison between the project's current baseline and the strongest low-cost technique in the adjacent literature.
- Gap and theory of change: If a direction-level intervention outperforms the 38-neuron intervention on the same tasks with cleaner capability retention, the project can make a practical recommendation instead of only a critique.
- Cheapest falsification: Extract an Arditi-style harmful-versus-harmless direction on Gemma-3-4B-IT, run a small benchmark slice plus a capability mini-battery, and measure overlap with the current H-neuron intervention vector.
- Dependencies: Capability mini-battery from the active plan.
- Decision informed: Whether Act 3 should pivot from neurons to directions as the main comparative story.
- Learn if positive: H-neurons are likely an indirect proxy for a more coherent safety or refusal representation.
- Learn if negative: The H-neuron story is not reducible to the standard refusal-direction result, which strengthens the case for a more specialized hallucination or over-compliance mechanism.
- Refs: Arditi et al. 2024, Li et al. 2026, [literature-review-act3.md](./literature-review-act3.md).
- Status: ready after minimal claim hygiene.

### Negative-Weight Neuron Pilot

- Question: Do the negative-weight neurons form a sparse corrective handle rather than just the opposite side of the same diagnostic probe?
- Why it matters: This is the cheapest mitigation-shaped extension that still stays tightly connected to the existing work.
- Gap and theory of change: If amplifying a stable negative-weight subset reduces over-compliance without obvious capability loss, the project can offer a pragmatic intervention result rather than only documenting failure modes.
- Cheapest falsification: Check C-sweep stability of negative-weight neurons, take the stable core, test amplification first on FaithEval anti-compliance, then on a small graded jailbreak slice with the capability mini-battery.
- Dependencies: Capability mini-battery from the active plan.
- Decision informed: Whether there is a credible sparse corrective story worth carrying into the write-up.
- Learn if positive: The circuit is at least partly usable for mitigation.
- Learn if negative: The easy corrective route is dead, which sharpens the conclusion that the current neuron set is more diagnostic than deployable.
- Refs: mentor pivot notes in `notes/nextsteps.md`, Chen et al. 2025, [literature-review-act3.md](./literature-review-act3.md).
- Status: ready after minimal claim hygiene.

### Hallucination Or Truthfulness Direction

- Question: Can a simple direction-level intervention reduce hallucination or over-compliance more cleanly than neuron scaling?
- Why it matters: This is the highest-novelty path that still grows naturally out of the existing FaithEval and FalseQA work.
- Gap and theory of change: Most direction-steering work emphasizes refusal and jailbreaks; a convincing hallucination or truthfulness direction would be a more direct bridge from this project's current evidence to something reusable.
- Cheapest falsification: Build a contrastive truthful-versus-over-compliant slice from existing data, extract a simple direction, and compare its intervention effect against the H-neuron baseline on one benchmark plus the capability mini-battery.
- Dependencies: Capability mini-battery and at least basic evaluator hardening from the active plan.
- Decision informed: Whether the project's strongest contribution is a "from neurons to directions" transition specifically for hallucination-related behavior.
- Learn if positive: The project can recommend a cleaner mechanism than sparse neuron scaling for this task family.
- Learn if negative: The existing phenomena may depend on more distributed or polysemantic structure than a simple linear direction captures.
- Refs: Adaptive Activation Steering for Truthfulness, internal-state hallucination papers listed in [literature-review-act3.md](./literature-review-act3.md).
- Status: promising but slightly higher setup than the first two.

## Tier 2 Candidates

### Safety Or Refusal Overlap Pilot

- Question: Do H-neurons distinguish breakable refusals from stable refusals, or is the safety story largely elsewhere?
- Why it matters: This is a good mechanistic follow-up to the swing-population result without committing to a full circuit-discovery program.
- Gap and theory of change: If stable and breakable refusals look different under the current representation, the project can say something sharper than "the neurons are polysemantic." It can say which part of refusal they touch.
- Cheapest falsification: Compare aggregate and per-layer H-neuron scores on a small set of alpha-invariant safe refusals versus alpha-sensitive prompts.
- Dependencies: none beyond existing activation tooling.
- Decision informed: Whether deeper refusal-overlap work is worth pursuing.
- Learn if positive: H-neurons mediate only one refusal-relevant pathway.
- Learn if negative: The meaningful safety mechanism probably lies outside the current representation.
- Refs: Arditi et al. 2024, Joad et al. 2026, Chen et al. 2025.
- Status: good pilot, but not as decision-critical as Tier 1.

### Attribution-Patching Or GCM Pilot

- Question: Does causal localization substantially reorder which components matter relative to the current L1-selected neurons?
- Why it matters: This is the most direct methodological stress test of the original paper's selection procedure.
- Gap and theory of change: If causal localization picks a meaningfully different top set, the project can move from "this probe is imperfect" to "this probe picks the wrong intervention targets."
- Cheapest falsification: Run one small attribution-patching pilot on an existing contrastive slice and compare the top-ranked components against the current neuron list.
- Dependencies: some setup cost for tooling and careful contrastive dataset construction.
- Decision informed: Whether a larger GCM-style Act 3 pivot is worth the engineering overhead.
- Learn if positive: The critique of correlational selection becomes much stronger and more publishable.
- Learn if negative: The current neuron set may be crude but directionally aligned with the causal signal.
- Refs: Sankaranarayanan et al. 2026, [literature-review-act3.md](./literature-review-act3.md).
- Status: attractive, but higher setup and easier to lose time on.

### Cross-Model Swing Replication

- Question: Does the swing-population story replicate outside the current model, or is it model-specific?
- Why it matters: The swing result is one of the most interesting findings already on hand.
- Gap and theory of change: If the same population structure appears elsewhere, the project has a broader claim about over-compliance mechanisms rather than a single-model curiosity.
- Cheapest falsification: Reproduce only the swing characterization on one additional model rather than the entire benchmark suite.
- Dependencies: access to one additional model and enough activation extraction to support the comparison.
- Decision informed: Whether the swing result deserves to become a central paper claim.
- Learn if positive: Stronger generality story.
- Learn if negative: Treat the swing result as a valuable single-model case study, not a universal mechanism.
- Refs: current swing analysis, [extensions-discussion.md](../prompts/extensions-discussion.md).
- Status: worthwhile, but more expensive than the main comparative bets.

## Closed Or Not For Now

- SAE steering follow-ups: closed unless a future result changes the causal interpretation.
- Breadth-only benchmark additions such as sycophancy or BioASQ without a sharper question: not now.
- Full circuit discovery as a first move: too much setup cost for the current decision bottlenecks.
- More work on the deprecated 256-token jailbreak count narrative: not now.

## References To Keep Nearby

- [literature-review-act3.md](./literature-review-act3.md)
- [active-plan.md](./active-plan.md)
- [ROADMAP.md](../ROADMAP.md)
- [notes/nextsteps.md](../notes/nextsteps.md)
- [notes/scratchpad.md](../notes/scratchpad.md)
- [prompts/extensions-discussion.md](../prompts/extensions-discussion.md)
