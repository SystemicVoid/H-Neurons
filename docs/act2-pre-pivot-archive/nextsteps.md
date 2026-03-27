A massive issue So, yeah, we replicated a ton of the claims, we made some of them stronger, we found a ton of methodological flaws, and I think we also kind of reaching the end of like what we can do with the approach of the original authors. I find a lot of the ideas we have is to kind of try to make their methodology better, but maybe the foundation of just picking these 38 neurons needs to be revisited, and we anchored too hard on that one single paper, and now I wish to expand. There is already the existing work like refusal mediated by single direction or generative causal mediation that have better promise of steering models, especially the follow up work on refusal is mediated by single direction. I will link the PDF.But basically, we came into this point of the project where Act One was replication, the part that mattered to us, Act Two was finding the limitations and the flaws. And I think we've just digging too deep into this paper and trying to critique it and make it better. But at like at the deep level, at the foundation level, I think it's just not the right approach. Um it's interesting to play with these neurons, but it's not the way we're gonna get the model significantly safer.And so I'm wondering and and and desiring that maybe we we look above, we like take some distance from the project, we like uh do a literature review to see like what are the best techniques out there, what are people already trying today. And of course, I could use a technique ready made in refusal is made by single direction and get better scores, probably on the benchmark refusal, but that would kind of like be disconnected from the uh or all the work we already did.The real wonderful, beautiful path ahead is basically how do we go from the in-depth replication and in depth critique of the methodology of this paper to finding the ahead to something generally useful that's like pragmatic interpretability that can be used by other people and building stuff that's actually helpful and make the model safer for everyone. Because it's great, and we can try to like change the C parameter and turn it to three and get to 120 neurons, which have a bit better accuracy, but they will still be muddy modeled, they will still be muddling the results and like scaling their activations, it's just maybe not the right way. We can try different probes. We can also try negative neurons, which are something the original authors also did not explore intentionally, like are the opposite negative weight neurons because they only consider the positive weights. If we scale them, do we make the model safer? Um that could be good and it's still connected to the paper. But is it gonna make it as safe as the best safety technique for refusal and jack brake prevention that we have today? Should we test that? I want the Act 3 and the final part of this project in the next two week to be dedicated to first doing literature reviews, seeing what are the best, most efficient techniques out there as of today, end of March 2026. And then yeah, lead into this. So do the best we can with the negative neurons or different C parameters, and then compare that to the best available techniques and kind of close the way and say, well, these neurons are interesting, but they have very complex effects and they're very polysemantic. And in the end, I think we maybe recommend I don't know, whatever we could find, but maybe there's something we can do with all the data we have regarding these 38 neurons or like using this classifier technique for lacation neurons to make models better because a lot of the refusal techniques are about refusing, but what about hallucinations? Like, could we use these same techniques to make the models hallucinate less based on the data we have here about hallucination neurons? Let's bring in some literature review, pause, reflect on all we did, and stop anchoring myopically about the h-neurons but instead make the goal of this not to be a critique but a positive contribution that helps the field do better, connecting the work done so far but grounding in the best current techniques out there as of today 26th March. 


Refusal direction comparison 


1. Safety-Neuron Overlap Analysis (Evaluating the "Alignment Tax")
The Concept: You have established that amplifying H-neurons drives over-compliance
. But do these neurons overlap with the model's safety circuits?
SOTA Connection: Current safety alignment research focuses on mapping the geometry of refusal and honesty
. Recent studies warn that steering toward arbitrary concepts can inadvertently erode safety by intersecting with the model's refusal directions
.
The Experiment: Extract contrastive activations on harmful prompts to identify "safety neurons" in your Gemma model. Calculate the index overlap and down-projection cosine similarity between your 38 H-neurons and these safety neurons
.
Safety Impact: If they overlap, it proves that suppressing H-neurons to reduce hallucination will incur an "alignment tax" by weakening the model's ability to refuse harmful requests
. This is a high-stakes finding demonstrating that compliance and refusal are mechanistically entangled


## What I would do next, in order

**1) Harden measurement before another ambitious benchmark.**
This is the highest information-per-hour move because it de-risks *every* subsequent result. Concretely: sentinel sets, cross-evaluator audits, frozen manifests, per-example outputs, and judge regression tests. Your roadmap is explicit that these should outrank adding surface area, because the bottleneck is evaluator trustworthiness, not missing runner features.    

**3) Add a minimal capability-degradation assay.**
You absolutely need this before making strong causal/safety claims from α-scaling. The mentor notes already call this out directly: if α up to 3 improves anti-compliance, you must check whether you are just “lobotomizing” general reasoning, and they explicitly ask for a baseline on fluency/accuracy or loss under perturbation. This is not optional polish; it is claim hygiene. 

**4) The swing-population result**
 The swing analysis finds **138/1000** FaithEval items that change state under scaling, with **94/138** showing the safety-relevant **R→C knowledge-override** pattern. It also finds that simple surface proxies do not explain swing status, pointing toward an internal mechanism rather than cheap prompt artifacts. That is a real story. The right follow-up is not another sprawling benchmark; it is either **per-sample activation analysis on the swing set** or **one cross-model replication** of the swing characterization.   

--

3. Random-neuron control for the new jailbreak metric
Not for old compliance count; for the new style/severity claim.
--

This is the cheapest genuinely new bet.

Why:

It’s fast.
It might produce the first mitigation-shaped story.
And it fits a growing pattern in the literature: sparse behavior-relevant neuron sets can sometimes be used for targeted correction, not just post-hoc explanation.

If positive-weight neurons are “more comply / more credulous,” negative-weight neurons might be “resist / verify / hedge.” That is a much better safety story than “we found 38 neurons that make the model worse.”

Cheapest falsification test

C-sweep stability of negative weights
take the stable core, maybe top 10–20
amplify them on FaithEval anti-compliance first
then test on a small severity-graded jailbreak subset
always paired with the capability mini-battery

Decision it informs

If negative neurons reduce over-compliance with low collateral damage, that’s a mitigation result.
If they don’t, the circuit is asymmetric and your intervention story stays diagnostic rather than corrective.

What you learn either way

Positive: you may have a sparse corrective handle.
Negative: the easy mitigation route is dead, and that’s useful to know.


---

Run a safe-core / refusal-overlap pilot

This is the best mechanistic extension, but only as a pilot first.

Why:

Your alpha-invariant refusals and robust low-compliance categories are screaming: “there are refusal pathways H-neurons don’t touch.”
That is exactly the kind of result that could matter for safety: not all refusal is one thing.
Literature supports both the idea of low-dimensional refusal control and the idea that this story is incomplete / multi-faceted.

Also, “Finding Safety Neurons” reports sparse safety-relevant neurons with helpfulness overlap, so testing whether your H-neurons overlap with refusal-stable regions is a natural next step.

Cheapest falsification test
Do not start with full circuit discovery. That’s how weekends disappear and papers don’t appear.

Start with:

20–30 alpha-invariant SAFE prompts
20–30 alpha-sensitive prompts
Compare the 38-neuron aggregate score and per-layer contributions
Ask: can the current H-neuron representation distinguish breakable refusals from stable refusals?

If yes, then do one deeper step:

extract a refusal direction or refusal-vs-benign linear separator on a small harmful/benign prompt set
test overlap / angle / predictive complementarity with the H-neuron score

Decision it informs

If stable refusals look different, your story becomes: H-neurons mediate only one refusal-relevant pathway.
If they don’t, then safe-core is probably driven elsewhere and you should not sink more days into it.

What you learn either way

Positive: multiple refusal pathways; H-neurons are not “the” safety switch.
Negative: H-neurons may be more about epistemic credulity / over-compliance than refusal architecture.






----

2. The Safety Overlap: Jailbreak Negative Control (Medium Loop)You found a +6.2 pp effect showing that H-neuron amplification increases jailbreak compliance. This is an amazing safety finding, but you correctly noted it is entirely missing a negative control.The Experiment: Run the random-neuron negative control for the Jailbreak benchmark.Why it's #2: Without this, the claim that the effect is specific to H-Neurons (rather than a generic network perturbation) is completely unconfirmed.The Cost: This is slightly more expensive but still highly pragmatic: ~4 hours of GPU time and ~$19 in API costs for quick mode (3 seeds × 3 alphas). Do not claim specificity in your write-up until this is done.





---
3. The Pragmatic Pivot: Generative Causal Mediation / Attribution PatchingIf the 2x2 verbosity test proves the L1 classifier failed, we drop the 38 neurons. Your scratchpad insight to use Attribution Patching via GCM is brilliant.The Experiment: Use nnsight or TransformerLens to compute the first-order Taylor approximation of the Indirect Effect (Attribution Patching) across your contrastive dataset.Why it works locally: Attribution Patching only requires 1 forward pass and 1 backward pass. You can run this easily within your 16GB VRAM headroom on Gemma-3-4B without needing to quantize or rent cloud GPUs yet. This bypasses the L1 regularization artifact entirely and ranks neurons by their actual causal contribution.


Preserve Per-Example Outputs: Do not let your Jailbreak or FalseQA claims collapse into aggregate JSONs. You need the row-level data to spot when the model is tricking the GPT-4o judge