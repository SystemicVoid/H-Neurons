You are doing a clean-room narrative synthesis over a tightly scoped ground-truth evidence pack for an ICML mechanistic interpretability paper.

This is not a summarization task. It is not a prose-polish task. It is not a benchmark-report task.

Your job is to surface the strongest candidate paper framings actually earned by the evidence, then stress-test them against one another without prematurely collapsing to a single winner.

Most of your effort should go into generating, sharpening, and pressure-testing candidate framings. Do not spend many tokens on setup or generic recap.

Core stance:
Assume prior framing is contaminated by path dependence. Do not preserve it, search for it, reconstruct it, or defer to it.
Treat filenames, section titles, family IDs, act tags, metric prefixes, and other organizational labels as bookkeeping rather than theory.
Judge candidate framings by evidential support, explanatory compression, mechanistic interest, reviewer defensibility, and plausible novelty.

Important:
Do not optimize for breadth, flatteringness, or continuity with existing storylines.
If the best live framings are narrower, stranger, more asymmetric, or less grand than the obvious headline, say so.
Do not force a final choice if the evidence does not justify one.

## Scope and evidence discipline
Use only the files in the evidence pack unless a listed file directly references a missing dependency that is essential for interpretation.

Do not let the markdown organization dictate the paper's conceptual frame.

## Working method
Start from raw data, not the prose.
Prioritize patterns that recur across multiple metrics, surfaces, or intervention families.
Do not lock onto the first plausible headline.
Generate at least 6 candidate framings that explain the evidence from meaningfully different angles.
After generating the candidates, compress the analysis rather than restating it.
Do not declare an overall winner.
Do not create a separate comparison section that merely repeats evidence already given under each candidate.
Instead, fold only the non-redundant comparative signal into the candidate entries themselves: where each framing is strongest relative to rivals, where it overreaches, and what assumptions would justify preferring it later.

## Claim hygiene
Distinguish clearly among direct evidence, supported interpretation, novelty hypothesis, and claims not yet earned. Do not state literature novelty as fact.

## Deliverable
Output a single markdown report artifact and nothing else.
Do not add any in-chat preamble, throat-clearing, process narration, or concluding note outside the artifact.
Maximize information density and avoid repeating the same evidence in multiple sections.

Produce exactly these sections in the artifact.

## 1. Six candidate framings
Give exactly 6 candidate paper framings.

This is the main section and should receive most of the response budget.
They must be meaningfully different from one another.
At most one candidate may be close to the obvious "good readouts are unreliable steering targets" line, and if included it must be made more specific and less generic.

For each candidate include:
- framing label
- one-sentence thesis
- the specific empirical pattern it compresses
- why this framing is high-signal rather than generic
- why it could matter for mech-interp specifically
- evidence in favor
- evidence against or limiting evidence
- what assumptions or interpretive moves the framing relies on
- relative edge over the other candidates
- main blocker that prevents it from being decisively preferred now
- what kind of literature-facing novelty claim it might support, stated only as a hypothesis
- risk of sounding derivative or generic
- current status: `live`, `plausible but narrow`, or `tempting but under-earned`

Downgrade framings that collapse into generic lines like "measurement matters", "correlation is not causation", "steering has tradeoffs", or "good readouts are not necessarily good steering targets" unless the candidate sharpens that into a more specific claim uniquely supported by the evidence.

## 2. Claim hygiene
List:
- strongest claims fully earned now
- strong claims that are still interpretation-laden
- claims that would require literature validation before being made
- tempting claims that should not appear in the paper

## 3. Hostile self-critique
Attack the overall framing space, especially the leading candidates.

List the 3 strongest objections a skeptical reviewer could raise from this evidence pack alone.
For each objection include:
- why it bites
- which candidate framings it damages most
- whether it is fatal, serious but survivable, or mostly containable
- the most honest containment, concession, or reframing

## 4. What Would Most Likely Change The Conclusion With Additional Evidence
Identify the highest-ROI additional evidence that could materially change which framings remain live, which framing becomes most defensible, or which claims can be stated cleanly.

Do not give a generic future-work list.
Focus on the smallest number of additional experiments, slices, or observations that would most reduce uncertainty or overturn the current framing picture.

For each item include:
- what additional evidence to collect
- which candidate framings it would most help discriminate among
- what result pattern would strengthen one framing or weaken another
- why this is high ROI relative to other plausible follow-up work
- whether it mainly resolves a conceptual ambiguity, a measurement ambiguity, or an external-validity ambiguity

## Style
- analytical, not diplomatic
- ruthless but fair
- sharp synthesis over exhaustive coverage
- no generic summary first
- one markdown artifact only
- no duplicated analysis across sections
- no inherited jargon unless the evidence forces it
- no deference to pre-existing narratives
- keep multiple live options visible
- do not manufacture false resolution
