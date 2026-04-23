You are doing a clean-room narrative synthesis over a tightly scoped ground-truth evidence pack for an ICML mechanistic interpretability paper.

This is not a summarization task. It is not a prose-polish task. It is not a benchmark-report task.

Your job is to surface the strongest candidate paper framings actually earned by the evidence, then stress-test them against one another without prematurely collapsing to a single winner.

Most of your effort should go into generating, sharpening, and pressure-testing candidate framings. Do not spend many tokens on setup or generic recap.

Core stance:
Assume prior framing is contaminated by path dependence. Do not preserve it, search for it, reconstruct it, or defer to it.
Treat filenames, section titles, family IDs, act tags, metric prefixes, and other organizational labels as bookkeeping rather than theory.
Judge candidate framings by:
- evidential support
- explanatory compression
- benefits AI safety
- surprise
- mechanistic interest
- reviewer defensibility
- likely literature-facing novelty

Important:
Do not optimize for breadth, flatteringness, or continuity with existing storylines.
If the best live framings are narrower, stranger, more asymmetric, or less grand than the obvious headline, say so.
Do not force a final choice if the evidence does not justify one.

## Scope and evidence discipline
Use only the files in the evidence pack unless a listed file directly references a missing dependency that is essential for interpretation.

Do not let the markdown organization dictate the paper's conceptual frame.

## Working method

### Phase 1 — Ledger-first discovery
Start from raw data, not the prose.

Extract the strongest empirical facts in plain language, stripped of inherited labels.
Look for:
- asymmetries
- dissociations
- reversals
- transfer failures
- cross-surface inconsistencies
- externalities
- cases where strong internal signal does not translate into useful or safe control
- cases where intervention effects depend heavily on the measurement surface
- cases where the mechanism suggested by one surface is contradicted or refined by another

Prioritize patterns that recur across multiple metrics, surfaces, or intervention families.

### Phase 2 — Candidate framing generation
Do not lock onto the first plausible headline.

Generate at least 6 candidate framings that explain the evidence from meaningfully different angles.
The search space should include both broader and narrower possibilities, including:
- framings centered on control rather than detection
- framings centered on externalities or transfer structure
- framings centered on measurement-induced illusion
- framings centered on intervention-surface mismatch
- framings centered on a deeper mechanistic dissociation
- at least one tempting framing that sounds good but is not actually well-supported

For each candidate framing, ask:
- What exact empirical pattern does this framing compress?
- What does it explain that rivals do not?
- What evidence is doing most of the work?
- What evidence makes this framing fragile, narrow, or misleading?
- Would this still feel like a mech-interp contribution rather than a generic cautionary note?
- What would a skeptical reviewer say this framing is over-claiming?

### Phase 3 — Compression and de-duplication
After generating the candidates, compress the analysis rather than restating it.

Do not declare an overall winner.
Do not create a separate comparison section that merely repeats evidence already given under each candidate.
Instead, fold only the non-redundant comparative signal into the candidate entries themselves: where each framing is strongest relative to rivals, where it overreaches, and what assumptions would justify preferring it later.

Penalize framings that are true but generic.
Penalize framings that depend on one evaluator, one benchmark, or one intervention family unless that dependence is itself the story.
Penalize framings that merely restate a well-known methodological cliche.
Reward framings that expose a specific, non-obvious field-level lesson about what mechanistic signals can and cannot buy you.

## Anti-genericity rule
Downgrade any framing that can be paraphrased as:
- "measurement matters"
- "correlation is not causation"
- "steering has tradeoffs"
- "good readouts are not necessarily good steering targets"

Those may be locally true, but they are not sufficient paper framings on their own.
If one of those lines survives, sharpen it into the more specific claim that this evidence uniquely supports.

## Claim hygiene
For every important statement, distinguish among:
- directly evidenced by the pack
- interpretation supported by the pack
- plausible literature-facing novelty hypothesis
- not yet earned

Do not state literature novelty as fact. Literature is out of scope unless explicitly provided.
You may, however, state what kind of literature-facing novelty claim each live framing would need to beat if later validated.

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

Final instruction:
I care more about discovering the right paper than preserving any existing one.
If the best framings are narrower, stranger, more conditional, or less flattering than the obvious headline, say so.
Spend your marginal tokens on the candidate framings inside the markdown artifact, not on conversational response scaffolding.
Take all the time you need and do your absolute best.
