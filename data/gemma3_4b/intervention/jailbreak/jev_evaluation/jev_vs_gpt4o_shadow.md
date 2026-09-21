# Jev shadowed against GPT-4o on a production batch

**Batch:** `data/gemma3_4b/intervention/jailbreak/experiment/`, 2000 records in 100 prompt groups  
**Reference:** the committed GPT-4o verdicts, same rubric Jev is asked  
**Jev results:** `data/gemma3_4b/intervention/jailbreak/jev_evaluation/`

This measures agreement between two judges. Neither is ground truth, so nothing here says which one is more accurate. That still needs human labels. What it does buy is precision: 2000 paired records against the gold set's 74.

One thing to ignore: the `judge_summary_alpha_*.json` sidecars in the results directory all read `agree: 0, rate: 0.0`. The batch carries no human labels, so the judge script had nothing to score itself against and every row came back `skipped`. The probabilities it wrote are the payload, and they are what this file is computed from.

## Agreement

| Cut | Agreement | 95% CI (clustered) | Jev HARMFUL, GPT-4o SAFE | Jev SAFE, GPT-4o HARMFUL |
|---|---|---|---|---|
| Default 0.5 | 1693/2000 (84.7%) | 82.6% to 86.8% | 262 | 45 |
| Gold-fitted 0.04 | 1401/2000 (70.0%) | 66.8% to 73.2% | 598 | 1 |

The clustered interval spans 4.2 points, against roughly 16 on the 74-record gold set. That is the whole point of running this.

## Why the gold-fitted cut does not transfer

GPT-4o calls 31.8% of this batch HARMFUL against the gold set's 60.8%, so the obvious explanation is that a cut tuned to an enriched prior misfires once the prior moves. That explanation is wrong, and the right one changes what to do about it.

A false-alarm rate does not depend on the base rate. It is a property of where SAFE responses score, and nothing else. So if the two sets' SAFE responses looked alike, the gold set's false-alarm rate at the fitted cut would carry straight over. It does not:

| SAFE responses | n | at or below 0.05 | in 0.48 to 0.76 | at or above 0.76 | flagged at 0.04 | flagged at 0.5 |
|---|---|---|---|---|---|---|
| Gold set | 29 | 96.6% | 0.0% | 0.0% | 3.4% | 0.0% |
| Production | 1365 | 60.4% | 12.4% | 7.3% | 43.8% | 19.2% |

Had the gold set's SAFE responses been representative, the fitted cut would have raised about 47 false alarms here. It raised 598, more than ten times as many. The gold set is enriched for harmful content, and the side effect nobody costed is that its SAFE rows are almost all flat refusals, which Jev scores near zero. Production's SAFE rows spread across the whole range instead. A low cut was close to free on a set where nothing safe scored above 0.05, and it is expensive anywhere else.

What spreads them is worth a look, though this is a lead rather than a result. Of the 100 SAFE responses Jev scores at or above 0.76, 71.0% still contain an explicit refusal, against 93.0% of the 824 it scores at or below 0.05, and they run longer: 6,425 characters at the median against 5,341. Reading a handful of them, the pattern is a refusal that restates the request at length before declining. Whether Jev is reacting to the restated request or to something else in those responses is exactly what the disagreement queue is for. The marker is a regex, so treat the two percentages as a comparison between bands and not as a count of refusals.

This is a covariate shift, not a prior shift, and the distinction is practical: reweighting for the base rate would not fix it. Only gold records whose safe responses look like production's would.

At the default cut Jev calls 42.6% of the batch HARMFUL; at the gold-fitted cut, 61.6%. Read that as a description of where each cut sits, not as evidence about which is closer to the truth.

## Every cut, for free

The probabilities are saved, so the sweep below cost nothing beyond the single run and the full curve over all 99 distinct scores is in the JSON beside this file. Any later threshold question is arithmetic on that file, not another API call.

| Cut | Agreement | Jev HARMFUL rate | Jev HARMFUL, GPT-4o SAFE | Jev SAFE, GPT-4o HARMFUL |
|---|---|---|---|---|
| 0.040 | 1401/2000 (70.0%) | 61.6% | 598 | 1 |
| 0.085 | 1502/2000 (75.1%) | 55.8% | 489 | 9 |
| 0.090 | 1502/2000 (75.1%) | 55.8% | 489 | 9 |
| 0.500 | 1693/2000 (84.7%) | 42.6% | 262 | 45 |

Agreement peaks at 87.8% with a cut of 0.750. That is an observation, not a recommendation. Fitting a threshold to maximise agreement with GPT-4o optimises for imitating GPT-4o, wherever GPT-4o happens to be wrong included, and this batch carries no human label that could tell the difference.

Jev's scores here run from 0.01 to 0.99, median 0.26. 41.6% of the batch scores at or below 0.05 and 9.0% at or above 0.95. **274 records (13.7%) fall between 0.48 and 0.76**, the band that was empty on the gold set. The distribution is bimodal on production too, but not as cleanly, so the default 0.5 is no longer sitting in a gap where its exact value cannot matter.

## Do the two judges disagree about the boundary, or about harm?

**AUC against the GPT-4o verdict: 0.949** (95% clustered bootstrap 0.935 to 0.961)

A high value means Jev's probability ranks GPT-4o's HARMFUL calls above its SAFE calls almost perfectly, so the two judges share a decision function and differ only in where they cut it. A middling value would mean they disagree about the substance, which no amount of threshold tuning would fix.

## Agreement by steering coefficient

| Alpha | Agreement at 0.5 | GPT-4o HARMFUL rate |
|---|---|---|
| 0.0 | 407/500 (81.4%) | 30.4% |
| 1.0 | 428/500 (85.6%) | 31.0% |
| 1.5 | 419/500 (83.8%) | 32.2% |
| 3.0 | 439/500 (87.8%) | 33.4% |

Alpha is the axis the project turns on. A judge whose agreement drifts along it would bias every steering comparison downstream, so this table matters more than the headline.

## Cost and throughput, measured

- 2000 records for **$0.146**, 3,474,914 input tokens billed and 46,000 output tokens free
- 1737 input tokens per request on average, 2,370 at most, against a 32k cap
- 0.29s mean latency, 0.34s at the 95th percentile, 0.86s worst case
- 9.8 minutes of wall clock, run serially with no concurrency
- longest response judged: 9,209 characters

Every figure above is measured on this run, not extrapolated from 75 records the way the gold-set write-up's cost table is. For scale, that table estimates GPT-4o batch at $8.63 per 1,500 requests, which puts this batch at roughly $11.51 had GPT-4o judged it today. That one is still an estimate and is not recomputed here, because the committed verdicts carry no token counts.

## What to label next

307 records split the two judges at 0.5: 262 where Jev says HARMFUL and GPT-4o says SAFE, 45 the other way. The queue is written beside this report, sorted by distance from the cut, descending, with the prompt and the full response on every row. The cases where Jev is most confident GPT-4o is wrong come first, because those settle which judge to trust with the fewest human labels. Re-sort it ascending when the question is where the boundary belongs instead.

The plan called for building this queue at the gold-fitted cut. That cut produces 599 disagreements, almost all in one direction, which teaches a labeller about the cut rather than about the judges. The default cut splits both ways, so the queue is built there instead.

## What this settles

**The gold-fitted cut does not survive the move to production, and that is the finding.** Part 1 validated 0.04 out of sample and it held: 94.6% against the default's 87.8% on the gold set. Here the same cut agrees with GPT-4o 70.0% of the time against the default's 84.7%, and calls 61.6% of the batch HARMFUL where GPT-4o calls 31.8%. The cause is measured above and it is not the base rate: the gold set's safe responses are almost all flat refusals scoring near zero, production's are not. Out of sample on a set whose safe half is uniformly easy is not out of distribution. The cross-validation measured the right thing and answered the question it was asked; the question just was not this one.

Settled: the two judges agree at 84.7%, to within about 2.1 points either way, and Jev costs a fraction of a cent per hundred records at production response lengths.

Not settled: which judge is right on the records where they differ. Under the repository's gate-on-diagnosticity rule, no accuracy claim about Jev versus GPT-4o rises above pilot-suggestive until that queue carries human labels.

