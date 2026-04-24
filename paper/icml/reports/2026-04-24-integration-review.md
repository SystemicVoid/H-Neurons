# Integration review — 2026-04-24

## Numerical drift table

| file:line | Drift | source-report value | patch |
|---|---|---|---|
| `paper/icml/main.tex:536` | The row is under `Delta margin vs. no-op`, but `[-1.75,+0.14]` is the `utility - matched_random` seed range, not a matched-random-vs-noop delta. | SAE report §2.1/§2.3: utility-minus-random seed range `[-1.753,+0.143]`; matched-random margins are `+8.33 ± 0.53` around noop `+8.309`. | `old_string`: `Matched-random (10 seeds)  & $66.4\% \pm 0.7$     & $+8.33 \pm 0.53$       & $\approx 0$ (per-seed range $[-1.75, +0.14]$) \\` / `new_string`: `Matched-random (10 seeds)  & $66.4\% \pm 0.7$     & $+8.33 \pm 0.53$       & utility$-$random seed range $[-1.75, +0.14]$ \\` and change the column header to `\textbf{Margin contrast (nats)}`. |
| `paper/icml/main.tex:599` | B-cohort CI is rounded more aggressively than the canonical table. | Bridge-margin report §3.2: `[-50.09, -30.37]`. | `old_string`: `$[-50.1, -30.4]$` / `new_string`: `$[-50.09, -30.37]$`. |

No other checked new headline numbers drift from the reports.

## Prose / framing issues

- `paper/icml/main.tex:251` — Offending text gives the margin result main-text weight and uses “logprob,” against the bridge report's prescribed one-line pointer / log-likelihood wording. `old_string`: `A teacher-forced gold-vs-wrong logprob-margin analysis on the same 57 discordant cases plus 200 length-matched random wrong-entity controls (Appendix~\ref{app:bridge_margin}) confirms that ITI causally compresses the gold-vs-wrong margin on right-to-wrong flips and expands it on rescues, but the behavioral taxonomy does not index a distinct mechanistic signature: non-substitution flips show \emph{larger} margin compression than substitution flips, so the taxonomy remains a reliable behavioral description rather than a margin-level fingerprint.` / `new_string`: `A teacher-forced gold-vs-wrong log-likelihood-margin check (Appendix~\ref{app:bridge_margin}) has the same directional R$\to$W/W$\to$R sign, but no substitution-specific signature: non-substitution flips show larger compression than substitution flips.`

- `paper/icml/main.tex:178,537,545` — The path-drift point estimate is a quantitative claim without its CI; line 537 also contradicts the table caption promising paired CIs. `old_string`: `$+8.2\times 10^{-8}$~nats` / `new_string`: `$+8.2\times 10^{-8}$~nats $[-2.4\times10^{-3}, +2.1\times10^{-3}]$` (apply analogously to `+8.2{\times}10^{-8}`).

- `paper/icml/main.tex:179,552` — “two held-out readout criteria” mislabels the utility selector as a readout criterion. `old_string`: `two held-out readout criteria pick features with opposite-sign margin effects` / `new_string`: `readout- and utility-based criteria pick features with opposite-sign margin effects`.

- `paper/icml/main.tex:189` — “layer coverage held fixed” overstates the control; the available extraction layers are fixed, not the selected layer histogram for readout-vs-utility. `old_string`: `with operator form and layer coverage held fixed` / `new_string`: `with operator form and the available extraction layers held fixed`.

- `paper/icml/main.tex:547` — The random-positive seed range lacks the seed-level CIs required for quantitative reporting. `old_string`: `with three layer-matched random-positive seeds yielding paired deltas $-0.42$ to $-0.81$~nats (all CIs exclude zero).` / `new_string`: `with three layer-matched random-positive seeds yielding paired deltas $-0.74$ $[-1.10, -0.36]$, $-0.81$ $[-1.16, -0.46]$, and $-0.42$ $[-0.75, -0.08]$~nats.`

- `paper/icml/main.tex:612` — Position-0 quantitative comparison lacks uncertainty. `old_string`: `($B$ position-$0 = -21.74$~nats vs.\ $A$ position-$0 = -3.49$~nats)` / `new_string`: `($B$ position-$0 = -21.74$~nats $[-29.48, -14.85]$ vs.\ $A$ position-$0 = -3.49$~nats $[-5.14, -1.97]$)`.

- `paper/icml/main.tex:614` — “mechanistic reading” is too ambitious for a negative margin-signature result. `old_string`: `broadens the mechanistic reading from substitution-specific entity suppression to answer-commitment margin compression on R$\to$W flips in general.` / `new_string`: `changes the margin-level reading from substitution-specific entity suppression to answer-commitment margin compression on R$\to$W flips in general.`

## Strongest reviewer objection

The strongest objection to the new material is that the SAE ablation optimizes and reports a margin-level selector while the behavioral endpoint remains null, so it may look like a post-hoc auxiliary metric inserted to rescue a failed SAE intervention and dilute the neuron-vs-SAE anchor.

The counter is that the paper does not claim SAE steering success: the ablation is a robustness check against the target-selection-artifact objection, uses a frozen validation/test split, preserves the accuracy null, and shows only that readout and utility selectors have opposite margin effects under the same operator. Pre-empt in text with the wording fixes above; do not expand it into another anchor.

## Verdict

Minor revisions. The added material meets the ICML-workshop bar for rigorous negative results and honest limitations, but the table-label drift, missing CIs, and a few over-ambitious phrases should be fixed before submission.

## Out-of-scope ideas

- do-not-apply-now: If space allows later, move the bridge margin pointer into an actual footnote to match the report's placement prescription exactly.
- do-not-apply-now: Consider a future 10-seed extension for the k=154 augment; current 3-seed result is useful but descriptively bounded.
