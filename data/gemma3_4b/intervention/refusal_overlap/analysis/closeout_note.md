# D3.5 Refusal-Overlap Closeout

## Geometry

- Canonical direction orientation: D2 stores `harmful - harmless`, so negative signed cosine means anti-refusal / harmless-ward alignment.
- Canonical signed cosine mean: -0.017291
- Canonical overlap gap vs null mean: -0.018253
- Refusal-subspace energy fraction mean: 0.038760
- Refusal-subspace gap vs null mean: 0.036118

## FaithEval Mediation

- Canonical Spearman(overlap, compliance slope): -0.086903
- Refusal-subspace Spearman(overlap, compliance slope): 0.085665
- Secondary Spearman(overlap, endpoint delta): -0.101678

## Jailbreak Externality

- Canonical Spearman(overlap, csv2_yes slope): -0.116404
- Refusal-subspace Spearman(overlap, csv2_yes slope): 0.111993
- Secondary Spearman(overlap, endpoint delta): -0.151087

## Dominant-Layer Fragility Check

- Dominant layer by subspace gap: 33
- FaithEval Spearman after excluding dominant layer: -0.004759 (canonical), -0.021826 (subspace)
- Jailbreak Spearman after excluding dominant layer: 0.030285 (canonical), -0.158750 (subspace)


## D4 Gate

- Decision: proceed_with_d4_unchanged
- Interpretation: Refusal overlap is too weak to explain Baseline A.
