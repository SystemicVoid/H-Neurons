# ITI Audit — Baseline Snapshot (2026-03-30)

## Artifact families

### iti_triviaqa_transfer
- **Source**: TriviaQA consistency responses (closed-book factuality)
- **Examples**: 2642 train (1063 T / 1579 F), 782 val (391 T / 391 F)
- **Activation surface**: pre-o_proj head outputs
- **Position selection**: best of first_answer_token, mean_answer_span, last_answer_token
- **Ranking**: AUROC primary, balanced_accuracy secondary
- **Direction**: mass_mean (truthful − untruthful centroid, L2-normalized)
- **Top 5 heads**:

| Rank | Layer | Head | Position | AUROC | Bal Acc | σ |
|------|-------|------|----------|-------|---------|------|
| 1 | 11 | 2 | mean_answer_span | 0.768 | 0.708 | 1.55 |
| 2 | 11 | 3 | mean_answer_span | 0.768 | 0.698 | 1.38 |
| 3 | 11 | 0 | last_answer_token | 0.751 | 0.696 | 1.07 |
| 4 | 9 | 3 | last_answer_token | 0.750 | 0.691 | 9.42 |
| 5 | 22 | 4 | mean_answer_span | 0.749 | 0.684 | 1.82 |

- **Artifact**: `data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt`

### iti_context_grounded
- **Source**: SQuAD-v2 (answerable vs impossible questions)
- **Examples**: 800 train (400 T / 400 F), 200 val (100 T / 100 F)
- **Same activation surface, position selection, ranking, direction as above**
- **Top 5 heads**:

| Rank | Layer | Head | Position | AUROC | Bal Acc | σ |
|------|-------|------|----------|-------|---------|------|
| 1 | 14 | 7 | mean_answer_span | 0.9999 | 0.990 | 2.43 |
| 2 | 5 | 1 | last_answer_token | 0.9998 | 0.980 | 7.47 |
| 3 | 13 | 1 | last_answer_token | 0.9991 | 0.985 | 2.64 |
| 4 | 6 | 7 | last_answer_token | 0.9989 | 0.980 | 5.58 |
| 5 | 18 | 4 | mean_answer_span | 0.9988 | 0.980 | 1.97 |

- **Note**: Near-perfect AUROC (~1.0) suggests the probe is picking up a trivial signal (likely lexical overlap between context and answer), not truthfulness.
- **Artifact**: `data/contrastive/truthfulness/iti_context/iti_heads.pt`

## FaithEval calibration runs (20 samples)

Both runs used:
- Prompt style: `anti_compliance`
- K: 16 (ranked)
- Alphas: 0.0, 0.1, 0.5, 1.0, 2.0
- Max samples: 20
- Sample IDs: `data/manifests/faitheval_iti_calibration_ids_seed42_n20.json`

### Results

| Alpha | triviaqa compliance | context compliance | Wilson 95% CI |
|-------|--------------------|--------------------|---------------|
| 0.0 | 85.0% (17/20) | 85.0% (17/20) | [0.640, 0.948] |
| 0.1 | 85.0% (17/20) | 85.0% (17/20) | [0.640, 0.948] |
| 0.5 | 85.0% (17/20) | 85.0% (17/20) | [0.640, 0.948] |
| 1.0 | 85.0% (17/20) | 85.0% (17/20) | [0.640, 0.948] |
| 2.0 | 85.0% (17/20) | 85.0% (17/20) | [0.640, 0.948] |

Completely flat. Zero parse failures at all alphas.

### Provenance

| Run | Git SHA | Started |
|-----|---------|---------|
| triviaqa | `0d26f5b` | 2026-03-30T15:52:46Z |
| context_grounded | `4a74bf3` | 2026-03-30T16:38:02Z |

## Key observations

1. **Both artifacts produce identical outputs** at every alpha on these 20 samples — not just the same compliance rate but the same per-sample responses.
2. **Wide CIs** (±15pp) mean we cannot distinguish a 5pp effect from zero at n=20.
3. **AUROC ≈ 1.0 on context_grounded** is a red flag: the probe is learning something trivial about SQuAD-v2 structure, not a general truthfulness signal.
4. **The FaithEval harness uses generation + regex**, which introduces a coupling between prompt style and parse reliability that contaminates interpretation.
