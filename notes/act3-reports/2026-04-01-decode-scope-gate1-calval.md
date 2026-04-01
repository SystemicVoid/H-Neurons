# Decode-Scope Gate 1: TruthfulQA MC1 cal-val

> **Historical gate snapshot only.**
>
> The canonical review now lives in
> [2026-04-01-decode-scope-gate1-audit.md](./2026-04-01-decode-scope-gate1-audit.md).
> Use that file for interpretation, caveats, and next-step decisions.

- Generated: 2026-04-01T22:22:48Z
- Artifact: `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt`
- Manifest: `data/manifests/truthfulqa_cal_val_mc1_seed42.json`
- Locked config: `K=12`, `alpha=8.0`
- Purpose: cheap answer-selection gate before any SimpleQA generation or batch judging

| Scope | MC1 @ α=0.0 | MC1 @ α=8.0 | Δ pp | Retained vs full_decode |
| --- | --- | --- | ---: | ---: |
| `full_decode` | 22.2% [14.5, 32.4] (18/81) | 35.8% [26.2, 46.7] (29/81) | +13.6 | 100% |
| `first_token_only` | 22.2% [14.5, 32.4] (18/81) | 25.9% [17.6, 36.4] (21/81) | +3.7 | 27% |
| `first_3_tokens` | 22.2% [14.5, 32.4] (18/81) | 34.6% [25.1, 45.4] (28/81) | +12.3 | 91% |
| `first_8_tokens` | 22.2% [14.5, 32.4] (18/81) | 34.6% [25.1, 45.4] (28/81) | +12.3 | 91% |
