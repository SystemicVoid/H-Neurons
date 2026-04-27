# SIMID Report

Generated: 2026-04-26T12:09:54.982367+00:00

All primary estimates use base-item pairing and bootstrap 95% CIs. The primary MC endpoint is the lettered forced-choice likelihood prompt.
Open correctness is reported as adjudicated_open_correct: judge adjudication is used when present, deterministic alias grading is only a fallback for unadjudicated rows, and deterministic alias diagnostics are kept separate.

Open grade sources: adjudication=3, deterministic_alias=61
Open correctness claimability: blocked (mixed_adjudication_and_deterministic_sources).

## selected
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 16
Baseline rates: lettered MC=0.6875 [0.4688, 0.8750], adjudicated_open_correct=0.4375 [0.1875, 0.6875]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=8; lettered MC=0.9375 [0.8125, 1.0000]; adjudicated_open_correct=0.7500 [0.3750, 1.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=allow_fitted): n=4; lettered MC=0.6250 [0.2500, 1.0000]; adjudicated_open_correct=0.2500 [0.0000, 0.7500]
- truthfulqa / truthfulqa_mc1 (artifact_split=train, seen_in_iti_fit=true, leakage_policy=allow_fitted): n=4; lettered MC=0.2500 [0.0000, 0.7500]; adjudicated_open_correct=0.0000 [0.0000, 0.0000]

## unhooked
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 16
Baseline rates: lettered MC=0.6875 [0.4688, 0.8750], adjudicated_open_correct=0.3750 [0.1250, 0.6250]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=8; lettered MC=0.9375 [0.8125, 1.0000]; adjudicated_open_correct=0.7500 [0.3750, 1.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=allow_fitted): n=4; lettered MC=0.6250 [0.2500, 1.0000]; adjudicated_open_correct=0.0000 [0.0000, 0.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=train, seen_in_iti_fit=true, leakage_policy=allow_fitted): n=4; lettered MC=0.2500 [0.0000, 0.7500]; adjudicated_open_correct=0.0000 [0.0000, 0.0000]
