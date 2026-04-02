#!/usr/bin/env bash
# E2 ITI pipeline (TriviaQA source-isolated extraction policy).
#
# Chain:
#   splits -> calibration extraction -> Kxalpha sweep
#   -> disjoint 100-ID shortlist pilot + judge + pilot report
#   -> lock promotion with poison gate
#   -> final fold extraction + MC eval (first_3_tokens)
#   -> production extraction
#   -> SimpleQA-200 final eval + judge
#   -> 2-fold MC reports + canonical E2 report.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="E2 ITI pipeline (~multi-hour GPU + judge wait)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_triviaqa_transfer"
ITI_FAMILY_ARG="triviaqa_transfer"
CAL_DIR="data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_triviaqa_source_isolated_production"
FOLD_ROOT="data/contrastive/truthfulness/iti_triviaqa_source_isolated"
MANIFESTS="data/manifests"
SIMPLEQA_PATH="data/benchmarks/simpleqa_verified.csv"
SIMPLEQA_PANEL_200="${MANIFESTS}/simpleqa_verified_control200_seed42.json"
SIMPLEQA_PILOT_100="${MANIFESTS}/simpleqa_verified_pilot100_disjoint_seed42.json"
SIMPLEQA_PROMPT_STYLE="factual_phrase"
DECODE_SCOPE="first_3_tokens"
E2_REPORT_JSON="notes/act3-reports/e2_triviaqa_source_isolated_report.json"
E2_REPORT_MD="notes/act3-reports/2026-04-02-e2-triviaqa-source-isolated-audit.md"

echo "=== Phase 0: Build TruthfulQA calibration/final splits ==="
PYTHONUNBUFFERED=1 uv run python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    2>&1 | tee logs/e2_build_calibration_splits.log

echo ""
echo "=== Phase 1a: Extract E2 calibration artifact (source-isolated selectors) ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --output_dir "${CAL_DIR}" \
    --ranking_metric_override val_accuracy \
    --position_policy_override last_answer_token \
    2>&1 | tee logs/e2_extract_calibration.log

CAL_ARTIFACT="${CAL_DIR}/iti_heads.pt"
if [ ! -f "${CAL_ARTIFACT}" ]; then
    echo "FATAL: Calibration artifact not found at ${CAL_ARTIFACT}" >&2
    exit 1
fi

echo ""
echo "=== Phase 1b: K x alpha sweep (resolution-aware shortlist) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_calibration_sweep.py \
    --artifact_path "${CAL_ARTIFACT}" \
    --cal_val_mc1_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
    --cal_val_mc2_manifest "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" \
    --k_values 8 12 16 24 32 40 \
    --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/e2_calibration_sweep.log

echo ""
echo "=== Phase 1c: Build disjoint 100-ID pilot manifest ==="
PYTHONUNBUFFERED=1 uv run python - <<'PY'
import json
from pathlib import Path
import pandas as pd

seed = 42
final_manifest = Path("data/manifests/simpleqa_verified_control200_seed42.json")
pilot_manifest = Path("data/manifests/simpleqa_verified_pilot100_disjoint_seed42.json")
csv_path = Path("data/benchmarks/simpleqa_verified.csv")

final_ids = set(json.loads(final_manifest.read_text(encoding="utf-8")))
df = pd.read_csv(csv_path)
all_ids = [f"simpleqa_{int(row.get('original_index', idx))}" for idx, row in df.iterrows()]
candidate_ids = sorted(set(all_ids) - final_ids)
if len(candidate_ids) < 100:
    raise RuntimeError(f"Need at least 100 disjoint IDs, found {len(candidate_ids)}")

import random
rng = random.Random(seed)
rng.shuffle(candidate_ids)
pilot_ids = sorted(candidate_ids[:100])
pilot_manifest.write_text(json.dumps(pilot_ids, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {pilot_manifest} with {len(pilot_ids)} IDs")
PY

echo ""
echo "=== Phase 1d: Shortlist generation pilot (100 disjoint IDs) ==="
SHORTLIST_TSV="$(mktemp)"
PYTHONUNBUFFERED=1 uv run python - <<PY > "${SHORTLIST_TSV}"
import json
from pathlib import Path

sweep_path = Path("${CAL_DIR}") / "sweep_results.json"
data = json.loads(sweep_path.read_text(encoding="utf-8"))
shortlist = data.get("selection_diagnostics", {}).get("shortlist", [])
if not shortlist:
    raise RuntimeError("Shortlist is empty in sweep_results.json")
for row in shortlist:
    print(f"{int(row['k'])}\t{float(row['alpha'])}")
PY

resolve_iti_output_dir() {
    local HEAD_PATH=$1
    local BENCHMARK_NAME=$2
    local K=$3
    local SCOPE=$4
    uv run python - <<PY
import sys
sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix
suffix = build_iti_output_suffix(
    "${HEAD_PATH}",
    "${ITI_FAMILY_ARG}",
    ${K},
    "ranked",
    42,
    "artifact",
    None,
    "${SCOPE}",
)
print(f"data/gemma3_4b/intervention/${BENCHMARK_NAME}_{suffix}/experiment")
PY
}

CANDIDATE_ARGS=()
while IFS=$'\t' read -r K ALPHA; do
    [ -z "${K}" ] && continue
    echo "--- Pilot candidate K=${K}, alpha=${ALPHA} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${CAL_ARTIFACT}" \
        --iti_family "${ITI_FAMILY_ARG}" \
        --benchmark simpleqa \
        --simpleqa_path "${SIMPLEQA_PATH}" \
        --simpleqa_prompt_style "${SIMPLEQA_PROMPT_STYLE}" \
        --iti_decode_scope "${DECODE_SCOPE}" \
        --iti_k "${K}" \
        --alphas 0.0 "${ALPHA}" \
        --sample_manifest "${SIMPLEQA_PILOT_100}" \
        2>&1 | tee "logs/e2_pilot_simpleqa_k${K}_a${ALPHA}.log"

    PILOT_DIR="$(resolve_iti_output_dir "${CAL_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${K}" "${DECODE_SCOPE}")"
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
        --benchmark simpleqa \
        --input_dir "${PILOT_DIR}" \
        --alphas 0.0 "${ALPHA}" \
        --api-mode batch \
        2>&1 | tee "logs/e2_pilot_simpleqa_judge_k${K}_a${ALPHA}.log"

    CANDIDATE_ARGS+=("--candidate" "${K}:${ALPHA}:${PILOT_DIR}")
done < "${SHORTLIST_TSV}"

rm -f "${SHORTLIST_TSV}"

PILOT_REPORT="${CAL_DIR}/shortlist_generation_pilot_report.json"
PYTHONUNBUFFERED=1 uv run python scripts/report_simpleqa_shortlist_pilot.py \
    "${CANDIDATE_ARGS[@]}" \
    --output_path "${PILOT_REPORT}" \
    --baseline_alpha 0.0 \
    --attempt_gate_pp -10.0 \
    --precision_gate_pp 0.0 \
    --not_attempted_gate_n 15 \
    2>&1 | tee logs/e2_shortlist_pilot_report.log

echo ""
echo "=== Phase 1e: Lock promotion with pilot poison gate ==="
PYTHONUNBUFFERED=1 uv run python scripts/lock_config.py \
    --state_dir "${CAL_DIR}" \
    --pilot_report "${PILOT_REPORT}" \
    2>&1 | tee logs/e2_lock_config.log

LOCKED_CONFIG="${CAL_DIR}/locked_iti_config.json"
LOCKED_K=$(jq -r '.K_locked' "${LOCKED_CONFIG}")
LOCKED_ALPHA=$(jq -r '.alpha_locked' "${LOCKED_CONFIG}")
if [ "${LOCKED_K}" = "null" ] || [ "${LOCKED_ALPHA}" = "null" ]; then
    echo "FATAL: lock_config did not produce a locked candidate." >&2
    exit 1
fi
echo "Locked: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"

echo ""
echo "=== Phase 2a: Extract final fold artifacts (source-isolated selectors) ==="
for FOLD in 0 1; do
    FOLD_DIR="${FOLD_ROOT}/final_fold${FOLD}"
    echo "--- Fold ${FOLD} -> ${FOLD_DIR} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --output_dir "${FOLD_DIR}" \
        --ranking_metric_override val_accuracy \
        --position_policy_override last_answer_token \
        2>&1 | tee "logs/e2_extract_final_fold${FOLD}.log"
done

echo ""
echo "=== Phase 2b: Final 2-fold TruthfulQA MC eval (first_3_tokens) ==="
for FOLD in 0 1; do
    FOLD_ARTIFACT="${FOLD_ROOT}/final_fold${FOLD}/iti_heads.pt"
    FOLD_DEF="${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json"
    for VARIANT in mc1 mc2; do
        echo "--- Fold ${FOLD}, ${VARIANT} ---"
        PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
            --intervention_mode iti_head \
            --iti_head_path "${FOLD_ARTIFACT}" \
            --iti_family "${ITI_FAMILY_ARG}" \
            --benchmark truthfulqa_mc \
            --truthfulqa_variant "${VARIANT}" \
            --truthfulqa_fold_path "${FOLD_DEF}" \
            --iti_decode_scope "${DECODE_SCOPE}" \
            --iti_k "${LOCKED_K}" \
            --alphas 0.0 "${LOCKED_ALPHA}" \
            --sample_manifest "${MANIFESTS}/truthfulqa_final_fold${FOLD}_heldout_${VARIANT}_seed${SEED}.json" \
            2>&1 | tee "logs/e2_final_fold${FOLD}_${VARIANT}.log"
    done
done

echo ""
echo "=== Phase 2c: Build E2 production artifact (source-isolated selectors) ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --output_dir "${PROD_DIR}" \
    --ranking_metric_override val_accuracy \
    --position_policy_override last_answer_token \
    2>&1 | tee logs/e2_extract_production.log

PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

echo ""
echo "=== Phase 3a: 2-fold TruthfulQA report ==="
for VARIANT in mc1 mc2; do
    FOLD0_DIR="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_${VARIANT}" "${LOCKED_K}" "${DECODE_SCOPE}")"
    FOLD1_DIR="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_${VARIANT}" "${LOCKED_K}" "${DECODE_SCOPE}")"
    PYTHONUNBUFFERED=1 uv run python scripts/report_iti_2fold.py \
        --fold0_dir "${FOLD0_DIR}" \
        --fold1_dir "${FOLD1_DIR}" \
        --locked_alpha "${LOCKED_ALPHA}" \
        --locked_k "${LOCKED_K}" \
        --variant "${VARIANT}" \
        --output_dir notes/act3-reports \
        2>&1 | tee "logs/e2_report_2fold_${VARIANT}.log"
done

echo ""
echo "=== Phase 3b: SimpleQA-200 final run (first_3_tokens) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family "${ITI_FAMILY_ARG}" \
    --benchmark simpleqa \
    --simpleqa_path "${SIMPLEQA_PATH}" \
    --simpleqa_prompt_style "${SIMPLEQA_PROMPT_STYLE}" \
    --iti_decode_scope "${DECODE_SCOPE}" \
    --iti_k "${LOCKED_K}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --sample_manifest "${SIMPLEQA_PANEL_200}" \
    2>&1 | tee logs/e2_simpleqa_200.log

E2_SIMPLEQA_DIR="$(resolve_iti_output_dir "${PROD_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${LOCKED_K}" "${DECODE_SCOPE}")"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${E2_SIMPLEQA_DIR}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --api-mode batch \
    2>&1 | tee logs/e2_simpleqa_200_judge.log

echo ""
echo "=== Phase 3c: Canonical E2 consolidated report ==="

PAPER_LOCK="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/locked_iti_config.json"
E1_LOCK="data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/locked_iti_config.json"
PAPER_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
E1_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_modernized_production/iti_heads.pt"

PAPER_MC1_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment"
PAPER_MC1_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment"
PAPER_MC2_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment"
PAPER_MC2_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment"
PAPER_SIMPLEQA="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment"

E1_MC1_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold0-iti-heads_e1f7e6ab4f/experiment"
E1_MC1_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold1-iti-heads_efccb19f6c/experiment"
E1_MC2_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold0-iti-heads_e1f7e6ab4f/experiment"
E1_MC2_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold1-iti-heads_efccb19f6c/experiment"
E1_SIMPLEQA="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-modernized-production-iti-heads_7a0136f3d3/experiment"

E2_MC1_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc1" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2_MC1_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc1" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2_MC2_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc2" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2_MC2_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc2" "${LOCKED_K}" "${DECODE_SCOPE}")"

PYTHONUNBUFFERED=1 uv run python scripts/report_e2_canonical.py \
    --e2_locked_config "${LOCKED_CONFIG}" \
    --e2_pilot_report "${PILOT_REPORT}" \
    --e2_mc1_fold0_dir "${E2_MC1_F0}" \
    --e2_mc1_fold1_dir "${E2_MC1_F1}" \
    --e2_mc2_fold0_dir "${E2_MC2_F0}" \
    --e2_mc2_fold1_dir "${E2_MC2_F1}" \
    --e2_simpleqa_dir "${E2_SIMPLEQA_DIR}" \
    --e2_artifact "${PROD_ARTIFACT}" \
    --paper_locked_config "${PAPER_LOCK}" \
    --paper_mc1_fold0_dir "${PAPER_MC1_F0}" \
    --paper_mc1_fold1_dir "${PAPER_MC1_F1}" \
    --paper_mc2_fold0_dir "${PAPER_MC2_F0}" \
    --paper_mc2_fold1_dir "${PAPER_MC2_F1}" \
    --paper_simpleqa_dir "${PAPER_SIMPLEQA}" \
    --paper_artifact "${PAPER_ARTIFACT}" \
    --e1_locked_config "${E1_LOCK}" \
    --e1_mc1_fold0_dir "${E1_MC1_F0}" \
    --e1_mc1_fold1_dir "${E1_MC1_F1}" \
    --e1_mc2_fold0_dir "${E1_MC2_F0}" \
    --e1_mc2_fold1_dir "${E1_MC2_F1}" \
    --e1_simpleqa_dir "${E1_SIMPLEQA}" \
    --e1_artifact "${E1_ARTIFACT}" \
    --output_json "${E2_REPORT_JSON}" \
    --output_md "${E2_REPORT_MD}" \
    --materiality_pp 1.5 \
    2>&1 | tee logs/e2_canonical_report.log

echo ""
echo "=== E2 pipeline complete ==="
echo "Family: ${FAMILY} (source-isolated selectors)"
echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Pilot report: ${PILOT_REPORT}"
echo "SimpleQA dir: ${E2_SIMPLEQA_DIR}"
echo "Canonical report: ${E2_REPORT_MD}"

