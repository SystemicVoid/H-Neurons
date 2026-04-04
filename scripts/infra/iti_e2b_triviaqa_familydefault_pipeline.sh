#!/usr/bin/env bash
# E2-B ITI diagnostic pipeline (TriviaQA family-default selectors).
#
# Hypothesis: E2-A's null may be due to paper-faithful selector overrides
# (val_accuracy + last_answer_token), not TriviaQA source data itself.
# E2-B removes all selector overrides, using the family defaults
# (AUROC ranking + all_answer_positions = first/mean/last).
#
# Chain: identical to E2-A but without --ranking_metric_override
# and --position_policy_override flags on extraction calls.
#
#   splits -> calibration extraction -> Kxalpha sweep
#   -> disjoint 100-ID shortlist pilot + judge + pilot report
#   -> lock promotion with poison gate
#   -> final fold extraction + MC eval (first_3_tokens)
#   -> sidecar MC eval (K=40,a=6.0 + K=16,a=6.0)
#   -> production extraction
#   -> SimpleQA-200 final eval + judge
#   -> 2-fold MC reports + E2-B diagnostic report.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="E2-B ITI diagnostic pipeline (~multi-hour GPU + judge wait)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_triviaqa_transfer"
ITI_FAMILY_ARG="triviaqa_transfer"
CAL_DIR="data/contrastive/truthfulness/iti_triviaqa_familydefault_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_triviaqa_familydefault_production"
FOLD_ROOT="data/contrastive/truthfulness/iti_triviaqa_familydefault"
MANIFESTS="data/manifests"
SIMPLEQA_PATH="data/benchmarks/simpleqa_verified.csv"
SIMPLEQA_PANEL_200="${MANIFESTS}/simpleqa_verified_control200_seed42.json"
SIMPLEQA_PILOT_100="${MANIFESTS}/simpleqa_verified_pilot100_disjoint_seed42.json"
SIMPLEQA_PROMPT_STYLE="factual_phrase"
DECODE_SCOPE="first_3_tokens"
E2B_REPORT_JSON="notes/act3-reports/e2b_triviaqa_familydefault_report.json"
E2B_REPORT_MD="notes/act3-reports/2026-04-03-e2b-triviaqa-familydefault-diagnostic.md"

ALLOW_NONIDENTICAL_ARTIFACTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --allow-nonidentical-artifacts)
            ALLOW_NONIDENTICAL_ARTIFACTS=1
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: scripts/infra/iti_e2b_triviaqa_familydefault_pipeline.sh [--allow-nonidentical-artifacts]

Options:
  --allow-nonidentical-artifacts
      Continue past artifact SHA mismatch. Report is marked non-canonical and
      classification is forced to ambiguous.
USAGE
            exit 0
            ;;
        *)
            echo "FATAL: Unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

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

alpha_jsonl_path() {
    local EXPERIMENT_DIR=$1
    local ALPHA=$2
    uv run python - <<PY
import sys
sys.path.insert(0, "scripts")
from utils import format_alpha_label
print("${EXPERIMENT_DIR}/alpha_" + format_alpha_label(float("${ALPHA}")) + ".jsonl")
PY
}

require_file() {
    local PATH_VALUE=$1
    local LABEL=$2
    if [ ! -f "${PATH_VALUE}" ]; then
        echo "MISSING FILE [${LABEL}]: ${PATH_VALUE}" >&2
        PREFLIGHT_MISSING=1
    fi
}

require_dir() {
    local PATH_VALUE=$1
    local LABEL=$2
    if [ ! -d "${PATH_VALUE}" ]; then
        echo "MISSING DIR [${LABEL}]: ${PATH_VALUE}" >&2
        PREFLIGHT_MISSING=1
    fi
}

require_alpha_jsonl() {
    local EXPERIMENT_DIR=$1
    local ALPHA=$2
    local LABEL=$3
    local ALPHA_PATH
    ALPHA_PATH="$(alpha_jsonl_path "${EXPERIMENT_DIR}" "${ALPHA}")"
    require_file "${ALPHA_PATH}" "${LABEL} alpha=${ALPHA}"
}

# E0 (paper-faithful) reference paths
PAPER_LOCK="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/locked_iti_config.json"
PAPER_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
PAPER_MC1_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment"
PAPER_MC1_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment"
PAPER_MC2_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment"
PAPER_MC2_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment"
PAPER_SIMPLEQA="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment"

# E1 (modernized) reference paths
E1_LOCK="data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/locked_iti_config.json"
E1_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_modernized_production/iti_heads.pt"
E1_MC1_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold0-iti-heads_e1f7e6ab4f/experiment"
E1_MC1_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold1-iti-heads_efccb19f6c/experiment"
E1_MC2_F0="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold0-iti-heads_e1f7e6ab4f/experiment"
E1_MC2_F1="data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-full-decode_final-fold1-iti-heads_efccb19f6c/experiment"
E1_SIMPLEQA="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-modernized-production-iti-heads_7a0136f3d3/experiment"

# E2-A (source-isolated, paper-faithful overrides) reference paths
E2A_LOCK="data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration/locked_iti_config.json"
E2A_ARTIFACT="data/contrastive/truthfulness/iti_triviaqa_source_isolated_production/iti_heads.pt"
E2A_CAL_META="data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration/extraction_metadata.json"
MARCH30_META="data/contrastive/truthfulness/iti_triviaqa/extraction_metadata.json"

echo "=== Preflight: required inputs and references ==="
PREFLIGHT_MISSING=0

require_file "scripts/build_truthfulqa_calibration_splits.py" "phase0 script"
require_file "scripts/extract_truthfulness_iti.py" "extraction script"
require_file "scripts/run_calibration_sweep.py" "sweep script"
require_file "scripts/run_intervention.py" "intervention script"
require_file "scripts/evaluate_intervention.py" "judge script"
require_file "scripts/lock_config.py" "lock script"
require_file "scripts/report_iti_2fold.py" "2fold report script"
require_file "scripts/report_e2b_diagnostic.py" "diagnostic report script"
require_file "${SIMPLEQA_PATH}" "simpleqa csv"
require_file "${SIMPLEQA_PANEL_200}" "simpleqa 200 manifest"
require_file "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" "cal manifest mc1"
require_file "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" "cal manifest mc2"

require_file "${PAPER_LOCK}" "paper lock"
require_file "${PAPER_ARTIFACT}" "paper artifact"
require_file "${E1_LOCK}" "e1 lock"
require_file "${E1_ARTIFACT}" "e1 artifact"
require_file "${E2A_LOCK}" "e2a lock"
require_file "${E2A_ARTIFACT}" "e2a artifact"
require_file "${E2A_CAL_META}" "e2a metadata"
require_file "${MARCH30_META}" "march30 metadata"

if [ "${PREFLIGHT_MISSING}" -eq 1 ]; then
    echo "FATAL: Preflight missing required files." >&2
    exit 1
fi

E2A_LOCKED_K="$(jq -r '.K_locked' "${E2A_LOCK}")"
E2A_LOCKED_ALPHA="$(jq -r '.alpha_locked' "${E2A_LOCK}")"
PAPER_LOCKED_ALPHA="$(jq -r '.alpha_locked' "${PAPER_LOCK}")"
E1_LOCKED_ALPHA="$(jq -r '.alpha_locked' "${E1_LOCK}")"
for VALUE in "${E2A_LOCKED_K}" "${E2A_LOCKED_ALPHA}" "${PAPER_LOCKED_ALPHA}" "${E1_LOCKED_ALPHA}"; do
    if [ -z "${VALUE}" ] || [ "${VALUE}" = "null" ]; then
        echo "FATAL: Missing locked config fields in reference locks." >&2
        exit 1
    fi
done

E2A_MC1_F0="$(resolve_iti_output_dir "data/contrastive/truthfulness/iti_triviaqa_source_isolated/final_fold0/iti_heads.pt" "truthfulqa_mc_mc1" "${E2A_LOCKED_K}" "${DECODE_SCOPE}")"
E2A_MC1_F1="$(resolve_iti_output_dir "data/contrastive/truthfulness/iti_triviaqa_source_isolated/final_fold1/iti_heads.pt" "truthfulqa_mc_mc1" "${E2A_LOCKED_K}" "${DECODE_SCOPE}")"
E2A_MC2_F0="$(resolve_iti_output_dir "data/contrastive/truthfulness/iti_triviaqa_source_isolated/final_fold0/iti_heads.pt" "truthfulqa_mc_mc2" "${E2A_LOCKED_K}" "${DECODE_SCOPE}")"
E2A_MC2_F1="$(resolve_iti_output_dir "data/contrastive/truthfulness/iti_triviaqa_source_isolated/final_fold1/iti_heads.pt" "truthfulqa_mc_mc2" "${E2A_LOCKED_K}" "${DECODE_SCOPE}")"
E2A_SIMPLEQA="$(resolve_iti_output_dir "${E2A_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${E2A_LOCKED_K}" "${DECODE_SCOPE}")"

for REF_DIR in \
    "${PAPER_MC1_F0}" "${PAPER_MC1_F1}" "${PAPER_MC2_F0}" "${PAPER_MC2_F1}" "${PAPER_SIMPLEQA}" \
    "${E1_MC1_F0}" "${E1_MC1_F1}" "${E1_MC2_F0}" "${E1_MC2_F1}" "${E1_SIMPLEQA}" \
    "${E2A_MC1_F0}" "${E2A_MC1_F1}" "${E2A_MC2_F0}" "${E2A_MC2_F1}" "${E2A_SIMPLEQA}"; do
    require_dir "${REF_DIR}" "reference experiment dir"
done

require_alpha_jsonl "${PAPER_MC1_F0}" "${PAPER_LOCKED_ALPHA}" "paper mc1 fold0"
require_alpha_jsonl "${PAPER_MC1_F1}" "${PAPER_LOCKED_ALPHA}" "paper mc1 fold1"
require_alpha_jsonl "${PAPER_MC2_F0}" "${PAPER_LOCKED_ALPHA}" "paper mc2 fold0"
require_alpha_jsonl "${PAPER_MC2_F1}" "${PAPER_LOCKED_ALPHA}" "paper mc2 fold1"
require_alpha_jsonl "${PAPER_SIMPLEQA}" "${PAPER_LOCKED_ALPHA}" "paper simpleqa"

require_alpha_jsonl "${E1_MC1_F0}" "${E1_LOCKED_ALPHA}" "e1 mc1 fold0"
require_alpha_jsonl "${E1_MC1_F1}" "${E1_LOCKED_ALPHA}" "e1 mc1 fold1"
require_alpha_jsonl "${E1_MC2_F0}" "${E1_LOCKED_ALPHA}" "e1 mc2 fold0"
require_alpha_jsonl "${E1_MC2_F1}" "${E1_LOCKED_ALPHA}" "e1 mc2 fold1"
require_alpha_jsonl "${E1_SIMPLEQA}" "${E1_LOCKED_ALPHA}" "e1 simpleqa"

for E2A_DIR in "${E2A_MC1_F0}" "${E2A_MC1_F1}" "${E2A_MC2_F0}" "${E2A_MC2_F1}" "${E2A_SIMPLEQA}"; do
    require_alpha_jsonl "${E2A_DIR}" "0.0" "e2a baseline"
    require_alpha_jsonl "${E2A_DIR}" "${E2A_LOCKED_ALPHA}" "e2a locked"
done

if [ "${PREFLIGHT_MISSING}" -eq 1 ]; then
    echo "FATAL: Preflight failed; missing reference outputs would invalidate report stage." >&2
    exit 1
fi
echo "Preflight OK: all required inputs and reference outputs are present."
echo ""

echo "=== Phase 0: Build TruthfulQA calibration/final splits ==="
PYTHONUNBUFFERED=1 uv run python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    2>&1 | tee logs/e2b_build_calibration_splits.log

echo ""
echo "=== Phase 1a: Extract E2-B calibration artifact (family-default selectors) ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/e2b_extract_calibration.log

CAL_ARTIFACT="${CAL_DIR}/iti_heads.pt"
if [ ! -f "${CAL_ARTIFACT}" ]; then
    echo "FATAL: Calibration artifact not found at ${CAL_ARTIFACT}" >&2
    exit 1
fi

echo ""
echo "--- Verifying extraction metadata (family-default selectors, no overrides) ---"
PYTHONUNBUFFERED=1 uv run python - <<PY
import json, sys
from pathlib import Path

meta = json.loads(Path("${CAL_DIR}/extraction_metadata.json").read_text(encoding="utf-8"))
errors = []
if meta.get("head_ranking_metric") != "auroc":
    errors.append(f"Expected head_ranking_metric=auroc, got {meta.get('head_ranking_metric')}")
if meta.get("position_policy") != "all_answer_positions":
    errors.append(f"Expected position_policy=all_answer_positions, got {meta.get('position_policy')}")
overrides = meta.get("selector_overrides", {})
if overrides.get("ranking_metric_override") is not None:
    errors.append(f"ranking_metric_override should be null, got {overrides.get('ranking_metric_override')}")
if overrides.get("position_policy_override") is not None:
    errors.append(f"position_policy_override should be null, got {overrides.get('position_policy_override')}")
summaries = meta.get("position_summaries", [])
expected = ["first_answer_token", "mean_answer_span", "last_answer_token"]
if summaries != expected:
    errors.append(f"Expected position_summaries={expected}, got {summaries}")
if errors:
    print("FATAL: Extraction metadata assertions failed:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
print("OK: Extraction metadata confirms family-default selectors (auroc + all_answer_positions)")
PY

echo ""
echo "=== Phase 1b: K x alpha sweep (resolution-aware shortlist) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_calibration_sweep.py \
    --artifact_path "${CAL_ARTIFACT}" \
    --cal_val_mc1_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
    --cal_val_mc2_manifest "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" \
    --k_values 8 12 16 24 32 40 \
    --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/e2b_calibration_sweep.log

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
        2>&1 | tee "logs/e2b_pilot_simpleqa_k${K}_a${ALPHA}.log"

    PILOT_DIR="$(resolve_iti_output_dir "${CAL_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${K}" "${DECODE_SCOPE}")"
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
        --benchmark simpleqa \
        --input_dir "${PILOT_DIR}" \
        --alphas 0.0 "${ALPHA}" \
        --api-mode batch \
        2>&1 | tee "logs/e2b_pilot_simpleqa_judge_k${K}_a${ALPHA}.log"

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
    2>&1 | tee logs/e2b_shortlist_pilot_report.log

echo ""
echo "=== Phase 1e: Lock promotion with pilot poison gate ==="
PYTHONUNBUFFERED=1 uv run python scripts/lock_config.py \
    --state_dir "${CAL_DIR}" \
    --pilot_report "${PILOT_REPORT}" \
    2>&1 | tee logs/e2b_lock_config.log

LOCKED_CONFIG="${CAL_DIR}/locked_iti_config.json"
LOCKED_K=$(jq -r '.K_locked' "${LOCKED_CONFIG}")
LOCKED_ALPHA=$(jq -r '.alpha_locked' "${LOCKED_CONFIG}")
if [ "${LOCKED_K}" = "null" ] || [ "${LOCKED_ALPHA}" = "null" ]; then
    echo "FATAL: lock_config did not produce a locked candidate." >&2
    exit 1
fi
echo "Locked: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"

echo ""
echo "=== Phase 2a: Extract final fold artifacts (family-default selectors) ==="
for FOLD in 0 1; do
    FOLD_DIR="${FOLD_ROOT}/final_fold${FOLD}"
    echo "--- Fold ${FOLD} -> ${FOLD_DIR} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --output_dir "${FOLD_DIR}" \
        2>&1 | tee "logs/e2b_extract_final_fold${FOLD}.log"
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
            2>&1 | tee "logs/e2b_final_fold${FOLD}_${VARIANT}.log"
    done
done

echo ""
echo "=== Phase 2b-sidecars: Frozen comparator MC reruns (K=40,a=6.0 + K=16,a=6.0) ==="
for SIDECAR_K in 40 16; do
    for FOLD in 0 1; do
        FOLD_ARTIFACT="${FOLD_ROOT}/final_fold${FOLD}/iti_heads.pt"
        FOLD_DEF="${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json"
        for VARIANT in mc1 mc2; do
            echo "--- Sidecar K=${SIDECAR_K}, Fold ${FOLD}, ${VARIANT} ---"
            PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
                --intervention_mode iti_head \
                --iti_head_path "${FOLD_ARTIFACT}" \
                --iti_family "${ITI_FAMILY_ARG}" \
                --benchmark truthfulqa_mc \
                --truthfulqa_variant "${VARIANT}" \
                --truthfulqa_fold_path "${FOLD_DEF}" \
                --iti_decode_scope "${DECODE_SCOPE}" \
                --iti_k "${SIDECAR_K}" \
                --alphas 0.0 6.0 \
                --sample_manifest "${MANIFESTS}/truthfulqa_final_fold${FOLD}_heldout_${VARIANT}_seed${SEED}.json" \
                2>&1 | tee "logs/e2b_sidecar_k${SIDECAR_K}_fold${FOLD}_${VARIANT}.log"
        done
    done
done

echo ""
echo "=== Phase 2c: Build E2-B production artifact (family-default selectors) ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --output_dir "${PROD_DIR}" \
    2>&1 | tee logs/e2b_extract_production.log

PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

echo ""
echo "--- Verifying artifact identity across all four extractions ---"
CAL_SHA="$(sha256sum "${CAL_DIR}/iti_heads.pt" | cut -d' ' -f1)"
FOLD0_SHA="$(sha256sum "${FOLD_ROOT}/final_fold0/iti_heads.pt" | cut -d' ' -f1)"
FOLD1_SHA="$(sha256sum "${FOLD_ROOT}/final_fold1/iti_heads.pt" | cut -d' ' -f1)"
PROD_SHA="$(sha256sum "${PROD_DIR}/iti_heads.pt" | cut -d' ' -f1)"
echo "CAL:   ${CAL_SHA}"
echo "FOLD0: ${FOLD0_SHA}"
echo "FOLD1: ${FOLD1_SHA}"
echo "PROD:  ${PROD_SHA}"
NONCANONICAL_ARTIFACT_MISMATCH=0
ARTIFACT_MISMATCH_PAIRS=()
if [ "${CAL_SHA}" != "${FOLD0_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("CAL!=FOLD0"); fi
if [ "${CAL_SHA}" != "${FOLD1_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("CAL!=FOLD1"); fi
if [ "${CAL_SHA}" != "${PROD_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("CAL!=PROD"); fi
if [ "${FOLD0_SHA}" != "${FOLD1_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("FOLD0!=FOLD1"); fi
if [ "${FOLD0_SHA}" != "${PROD_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("FOLD0!=PROD"); fi
if [ "${FOLD1_SHA}" != "${PROD_SHA}" ]; then ARTIFACT_MISMATCH_PAIRS+=("FOLD1!=PROD"); fi
if [ "${#ARTIFACT_MISMATCH_PAIRS[@]}" -gt 0 ]; then
    echo "Artifact mismatch pairs: ${ARTIFACT_MISMATCH_PAIRS[*]}" >&2
    if [ "${ALLOW_NONIDENTICAL_ARTIFACTS}" -eq 1 ]; then
        NONCANONICAL_ARTIFACT_MISMATCH=1
        echo "WARNING: Continuing due to --allow-nonidentical-artifacts; report will be non-canonical and forced ambiguous." >&2
    else
        echo "FATAL: Artifact identity mismatch violates source-isolation assumptions." >&2
        echo "Re-run with --allow-nonidentical-artifacts only for debugging." >&2
        exit 1
    fi
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
        2>&1 | tee "logs/e2b_report_2fold_${VARIANT}.log"
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
    2>&1 | tee logs/e2b_simpleqa_200.log

E2B_SIMPLEQA_DIR="$(resolve_iti_output_dir "${PROD_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${LOCKED_K}" "${DECODE_SCOPE}")"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${E2B_SIMPLEQA_DIR}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --api-mode batch \
    2>&1 | tee logs/e2b_simpleqa_200_judge.log

echo ""
echo "=== Phase 3c: E2-B diagnostic report ==="

# E2-B own MC fold dirs (optimized lock)
E2B_MC1_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc1" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2B_MC1_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc1" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2B_MC2_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc2" "${LOCKED_K}" "${DECODE_SCOPE}")"
E2B_MC2_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc2" "${LOCKED_K}" "${DECODE_SCOPE}")"

# Sidecar dirs (K=40 and K=16 at alpha=6.0)
E2B_K40_MC1_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc1" 40 "${DECODE_SCOPE}")"
E2B_K40_MC1_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc1" 40 "${DECODE_SCOPE}")"
E2B_K40_MC2_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc2" 40 "${DECODE_SCOPE}")"
E2B_K40_MC2_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc2" 40 "${DECODE_SCOPE}")"

E2B_K16_MC1_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc1" 16 "${DECODE_SCOPE}")"
E2B_K16_MC1_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc1" 16 "${DECODE_SCOPE}")"
E2B_K16_MC2_F0="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_mc2" 16 "${DECODE_SCOPE}")"
E2B_K16_MC2_F1="$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_mc2" 16 "${DECODE_SCOPE}")"

E2B_CAL_META="${CAL_DIR}/extraction_metadata.json"

REPORT_EXTRA_ARGS=()
if [ "${NONCANONICAL_ARTIFACT_MISMATCH}" -eq 1 ]; then
    REPORT_EXTRA_ARGS+=(
        --noncanonical_artifact_mismatch
        --artifact_sha_cal "${CAL_SHA}"
        --artifact_sha_fold0 "${FOLD0_SHA}"
        --artifact_sha_fold1 "${FOLD1_SHA}"
        --artifact_sha_prod "${PROD_SHA}"
    )
fi

PYTHONUNBUFFERED=1 uv run python scripts/report_e2b_diagnostic.py \
    --e2b_name "E2-B TriviaQA Family-Default Selectors (Diagnostic)" \
    --e2b_locked_config "${LOCKED_CONFIG}" \
    --e2b_pilot_report "${PILOT_REPORT}" \
    --e2b_mc1_fold0_dir "${E2B_MC1_F0}" \
    --e2b_mc1_fold1_dir "${E2B_MC1_F1}" \
    --e2b_mc2_fold0_dir "${E2B_MC2_F0}" \
    --e2b_mc2_fold1_dir "${E2B_MC2_F1}" \
    --e2b_simpleqa_dir "${E2B_SIMPLEQA_DIR}" \
    --e2b_artifact "${PROD_ARTIFACT}" \
    --e2b_extraction_metadata "${E2B_CAL_META}" \
    --e2b_k40_mc1_fold0_dir "${E2B_K40_MC1_F0}" \
    --e2b_k40_mc1_fold1_dir "${E2B_K40_MC1_F1}" \
    --e2b_k40_mc2_fold0_dir "${E2B_K40_MC2_F0}" \
    --e2b_k40_mc2_fold1_dir "${E2B_K40_MC2_F1}" \
    --e2b_k16_mc1_fold0_dir "${E2B_K16_MC1_F0}" \
    --e2b_k16_mc1_fold1_dir "${E2B_K16_MC1_F1}" \
    --e2b_k16_mc2_fold0_dir "${E2B_K16_MC2_F0}" \
    --e2b_k16_mc2_fold1_dir "${E2B_K16_MC2_F1}" \
    --e2a_locked_config "${E2A_LOCK}" \
    --e2a_mc1_fold0_dir "${E2A_MC1_F0}" \
    --e2a_mc1_fold1_dir "${E2A_MC1_F1}" \
    --e2a_mc2_fold0_dir "${E2A_MC2_F0}" \
    --e2a_mc2_fold1_dir "${E2A_MC2_F1}" \
    --e2a_simpleqa_dir "${E2A_SIMPLEQA}" \
    --e2a_artifact "${E2A_ARTIFACT}" \
    --e2a_extraction_metadata "${E2A_CAL_META}" \
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
    --march30_extraction_metadata "${MARCH30_META}" \
    --output_json "${E2B_REPORT_JSON}" \
    --output_md "${E2B_REPORT_MD}" \
    --materiality_pp 1.5 \
    "${REPORT_EXTRA_ARGS[@]}" \
    2>&1 | tee logs/e2b_diagnostic_report.log

echo ""
echo "=== E2-B diagnostic pipeline complete ==="
echo "Family: ${FAMILY} (family-default selectors: AUROC + all_answer_positions)"
echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Pilot report: ${PILOT_REPORT}"
echo "SimpleQA dir: ${E2B_SIMPLEQA_DIR}"
echo "Diagnostic report: ${E2B_REPORT_MD}"
