#!/usr/bin/env bash
# Bridge logprob-margin mechanism check (Priority 3).
#
# Phase 1: Gate 2 sanity run on 5 deterministic case_ids (3 R→W sub +
#          1 R→W non-sub + 1 W→R rescue). Prints each case.
# Phase 2: Full run on 57 adjudicated + 200 controls.
# Phase 3: Analysis + figures.
#
# Usage:
#   bash scripts/infra/run_bridge_margins.sh sanity    # Phase 1 only
#   bash scripts/infra/run_bridge_margins.sh full      # Phase 2 + 3
#   bash scripts/infra/run_bridge_margins.sh analyze   # Phase 3 only
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Bridge logprob-margin scoring (Gemma-3-4B-IT, ~10min GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

MODE="${1:-}"
if [ -z "${MODE}" ]; then
    echo "usage: $0 {sanity|full|analyze}" >&2
    exit 2
fi

PIPELINE="uv run python -m scripts.lib.pipeline"
OUT_DIR="data/gemma3_4b/analysis/bridge_margins/test"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# Deterministic Gate 2 sanity cases (3 R→W sub + 1 R→W non-sub + 1 W→R
# rescue, each drawn by sorting case_ids and taking the first N). Resolved
# at run time from the adjudicated labels so the set is reproducible without
# hand-maintained IDs in this script.
resolve_sanity_case_ids() {
    uv run python - <<'PY'
import json
from pathlib import Path

adj = [json.loads(l) for l in Path(
    "data/judge_validation/bridge_irr/adjudicated_labels.jsonl"
).read_text().splitlines() if l.strip()]

def first_n(cases, n):
    return sorted(cases, key=lambda r: r["case_id"])[:n]

a = first_n(
    [r for r in adj
     if r["transition"] == "right_to_wrong"
     and r["label"] == "wrong_entity_substitution"],
    3,
)
b = first_n(
    [r for r in adj
     if r["transition"] == "right_to_wrong"
     and r["label"] != "wrong_entity_substitution"],
    1,
)
c = first_n(
    [r for r in adj if r["transition"] == "wrong_to_right"],
    1,
)
ids = [r["case_id"] for r in (a + b + c)]
print(" ".join(ids))
PY
}

${PIPELINE} gpu-preflight || true

if [ "${MODE}" = "sanity" ]; then
    # shellcheck disable=SC2207  # ids are whitespace-safe (bridge_test_case_XXXX).
    IDS=($(resolve_sanity_case_ids))
    echo "=== Gate 2 sanity run on ${#IDS[@]} cases ==="
    echo "Cases: ${IDS[*]}"
    uv run python scripts/score_bridge_margins.py \
        --output-dir "${OUT_DIR}_sanity" \
        --n-controls 0 \
        --only-case-ids "${IDS[@]}" \
        --dry-run-print-each-case \
        2>&1 | tee "${LOG_DIR}/bridge_margins_sanity_${LOG_TS}.log"
    echo ""
    echo "Sanity run complete. Review ${OUT_DIR}_sanity/margins.jsonl and the log."
    exit 0
fi

if [ "${MODE}" = "full" ]; then
    echo "=== Full scoring run ==="
    uv run python scripts/score_bridge_margins.py \
        --output-dir "${OUT_DIR}" \
        --n-controls 200 \
        2>&1 | tee "${LOG_DIR}/bridge_margins_full_${LOG_TS}.log"

    echo ""
    echo "=== Analysis ==="
    uv run python scripts/analyze_bridge_margins.py \
        --input "${OUT_DIR}/margins.jsonl" \
        --output-dir "${OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/bridge_margins_analysis_${LOG_TS}.log"

    ${PIPELINE} log-run \
        --run-dir "${OUT_DIR}" \
        --description "Bridge logprob-margin mechanism check (alpha=8.0, k=12, first_3_tokens)" \
        --key-files "margins.jsonl, results.json, *.provenance.json, margin_shift_by_cohort_*.pdf"
    exit 0
fi

if [ "${MODE}" = "analyze" ]; then
    echo "=== Analysis only ==="
    uv run python scripts/analyze_bridge_margins.py \
        --input "${OUT_DIR}/margins.jsonl" \
        --output-dir "${OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/bridge_margins_analysis_${LOG_TS}.log"
    exit 0
fi

echo "Unknown mode: ${MODE}" >&2
exit 2
