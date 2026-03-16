#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="/home/hugo/.ob1/tmp/02-h-neurons/falseqa_watch.log"
CHECK_INTERVAL=60
EXPECTED_LINES=687
ALPHAS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
MODEL_DIR="data/gemma3_4b"
INPUT_DIR="$MODEL_DIR/intervention/falseqa"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$PROJECT_DIR"

log "Starting FalseQA watcher in $PROJECT_DIR"
log "Monitoring ${#ALPHAS[@]} files in $INPUT_DIR for exactly $EXPECTED_LINES lines each"

while true; do
  all_complete=true
  status_parts=()

  for alpha in "${ALPHAS[@]}"; do
    file="$INPUT_DIR/alpha_${alpha}.jsonl"

    if [[ ! -f "$file" ]]; then
      all_complete=false
      status_parts+=("alpha_${alpha}:missing")
      continue
    fi

    lines=$(wc -l < "$file")
    if [[ "$lines" -eq "$EXPECTED_LINES" ]]; then
      status_parts+=("alpha_${alpha}:$lines")
    else
      all_complete=false
      status_parts+=("alpha_${alpha}:$lines/$EXPECTED_LINES")
    fi
  done

  log "Status -> ${status_parts[*]}"

  if [[ "$all_complete" == true ]]; then
    log "All FalseQA generation files are complete. Starting Phase 2 evaluation."

    uv run python scripts/evaluate_intervention.py \
      --benchmark falseqa \
      --input_dir "$MODEL_DIR/intervention/falseqa" \
      --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 2>&1 | tee -a "$LOG_FILE"

    log "Phase 2 completed successfully."

    # ── Phase 3: FaithEval standard prompt re-run ──
    log "Starting Phase 3: FaithEval standard prompt re-run (pro-context retrieval QA)."

    uv run python scripts/run_intervention.py \
      --model_path google/gemma-3-4b-it \
      --classifier_path models/gemma3_4b_classifier.pkl \
      --device_map cuda:0 \
      --benchmark faitheval \
      --prompt_style standard \
      --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 \
      --output_dir "$MODEL_DIR/intervention/faitheval_standard" 2>&1 | tee -a "$LOG_FILE"

    log "Phase 3 (FaithEval standard) completed successfully. Starting Phase 4 plotting."

    uv run python scripts/plot_intervention.py \
      --input_dir "$MODEL_DIR/intervention" \
      --output "$MODEL_DIR/intervention/figure3_compliance.png" 2>&1 | tee -a "$LOG_FILE"

    log "Phase 4 completed successfully. Watcher exiting."
    exit 0
  fi

  log "Not complete yet; sleeping ${CHECK_INTERVAL}s before next check."
  sleep "$CHECK_INTERVAL"
done
