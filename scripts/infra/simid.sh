#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUN_ROOT="data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
MODEL_PATH="${SIMID_MODEL_PATH:-google/gemma-3-4b-it}"
ITI_ARTIFACT="${SIMID_ITI_ARTIFACT:-data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/infra/simid.sh smoke
  scripts/infra/simid.sh phase0
  scripts/infra/simid.sh mvp
  scripts/infra/simid.sh analyze <run_dir>

Environment overrides:
  SIMID_RUN_NAME, SIMID_RUN_DIR, SIMID_MANIFEST, SIMID_MODEL_PATH,
  SIMID_ITI_ARTIFACT, SIMID_DEVICE_MAP, SIMID_TRUTHFULQA_LEAKAGE_POLICY
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

gpu_preflight() {
  echo "=== GPU preflight ==="
  if command -v nvitop >/dev/null 2>&1; then
    nvitop -1 || true
  else
    echo "nvitop not available; skipping GPU process snapshot"
  fi
}

maybe_inhibit() {
  local why="$1"
  shift
  if [[ -z "${INHIBIT_WRAPPED:-}" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit --what=idle:sleep --why="$why" "$@"
  fi
}

ensure_not_analyzed() {
  local run_dir="$1"
  if [[ -f "${run_dir}/results.json" ]]; then
    die "${run_dir} already contains results.json; archive it or choose a new SIMID_RUN_NAME"
  fi
}

build_manifest() {
  local output="$1"
  local truthfulqa_n="$2"
  local bridge_n="$3"
  local replicates="$4"
  local truthfulqa_leakage_policy="$5"
  uv run python scripts/build_simid_manifest.py \
    --seed 42 \
    --truthfulqa-n "$truthfulqa_n" \
    --truthfulqa-leakage-policy "$truthfulqa_leakage_policy" \
    --bridge-n "$bridge_n" \
    --option-order-replicates "$replicates" \
    --model-path "$MODEL_PATH" \
    --iti-artifact-path "$ITI_ARTIFACT" \
    --output "$output"
}

append_run_note() {
  local run_dir="$1"
  local iso_ts
  iso_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat >> notes/runs_to_analyse.md <<EOF

## ${iso_ts} | ${run_dir}
What: SIMID same-item held-out TruthfulQA MC1 when available + TriviaQA Bridge, ITI paper-faithful k=12 first-3-token scope, alpha grid -8/0/4/8, selected + random-head + random-direction controls
Key files: selected/alpha_0.0.jsonl, selected/alpha_8.0.jsonl, random_head_seed1/alpha_8.0.jsonl, random_direction_seed1/alpha_8.0.jsonl, run_config.json, manifest.locked.json
Status: awaiting analysis
EOF
}

cd "$PROJECT_ROOT"

case "$MODE" in
  smoke)
    RUN_DIR="${SIMID_RUN_DIR:-${BASE_RUN_ROOT}/smoke_${TIMESTAMP}_nonclaimable}"
    MANIFEST="${SIMID_MANIFEST:-data/manifests/simid_smoke_seed42.json}"
    ensure_not_analyzed "$RUN_DIR"
    build_manifest "$MANIFEST" 2 2 1 allow_fitted
    gpu_preflight
    uv run python scripts/run_simid.py \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_DIR" \
      --model-path "$MODEL_PATH" \
      --iti-artifact-path "$ITI_ARTIFACT" \
      --device-map "${SIMID_DEVICE_MAP:-cuda:0}" \
      --alphas 0.0 8.0 \
      --conditions selected \
      --top-k-first-token 5 \
      --mc-max-new-tokens 4 \
      --open-max-new-tokens 64
    uv run python scripts/analyze_simid.py \
      --run-dir "$RUN_DIR" \
      --conditions selected \
      --alphas 0.0 8.0 \
      --n-resamples 2000
    echo "Smoke run complete: ${RUN_DIR}"
    ;;
  phase0)
    RUN_DIR="${SIMID_RUN_DIR:-${BASE_RUN_ROOT}/phase0_${TIMESTAMP}_gates}"
    MANIFEST="${SIMID_MANIFEST:-data/manifests/simid_phase0_seed42.json}"
    ensure_not_analyzed "$RUN_DIR"
    build_manifest "$MANIFEST" 8 8 2 allow_fitted
    gpu_preflight
    uv run python scripts/run_simid.py \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_DIR" \
      --model-path "$MODEL_PATH" \
      --iti-artifact-path "$ITI_ARTIFACT" \
      --device-map "${SIMID_DEVICE_MAP:-cuda:0}" \
      --alphas 0.0 \
      --conditions selected \
      --include-unhooked \
      --noop-check \
      --top-k-first-token 5 \
      --mc-max-new-tokens 4 \
      --open-max-new-tokens 64
    uv run python scripts/analyze_simid.py \
      --run-dir "$RUN_DIR" \
      --conditions unhooked selected \
      --alphas 0.0 \
      --phase0-gates \
      --n-resamples 2000
    echo "Phase 0 gate run complete: ${RUN_DIR}"
    ;;
  mvp)
    maybe_inhibit "SIMID MVP GPU run" "$0" "$@"
    RUN_NAME="${SIMID_RUN_NAME:-mvp_${TIMESTAMP}}"
    RUN_DIR="${SIMID_RUN_DIR:-${BASE_RUN_ROOT}/${RUN_NAME}}"
    MANIFEST="${SIMID_MANIFEST:-data/manifests/simid_truthfulqa_bridge_seed42.json}"
    ensure_not_analyzed "$RUN_DIR"
    uv run python scripts/build_simid_manifest.py \
      --seed 42 \
      --truthfulqa-leakage-policy "${SIMID_TRUTHFULQA_LEAKAGE_POLICY:-heldout_only}" \
      --model-path "$MODEL_PATH" \
      --iti-artifact-path "$ITI_ARTIFACT" \
      --output "$MANIFEST"
    gpu_preflight
    uv run python scripts/run_simid.py \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_DIR" \
      --model-path "$MODEL_PATH" \
      --iti-artifact-path "$ITI_ARTIFACT" \
      --device-map "${SIMID_DEVICE_MAP:-cuda:0}" \
      --alphas -8.0 0.0 4.0 8.0 \
      --conditions selected random_head_seed1 random_direction_seed1 \
      --top-k-first-token 10 \
      --mc-max-new-tokens 4 \
      --open-max-new-tokens 64
    append_run_note "$RUN_DIR"
    echo "MVP run complete: ${RUN_DIR}"
    ;;
  analyze)
    RUN_DIR="${2:-${SIMID_RUN_DIR:-}}"
    [[ -n "$RUN_DIR" ]] || die "analyze mode requires <run_dir> or SIMID_RUN_DIR"
    uv run python scripts/analyze_simid.py --run-dir "$RUN_DIR"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage
    die "unknown mode: ${MODE}"
    ;;
esac
