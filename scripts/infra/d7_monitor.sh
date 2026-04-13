#!/usr/bin/env bash
# Quick D7 pipeline status check.
# Usage: bash scripts/infra/d7_monitor.sh [--watch]
set -euo pipefail

ROOT="/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons"
LOG=$(ls -t "${ROOT}"/logs/d7_*.log 2>/dev/null | head -1 || true)
PILOT_ROOT="${ROOT}/data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical"
FULL_ROOT="${ROOT}/data/gemma3_4b/intervention/jailbreak_d7/full500_canonical"
PIPELINE=(uv run python -m scripts.lib.pipeline)
PILOT_MANIFEST="${ROOT}/data/manifests/jbb_d7_pilot_harmful100_seed42.json"
FULL_MANIFEST="${ROOT}/data/manifests/jbb_d7_full_harmful500_seed42.json"
PROBE_LOCK="${PILOT_ROOT}/probe_lock.json"
CAUSAL_LOCK="${PILOT_ROOT}/causal_lock.json"
PROBE_LOCKED_ALPHA="1.0"
CAUSAL_LOCKED_ALPHA="4.0"

if [[ -f "${PROBE_LOCK}" ]]; then
    PROBE_LOCKED_ALPHA=$(python3 -c "import json; print(json.load(open('${PROBE_LOCK}'))['selected_alpha'])" 2>/dev/null || echo "1.0")
fi
if [[ -f "${CAUSAL_LOCK}" ]]; then
    CAUSAL_LOCKED_ALPHA=$(python3 -c "import json; print(json.load(open('${CAUSAL_LOCK}'))['selected_alpha'])" 2>/dev/null || echo "4.0")
fi

count_alpha_files() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        echo 0
        return
    fi
    find "${dir}" -maxdepth 1 -type f -name 'alpha_*.jsonl' 2>/dev/null | wc -l
}

stage_status() {
    local dir="$1"
    local manifest="$2"
    shift 2
    local alphas=("$@")

    if [[ ! -d "${dir}" ]]; then
        echo "not started"
        return
    fi
    if "${PIPELINE[@]}" check-stage --output-dir "${dir}" --manifest "${manifest}" --alphas "${alphas[@]}" >/dev/null 2>&1; then
        echo "complete"
    else
        echo "partial"
    fi
}

line_count_for_alpha() {
    local dir="$1"
    local alpha="$2"
    local file="${dir}/alpha_${alpha}.jsonl"
    if [[ -f "${file}" ]]; then
        wc -l < "${file}"
    else
        echo 0
    fi
}

csv2_stage_status() {
    local dir="$1"
    local manifest="$2"
    shift 2
    local alphas=("$@")

    if [[ ! -d "${dir}" ]]; then
        echo "not started"
        return
    fi

    if python3 - "${dir}" "${manifest}" "${alphas[@]}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.utils import format_alpha_label

directory = Path(sys.argv[1])
manifest = Path(sys.argv[2])
alphas = [float(alpha) for alpha in sys.argv[3:]]
expected = len(json.loads(manifest.read_text()))

found_any = False
all_complete = True

for alpha in alphas:
    path = directory / f"alpha_{format_alpha_label(alpha)}.jsonl"
    if not path.exists():
        all_complete = False
        continue

    found_any = True
    rows = 0
    annotated = 0
    try:
        with path.open() as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                rows += 1
                record = json.loads(raw_line)
                if "csv2" in record:
                    annotated += 1
    except (OSError, json.JSONDecodeError):
        all_complete = False
        continue

    if rows < expected or annotated < expected:
        all_complete = False

if found_any and all_complete:
    sys.exit(0)
sys.exit(1)
PY
    then
        echo "complete"
    elif (( $(count_alpha_files "${dir}") > 0 )); then
        echo "partial"
    else
        echo "not started"
    fi
}

echo "═══════════════════════════════════════════════════════"
echo "  D7 Status @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════"

RUNNER_PATTERN="scripts/infra/(d7_causal_pilot|d7_causal_pilot_trimmed|d7_probe_locked_full500|d7_layer_matched_control|d7_minimal_debt_audit)\\.sh"
if pgrep -f "${RUNNER_PATTERN}" >/dev/null 2>&1; then
    echo "  Pipeline: RUNNING (PID $(pgrep -f "${RUNNER_PATTERN}" | head -1))"
else
    echo "  Pipeline: NOT RUNNING"
fi

if command -v nvidia-smi &>/dev/null; then
    read -r gpu_pct mem_used mem_total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ') || true
    echo "  GPU: ${gpu_pct:-?}% util, ${mem_used:-?}/${mem_total:-?} MiB"
fi

echo ""
echo "── Stage 1: Manifests ──"
for f in jbb_d7_extraction_pairs_seed42.jsonl jbb_d7_pilot_harmful100_seed42.json jbb_d7_full_harmful500_seed42.json jbb_d7_metadata_seed42.json; do
    [[ -f "${ROOT}/data/manifests/${f}" ]] && echo "  ✓ ${f}" || echo "  ✗ ${f}"
done

echo ""
echo "── Stage 2: Extraction Artifacts ──"
for d in iti_refusal_probe_d7 iti_refusal_causal_d7; do
    [[ -f "${ROOT}/data/contrastive/refusal/${d}/iti_heads.pt" ]] && echo "  ✓ ${d}" || echo "  ✗ ${d}"
done

echo ""
echo "── Stage 3: Pilot 100 ──"
ALPHAS=(0.0 1.0 2.0 4.0 8.0)

for cond in probe causal; do
    dir="${PILOT_ROOT}/${cond}/experiment"
    count=$(count_alpha_files "${dir}")
    status=$(stage_status "${dir}" "${PILOT_MANIFEST}" "${ALPHAS[@]}")
    echo "  ${cond} generation: ${count}/${#ALPHAS[@]} alpha file(s) (${status})"
done

for cond in probe causal; do
    csv2_dir="${PILOT_ROOT}/${cond}/csv2_evaluation"
    count=$(count_alpha_files "${csv2_dir}")
    status=$(csv2_stage_status "${csv2_dir}" "${PILOT_MANIFEST}" "${ALPHAS[@]}")
    echo "  ${cond} CSV2 eval: ${count}/${#ALPHAS[@]} alpha file(s) (${status})"
done

if [[ -f "${PROBE_LOCK}" ]]; then
    echo "  Probe locked α: ${PROBE_LOCKED_ALPHA}"
else
    echo "  Probe locked α: pending"
fi
if [[ -f "${CAUSAL_LOCK}" ]]; then
    echo "  Causal locked α: ${CAUSAL_LOCKED_ALPHA}"
else
    echo "  Causal locked α: pending"
fi

echo ""
echo "── Stage 4: Full 500 ──"
BASELINE_DIR="${FULL_ROOT}/baseline_noop/experiment"
L1_DIR="${FULL_ROOT}/l1_neuron/experiment"
PROBE_DIR="${FULL_ROOT}/probe_locked/experiment"
CAUSAL_DIR="${FULL_ROOT}/causal_locked/experiment"
echo "  baseline_noop: $(count_alpha_files "${BASELINE_DIR}") alpha file(s) ($(stage_status "${BASELINE_DIR}" "${FULL_MANIFEST}" 1.0))"
echo "  l1_neuron: $(count_alpha_files "${L1_DIR}") alpha file(s) ($(stage_status "${L1_DIR}" "${FULL_MANIFEST}" 3.0))"
echo "  probe_locked: $(count_alpha_files "${PROBE_DIR}") alpha file(s) ($(stage_status "${PROBE_DIR}" "${FULL_MANIFEST}" "${PROBE_LOCKED_ALPHA}"))"
if [[ -d "${PROBE_DIR}" ]]; then
    echo "    alpha_${PROBE_LOCKED_ALPHA} rows: $(line_count_for_alpha "${PROBE_DIR}" "${PROBE_LOCKED_ALPHA}")/500"
fi
echo "  causal_locked: $(count_alpha_files "${CAUSAL_DIR}") alpha file(s) ($(stage_status "${CAUSAL_DIR}" "${FULL_MANIFEST}" "${CAUSAL_LOCKED_ALPHA}"))"

for seed in 1 2; do
    dir="${FULL_ROOT}/causal_random_head_layer_matched/seed_${seed}/experiment"
    if [[ -d "${dir}" ]]; then
        echo "  causal_random_head_layer_matched/seed_${seed}: $(count_alpha_files "${dir}") alpha file(s) ($(stage_status "${dir}" "${FULL_MANIFEST}" "${CAUSAL_LOCKED_ALPHA}"))"
    else
        echo "  causal_random_head_layer_matched/seed_${seed}: not started"
    fi
done

echo ""
echo "── Stage 5: Report ──"
for report in \
    "${FULL_ROOT}/d7_csv2_report.json" \
    "${FULL_ROOT}/d7_csv2_report_layer_matched_control.json" \
    "${FULL_ROOT}/d7_csv2_report_with_probe.json"; do
    if [[ -f "${report}" ]]; then
        echo "  ✓ $(basename "${report}")"
    else
        echo "  ✗ $(basename "${report}")"
    fi
done

echo ""
echo "── Current Activity ──"
if [[ -n "${LOG}" ]]; then
    latest_progress=$(
        grep -oP 'Jailbreak α=\S+ +\d+%\|[^|]+\| \d+/\d+ \[[^\]]+\]' "${LOG}" 2>/dev/null | tail -1 || true
    )
    if [[ -n "${latest_progress}" ]]; then
        echo "  ${latest_progress}"
    else
        latest_marker=$(
            grep -E '(Stage|Locked|Wrote|Saved|Loading model)' "${LOG}" 2>/dev/null | tail -1 || true
        )
        if [[ -n "${latest_marker}" ]]; then
            echo "  ${latest_marker}"
        else
            echo "  No progress marker found in latest log"
        fi
    fi
    echo ""
    echo "── Last 5 log lines ──"
    tail -5 "${LOG}" 2>/dev/null | sed 's/^/  /'
else
    echo "  No D7 log found yet"
    echo ""
    echo "── Last 5 log lines ──"
    echo "  No D7 log found yet"
fi

echo ""

if [[ "${1:-}" == "--watch" ]]; then
    echo "(Refreshing every 5 minutes. Ctrl+C to stop.)"
    sleep 300
    exec "$0" --watch
fi
