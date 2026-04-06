#!/usr/bin/env bash
# Quick D7 pipeline status check.
# Usage: bash scripts/infra/d7_monitor.sh [--watch]
set -euo pipefail

ROOT="/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons"
LOG="${ROOT}/logs/d7_causal_pilot_20260406_090209.log"
PILOT_ROOT="${ROOT}/data/gemma3_4b/intervention/jailbreak_d7/pilot100"
FULL_ROOT="${ROOT}/data/gemma3_4b/intervention/jailbreak_d7/full500"

echo "═══════════════════════════════════════════════════════"
echo "  D7 Causal Pilot — Status @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════"

# Process alive?
if pgrep -f "d7_causal_pilot.sh" >/dev/null 2>&1; then
    echo "  Pipeline: RUNNING (PID $(pgrep -f 'd7_causal_pilot.sh' | head -1))"
else
    echo "  Pipeline: NOT RUNNING"
fi

# GPU
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
ALPHA_FILES=()
for a in "${ALPHAS[@]}"; do ALPHA_FILES+=("alpha_${a}.jsonl"); done

for cond in probe causal; do
    dir="${PILOT_ROOT}/${cond}/experiment"
    count=0; total=${#ALPHA_FILES[@]}
    for f in "${ALPHA_FILES[@]}"; do [[ -f "${dir}/${f}" ]] && count=$((count + 1)); done
    echo "  ${cond} generation: ${count}/${total} alphas"
done

# CSV-v2 eval
for cond in probe causal; do
    csv2_dir="${PILOT_ROOT}/${cond}/csv2_evaluation"
    count=0
    for a in "${ALPHAS[@]}"; do [[ -f "${csv2_dir}/alpha_${a}_csv2.json" ]] && count=$((count + 1)); done
    echo "  ${cond} CSV-v2 eval: ${count}/${#ALPHAS[@]} alphas"
done

# Lock files
PROBE_LOCK="${PILOT_ROOT}/probe_lock.json"
CAUSAL_LOCK="${PILOT_ROOT}/causal_lock.json"
if [[ -f "${PROBE_LOCK}" ]]; then
    pa=$(python3 -c "import json; print(json.load(open('${PROBE_LOCK}'))['selected_alpha'])" 2>/dev/null || echo "?")
    echo "  Probe locked α: ${pa}"
else
    echo "  Probe locked α: pending"
fi
if [[ -f "${CAUSAL_LOCK}" ]]; then
    ca=$(python3 -c "import json; print(json.load(open('${CAUSAL_LOCK}'))['selected_alpha'])" 2>/dev/null || echo "?")
    echo "  Causal locked α: ${ca}"
else
    echo "  Causal locked α: pending"
fi

echo ""
echo "── Stage 4: Full 500 ──"
CONDITIONS=(baseline l1 probe causal)
for cond in "${CONDITIONS[@]}"; do
    dir="${FULL_ROOT}/${cond}/experiment"
    if [[ -d "${dir}" ]]; then
        n=$(find "${dir}" -name 'alpha_*.jsonl' 2>/dev/null | wc -l)
        echo "  ${cond}: ${n} alpha file(s)"
    else
        echo "  ${cond}: not started"
    fi
done
# Random seeds
for seed in 1 2 3; do
    dir="${FULL_ROOT}/random/seed_${seed}/experiment"
    if [[ -d "${dir}" ]]; then
        n=$(find "${dir}" -name 'alpha_*.jsonl' 2>/dev/null | wc -l)
        echo "  random/seed_${seed}: ${n} alpha file(s)"
    else
        echo "  random/seed_${seed}: not started"
    fi
done

echo ""
echo "── Stage 5: Report ──"
report="${FULL_ROOT}/d7_paired_report.json"
[[ -f "${report}" ]] && echo "  ✓ Report generated" || echo "  ✗ Not yet"

echo ""
echo "── Current Activity ──"
# Extract latest progress line (last line containing a percentage)
latest_progress=$(grep -oP 'Jailbreak α=\S+ +\d+%\|[^|]+\| \d+/\d+ \[[^\]]+\]' "${LOG}" 2>/dev/null | tail -1)
if [[ -n "${latest_progress}" ]]; then
    echo "  ${latest_progress}"
else
    # Fallback to latest stage marker
    grep -E '(Stage|Locked|Wrote|Saved|Loading model)' "${LOG}" 2>/dev/null | tail -1 | sed 's/^/  /'
fi
echo ""
echo "── Last 5 log lines ──"
tail -5 "${LOG}" 2>/dev/null | sed 's/^/  /'

echo ""

# Watch mode
if [[ "${1:-}" == "--watch" ]]; then
    echo "(Refreshing every 5 minutes. Ctrl+C to stop.)"
    sleep 300
    exec "$0" --watch
fi
