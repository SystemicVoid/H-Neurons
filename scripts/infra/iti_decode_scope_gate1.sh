#!/usr/bin/env bash
# Stage 5.2 Gate 1: TruthfulQA MC1 cal-val decode-scope panel
#
# Runs the cheap local gate before any SimpleQA generation or API judging.
# This script stops after the cal-val scope panel and writes a compact review
# note so the next decision is explicit.
set -euo pipefail

# Wrap in systemd-inhibit (~20-40 min depending on cache/GPU state)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Decode-scope gate 1 (~4 cal-val scope runs x 2 alphas)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"
STATE_FILE="${CAL_DIR}/pipeline_state.json"
MANIFEST="data/manifests/truthfulqa_cal_val_mc1_seed${SEED}.json"
PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
SCOPES=(full_decode first_token_only first_3_tokens first_8_tokens)
LOG_STEM="decode_scope_gate1"
NOTE_PATH="notes/act3-reports/$(date -u +%Y-%m-%d)-decode-scope-gate1-calval.md"

mkdir -p logs notes notes/act3-reports

if [ ! -f "${STATE_FILE}" ]; then
    echo "FATAL: ${STATE_FILE} not found." >&2
    exit 1
fi
if [ ! -f "${MANIFEST}" ]; then
    echo "FATAL: cal-val manifest not found at ${MANIFEST}" >&2
    exit 1
fi
if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

LOCKED_K=$(jq -r '.gate_1_sweep.locked.K' "${STATE_FILE}")
LOCKED_ALPHA=$(jq -r '.gate_1_sweep.locked.alpha' "${STATE_FILE}")
if [ "${LOCKED_K}" = "null" ] || [ "${LOCKED_ALPHA}" = "null" ]; then
    echo "FATAL: gate_1_sweep.locked missing from ${STATE_FILE}" >&2
    exit 1
fi
if [ "${LOCKED_K}" != "12" ] || [ "${LOCKED_ALPHA}" != "8.0" ]; then
    echo "FATAL: Stage 5.2 is locked to K=12 and alpha=8.0, got K=${LOCKED_K}, alpha=${LOCKED_ALPHA}" >&2
    exit 1
fi

echo "=== Stage 5.2 Gate 1: Decode-scope panel ==="
echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Artifact: ${PROD_ARTIFACT}"
echo "Manifest: ${MANIFEST}"
echo "Scopes: ${SCOPES[*]}"

echo ""
echo "=== Local validation ==="
uv run pytest tests/test_truthfulness_iti.py tests/test_utils.py -q \
    2>&1 | tee "logs/${LOG_STEM}_pytest.log"
ruff check scripts/intervene_iti.py scripts/run_intervention.py tests/test_truthfulness_iti.py tests/test_utils.py \
    2>&1 | tee "logs/${LOG_STEM}_ruff.log"
ty check scripts \
    2>&1 | tee "logs/${LOG_STEM}_ty.log"
bash -n scripts/infra/simpleqa_standalone.sh
shellcheck scripts/infra/simpleqa_standalone.sh \
    2>&1 | tee "logs/${LOG_STEM}_shellcheck.log"

echo ""
echo "=== GPU preflight ==="
if command -v nvitop &>/dev/null; then
    nvitop -1 2>&1 | tee "logs/${LOG_STEM}_nvitop.log"
fi
if command -v nvidia-smi &>/dev/null; then
    PMON_OUTPUT=$(nvidia-smi pmon -c 1 2>&1 || true)
    printf '%s\n' "${PMON_OUTPUT}" | tee "logs/${LOG_STEM}_nvidia_pmon.log"
    UNEXPECTED_COMPUTE=$(
        printf '%s\n' "${PMON_OUTPUT}" \
            | awk 'NR > 2 && $2 ~ /^[0-9]+$/ && $3 ~ /C/ {print}' \
            | rg -v 'xdg-desktop-por|cosmic-edit|cosmic-comp|cosmic-panel|Xwayland|ghostty|renderD|code|spo' \
            || true
    )
    if [ -n "${UNEXPECTED_COMPUTE}" ]; then
        echo "FATAL: unexpected compute-like GPU processes detected:" >&2
        printf '%s\n' "${UNEXPECTED_COMPUTE}" >&2
        exit 1
    fi
fi

echo ""
echo "=== Resolve output dirs / freshness check ==="
mapfile -t PANEL_DIRS < <(
    LOCKED_K="${LOCKED_K}" \
    PROD_ARTIFACT="${PROD_ARTIFACT}" \
    uv run python - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix

scopes = ("full_decode", "first_token_only", "first_3_tokens", "first_8_tokens")
for scope in scopes:
    suffix = build_iti_output_suffix(
        os.environ["PROD_ARTIFACT"],
        "truthfulqa_paperfaithful",
        int(os.environ["LOCKED_K"]),
        "ranked",
        42,
        "artifact",
        None,
        scope,
    )
    print(
        "data/gemma3_4b/intervention/"
        f"truthfulqa_mc_mc1_{suffix}/experiment"
    )
PY
)

for dir in "${PANEL_DIRS[@]}"; do
    if [ -e "${dir}/alpha_0.0.jsonl" ] || compgen -G "${dir}/results.*.json" >/dev/null; then
        echo "FATAL: existing scope-panel output found at ${dir}" >&2
        echo "Archive or remove the old run before launching a fresh gate-1 panel." >&2
        exit 1
    fi
done

echo ""
echo "=== TruthfulQA MC1 cal-val scope panel ==="
for scope in "${SCOPES[@]}"; do
    echo ""
    echo "--- Scope: ${scope} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${PROD_ARTIFACT}" \
        --iti_family truthfulqa_paperfaithful \
        --benchmark truthfulqa_mc \
        --truthfulqa_variant mc1 \
        --iti_k "${LOCKED_K}" \
        --iti_decode_scope "${scope}" \
        --alphas 0.0 "${LOCKED_ALPHA}" \
        --sample_manifest "${MANIFEST}" \
        2>&1 | tee "logs/${LOG_STEM}_${scope}.log"
done

echo ""
echo "=== Build review note ==="
LOCKED_K="${LOCKED_K}" \
LOCKED_ALPHA="${LOCKED_ALPHA}" \
PROD_ARTIFACT="${PROD_ARTIFACT}" \
MANIFEST="${MANIFEST}" \
NOTE_PATH="${NOTE_PATH}" \
uv run python - <<'PY'
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix

scopes = ("full_decode", "first_token_only", "first_3_tokens", "first_8_tokens")
artifact = os.environ["PROD_ARTIFACT"]
locked_k = int(os.environ["LOCKED_K"])
locked_alpha = os.environ["LOCKED_ALPHA"]
manifest = os.environ["MANIFEST"]
note_path = os.environ["NOTE_PATH"]


def latest_summary(path_glob: str) -> dict:
    matches = sorted(glob.glob(path_glob))
    if not matches:
        raise FileNotFoundError(f"No summary files matched {path_glob}")
    with open(matches[-1], encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_ci(ci: dict) -> str:
    return f"[{100.0 * ci['lower']:.1f}, {100.0 * ci['upper']:.1f}]"


rows: list[dict[str, str | float]] = []
for scope in scopes:
    suffix = build_iti_output_suffix(
        artifact,
        "truthfulqa_paperfaithful",
        locked_k,
        "ranked",
        42,
        "artifact",
        None,
        scope,
    )
    out_dir = (
        "data/gemma3_4b/intervention/"
        f"truthfulqa_mc_mc1_{suffix}/experiment"
    )
    summary = latest_summary(f"{out_dir}/results.*.json")
    base = summary["results"]["0.0"]
    active = summary["results"][locked_alpha]
    delta = float(active["metric_mean"]) - float(base["metric_mean"])
    rows.append(
        {
            "scope": scope,
            "output_dir": out_dir,
            "base_rate": float(base["metric_mean"]),
            "base_ci": fmt_ci(base["compliance"]["ci"]),
            "active_rate": float(active["metric_mean"]),
            "active_ci": fmt_ci(active["compliance"]["ci"]),
            "delta_pp": 100.0 * delta,
            "base_n": f"{base['n_compliant']}/{base['n_total']}",
            "active_n": f"{active['n_compliant']}/{active['n_total']}",
        }
    )

full_decode_gain = next(row["delta_pp"] for row in rows if row["scope"] == "full_decode")
for row in rows:
    if full_decode_gain > 0:
        row["retained_gain"] = row["delta_pp"] / full_decode_gain
    else:
        row["retained_gain"] = float("nan")

lines = [
    "# Decode-Scope Gate 1: TruthfulQA MC1 cal-val",
    "",
    f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    f"- Artifact: `{artifact}`",
    f"- Manifest: `{manifest}`",
    f"- Locked config: `K={locked_k}`, `alpha={locked_alpha}`",
    "- Purpose: cheap answer-selection gate before any SimpleQA generation or batch judging.",
    "",
    "| Scope | MC1 @ α=0.0 | MC1 @ α=8.0 | Δ pp | Retained vs full_decode | Output dir |",
    "| --- | --- | --- | ---: | ---: | --- |",
]

for row in rows:
    retained = "n/a" if row["retained_gain"] != row["retained_gain"] else f"{100.0 * row['retained_gain']:.0f}%"
    lines.append(
        "| "
        f"`{row['scope']}` | "
        f"{fmt_pct(row['base_rate'])} {row['base_ci']} ({row['base_n']}) | "
        f"{fmt_pct(row['active_rate'])} {row['active_ci']} ({row['active_n']}) | "
        f"{row['delta_pp']:+.1f} | "
        f"{retained} | "
        f"`{row['output_dir']}` |"
    )

lines.extend(
    [
        "",
        "## Review Gate",
        "",
        "- Proceed to the 200-ID forced-commitment SimpleQA pilot only if a narrower scope retains a material fraction of the `full_decode` MC1 gain.",
        "- If the cal-val result is weak, negative, or unstable, stop here and review before spending GPU time on generation or API budget on judging.",
    ]
)

with open(note_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote review note to {note_path}")
PY

RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat >> notes/runs_to_analyse.md <<EOF

## ${RUN_TS} | data/gemma3_4b/intervention/truthfulqa_mc_mc1_*scope-*/experiment
What: TruthfulQA MC1 cal-val decode-scope gate-1 panel — paper-faithful ITI K=${LOCKED_K}, alpha=[0.0,${LOCKED_ALPHA}], scopes=[${SCOPES[*]}], ranked heads, artifact directions
Key files: per-scope results.*.json, alpha_*.jsonl, ${NOTE_PATH}
Status: awaiting analysis
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GATE 1: TruthfulQA MC1 cal-val scope panel complete            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Review note: ${NOTE_PATH}"
echo "║  Stop here before any SimpleQA generation or API judging."
echo "╚══════════════════════════════════════════════════════════════════╝"
