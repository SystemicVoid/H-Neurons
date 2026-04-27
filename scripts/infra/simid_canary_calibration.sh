#!/usr/bin/env bash
# Small real Batch API canary for the SIMID open-correctness calibration
# secondary/adjudication pipeline.
#
# Complements scripts/infra/check_judge_models.sh (which only calls
# models.retrieve) by exercising the full wire path: file upload, batch
# enqueue/poll, max_completion_tokens acceptance, and response parsing,
# for both `--mode secondary` and `--mode adjudicate` of
# scripts/label_simid_open_calibration.py.
#
# This DOES spend a tiny real budget: the synthetic queue makes 2 secondary
# calls and normally 1 adjudication call against the configured Batch model.
# At gpt-5.5 Batch pricing ($1.25/M input, $7.50/M output) that is roughly
# $0.002-$0.010 per synthetic run.
#
# Usage:
#   scripts/infra/simid_canary_calibration.sh                      # synthetic queue
#   scripts/infra/simid_canary_calibration.sh path/to/queue.jsonl  # custom small queue
#
# Env knobs (read from .env or shell):
#   SIMID_SECONDARY_JUDGE_MODEL   default gpt-5.5
#   SIMID_CANARY_MAX_ROWS         default 10
#   SIMID_CANARY_TIMEOUT_SECONDS  default 600
#   OPENAI_API_KEY                required

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source .env BEFORE resolving SIMID_SECONDARY_JUDGE_MODEL so .env overrides
# are honored, mirroring scripts/infra/check_judge_models.sh.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY not set and not found in .env" >&2
  exit 1
fi

MODEL="${SIMID_SECONDARY_JUDGE_MODEL:-gpt-5.5}"
MAX_CANARY_ROWS="${SIMID_CANARY_MAX_ROWS:-10}"
TIMEOUT_SECONDS="${SIMID_CANARY_TIMEOUT_SECONDS:-600}"

RULE_PATH="data/judge_validation/simid_open_calibration/adjudication_rule.md"
if [[ ! -f "$RULE_PATH" ]]; then
  echo "FATAL: frozen calibration rule missing at $RULE_PATH" >&2
  echo "  This canary requires the real rule file the production labeler uses." >&2
  exit 1
fi

TMP="$(mktemp -d -t simid_canary.XXXXXX)"
cleanup() {
  local status="$?"
  if [[ "$status" -eq 0 ]]; then
    rm -rf "$TMP"
  else
    echo "Canary failed or was interrupted; preserving temp dir for resume/inspection:" >&2
    echo "  $TMP" >&2
    if compgen -G "${TMP}/*.batch_state.json" > /dev/null; then
      echo "Saved Batch resume state:" >&2
      local state_path
      for state_path in "$TMP"/*.batch_state.json; do
        [[ -e "$state_path" ]] || continue
        echo "  $state_path" >&2
      done
    fi
    echo "Inspect active-run locks with:" >&2
    echo "  uv run python -m scripts.lib.pipeline active-run-status" >&2
  fi
}
trap cleanup EXIT

QUEUE_ARG="${1:-}"
if [[ -n "$QUEUE_ARG" ]]; then
  if [[ ! -f "$QUEUE_ARG" ]]; then
    echo "FATAL: provided queue file does not exist: $QUEUE_ARG" >&2
    exit 1
  fi
  QUEUE_PATH="$QUEUE_ARG"
  echo "  Using caller-supplied queue: $QUEUE_PATH"
else
  QUEUE_PATH="$TMP/canary_queue.jsonl"
  # Synthetic 2-row queue: it is undersized, but otherwise satisfies the
  # launch validator's non-bypassable source-mix and distinct-grade checks.
  # The first row forces primary=INCORRECT for an obviously correct answer so
  # the secondary's natural CORRECT verdict should create an adjudication case.
  # The second row supplies the stratified-panel source and the second primary
  # grade class while normally agreeing with the secondary.
  cat > "$QUEUE_PATH" <<'JSONL'
{"schema_version": "simid_open_calibration_queue/v1", "calibration_case_id": "simid_open_cal_canary_e2e_forced_disagreement", "sample_source": "audit_queue_disagreement", "sampling_stratum": "canary::forced_disagreement", "sample_id": "canary_sample_forced_disagreement", "condition": "selected", "alpha": 0.0, "base_sample_id": "canary_sample_forced_disagreement", "dataset": "triviaqa_bridge", "mc_endpoint": "synthetic_mc1", "question": "What is the capital of France?", "gold_aliases": ["Paris", "Paris, France"], "response": "Paris", "primary_judge_grade": "INCORRECT", "judge_grade": "INCORRECT"}
{"schema_version": "simid_open_calibration_queue/v1", "calibration_case_id": "simid_open_cal_canary_e2e_consensus", "sample_source": "stratified_panel::canary", "sampling_stratum": "canary::consensus", "sample_id": "canary_sample_consensus", "condition": "selected", "alpha": 0.0, "base_sample_id": "canary_sample_consensus", "dataset": "triviaqa_bridge", "mc_endpoint": "synthetic_mc1", "question": "What is 2 + 2?", "gold_aliases": ["4", "four"], "response": "4", "primary_judge_grade": "CORRECT", "judge_grade": "CORRECT"}
JSONL
  echo "  Built synthetic 2-row canary queue: $QUEUE_PATH"
fi

QUEUE_ROWS="$(grep -c -v '^[[:space:]]*$' "$QUEUE_PATH" || true)"
if (( QUEUE_ROWS < 2 || QUEUE_ROWS > MAX_CANARY_ROWS )); then
  echo "FATAL: canary queue must have between 2 and $MAX_CANARY_ROWS rows, got $QUEUE_ROWS" >&2
  echo "  It must satisfy the non-size launch checks and contain at least one" >&2
  echo "  primary/secondary disagreement so adjudication is exercised." >&2
  exit 1
fi

SECONDARY_OUT="$TMP/secondary_labels.jsonl"
ADJUDICATION_OUT="$TMP/adjudication.jsonl"

echo "--- SIMID Batch canary preflight ---"
echo "  model       = $MODEL"
echo "  queue       = $QUEUE_PATH"
echo "  rows        = $QUEUE_ROWS"
echo "  rule        = $RULE_PATH"
echo "  tmp         = $TMP"
echo "  timeout     = ${TIMEOUT_SECONDS}s per stage"
echo "  budget      = tiny; scales with queue rows and disagreement count"
echo

# --- Stage 1: secondary labeling -------------------------------------------
# `--model` selects the Batch API model for this stage. We also pass
# `--secondary-model` so provenance records the configured secondary judge
# id even though it is unused in `--mode secondary`.
echo "[1/2] Running --mode secondary (real Batch submission, up to ${TIMEOUT_SECONDS}s)..."
timeout --signal=INT --kill-after=60s "$TIMEOUT_SECONDS" \
  uv run python scripts/label_simid_open_calibration.py \
  --mode secondary \
  --queue-path "$QUEUE_PATH" \
  --rule-path "$RULE_PATH" \
  --output "$SECONDARY_OUT" \
  --model "$MODEL" \
  --secondary-model "$MODEL" \
  --allow-undersized-queue

if [[ ! -s "$SECONDARY_OUT" ]]; then
  echo "FAIL: secondary output is empty: $SECONDARY_OUT" >&2
  exit 1
fi

SECONDARY_ROWS="$(grep -c -v '^[[:space:]]*$' "$SECONDARY_OUT" || true)"
if [[ "$SECONDARY_ROWS" != "$QUEUE_ROWS" ]]; then
  echo "FAIL: secondary output must have $QUEUE_ROWS rows, got $SECONDARY_ROWS" >&2
  exit 1
fi

SECONDARY_LABELS="$(
  SECONDARY_OUT="$SECONDARY_OUT" uv run python - <<'PY'
from collections import Counter
import json
import os
import sys

path = os.environ["SECONDARY_OUT"]
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if not rows:
    print("expected non-empty secondary rows", file=sys.stderr)
    sys.exit(1)
case_ids = set()
labels = []
allowed = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
for row in rows:
    case_id = row.get("calibration_case_id")
    if not isinstance(case_id, str) or not case_id:
        print(f"invalid secondary case id: {case_id!r}", file=sys.stderr)
        sys.exit(1)
    if case_id in case_ids:
        print(f"duplicate secondary case id: {case_id}", file=sys.stderr)
        sys.exit(1)
    case_ids.add(case_id)
    label = row.get("label") or row.get("judge_grade")
    if label not in allowed:
        print(f"invalid secondary label: {label!r}", file=sys.stderr)
        sys.exit(1)
    labels.append(label)
counts = Counter(labels)
print(", ".join(f"{label}={counts[label]}" for label in sorted(counts)))
PY
)"
echo "  secondary labels = $SECONDARY_LABELS"

DISAGREEMENT_INFO="$(
  QUEUE_PATH="$QUEUE_PATH" SECONDARY_OUT="$SECONDARY_OUT" uv run python - <<'PY'
import json
import os
import sys

allowed = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}


def grade(row: dict, *, row_kind: str) -> str:
    raw = (
        row.get("judge_grade")
        or row.get("primary_judge_grade")
        or row.get("label")
        or row.get("verdict")
    )
    if raw is None and isinstance(row.get("primary_effective_open_grade"), dict):
        raw = row["primary_effective_open_grade"].get("judge_grade")
    label = str(raw).strip().upper() if raw is not None else ""
    if label not in allowed:
        print(f"invalid {row_kind} grade: {raw!r}", file=sys.stderr)
        sys.exit(1)
    return label


with open(os.environ["QUEUE_PATH"], encoding="utf-8") as handle:
    queue = [json.loads(line) for line in handle if line.strip()]
with open(os.environ["SECONDARY_OUT"], encoding="utf-8") as handle:
    secondary = [json.loads(line) for line in handle if line.strip()]
secondary_by_case = {str(row.get("calibration_case_id")): row for row in secondary}
ids = []
for row in queue:
    case_id = str(row.get("calibration_case_id"))
    if case_id not in secondary_by_case:
        print(f"missing secondary row for {case_id}", file=sys.stderr)
        sys.exit(1)
    if grade(row, row_kind="primary") != grade(
        secondary_by_case[case_id], row_kind="secondary"
    ):
        ids.append(case_id)
if not ids:
    print(
        "secondary labels produced no disagreements; adjudication path would not run",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"{len(ids)} {' '.join(ids)}")
PY
)"
DISAGREEMENT_ROWS="${DISAGREEMENT_INFO%% *}"
DISAGREEMENT_IDS="${DISAGREEMENT_INFO#* }"
echo "  disagreements   = $DISAGREEMENT_ROWS ($DISAGREEMENT_IDS)"

# --- Stage 2: adjudication --------------------------------------------------
echo
echo "[2/2] Running --mode adjudicate (real Batch submission, up to ${TIMEOUT_SECONDS}s)..."
timeout --signal=INT --kill-after=60s "$TIMEOUT_SECONDS" \
  uv run python scripts/label_simid_open_calibration.py \
  --mode adjudicate \
  --queue-path "$QUEUE_PATH" \
  --rule-path "$RULE_PATH" \
  --secondary-rater-path "$SECONDARY_OUT" \
  --output "$ADJUDICATION_OUT" \
  --model "$MODEL" \
  --secondary-model "$MODEL"

if [[ ! -s "$ADJUDICATION_OUT" ]]; then
  echo "FAIL: adjudication output is empty (no disagreement was forced — synthetic" >&2
  echo "      queue may have flipped or secondary already agreed with primary). File:" >&2
  echo "      $ADJUDICATION_OUT" >&2
  exit 1
fi

ADJUDICATION_ROWS="$(grep -c -v '^[[:space:]]*$' "$ADJUDICATION_OUT" || true)"
if [[ "$ADJUDICATION_ROWS" != "$DISAGREEMENT_ROWS" ]]; then
  echo "FAIL: adjudication output must have $DISAGREEMENT_ROWS rows, got $ADJUDICATION_ROWS" >&2
  exit 1
fi

ADJUDICATION_LABEL="$(
  ADJUDICATION_OUT="$ADJUDICATION_OUT" EXPECTED_ROWS="$DISAGREEMENT_ROWS" uv run python - <<'PY'
from collections import Counter
import json
import os
import sys

path = os.environ["ADJUDICATION_OUT"]
expected_rows = int(os.environ["EXPECTED_ROWS"])
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if len(rows) != expected_rows:
    print(f"expected {expected_rows} adjudication rows, got {len(rows)}", file=sys.stderr)
    sys.exit(1)
allowed = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
labels = []
rule_gap_count = 0
for row in rows:
    label = row.get("label") or row.get("judge_grade")
    if label not in allowed:
        print(f"invalid adjudication label: {label!r}", file=sys.stderr)
        sys.exit(1)
    rule_gap = row.get("rule_gap")
    if not isinstance(rule_gap, bool):
        print(f"adjudication rule_gap must be boolean, got {rule_gap!r}", file=sys.stderr)
        sys.exit(1)
    labels.append(label)
    rule_gap_count += int(rule_gap)
counts = Counter(labels)
print(
    ", ".join(f"{label}={counts[label]}" for label in sorted(counts))
    + f"; rule_gap={rule_gap_count}/{len(rows)}"
)
PY
)"
echo "  adjudication = $ADJUDICATION_LABEL"

echo
echo "============================================================"
echo "PASS  SIMID Batch canary preflight"
echo "  model              = $MODEL"
echo "  secondary labels   = $SECONDARY_LABELS"
echo "  disagreements      = $DISAGREEMENT_ROWS ($DISAGREEMENT_IDS)"
echo "  adjudication       = $ADJUDICATION_LABEL"
echo "  cost estimate      = tiny; see OpenAI Batch usage for exact cost"
echo "============================================================"
