#!/usr/bin/env bash
# Pre-flight smoke: confirm OpenAI returns metadata for the SIMID primary
# (gpt-4o) and secondary (default gpt-5.5) judge model ids before the
# adjudicated-analysis stage submits a real Batch API job.
#
# Fails fast with a non-zero exit and the wire error if either model id
# is rejected (e.g. typo, deprecation, missing org access). Cheap (<5 s)
# and read-only — it only calls models.retrieve, never enqueues work.
#
# Usage:
#   scripts/infra/check_judge_models.sh                  # gpt-4o + gpt-5.5
#   scripts/infra/check_judge_models.sh gpt-4o gpt-5     # custom secondary
#   SIMID_JUDGE_MODEL=gpt-4o-2024-08-06 \
#     SIMID_SECONDARY_JUDGE_MODEL=gpt-5.1 scripts/infra/check_judge_models.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source .env BEFORE resolving model env vars so SIMID_*_JUDGE_MODEL overrides
# stored there are honored.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PRIMARY_MODEL="${1:-${SIMID_JUDGE_MODEL:-gpt-4o}}"
SECONDARY_MODEL="${2:-${SIMID_SECONDARY_JUDGE_MODEL:-gpt-5.5}}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY not set and not found in .env" >&2
  exit 1
fi

echo "--- SIMID judge-model pre-flight ---"
echo "  primary   = ${PRIMARY_MODEL}"
echo "  secondary = ${SECONDARY_MODEL}"

PRIMARY_MODEL="${PRIMARY_MODEL}" \
SECONDARY_MODEL="${SECONDARY_MODEL}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
uv run python - <<'PY'
import os
import sys

from openai import OpenAI

primary = os.environ["PRIMARY_MODEL"]
secondary = os.environ["SECONDARY_MODEL"]
client = OpenAI()

failures: list[str] = []
for label, model_id in (("primary", primary), ("secondary", secondary)):
    try:
        meta = client.models.retrieve(model_id)
    except Exception as exc:  # noqa: BLE001 - surface any wire error verbatim
        failures.append(f"{label}={model_id!r}: {type(exc).__name__}: {exc}")
        continue
    owned_by = getattr(meta, "owned_by", "?")
    print(f"  OK  {label:<9} {model_id:<28} owned_by={owned_by}")

if failures:
    print("FATAL: judge-model pre-flight failed:", file=sys.stderr)
    for line in failures:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)

print("All judge model ids resolved.")
PY
