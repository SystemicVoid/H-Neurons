#!/usr/bin/env bash
# Wrapper: comprehensive codebase prompt with noise excluded.
# Usage: ./code2prompt.sh [extra code2prompt flags...]
#   e.g. ./code2prompt.sh --output-file=prompt_out.md
#        ./code2prompt.sh --line-numbers --token-format format

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── File-type exclusions (binary, data, artefact noise) ──────────────────
EXCLUDE_TYPES=(
  "*.pdf" "*.png" "*.jpg" "*.jpeg" "*.gif" "*.svg" "*.ico"
  "*.pkl" "*.parquet" "*.csv" "*.npy" "*.safetensors" "*.bin" "*.gguf"
  "*.pyc" "*.log"
  "*.sample" "*.TAG" "*.gitkeep" "*.env"
  "*.factory-scm-trigger"
)

# ── Directory / path exclusions (generated, caches, large blobs) ─────────
EXCLUDE_DIRS=(
  "data/**"
  "models/**"
  "logs/**"
  "costs/**"
  ".git/**"
  ".venv/**"
  ".pytest_cache/**"
  ".ruff_cache/**"
  ".herenow/**"
  ".ob1/**"
  ".claude/**"
  ".agents/**"
  "site/lib/**"
  "docs/deep-research/**"
  "prompt.txt"
  "verbosity-test-data.json"
)

# Build args
args=()
for p in "${EXCLUDE_TYPES[@]}" "${EXCLUDE_DIRS[@]}"; do
  args+=(-e "$p")
done

exec code2prompt "$REPO_ROOT" \
  "${args[@]}" \
  --full-directory-tree \
  --line-numbers \
  --token-format format \
  "$@"
