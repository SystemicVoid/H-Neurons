#!/usr/bin/env bash
# One-shot: run batch judge on the three decode-scope pilot experiment dirs.
# Inference already complete; this is API-only (no GPU needed).
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

DIR_FD="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-full-decode_iti-truthfulqa-paperfaithful-production-iti-head_a0b1088812/experiment"
DIR_F3="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment"
DIR_F8="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-8-tokens_iti-truthfulqa-paperfaithful-production-iti-head_c15089a5d6/experiment"

echo "=== Sanity checks ==="
for dir in "$DIR_FD" "$DIR_F3" "$DIR_F8"; do
    if [ ! -f "${dir}/alpha_0.0.jsonl" ] || [ ! -f "${dir}/alpha_8.0.jsonl" ]; then
        echo "FATAL: missing JSONL files in ${dir}" >&2
        exit 1
    fi
    echo "  OK: ${dir}"
done

mkdir -p logs

echo ""
echo "=== [1/3] Judge: full_decode ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${DIR_FD}" \
    --alphas 0.0 8.0 \
    --api-mode batch \
    --batch-max-enqueued-tokens 1000000 \
    2>&1 | tee logs/decode_scope_judge_full_decode.log
echo "=== [1/3] DONE: full_decode ==="

echo ""
echo "=== [2/3] Judge: first_3_tokens ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${DIR_F3}" \
    --alphas 0.0 8.0 \
    --api-mode batch \
    --batch-max-enqueued-tokens 1000000 \
    2>&1 | tee logs/decode_scope_judge_first_3_tokens.log
echo "=== [2/3] DONE: first_3_tokens ==="

echo ""
echo "=== [3/3] Judge: first_8_tokens ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${DIR_F8}" \
    --alphas 0.0 8.0 \
    --api-mode batch \
    --batch-max-enqueued-tokens 1000000 \
    2>&1 | tee logs/decode_scope_judge_first_8_tokens.log
echo "=== [3/3] DONE: first_8_tokens ==="

echo ""
echo "=============================="
echo "All three judge runs complete."
echo "=============================="
