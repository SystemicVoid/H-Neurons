#!/usr/bin/env bash
# =============================================================================
# Lambda Cloud Bootstrap — H-Neurons Pipeline for Mistral-7B-v0.3-Instruct
# =============================================================================
# Target: 1×A100-40GB SXM, Lambda Stack 24.04 (Python 3.12, CUDA pre-installed)
#
# Usage:
#   1. Launch instance in Lambda console
#   2. SSH in: ssh ubuntu@<INSTANCE_IP>
#   3. Upload this script: scp scripts/lambda-bootstrap.sh ubuntu@<IP>:~/
#   4. Run: bash ~/lambda-bootstrap.sh
#
# After bootstrap completes, start claude code in the project tmux session:
#   tmux attach -t hneurons
#   claude
# =============================================================================

set -euo pipefail

echo "=== H-Neurons Lambda Bootstrap ==="
echo "Instance: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# ---------------------------------------------------------------------------
# 1. Install uv (Python package manager)
# ---------------------------------------------------------------------------
echo "[1/6] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ---------------------------------------------------------------------------
# 2. Install Claude Code
# ---------------------------------------------------------------------------
echo "[2/6] Installing Claude Code..."
# Lambda Stack 24.04 has node pre-installed via Lambda Stack
npm install -g @anthropic/claude-code 2>/dev/null || {
    # Fallback: install node via nvm if not present
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
    nvm install --lts
    npm install -g @anthropic/claude-code
}

# ---------------------------------------------------------------------------
# 3. Clone repo and install Python dependencies
# ---------------------------------------------------------------------------
echo "[3/6] Setting up project..."
cd ~
git clone https://github.com/thunlp/H-Neurons.git h-neurons-ref 2>/dev/null || echo "Ref repo already cloned"

mkdir -p ~/h-neurons && cd ~/h-neurons
mkdir -p data/TriviaQA/rc.nocontext data/activations scripts models

# AGENTS.md will be uploaded via scp alongside the scripts
# It goes in the project root so Claude Code reads it on launch

# Set up Python env
uv init --no-readme --python 3.12 2>/dev/null || true
uv add torch transformers datasets accelerate scikit-learn joblib openai tqdm numpy

# ---------------------------------------------------------------------------
# 4. Download Mistral-7B-v0.3-Instruct (~13.5GB)
# ---------------------------------------------------------------------------
echo "[4/6] Downloading Mistral-7B-v0.3-Instruct..."
uv add huggingface-hub
uv run huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir ~/models/Mistral-7B-Instruct-v0.3 \
    --local-dir-use-symlinks False

# ---------------------------------------------------------------------------
# 5. Download TriviaQA dataset
# ---------------------------------------------------------------------------
echo "[5/6] Downloading TriviaQA..."
uv run python3 -c "
from datasets import load_dataset
ds = load_dataset('trivia_qa', 'rc.nocontext', split='train')
ds.to_parquet('data/TriviaQA/rc.nocontext/triviaqa_train.parquet')
print(f'TriviaQA saved: {len(ds)} questions')
"

# ---------------------------------------------------------------------------
# 6. Set up tmux session
# ---------------------------------------------------------------------------
echo "[6/6] Setting up tmux session..."
tmux new-session -d -s hneurons -c ~/h-neurons 2>/dev/null || true

echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo "Model: ~/models/Mistral-7B-Instruct-v0.3 (~13.5GB bf16)"
echo "Project: ~/h-neurons"
echo "GPU memory free for activations: ~26GB"
echo ""
echo "Next steps:"
echo "  1. Upload your scripts + AGENTS.md (from local project root):"
echo "     scp scripts/*.py ubuntu@<IP>:~/h-neurons/scripts/"
echo "     scp scripts/lambda-AGENTS.md ubuntu@<IP>:~/h-neurons/AGENTS.md"
echo "  2. Attach to tmux:"
echo "     tmux attach -t hneurons"
echo "  3. Start Claude Code (full autonomy, no permission prompts):"
echo "     claude --dangerously-skip-permissions"
echo ""
echo "=== Pipeline Commands Reference ==="
echo ""
echo "# Step 1: Collect responses (10 samples/question, rule judge, ~3-5 hrs)"
echo "uv run python3 scripts/collect_responses.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --data_path data/TriviaQA/rc.nocontext/triviaqa_train.parquet "
echo "    --output_path data/mistral7b_TriviaQA_consistency_samples.jsonl "
echo "    --sample_num 10 --max_samples 3500 "
echo "    --backend transformers --judge_type rule"
echo ""
echo "# Step 2: Extract answer tokens (skip if training on all output tokens)"
echo "# Uses LLM judge — can point at local model or external API"
echo "uv run python3 scripts/extract_answer_tokens.py "
echo "    --input_path data/mistral7b_TriviaQA_consistency_samples.jsonl "
echo "    --output_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --tokenizer_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --api_key \$OPENAI_API_KEY --base_url https://api.openai.com/v1"
echo ""
echo "# Step 3: Sample balanced IDs"
echo "uv run python3 scripts/sample_balanced_ids.py "
echo "    --input_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --output_path data/mistral7b_train_qids.json "
echo "    --num_samples 1000"
echo ""
echo "# Step 4: Extract CETT activations (~4-6 hrs, GPU-intensive)"
echo "uv run python3 scripts/extract_activations.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --input_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --train_ids_path data/mistral7b_train_qids.json "
echo "    --output_root data/activations "
echo "    --locations answer_tokens all_except_answer_tokens"
echo ""
echo "# Step 5: Train classifier"
echo "uv run python3 scripts/classifier.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --train_ids data/mistral7b_train_qids.json "
echo "    --train_ans_acts data/activations/answer_tokens "
echo "    --train_other_acts data/activations/all_except_answer_tokens "
echo "    --train_mode 3-vs-1 --penalty l1 --C 1.0 "
echo "    --save_model models/mistral7b_classifier.pkl"
