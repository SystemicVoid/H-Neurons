#!/usr/bin/env bash
# =============================================================================
# Lambda Cloud Bootstrap — H-Neurons Pipeline for Mistral-7B-v0.3-Instruct
# =============================================================================
# Target: 1×A100-40GB SXM by default, with GH200/ARM64 notes baked in
#
# Usage:
#   1. Launch instance in Lambda console
#   2. SSH in: ssh ubuntu@<INSTANCE_IP>
#   3. Upload this script: scp scripts/lambda-bootstrap.sh ubuntu@<IP>:~/
#   4. Run: TAILSCALE_AUTH_KEY=tskey-... bash ~/lambda-bootstrap.sh
#      Optional:
#        TAILSCALE_HOSTNAME=hneurons-gh200
#        TAILSCALE_ADVERTISE_TAGS=tag:lambda
#        TAILSCALE_LOCKDOWN_SSH=1
#   5. One-time tailnet prep before the first private-only launch:
#        - enable MagicDNS in the Tailscale admin console
#        - add tag ownership / grants for tag:lambda
#        - mint a tagged auth key for tag:lambda
#
# After bootstrap completes, start claude code in the project tmux session:
#   tmux attach -t hneurons
#   claude
# =============================================================================

set -euo pipefail

TAILSCALE_AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-$(hostname)}"
TAILSCALE_ADVERTISE_TAGS="${TAILSCALE_ADVERTISE_TAGS:-tag:lambda}"
TAILSCALE_LOCKDOWN_SSH="${TAILSCALE_LOCKDOWN_SSH:-1}"
TAILSCALE_USE_TAILSCALE_SSH="${TAILSCALE_USE_TAILSCALE_SSH:-0}"
TAILSCALE_IPV4=""
TAILSCALE_IPV6=""
TAILSCALE_CONNECTED=0

lock_down_ssh_to_tailscale() {
    local listen_ipv4="$1"
    local listen_ipv6="$2"

    if [[ -z "$listen_ipv4" && -z "$listen_ipv6" ]]; then
        echo "Skipping SSH lockdown because no Tailscale IP was detected."
        return
    fi

    echo "Locking SSH to the Tailscale interface..."
    sudo install -d -m 755 /etc/ssh/sshd_config.d
    sudo tee /etc/ssh/sshd_config.d/90-tailscale-only.conf >/dev/null <<EOF
# Managed by scripts/infra/lambda-bootstrap.sh
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
$(if [[ -n "$listen_ipv4" ]]; then printf 'ListenAddress %s\n' "$listen_ipv4"; fi)$(if [[ -n "$listen_ipv6" ]]; then printf 'ListenAddress %s\n' "$listen_ipv6"; fi)
EOF
    sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd
}

echo "=== H-Neurons Lambda Bootstrap ==="
echo "Instance: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# ---------------------------------------------------------------------------
# 1. Install Tailscale
# ---------------------------------------------------------------------------
echo "[1/7] Installing Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled

if [[ -n "$TAILSCALE_AUTH_KEY" ]]; then
    echo "Enrolling instance into Tailscale..."
    tailscale_args=(
        --auth-key="$TAILSCALE_AUTH_KEY"
        --hostname="$TAILSCALE_HOSTNAME"
        --advertise-tags="$TAILSCALE_ADVERTISE_TAGS"
        --accept-dns=true
        --accept-routes=false
        --reset
    )

    if [[ "$TAILSCALE_USE_TAILSCALE_SSH" == "1" ]]; then
        tailscale_args+=(--ssh)
    fi

    sudo tailscale up "${tailscale_args[@]}"
    TAILSCALE_IPV4="$(tailscale ip -4 | head -n 1 || true)"
    TAILSCALE_IPV6="$(tailscale ip -6 | head -n 1 || true)"

    if [[ -n "$TAILSCALE_IPV4" || -n "$TAILSCALE_IPV6" ]]; then
        TAILSCALE_CONNECTED=1
        echo "Tailscale connected."
        [[ -n "$TAILSCALE_IPV4" ]] && echo "  IPv4: $TAILSCALE_IPV4"
        [[ -n "$TAILSCALE_IPV6" ]] && echo "  IPv6: $TAILSCALE_IPV6"
    else
        echo "WARNING: tailscale up completed but no Tailscale IP was detected."
    fi
else
    echo "TAILSCALE_AUTH_KEY not set; installed Tailscale but skipped enrollment."
    echo "Tailnet prep checklist (once per tailnet):"
    echo "  - enable MagicDNS"
    echo "  - add tag ownership / grants for $TAILSCALE_ADVERTISE_TAGS"
    echo "  - mint a tagged auth key for $TAILSCALE_ADVERTISE_TAGS"
    echo "Later, run:"
    echo "  sudo tailscale up --auth-key=tskey-... --hostname=$TAILSCALE_HOSTNAME --advertise-tags=$TAILSCALE_ADVERTISE_TAGS"
fi

if [[ "$TAILSCALE_CONNECTED" == "1" && "$TAILSCALE_LOCKDOWN_SSH" == "1" ]]; then
    lock_down_ssh_to_tailscale "$TAILSCALE_IPV4" "$TAILSCALE_IPV6"
elif [[ "$TAILSCALE_LOCKDOWN_SSH" == "1" ]]; then
    echo "Skipping SSH lockdown because Tailscale enrollment did not complete."
fi

# ---------------------------------------------------------------------------
# 2. Install uv (Python package manager)
# ---------------------------------------------------------------------------
echo "[2/7] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ---------------------------------------------------------------------------
# 3. Install Claude Code
# ---------------------------------------------------------------------------
echo "[3/7] Installing Claude Code..."
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
# 4. Clone repo and install Python dependencies
# ---------------------------------------------------------------------------
echo "[4/7] Setting up project..."
cd ~
git clone https://github.com/thunlp/H-Neurons.git h-neurons-ref 2>/dev/null || echo "Ref repo already cloned"

mkdir -p ~/h-neurons && cd ~/h-neurons
mkdir -p data/TriviaQA/rc.nocontext data/activations scripts models

# AGENTS.md will be uploaded via scp alongside the scripts
# It goes in the project root so Claude Code reads it on launch

# Set up Python env
uv init --no-readme --python 3.12 2>/dev/null || true
uv add torch transformers datasets accelerate scikit-learn joblib openai tqdm numpy

# Create a launcher script that can prefer a GH200 fallback env when needed.
cat > scripts/lambda-run-python.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU_VENV="${GPU_VENV:-$PROJECT_DIR/.venv-gpu}"
GPU_PYTHONPATH_DEFAULT="$GPU_VENV/lib/python3.10/site-packages"

if [[ -x "$GPU_VENV/bin/python" ]]; then
  export PYTHONPATH="${GPU_PYTHONPATH:-$GPU_PYTHONPATH_DEFAULT}"
  exec "$GPU_VENV/bin/python" "$@"
fi

exec uv run python3 "$@"
EOF
chmod +x scripts/lambda-run-python.sh

echo "Checking whether the project uv environment can see CUDA..."
if ! uv run python3 - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "uv project env is not CUDA-capable. Checking system Python..."
  if /usr/bin/python3 - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "System Python has CUDA torch; building .venv-gpu fallback..."
    ~/.local/bin/uv venv --python /usr/bin/python3 --system-site-packages .venv-gpu --allow-existing
    source .venv-gpu/bin/activate
    python -m ensurepip --upgrade
    python -m pip install transformers datasets openai accelerate joblib scikit-learn
    python -m pip install --upgrade pillow "jinja2>=3.1"
    deactivate || true
  else
    echo "WARNING: Neither uv env nor system Python sees CUDA. Investigate torch installation before long jobs."
  fi
fi

# ---------------------------------------------------------------------------
# 5. Download Mistral-7B-v0.3-Instruct (~13.5GB)
# ---------------------------------------------------------------------------
echo "[5/7] Downloading Mistral-7B-v0.3-Instruct..."
uv add huggingface-hub
uv run huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir ~/models/Mistral-7B-Instruct-v0.3 \
    --local-dir-use-symlinks False

# ---------------------------------------------------------------------------
# 6. Download TriviaQA dataset
# ---------------------------------------------------------------------------
echo "[6/7] Downloading TriviaQA..."
uv run python3 -c "
from datasets import load_dataset
ds = load_dataset('trivia_qa', 'rc.nocontext', split='train')
ds.to_parquet('data/TriviaQA/rc.nocontext/triviaqa_train.parquet')
print(f'TriviaQA saved: {len(ds)} questions')
"

# ---------------------------------------------------------------------------
# 7. Set up tmux session
# ---------------------------------------------------------------------------
echo "[7/7] Setting up tmux session..."
tmux new-session -d -s hneurons -c ~/h-neurons 2>/dev/null || true

echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo "Model: ~/models/Mistral-7B-Instruct-v0.3 (~13.5GB bf16)"
echo "Project: ~/h-neurons"
echo "GPU memory free for activations: ~26GB"
if [[ "$TAILSCALE_CONNECTED" == "1" ]]; then
    echo "Tailscale host: $TAILSCALE_HOSTNAME"
    [[ -n "$TAILSCALE_IPV4" ]] && echo "Tailscale IPv4: $TAILSCALE_IPV4"
    echo "SSH exposure: Tailscale-only"
else
    echo "Tailscale host: pending enrollment"
    echo "SSH exposure: unchanged"
fi
echo ""
echo "Next steps:"
if [[ "$TAILSCALE_CONNECTED" == "1" ]]; then
    echo "  1. Upload your scripts + AGENTS.md over Tailscale:"
    echo "     scp scripts/*.py ubuntu@$TAILSCALE_HOSTNAME:~/h-neurons/scripts/"
    echo "     scp scripts/lambda-AGENTS.md ubuntu@$TAILSCALE_HOSTNAME:~/h-neurons/AGENTS.md"
    echo "     # Or use the Tailscale IP directly:"
    [[ -n "$TAILSCALE_IPV4" ]] && echo "     scp scripts/*.py ubuntu@$TAILSCALE_IPV4:~/h-neurons/scripts/"
    attach_step=2
    claude_step=3
else
    echo "  1. Tailnet prep once: enable MagicDNS, add tag ownership / grants for $TAILSCALE_ADVERTISE_TAGS, mint a tagged auth key."
    echo "  2. Enroll the box into Tailscale, then upload your scripts + AGENTS.md:"
    echo "     sudo tailscale up --auth-key=tskey-... --hostname=$TAILSCALE_HOSTNAME --advertise-tags=$TAILSCALE_ADVERTISE_TAGS"
    echo "     scp scripts/*.py ubuntu@$TAILSCALE_HOSTNAME:~/h-neurons/scripts/"
    echo "     scp scripts/lambda-AGENTS.md ubuntu@$TAILSCALE_HOSTNAME:~/h-neurons/AGENTS.md"
    attach_step=3
    claude_step=4
fi
echo "  $attach_step. Attach to tmux:"
echo "     tmux attach -t hneurons"
echo "  $claude_step. Start Claude Code (full autonomy, no permission prompts):"
echo "     claude --dangerously-skip-permissions"
if [[ "$TAILSCALE_CONNECTED" == "1" && "$TAILSCALE_LOCKDOWN_SSH" == "1" ]]; then
    echo "  4. Public SSH is disabled. To reverse it:"
    echo "     sudo rm /etc/ssh/sshd_config.d/90-tailscale-only.conf && sudo systemctl reload ssh"
fi
echo ""
echo "=== Pipeline Commands Reference ==="
echo ""
echo "# Step 1: Collect responses (10 samples/question, rule judge, ~3-5 hrs)"
echo "bash scripts/lambda-run-python.sh scripts/collect_responses.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --data_path data/TriviaQA/rc.nocontext/triviaqa_train.parquet "
echo "    --output_path data/mistral7b_TriviaQA_consistency_samples.jsonl "
echo "    --sample_num 10 --max_samples 3500 "
echo "    --backend transformers --judge_type rule"
echo ""
echo "# Step 2: Extract answer tokens (skip if training on all output tokens)"
echo "# Uses LLM judge — can point at local model or external API"
echo "bash scripts/lambda-run-python.sh scripts/extract_answer_tokens.py "
echo "    --input_path data/mistral7b_TriviaQA_consistency_samples.jsonl "
echo "    --output_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --tokenizer_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --api_key \$OPENAI_API_KEY --base_url https://api.openai.com/v1"
echo ""
echo "# Step 3: Sample balanced IDs"
echo "bash scripts/lambda-run-python.sh scripts/sample_balanced_ids.py "
echo "    --input_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --output_path data/mistral7b_train_qids.json "
echo "    --num_samples 1000"
echo ""
echo "# Step 4: Extract CETT activations (~4-6 hrs, GPU-intensive)"
echo "bash scripts/lambda-run-python.sh scripts/extract_activations.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --input_path data/mistral7b_TriviaQA_answer_tokens.jsonl "
echo "    --train_ids_path data/mistral7b_train_qids.json "
echo "    --output_root data/activations "
echo "    --locations answer_tokens all_except_answer_tokens"
echo ""
echo "# Step 5: Train classifier"
echo "bash scripts/lambda-run-python.sh scripts/classifier.py "
echo "    --model_path ~/models/Mistral-7B-Instruct-v0.3 "
echo "    --train_ids data/mistral7b_train_qids.json "
echo "    --train_ans_acts data/activations/answer_tokens "
echo "    --train_other_acts data/activations/all_except_answer_tokens "
echo "    --train_mode 3-vs-1 --penalty l1 --C 1.0 "
echo "    --save_model models/mistral7b_classifier.pkl"
