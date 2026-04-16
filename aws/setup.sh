#!/bin/bash
# AWS EC2 setup script for MCTS-GRPO training
# Intended for: p4d.24xlarge (8x A100 80GB) or p5.48xlarge (8x H100)
# Single-GPU usage: g5.2xlarge (1x A10G 24GB) for dev, p4de.24xlarge for full runs
#
# Usage:
#   1. Launch an EC2 instance with Deep Learning AMI (Ubuntu, PyTorch 2.4+)
#   2. SSH in and run: bash aws/setup.sh
#   3. Activate: source .venv/bin/activate
#   4. Start: make download-data && make vllm-server

set -euo pipefail

echo "=== MCTS-GRPO AWS Setup ==="

# --- System dependencies ---
sudo apt-get update -qq
sudo apt-get install -y -qq git htop tmux nvtop

# --- Python virtual environment ---
python3 -m venv .venv
source .venv/bin/activate

# --- Install project ---
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# --- Verify GPU access ---
echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# --- Create data directories ---
mkdir -p data/{raw,processed,mcts_traces,training}
mkdir -p outputs/{checkpoints,eval_results,logs}

# --- Setup wandb (user must set WANDB_API_KEY in .env) ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY"
    echo "wandb authenticated"
else
    echo "WARNING: WANDB_API_KEY not set. Run 'wandb login' manually or set it in .env"
fi

# --- HuggingFace login (for gated models) ---
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace authenticated"
else
    echo "WARNING: HF_TOKEN not set. Set it in .env if you need gated model access."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  make download-data"
echo "  make vllm-server  (in a tmux pane)"
echo "  make mcts          (in another pane)"
