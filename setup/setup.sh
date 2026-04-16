#!/bin/bash
# Setup script for MCTS-GRPO training (RunPod / AWS EC2)
# Works as root (RunPod) or with sudo (EC2).
# Usage:
#   bash aws/setup.sh
#   source .venv/bin/activate  (EC2 only — RunPod uses system Python)

set -euo pipefail

echo "=== MCTS-GRPO Setup ==="

# --- System dependencies (skip if apt unavailable or already installed) ---
if command -v apt-get &>/dev/null; then
    APT="apt-get"
    # Use sudo only if we are not already root
    if [ "$(id -u)" -ne 0 ]; then APT="sudo apt-get"; fi
    $APT update -qq 2>/dev/null || true
    $APT install -y -qq git htop tmux nvtop 2>/dev/null || true
fi

# --- Pin pip to a known-good version before installing anything ---
python3 -m pip install "pip==24.3.1" --quiet --disable-pip-version-check 2>/dev/null || true

# --- Fix pyproject.toml build backend if still using the broken one ---
if [ -f pyproject.toml ]; then
    sed -i 's|setuptools.backends._legacy:_Backend|setuptools.build_meta|g' pyproject.toml
fi

# --- Install project dependencies ---
# Use --no-build-isolation to avoid the system Python's stale setuptools
python3 -m pip install --no-build-isolation --quiet ".[dev]"

# --- Verify GPU access ---
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# --- Create runtime directories ---
mkdir -p data/{raw,processed,mcts_traces,training}
mkdir -p outputs/{checkpoints,eval_results,logs}

# --- Authenticate services ---
if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" --relogin 2>/dev/null || true
    echo "wandb: authenticated"
else
    echo "WARNING: WANDB_API_KEY not set"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || \
        hf auth login --token "$HF_TOKEN" 2>/dev/null || true
    echo "HuggingFace: authenticated"
else
    echo "WARNING: HF_TOKEN not set"
fi

# --- Set PYTHONPATH permanently ---
REPO_DIR="$(pwd)"
if ! grep -q "MCTS_LLM" ~/.bashrc 2>/dev/null; then
    echo "export PYTHONPATH=${REPO_DIR}:\$PYTHONPATH" >> ~/.bashrc
fi
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# --- Download and preprocess data ---
echo "=== Downloading and preprocessing datasets ==="
make setup-data

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. In terminal 1: bash scripts/start_vllm_server.sh"
echo "  2. In terminal 2: make mcts"
