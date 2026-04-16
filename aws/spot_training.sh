#!/bin/bash
# Run training on AWS Spot Instance with automatic S3 checkpointing
# Usage: bash aws/spot_training.sh [round_num]
#
# This script handles:
# 1. Downloading latest data/checkpoints from S3
# 2. Running the training pipeline
# 3. Uploading results back to S3
# 4. Graceful shutdown on spot interruption

set -euo pipefail

ROUND=${1:-0}
S3_BUCKET="${S3_BUCKET:-your-mcts-grpo-bucket}"
S3_PREFIX="s3://${S3_BUCKET}/mcts-grpo"

echo "=== Spot Training: Round ${ROUND} ==="

# --- Spot interruption handler ---
cleanup() {
    echo "Caught interruption signal. Syncing to S3..."
    aws s3 sync outputs/ "${S3_PREFIX}/outputs/" --quiet
    aws s3 sync data/mcts_traces/ "${S3_PREFIX}/mcts_traces/" --quiet
    echo "Sync complete. Exiting."
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- Sync latest data from S3 ---
echo "Downloading data from S3..."
aws s3 sync "${S3_PREFIX}/outputs/" outputs/ --quiet 2>/dev/null || true
aws s3 sync "${S3_PREFIX}/mcts_traces/" data/mcts_traces/ --quiet 2>/dev/null || true
aws s3 sync "${S3_PREFIX}/processed/" data/processed/ --quiet 2>/dev/null || true

# --- Check if datasets are downloaded ---
if [ ! -d "data/processed/train_combined" ]; then
    echo "Processed data not found. Downloading and preprocessing..."
    make download-data
    python -c "from src.data.preprocess import preprocess_all; preprocess_all('data/raw', 'data/processed')"
    aws s3 sync data/processed/ "${S3_PREFIX}/processed/" --quiet
fi

# --- Run pipeline phase ---
source .venv/bin/activate

echo "Starting pipeline for round ${ROUND}..."
python scripts/run_loop.py --round "$ROUND"

# --- Upload results ---
echo "Uploading results to S3..."
aws s3 sync outputs/ "${S3_PREFIX}/outputs/" --quiet
aws s3 sync data/mcts_traces/ "${S3_PREFIX}/mcts_traces/" --quiet

echo "=== Round ${ROUND} complete ==="
