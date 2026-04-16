#!/bin/bash
# Launch vLLM server for MCTS data generation (full GPU mode)
# Usage: bash scripts/start_vllm_server.sh [model_path]

MODEL=${1:-"Qwen/Qwen2.5-Math-1.5B"}

echo "Starting vLLM server with model: $MODEL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --seed 42 \
    --enable-prefix-caching \
    --max-num-seqs 128
