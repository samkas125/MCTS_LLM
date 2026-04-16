.PHONY: install test lint vllm-server mcts grpo sft eval loop ablations

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

# vLLM server for MCTS data generation (full GPU)
vllm-server:
	bash scripts/start_vllm_server.sh

# Pipeline stages
mcts:
	python scripts/run_mcts.py

grpo:
	python scripts/run_grpo.py

sft:
	python scripts/run_sft.py

eval:
	python scripts/run_eval.py

# Full pipeline
loop:
	python scripts/run_loop.py

ablations:
	python scripts/run_ablations.py

# Data
download-data:
	python -c "from src.data.download import download_all_datasets; download_all_datasets('data/raw')"
