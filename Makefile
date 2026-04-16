.PHONY: install test lint vllm-server mcts grpo sft eval loop ablations download-data preprocess setup-data

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

install:
	pip install "pip==24.3.1" --quiet --disable-pip-version-check
	sed -i 's|setuptools.backends._legacy:_Backend|setuptools.build_meta|g' pyproject.toml || true
	pip install --no-build-isolation ".[dev]"

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

# Data pipeline
download-data:
	python -c "from src.data.download import download_all_datasets; download_all_datasets('data/raw')"

preprocess:
	python -c "from src.data.preprocess import preprocess_all; preprocess_all('data/raw', 'data/processed')"

# Download + preprocess in one step
setup-data: download-data preprocess

# Pipeline stages
mcts:
	python scripts/run_mcts.py

grpo:
	python scripts/run_grpo.py --max-problems 100

sft:
	python scripts/run_sft.py

eval:
	python scripts/run_eval.py

# Full pipeline
loop:
	python scripts/run_loop.py $(LOOP_ARGS)

ablations:
	python scripts/run_ablations.py
