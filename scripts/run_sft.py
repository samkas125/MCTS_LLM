"""Entry point: SFT baseline training on MCTS trajectories."""

import argparse
import logging

from datasets import load_from_disk
from rich.logging import RichHandler

from src.data.mcts_dataset import load_mcts_traces
from src.training.data_loader import prepare_sft_dataset
from src.training.sft_runner import run_sft_training

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run SFT baseline training")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/train_combined",
        help="Preprocessed training dataset",
    )
    parser.add_argument(
        "--mcts-traces",
        required=True,
        help="Path to MCTS traces dir (required for SFT)",
    )
    parser.add_argument(
        "--config", default="configs/sft_config.yaml", help="SFT config path"
    )
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--round", type=int, default=0)

    args = parser.parse_args()

    # Load data
    problems = load_from_disk(args.dataset)
    mcts_traces = load_mcts_traces(args.mcts_traces)
    train_dataset = prepare_sft_dataset(problems, mcts_traces)

    logger.info(f"SFT dataset: {len(train_dataset)} examples from MCTS traces")

    # Train
    model_path = run_sft_training(
        model_name_or_path=args.model,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        round_num=args.round,
        config_path=args.config,
    )

    logger.info(f"SFT training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
