"""Entry point: GRPO training with optional MCTS Q-value rewards."""

import argparse
import logging

from datasets import load_dataset, load_from_disk
from rich.logging import RichHandler

from src.rewards.accuracy import accuracy_reward_func
from src.rewards.format_reward import format_reward_func
from src.training.data_loader import prepare_grpo_dataset
from src.training.grpo_runner import run_grpo_training

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/train_combined.jsonl",
        help="Preprocessed training dataset",
    )
    parser.add_argument(
        "--mcts-traces",
        default=None,
        help="Path to MCTS traces dir (enables Q-value reward)",
    )
    parser.add_argument(
        "--config", default="configs/grpo_config.yaml", help="GRPO config path"
    )
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max-problems", type=int, default=500,
                        help="Max training problems per round (default: 500 for ~30min GRPO)")

    args = parser.parse_args()

    # Load dataset (supports both Arrow directories and JSONL files)
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        problems = load_dataset("json", data_files=args.dataset, split="train")
    else:
        problems = load_from_disk(args.dataset)

    if args.max_problems and args.max_problems < len(problems):
        problems = problems.select(range(args.max_problems))
    train_dataset = prepare_grpo_dataset(problems)

    # Build reward functions
    reward_funcs = [accuracy_reward_func, format_reward_func]
    reward_weights = [1.0, 0.1]

    if args.mcts_traces:
        from src.data.mcts_dataset import load_mcts_traces
        from src.rewards.qvalue_reward import build_qvalue_reward_from_traces

        mcts_traces = load_mcts_traces(args.mcts_traces)
        qvalue_reward = build_qvalue_reward_from_traces(mcts_traces)
        reward_funcs.append(qvalue_reward)
        reward_weights.append(0.5)
        logger.info("Q-value reward enabled from MCTS traces")

    # Train
    model_path = run_grpo_training(
        model_name_or_path=args.model,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        output_dir=args.output_dir,
        round_num=args.round,
        config_path=args.config,
    )

    logger.info(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
