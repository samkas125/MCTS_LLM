"""Entry point: Generate MCTS reasoning traces for training data."""

import argparse
import asyncio
import json
import logging

import yaml
from datasets import load_from_disk
from rich.logging import RichHandler

from src.data.mcts_dataset import save_mcts_traces
from src.inference.vllm_client import VLLMClient
from src.mcts.tree import MCTSConfig, MCTSTree

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


async def run_mcts(args):
    """Run MCTS data generation on training problems."""
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcts_cfg = MCTSConfig(
        num_rollouts=cfg["mcts"]["num_rollouts"],
        num_candidates=cfg["mcts"]["num_candidates"],
        exploration_constant=cfg["mcts"]["exploration_constant"],
        max_depth=cfg["mcts"]["max_depth"],
        temperature=cfg["mcts"]["temperature"],
        rollout_temperature=cfg["mcts"]["rollout_temperature"],
        top_k_trajectories=cfg["mcts"]["top_k_trajectories"],
        sandbox_timeout=cfg["sandbox"]["timeout_seconds"],
        max_tokens_per_step=cfg["inference"]["max_tokens_per_step"],
    )

    # Load dataset
    problems = load_from_disk(args.dataset)
    if args.max_problems:
        problems = problems.select(range(min(args.max_problems, len(problems))))

    logger.info(f"Running MCTS on {len(problems)} problems")
    logger.info(f"Config: {mcts_cfg}")

    # Connect to vLLM
    vllm_client = VLLMClient(
        base_url=f"http://{args.vllm_host}:{args.vllm_port}/v1",
        model=args.model,
        max_concurrent=cfg["inference"]["max_concurrent"],
    )

    all_trajectories = {}
    solved = 0

    for idx in range(len(problems)):
        problem = problems[idx]
        tree = MCTSTree(
            problem=problem["problem"],
            ground_truth=problem["solution"],
            config=mcts_cfg,
        )

        trajectories = await tree.run(vllm_client)
        if trajectories:
            all_trajectories[problem["problem_id"]] = trajectories
            solved += 1

        if (idx + 1) % 10 == 0:
            logger.info(
                f"Progress: {idx + 1}/{len(problems)} "
                f"(solved: {solved}, rate: {solved / (idx + 1):.1%})"
            )

        # Periodic save
        if (idx + 1) % 100 == 0:
            save_mcts_traces(all_trajectories, args.output_dir)

    # Final save
    save_mcts_traces(all_trajectories, args.output_dir)

    logger.info(
        f"MCTS complete: {solved}/{len(problems)} solved ({solved / len(problems):.1%})"
    )
    logger.info(f"Traces saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MCTS data generation")
    parser.add_argument(
        "--config", default="configs/mcts_config.yaml", help="MCTS config path"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/train_combined",
        help="Path to preprocessed dataset",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument("--output-dir", default="data/mcts_traces/round_0")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=8000)

    args = parser.parse_args()
    asyncio.run(run_mcts(args))


if __name__ == "__main__":
    main()
