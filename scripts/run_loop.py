"""Entry point: Full MCTS -> GRPO -> MCTS self-improvement loop."""

import argparse
import logging

from rich.logging import RichHandler

from src.mcts.tree import MCTSConfig
from src.pipeline.self_improvement_loop import run_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement loop")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-7B",
        help="Base model",
    )
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--mcts-rollouts", type=int, default=8)
    parser.add_argument("--mcts-candidates", type=int, default=4)
    parser.add_argument("--problems-per-round", type=int, default=100,
                        help="Problems to run MCTS on per round")
    parser.add_argument("--round", type=int, default=None, help="Run only this round")

    args = parser.parse_args()

    # If --round is specified, run only that round
    if args.round is not None:
        args.start_round = args.round
        args.num_rounds = args.round + 1

    results = run_loop(
        num_rounds=args.num_rounds,
        start_round=args.start_round,
        base_model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vllm_base_url=args.vllm_url,
        problems_per_round=args.problems_per_round,
    )

    logger.info(f"Self-improvement loop complete. {len(results)} rounds executed.")


if __name__ == "__main__":
    main()
