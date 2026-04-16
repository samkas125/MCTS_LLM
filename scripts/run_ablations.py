"""Entry point: Run all ablation experiments."""

import argparse
import logging

from rich.logging import RichHandler

from src.pipeline.ablation_runner import AblationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model",
    )
    parser.add_argument(
        "--mcts-traces",
        default=None,
        help="Path to MCTS traces dir (for MCTS-based ablations)",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args()

    runner = AblationRunner(
        base_model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mcts_traces_path=args.mcts_traces,
    )

    results = runner.run_all()
    logger.info(f"Ablations complete. {len(results)} experiments run.")


if __name__ == "__main__":
    main()
