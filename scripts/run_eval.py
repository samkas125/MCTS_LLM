"""Entry point: Evaluate a model checkpoint on GSM8K and MATH-500."""

import argparse
import json
import logging

from datasets import load_dataset, load_from_disk
from rich.logging import RichHandler

from src.evaluation.evaluator import evaluate_model, save_eval_results

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--gsm8k-test",
        default="data/processed/gsm8k_test.jsonl",
        help="Path to preprocessed GSM8K test set",
    )
    parser.add_argument(
        "--math500-test",
        default="data/processed/math500_test.jsonl",
        help="Path to preprocessed MATH-500 test set",
    )
    parser.add_argument("--output-dir", default="outputs/eval_results")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)

    args = parser.parse_args()

    def _load(path):
        if path.endswith(".jsonl") or path.endswith(".json"):
            return load_dataset("json", data_files=path, split="train")
        return load_from_disk(path)

    gsm8k_test = _load(args.gsm8k_test)
    math500_test = _load(args.math500_test) if args.math500_test else None

    results = evaluate_model(
        model_path=args.model,
        gsm8k_test=gsm8k_test,
        math500_test=math500_test,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Print results
    for benchmark, res in results.items():
        logger.info(f"{benchmark}: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")
        if "per_level" in res:
            for level, acc in res["per_level"].items():
                logger.info(f"  Level {level}: {acc:.4f}")

    # Save
    save_eval_results(results, args.output_dir, "eval", args.round)


if __name__ == "__main__":
    main()
