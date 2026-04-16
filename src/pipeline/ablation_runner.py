"""Run ablation experiments comparing MCTS+GRPO against baselines.

Ablations:
    1. MCTS+GRPO (main system) — full pipeline with Q-value reward
    2. MCTS+GRPO (no Q-value) — GRPO with MCTS data but no Q-value reward
    3. MCTS+SFT — SFT training on MCTS trajectories
    4. GRPO-only — GRPO without MCTS data (no Q-value reward)
    5. Base model — zero-shot evaluation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from src.evaluation.evaluator import evaluate_model
from src.evaluation.metrics import format_results_table
from src.rewards.accuracy import accuracy_reward_func
from src.rewards.format_reward import format_reward_func
from src.rewards.qvalue_reward import build_qvalue_reward_from_traces
from src.training.data_loader import (
    load_processed_datasets,
    prepare_grpo_dataset,
    prepare_sft_dataset,
)
from src.training.grpo_runner import run_grpo_training
from src.training.sft_runner import run_sft_training

logger = logging.getLogger(__name__)


class AblationRunner:
    """Run all ablation experiments for comparison."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Math-1.5B",
        data_dir: str = "data",
        output_dir: str = "outputs",
        mcts_traces_path: str | None = None,
    ):
        self.base_model = base_model
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.mcts_traces_path = mcts_traces_path

    def run_all(self) -> dict[str, dict]:
        """Run all ablation experiments.

        Returns dict mapping experiment name -> evaluation results.
        """
        datasets = load_processed_datasets(str(self.data_dir / "processed"))
        train_problems = datasets["train_combined"]
        gsm8k_test = datasets["gsm8k_test"]
        math500_test = datasets["math500_test"]

        # Load MCTS traces (needed for MCTS-based ablations)
        mcts_traces = None
        if self.mcts_traces_path:
            from src.data.mcts_dataset import load_mcts_traces

            mcts_traces = load_mcts_traces(self.mcts_traces_path)

        results = {}

        # Ablation 1: Base model (zero-shot)
        logger.info("=== Ablation: Base Model ===")
        results["base_model"] = evaluate_model(
            self.base_model, gsm8k_test, math500_test
        )

        # Ablation 2: GRPO-only (no MCTS data)
        logger.info("=== Ablation: GRPO-only ===")
        grpo_dataset = prepare_grpo_dataset(train_problems)
        grpo_only_path = run_grpo_training(
            model_name_or_path=self.base_model,
            train_dataset=grpo_dataset,
            reward_funcs=[accuracy_reward_func, format_reward_func],
            reward_weights=[1.0, 0.1],
            output_dir=str(self.output_dir / "ablations"),
            round_num=0,
        )
        results["grpo_only"] = evaluate_model(
            grpo_only_path, gsm8k_test, math500_test
        )

        if mcts_traces:
            # Ablation 3: MCTS+SFT
            logger.info("=== Ablation: MCTS+SFT ===")
            sft_dataset = prepare_sft_dataset(train_problems, mcts_traces)
            sft_path = run_sft_training(
                model_name_or_path=self.base_model,
                train_dataset=sft_dataset,
                output_dir=str(self.output_dir / "ablations"),
                round_num=0,
            )
            results["mcts_sft"] = evaluate_model(
                sft_path, gsm8k_test, math500_test
            )

            # Ablation 4: MCTS+GRPO without Q-value reward
            logger.info("=== Ablation: MCTS+GRPO (no Q-value) ===")
            no_q_path = run_grpo_training(
                model_name_or_path=self.base_model,
                train_dataset=grpo_dataset,
                reward_funcs=[accuracy_reward_func, format_reward_func],
                reward_weights=[1.0, 0.1],
                output_dir=str(self.output_dir / "ablations"),
                round_num=1,
            )
            results["mcts_grpo_no_qvalue"] = evaluate_model(
                no_q_path, gsm8k_test, math500_test
            )

            # Ablation 5: MCTS+GRPO with Q-value reward (main system)
            logger.info("=== Ablation: MCTS+GRPO (full) ===")
            qvalue_reward = build_qvalue_reward_from_traces(mcts_traces)
            full_path = run_grpo_training(
                model_name_or_path=self.base_model,
                train_dataset=grpo_dataset,
                reward_funcs=[
                    accuracy_reward_func,
                    format_reward_func,
                    qvalue_reward,
                ],
                reward_weights=[1.0, 0.1, 0.5],
                output_dir=str(self.output_dir / "ablations"),
                round_num=2,
            )
            results["mcts_grpo_full"] = evaluate_model(
                full_path, gsm8k_test, math500_test
            )

        # Save results
        self._save_results(results)
        self._print_comparison(results)

        return results

    def _save_results(self, results: dict) -> None:
        """Save ablation results to JSON."""
        output_path = self.output_dir / "ablations" / "comparison.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip non-serializable fields
        clean = {}
        for name, res in results.items():
            clean[name] = {}
            for benchmark in ["gsm8k", "math500"]:
                if benchmark in res:
                    entry = dict(res[benchmark])
                    entry.pop("wrong_examples", None)
                    clean[name][benchmark] = entry

        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2)

    def _print_comparison(self, results: dict) -> None:
        """Print a comparison table of all ablations."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Ablation Results")
        table.add_column("Experiment", style="cyan")
        table.add_column("GSM8K", justify="right")
        table.add_column("MATH-500", justify="right")

        for name, res in results.items():
            gsm = res.get("gsm8k", {}).get("accuracy", 0.0)
            math = res.get("math500", {}).get("accuracy", 0.0)
            table.add_row(name, f"{gsm:.4f}", f"{math:.4f}")

        console.print(table)
