"""Self-improvement loop: MCTS -> GRPO -> MCTS iterative training.

This is the top-level orchestrator that manages the full pipeline:
1. Evaluate current model
2. Generate MCTS data with current policy
3. Train with GRPO using MCTS-informed rewards
4. Evaluate trained model
5. Repeat with updated policy
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.mcts_dataset import save_mcts_traces
from src.evaluation.evaluator import evaluate_model, save_eval_results
from src.evaluation.metrics import compute_improvement, format_results_table
from src.inference.vllm_client import VLLMClient
from src.mcts.tree import MCTSConfig, MCTSTree
from src.rewards.accuracy import accuracy_reward_func
from src.rewards.format_reward import format_reward_func
from src.rewards.qvalue_reward import build_qvalue_reward_from_traces
from src.training.data_loader import load_processed_datasets, prepare_grpo_dataset
from src.training.grpo_runner import run_grpo_training

logger = logging.getLogger(__name__)
console = Console()


class SelfImprovementLoop:
    """Orchestrate the MCTS -> GRPO -> MCTS self-play loop."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Math-1.5B",
        data_dir: str = "data",
        output_dir: str = "outputs",
        num_rounds: int = 3,
        mcts_config: MCTSConfig | None = None,
        vllm_base_url: str = "http://localhost:8000/v1",
        reward_weights: list[float] | None = None,
    ):
        self.base_model = base_model
        self.current_model = base_model
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_rounds = num_rounds
        self.mcts_config = mcts_config or MCTSConfig()
        self.vllm_base_url = vllm_base_url
        self.reward_weights = reward_weights or [1.0, 0.1, 0.5]
        self.results_log: list[dict] = []

    async def run(self, start_round: int = 0) -> list[dict]:
        """Execute the full self-improvement pipeline.

        Args:
            start_round: Round to start from (for resuming after interruption).

        Returns:
            Results log with per-round evaluation metrics.
        """
        # Load datasets
        datasets = load_processed_datasets(str(self.data_dir / "processed"))
        train_problems = datasets["train_combined"]
        gsm8k_test = datasets["gsm8k_test"]
        math500_test = datasets["math500_test"]

        for round_num in range(start_round, self.num_rounds):
            console.rule(f"[bold blue]Round {round_num}")
            console.print(f"Model: {self.current_model}")

            # Phase 1: Evaluate current model
            console.print("[yellow]Phase 1: Pre-training evaluation[/yellow]")
            pre_results = evaluate_model(
                self.current_model, gsm8k_test, math500_test
            )

            # Phase 2: MCTS data generation
            console.print("[yellow]Phase 2: MCTS data generation[/yellow]")
            mcts_traces = await self._run_mcts_phase(
                train_problems, round_num
            )

            # Phase 3: GRPO training
            console.print("[yellow]Phase 3: GRPO training[/yellow]")
            new_model_path = self._run_grpo_phase(
                train_problems, mcts_traces, round_num
            )

            # Phase 4: Evaluate trained model
            console.print("[yellow]Phase 4: Post-training evaluation[/yellow]")
            post_results = evaluate_model(
                new_model_path, gsm8k_test, math500_test
            )

            # Log results
            round_log = {
                "round": round_num,
                "model": self.current_model,
                "new_model": new_model_path,
                "pre_eval": pre_results,
                "post_eval": post_results,
                "num_mcts_traces": len(mcts_traces),
            }
            self.results_log.append(round_log)

            # Save results
            save_eval_results(
                post_results,
                str(self.output_dir / "eval_results"),
                "mcts_grpo",
                round_num,
            )
            self._save_results_log()

            # Print summary
            console.print(format_results_table(self.results_log))

            # Update model for next round
            self.current_model = new_model_path

        # Final summary
        summary = compute_improvement(self.results_log)
        console.rule("[bold green]Training Complete")
        for key, value in summary.items():
            console.print(f"  {key}: {value}")

        return self.results_log

    async def _run_mcts_phase(
        self,
        problems: Dataset,
        round_num: int,
    ) -> dict[str, list]:
        """Run MCTS on training problems with current model.

        Returns dict mapping problem_id -> list of MCTSTrajectory objects.
        """
        vllm_client = VLLMClient(
            base_url=self.vllm_base_url,
            model=self.current_model,
        )

        all_trajectories = {}
        solved = 0
        total = len(problems)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"MCTS round {round_num}: 0/{total}", total=total
            )

            for idx in range(total):
                problem = problems[idx]
                tree = MCTSTree(
                    problem=problem["problem"],
                    ground_truth=problem["solution"],
                    config=self.mcts_config,
                )

                trajectories = await tree.run(vllm_client)
                if trajectories:
                    all_trajectories[problem["problem_id"]] = trajectories
                    solved += 1

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"MCTS round {round_num}: {idx + 1}/{total} "
                        f"(solved: {solved})"
                    ),
                )

        # Save traces
        traces_dir = self.data_dir / f"mcts_traces/round_{round_num}"
        save_mcts_traces(all_trajectories, traces_dir)

        logger.info(
            f"MCTS round {round_num}: {solved}/{total} problems solved "
            f"({solved / total:.1%})"
        )

        return all_trajectories

    def _run_grpo_phase(
        self,
        problems: Dataset,
        mcts_traces: dict[str, list],
        round_num: int,
    ) -> str:
        """Train model with GRPO using MCTS-informed rewards."""
        # Prepare prompt-only dataset
        train_dataset = prepare_grpo_dataset(problems)

        # Build Q-value reward from MCTS traces
        qvalue_reward = build_qvalue_reward_from_traces(mcts_traces)

        # Train
        new_model_path = run_grpo_training(
            model_name_or_path=self.current_model,
            train_dataset=train_dataset,
            reward_funcs=[accuracy_reward_func, format_reward_func, qvalue_reward],
            reward_weights=self.reward_weights,
            output_dir=str(self.output_dir / "checkpoints"),
            round_num=round_num,
        )

        return new_model_path

    def _save_results_log(self) -> None:
        """Save intermediate results log to disk."""
        log_path = self.output_dir / "results_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Make serializable (remove non-JSON objects)
        serializable = []
        for entry in self.results_log:
            s = {k: v for k, v in entry.items()}
            # Remove wrong_examples (can be large)
            for benchmark in ["gsm8k", "math500"]:
                for phase in ["pre_eval", "post_eval"]:
                    if phase in s and benchmark in s[phase]:
                        s[phase][benchmark].pop("wrong_examples", None)
            serializable.append(s)

        with open(log_path, "w") as f:
            json.dump(serializable, f, indent=2)


def run_loop(
    num_rounds: int = 3,
    start_round: int = 0,
    base_model: str = "Qwen/Qwen2.5-Math-1.5B",
    **kwargs,
) -> list[dict]:
    """Convenience function to run the self-improvement loop."""
    loop = SelfImprovementLoop(
        base_model=base_model,
        num_rounds=num_rounds,
        **kwargs,
    )
    return asyncio.run(loop.run(start_round=start_round))
