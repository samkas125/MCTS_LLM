"""Convert MCTS trajectories to training data for GRPO and SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset

if TYPE_CHECKING:
    from src.mcts.extract import MCTSTrajectory

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def trajectories_to_grpo_dataset(
    problems: Dataset,
    mcts_traces: dict[str, list[MCTSTrajectory]] | None = None,
) -> Dataset:
    """Convert problems + optional MCTS traces to prompt-only dataset for GRPO.

    GRPO generates its own completions. The MCTS traces inform the Q-value
    reward function via problem_id lookup, not through direct training data.

    Returns Dataset with columns:
        prompt: list[dict] (conversational format)
        solution: str (ground truth answer)
        problem_id: str
    """

    def format_row(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"],
            "problem_id": example["problem_id"],
        }

    return problems.map(format_row, remove_columns=problems.column_names)


def trajectories_to_sft_dataset(
    problems: Dataset,
    mcts_traces: dict[str, list],
) -> Dataset:
    """Convert MCTS trajectories to prompt-completion format for SFT baseline.

    Uses the best trajectory (highest avg Q-value) as the target completion.

    Returns Dataset with columns:
        messages: list[dict] (full conversation for SFT)
    """
    records = []
    for i in range(len(problems)):
        pid = problems[i]["problem_id"]
        if pid not in mcts_traces or not mcts_traces[pid]:
            continue

        best_traj = mcts_traces[pid][0]  # Already sorted by avg Q-value
        trajectory_text = best_traj.trajectory_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problems[i]["problem"]},
            {"role": "assistant", "content": trajectory_text},
        ]
        records.append({"messages": messages})

    return Dataset.from_list(records)


def save_mcts_traces(
    traces: dict[str, list],
    output_dir: str | Path,
) -> None:
    """Save MCTS traces to JSON for later use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for pid, traj_list in traces.items():
        serializable[pid] = [
            {
                "problem": t.problem,
                "steps": t.steps,
                "codes": t.codes,
                "final_answer": t.final_answer,
                "is_correct": t.is_correct,
                "q_values": t.q_values,
                "visit_counts": t.visit_counts,
                "avg_q_value": t.avg_q_value,
                "trajectory_text": t.trajectory_text,
            }
            for t in traj_list
        ]

    with open(output_dir / "traces.json", "w") as f:
        json.dump(serializable, f, indent=2)


def load_mcts_traces(traces_dir: str | Path) -> dict[str, list[dict]]:
    """Load saved MCTS traces from JSON."""
    traces_path = Path(traces_dir) / "traces.json"
    with open(traces_path) as f:
        return json.load(f)
