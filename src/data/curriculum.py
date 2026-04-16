"""MCTS-based curriculum learning and visit-count-weighted sampling."""

from __future__ import annotations

from datasets import Dataset


def compute_mcts_difficulty(
    problems: Dataset,
    mcts_traces: dict[str, list[dict]],
) -> Dataset:
    """Update problem difficulty scores using MCTS solve rates and Q-values.

    Difficulty is computed as: 1 - solve_rate (fraction of rollouts reaching correct answer).
    Problems not in mcts_traces keep their original difficulty.
    """

    def update_difficulty(example):
        pid = example["problem_id"]
        if pid in mcts_traces and mcts_traces[pid]:
            best_traj = mcts_traces[pid][0]
            avg_q = best_traj.get("avg_q_value", 0.0) if isinstance(best_traj, dict) else best_traj.avg_q_value
            # Map Q-value from [-1, 1] to difficulty [0, 1]
            # High Q-value = easy problem, low Q-value = hard problem
            difficulty = (1.0 - avg_q) / 2.0
            return {"difficulty": difficulty}
        return {"difficulty": example["difficulty"]}

    return problems.map(update_difficulty)


def sort_by_difficulty(dataset: Dataset, ascending: bool = True) -> Dataset:
    """Sort dataset by difficulty for curriculum learning.

    Args:
        ascending: If True, easy problems first (default for curriculum learning).
    """
    return dataset.sort("difficulty", reverse=not ascending)


def compute_visit_count_weights(
    problems: Dataset,
    mcts_traces: dict[str, list[dict]],
) -> Dataset:
    """Add sample weights based on MCTS visit counts.

    Problems whose MCTS trees contain high-visit-count "crux" nodes
    get higher weight, focusing training on harder decision points.
    """

    def add_weight(example):
        pid = example["problem_id"]
        if pid in mcts_traces and mcts_traces[pid]:
            best_traj = mcts_traces[pid][0]
            visit_counts = (
                best_traj.get("visit_counts", [1])
                if isinstance(best_traj, dict)
                else best_traj.visit_counts
            )
            # Weight proportional to max visit count (proxy for crux difficulty)
            max_visits = max(visit_counts) if visit_counts else 1
            weight = min(max_visits / 4.0, 5.0)  # Cap at 5x
            return {"sample_weight": weight}
        return {"sample_weight": 1.0}

    return problems.map(add_weight)
