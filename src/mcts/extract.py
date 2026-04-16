"""Extract top trajectories from MCTS tree with Q-value annotations."""

from __future__ import annotations

from dataclasses import dataclass

from src.mcts.node import MCTSNode


@dataclass
class MCTSTrajectory:
    """A complete reasoning trajectory extracted from the MCTS tree."""

    problem: str
    steps: list[str]  # Step texts (one per node, excluding root)
    codes: list[str]  # Code blocks per step
    final_answer: str
    is_correct: bool
    q_values: list[float]  # Q-value at each step
    visit_counts: list[int]  # Visit count at each step
    avg_q_value: float  # Average Q across all steps
    trajectory_text: str  # Full concatenated trajectory


def _find_correct_terminals(node: MCTSNode, results: list[MCTSNode]) -> None:
    """DFS to find all terminal nodes with correct answers."""
    if node.is_terminal and node.reward == 1.0:
        results.append(node)

    for child in node.children:
        _find_correct_terminals(child, results)


def _find_all_terminals(node: MCTSNode, results: list[MCTSNode]) -> None:
    """DFS to find all terminal nodes."""
    if node.is_terminal:
        results.append(node)

    for child in node.children:
        _find_all_terminals(child, results)


def extract_top_trajectories(
    root: MCTSNode,
    top_k: int = 2,
) -> list[MCTSTrajectory]:
    """Extract the top-k trajectories with highest average Q-values.

    Searches for all terminal nodes with correct answers, traces each
    back to the root, computes per-step Q-values, and returns the
    top-k ranked by average Q-value.

    Args:
        root: Root node of the MCTS tree.
        top_k: Number of trajectories to extract.

    Returns:
        List of MCTSTrajectory objects, sorted by avg_q_value descending.
    """
    correct_terminals: list[MCTSNode] = []
    _find_correct_terminals(root, correct_terminals)

    trajectories = []
    for terminal in correct_terminals:
        path = terminal.get_trajectory()
        # Skip root node (it only contains the problem text)
        steps = path[1:]

        if not steps:
            continue

        traj = MCTSTrajectory(
            problem=root.step_text,
            steps=[n.step_text for n in steps],
            codes=[n.code_text for n in steps],
            final_answer=terminal.final_answer or "",
            is_correct=True,
            q_values=[n.q_value for n in steps],
            visit_counts=[n.visit_count for n in steps],
            avg_q_value=sum(n.q_value for n in steps) / len(steps),
            trajectory_text="\n".join(
                n.step_text for n in steps if n.step_text
            ),
        )
        trajectories.append(traj)

    # Sort by average Q-value, take top-k
    trajectories.sort(key=lambda t: t.avg_q_value, reverse=True)
    return trajectories[:top_k]


def get_tree_stats(root: MCTSNode) -> dict:
    """Compute summary statistics for an MCTS tree.

    Useful for logging and debugging.
    """
    all_terminals: list[MCTSNode] = []
    _find_all_terminals(root, all_terminals)

    correct = [t for t in all_terminals if t.reward == 1.0]
    incorrect = [t for t in all_terminals if t.reward == -1.0]

    # Count total nodes
    total_nodes = 0
    max_depth = 0
    queue = [root]
    while queue:
        node = queue.pop()
        total_nodes += 1
        max_depth = max(max_depth, node.depth)
        queue.extend(node.children)

    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "total_terminals": len(all_terminals),
        "correct_terminals": len(correct),
        "incorrect_terminals": len(incorrect),
        "solve_rate": len(correct) / max(len(all_terminals), 1),
        "root_q_value": root.q_value,
        "root_visit_count": root.visit_count,
    }
