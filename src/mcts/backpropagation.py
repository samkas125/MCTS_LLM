"""Backpropagation of rewards through the MCTS tree."""

from __future__ import annotations

from src.mcts.node import MCTSNode


def backpropagate(node: MCTSNode, reward: float) -> None:
    """Propagate reward from a terminal/simulated node back to the root.

    Updates visit_count and total_value for every node along the path.
    This produces Q-values: Q(s) = total_value / visit_count.

    Args:
        node: The node where the reward was obtained (leaf or terminal).
        reward: The reward value (+1 for correct, -1 for incorrect).
    """
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_value += reward
        current = current.parent
