"""UCT selection policy for MCTS tree traversal."""

from __future__ import annotations

import math

from src.mcts.node import MCTSNode


def uct_score(node: MCTSNode, exploration_constant: float = 1.414) -> float:
    """Compute the Upper Confidence bound for Trees (UCT) score.

    UCT(s) = Q(s) + c * sqrt(ln(N_parent) / N(s))

    Unvisited nodes return infinity to guarantee initial exploration.

    Args:
        node: The node to score (must have a parent).
        exploration_constant: c in UCT formula. Higher = more exploration.
            sqrt(2) ~ 1.414 is theoretically optimal for rewards in [0,1].

    Returns:
        UCT score (float). Higher is better.
    """
    if node.visit_count == 0:
        return float("inf")

    exploitation = node.q_value
    exploration = exploration_constant * math.sqrt(
        math.log(node.parent.visit_count) / node.visit_count
    )
    return exploitation + exploration


def select_node(root: MCTSNode, exploration_constant: float = 1.414) -> MCTSNode:
    """Descend tree from root by selecting the child with highest UCT at each level.

    Stops when reaching a leaf (unexpanded) or terminal node.

    Args:
        root: Root node of the MCTS tree.
        exploration_constant: c in UCT formula.

    Returns:
        The selected leaf/terminal node.
    """
    current = root
    while not current.is_leaf and not current.is_terminal:
        current = max(
            current.children,
            key=lambda c: uct_score(c, exploration_constant),
        )
    return current
