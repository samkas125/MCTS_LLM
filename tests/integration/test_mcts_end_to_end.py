"""Integration test for MCTS tree with a mock LLM client."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.mcts.backpropagation import backpropagate
from src.mcts.extract import extract_top_trajectories, get_tree_stats
from src.mcts.node import MCTSNode
from src.mcts.selection import select_node


def test_manual_mcts_tree():
    """Test MCTS mechanics with a manually constructed tree."""
    # Build a small tree
    root = MCTSNode(step_text="What is 6*7?")

    # Simulate expansion: two children
    child1 = MCTSNode(step_text="Step 1: Multiply 6 by 7")
    child2 = MCTSNode(step_text="Step 1: Add 6 seven times")
    root.add_child(child1)
    root.add_child(child2)

    # child1 leads to correct terminal
    terminal_correct = MCTSNode(
        step_text="Step 2: 6*7 = \\boxed{42}",
        is_terminal=True,
        final_answer="42",
        reward=1.0,
    )
    child1.add_child(terminal_correct)

    # child2 leads to wrong terminal
    terminal_wrong = MCTSNode(
        step_text="Step 2: 6+6+6+6+6+6+6 = \\boxed{43}",
        is_terminal=True,
        final_answer="43",
        reward=-1.0,
    )
    child2.add_child(terminal_wrong)

    # Backpropagate
    backpropagate(terminal_correct, 1.0)
    backpropagate(terminal_wrong, -1.0)

    # Check Q-values
    assert root.visit_count == 2
    assert root.q_value == 0.0  # (1 + -1) / 2
    assert child1.q_value == 1.0
    assert child2.q_value == -1.0

    # Selection should prefer child1 path
    selected = select_node(root)
    # Should select an unvisited node or the one with higher UCT
    assert selected is not None

    # Extract trajectories
    trajectories = extract_top_trajectories(root, top_k=1)
    assert len(trajectories) == 1
    assert trajectories[0].is_correct
    assert trajectories[0].final_answer == "42"
    assert trajectories[0].avg_q_value == 1.0

    # Tree stats
    stats = get_tree_stats(root)
    assert stats["total_nodes"] == 5
    assert stats["correct_terminals"] == 1
    assert stats["incorrect_terminals"] == 1


def test_multiple_correct_paths():
    """Test extraction when multiple paths lead to correct answers."""
    root = MCTSNode(step_text="What is 2+3?")

    # Path 1: direct
    s1 = MCTSNode(step_text="2+3=5")
    root.add_child(s1)
    t1 = MCTSNode(
        step_text="\\boxed{5}", is_terminal=True, final_answer="5", reward=1.0
    )
    s1.add_child(t1)

    # Path 2: longer but also correct
    s2 = MCTSNode(step_text="Break it down")
    root.add_child(s2)
    s2a = MCTSNode(step_text="2+3 = 2+2+1 = 5")
    s2.add_child(s2a)
    t2 = MCTSNode(
        step_text="\\boxed{5}", is_terminal=True, final_answer="5", reward=1.0
    )
    s2a.add_child(t2)

    # Backpropagate with different visit patterns
    backpropagate(t1, 1.0)
    backpropagate(t1, 1.0)  # Visited twice
    backpropagate(t2, 1.0)

    # Extract top-2: both should be correct
    trajectories = extract_top_trajectories(root, top_k=2)
    assert len(trajectories) == 2
    assert all(t.is_correct for t in trajectories)
    # First should have higher avg Q due to more visits
    assert trajectories[0].avg_q_value >= trajectories[1].avg_q_value
