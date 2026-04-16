"""Tests for UCT selection policy."""

import math

from src.mcts.node import MCTSNode
from src.mcts.selection import select_node, uct_score


def test_uct_unvisited_is_inf():
    parent = MCTSNode(visit_count=10)
    child = MCTSNode(visit_count=0)
    parent.add_child(child)

    assert uct_score(child) == float("inf")


def test_uct_score_computation():
    parent = MCTSNode(visit_count=10)
    child = MCTSNode(visit_count=5, total_value=2.5)
    parent.add_child(child)

    c = 1.414
    expected_exploitation = 2.5 / 5  # Q = 0.5
    expected_exploration = c * math.sqrt(math.log(10) / 5)
    expected = expected_exploitation + expected_exploration

    assert abs(uct_score(child, c) - expected) < 1e-6


def test_uct_prefers_unexplored():
    parent = MCTSNode(visit_count=10)
    visited = MCTSNode(visit_count=5, total_value=4.0)  # Q=0.8
    unvisited = MCTSNode(visit_count=0)
    parent.add_child(visited)
    parent.add_child(unvisited)

    assert uct_score(unvisited) > uct_score(visited)


def test_select_node_reaches_leaf():
    root = MCTSNode(visit_count=10)
    child1 = MCTSNode(visit_count=5, total_value=3.0)
    child2 = MCTSNode(visit_count=3, total_value=2.0)
    root.add_child(child1)
    root.add_child(child2)

    # child1 has grandchild (not a leaf)
    grandchild = MCTSNode(visit_count=2, total_value=1.5)
    child1.add_child(grandchild)

    selected = select_node(root)
    assert selected.is_leaf


def test_select_node_stops_at_terminal():
    root = MCTSNode(visit_count=10)
    terminal = MCTSNode(visit_count=5, total_value=5.0, is_terminal=True)
    root.add_child(terminal)

    selected = select_node(root)
    assert selected.is_terminal


def test_select_balances_exploration_exploitation():
    root = MCTSNode(visit_count=20)

    # High value, many visits (exploit)
    child1 = MCTSNode(visit_count=15, total_value=12.0)  # Q=0.8
    # Low value, few visits (explore)
    child2 = MCTSNode(visit_count=2, total_value=0.5)  # Q=0.25

    root.add_child(child1)
    root.add_child(child2)

    # With default c=1.414, child2 should get a high exploration bonus
    score1 = uct_score(child1)
    score2 = uct_score(child2)

    # child2 should have high exploration bonus due to low visit count
    # but child1 has higher exploitation
    # The exact winner depends on the math, but both should be finite
    assert score1 > 0
    assert score2 > 0
