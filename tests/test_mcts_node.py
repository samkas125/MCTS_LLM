"""Tests for MCTSNode dataclass."""

from src.mcts.node import MCTSNode


def test_node_creation():
    node = MCTSNode(step_text="test problem")
    assert node.step_text == "test problem"
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.is_leaf
    assert node.is_root
    assert not node.is_terminal


def test_q_value_zero_visits():
    node = MCTSNode()
    assert node.q_value == 0.0


def test_q_value_computation():
    node = MCTSNode(visit_count=4, total_value=2.0)
    assert node.q_value == 0.5


def test_add_child():
    parent = MCTSNode(step_text="root")
    child = MCTSNode(step_text="step 1")
    parent.add_child(child)

    assert len(parent.children) == 1
    assert child.parent is parent
    assert child.depth == 1
    assert not parent.is_leaf
    assert child.is_leaf


def test_get_trajectory():
    root = MCTSNode(step_text="problem")
    step1 = MCTSNode(step_text="step 1")
    step2 = MCTSNode(step_text="step 2")
    root.add_child(step1)
    step1.add_child(step2)

    trajectory = step2.get_trajectory()
    assert len(trajectory) == 3
    assert trajectory[0] is root
    assert trajectory[1] is step1
    assert trajectory[2] is step2


def test_get_trajectory_text():
    root = MCTSNode(step_text="What is 2+2?")
    step1 = MCTSNode(step_text="Step 1: Add 2 and 2", code_output="4")
    step2 = MCTSNode(step_text="Step 2: The answer is 4")
    root.add_child(step1)
    step1.add_child(step2)

    text = step2.get_trajectory_text()
    assert "Step 1" in text
    assert "Step 2" in text
    assert "Output: 4" in text
    # Root text should not appear (it's the problem, not a step)


def test_terminal_node():
    node = MCTSNode(
        is_terminal=True,
        final_answer="42",
        reward=1.0,
    )
    assert node.is_terminal
    assert node.final_answer == "42"
    assert node.reward == 1.0


def test_deep_tree():
    root = MCTSNode(step_text="root")
    current = root
    for i in range(5):
        child = MCTSNode(step_text=f"step {i}")
        current.add_child(child)
        current = child

    assert current.depth == 5
    trajectory = current.get_trajectory()
    assert len(trajectory) == 6  # root + 5 steps
