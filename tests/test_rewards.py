"""Tests for reward functions."""

from src.rewards.accuracy import accuracy_reward_func, check_answer_equivalence
from src.rewards.format_reward import format_reward_func
from src.rewards.qvalue_reward import QValueRewardFunction


# --- check_answer_equivalence ---


def test_exact_match():
    assert check_answer_equivalence("42", "42")


def test_integer_decimal_equivalence():
    assert check_answer_equivalence("42.0", "42")


def test_different_answers():
    assert not check_answer_equivalence("41", "42")


def test_whitespace_handling():
    assert check_answer_equivalence("  42  ", "42")


def test_comma_handling():
    assert check_answer_equivalence("1,234", "1234")


def test_negative_numbers():
    assert check_answer_equivalence("-7", "-7")
    assert not check_answer_equivalence("-7", "7")


# --- accuracy_reward_func ---


def test_accuracy_correct():
    completions = ["The answer is \\boxed{42}"]
    solution = ["42"]
    rewards = accuracy_reward_func(completions, solution)
    assert rewards == [1.0]


def test_accuracy_wrong():
    completions = ["The answer is \\boxed{41}"]
    solution = ["42"]
    rewards = accuracy_reward_func(completions, solution)
    assert rewards == [0.0]


def test_accuracy_no_answer():
    completions = ["I'm not sure about this"]
    solution = ["42"]
    rewards = accuracy_reward_func(completions, solution)
    assert rewards == [0.0]


def test_accuracy_batch():
    completions = [
        "\\boxed{42}",
        "\\boxed{wrong}",
        "\\boxed{10}",
    ]
    solution = ["42", "42", "10"]
    rewards = accuracy_reward_func(completions, solution)
    assert rewards == [1.0, 0.0, 1.0]


def test_accuracy_conversational_format():
    completions = [[{"role": "assistant", "content": "Therefore \\boxed{42}"}]]
    solution = ["42"]
    rewards = accuracy_reward_func(completions, solution)
    assert rewards == [1.0]


# --- format_reward_func ---


def test_format_full_marks():
    completions = ["<think>reasoning</think> \\boxed{42}"]
    rewards = format_reward_func(completions)
    assert rewards == [1.0]


def test_format_with_steps():
    completions = ["Step 1: do this\n```python\nprint(1)\n```\n\\boxed{42}"]
    rewards = format_reward_func(completions)
    assert rewards == [1.0]


def test_format_boxed_only():
    completions = ["\\boxed{42}"]
    rewards = format_reward_func(completions)
    assert rewards == [0.5]


def test_format_no_boxed():
    completions = ["The answer is 42"]
    rewards = format_reward_func(completions)
    assert rewards == [0.0]


# --- QValueRewardFunction ---


def test_qvalue_reward():
    mcts_data = {
        "prob_1": {"avg_q_value": 0.8, "visit_counts": [2, 3, 4], "q_values": [0.7, 0.8, 0.9]},
    }
    reward_fn = QValueRewardFunction(mcts_data, step_weight_by_visits=False)

    rewards = reward_fn(
        completions=["some completion"],
        problem_id=["prob_1"],
    )
    assert len(rewards) == 1
    assert rewards[0] > 0  # Should be positive for positive Q-value


def test_qvalue_reward_missing_problem():
    mcts_data = {"prob_1": {"avg_q_value": 0.5, "visit_counts": [1], "q_values": [0.5]}}
    reward_fn = QValueRewardFunction(mcts_data, step_weight_by_visits=False)

    rewards = reward_fn(
        completions=["completion"],
        problem_id=["unknown_id"],
    )
    assert rewards == [0.0]


def test_qvalue_reward_visit_boost():
    mcts_data = {
        "easy": {"avg_q_value": 0.5, "visit_counts": [1, 1], "q_values": [0.5, 0.5]},
        "hard": {"avg_q_value": 0.5, "visit_counts": [10, 15], "q_values": [0.5, 0.5]},
    }
    reward_fn = QValueRewardFunction(mcts_data, step_weight_by_visits=True)

    easy_reward = reward_fn(["c"], problem_id=["easy"])[0]
    hard_reward = reward_fn(["c"], problem_id=["hard"])[0]

    # Hard problem (high visit counts) should get boosted
    assert hard_reward > easy_reward
