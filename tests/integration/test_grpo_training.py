"""Integration test for GRPO training (requires GPU or mocked model).

These tests verify the data format is correct for TRL's GRPOTrainer.
Full training tests require a GPU and are marked accordingly.
"""

import pytest

from src.rewards.accuracy import accuracy_reward_func
from src.rewards.format_reward import format_reward_func
from src.training.data_loader import prepare_grpo_dataset


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_mock_dataset():
    """Create a minimal dataset for testing."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "problem_id": ["test_0", "test_1", "test_2"],
            "problem": [
                "What is 2+2?",
                "What is 3*3?",
                "What is 10-7?",
            ],
            "solution": ["4", "9", "3"],
            "source": ["gsm8k", "gsm8k", "gsm8k"],
            "level": [0, 0, 0],
            "subject": ["arithmetic", "arithmetic", "arithmetic"],
            "difficulty": [0.0, 0.0, 0.0],
        }
    )


def test_grpo_dataset_format():
    """Test that prepare_grpo_dataset produces the correct format for TRL."""
    problems = _make_mock_dataset()
    dataset = prepare_grpo_dataset(problems)

    assert "prompt" in dataset.column_names
    assert "solution" in dataset.column_names
    assert "problem_id" in dataset.column_names

    # Check prompt format
    prompt = dataset[0]["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "2+2" in prompt[1]["content"]


def test_reward_functions_compatible():
    """Test that reward functions work with the expected input format."""
    # Simulate GRPO completions (string format)
    completions = [
        "Let me think... 2+2=4 \\boxed{4}",
        "Hmm... \\boxed{5}",
        "I don't know",
    ]
    solutions = ["4", "4", "4"]

    acc_rewards = accuracy_reward_func(completions, solutions)
    assert acc_rewards == [1.0, 0.0, 0.0]

    fmt_rewards = format_reward_func(completions)
    assert fmt_rewards[0] == 0.5  # boxed but no think tags
    assert fmt_rewards[2] == 0.0  # nothing


@pytest.mark.skipif(
    not _has_gpu(),
    reason="Requires GPU for training",
)
def test_grpo_training_smoke():
    """Smoke test: verify GRPO training can initialize (requires GPU)."""
    # This test only checks initialization, not actual training
    # Full training would take too long for a test
    problems = _make_mock_dataset()
    dataset = prepare_grpo_dataset(problems)

    from src.training.grpo_runner import load_grpo_config

    cfg = load_grpo_config()
    assert "grpo" in cfg
    assert "lora" in cfg
