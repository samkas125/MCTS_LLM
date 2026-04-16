"""Format reward function for reasoning structure."""

import re


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for proper formatting with reasoning and boxed answer.

    Scoring:
        1.0 - Has structured reasoning (think tags or step-by-step) AND \\boxed{} answer
        0.5 - Has \\boxed{} answer but no structured reasoning
        0.0 - Missing \\boxed{} answer

    Args:
        completions: list of completions (conversational or string format)

    Returns:
        list[float]: format reward for each completion
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[-1]["content"]
        else:
            content = completion

        has_reasoning = bool(
            re.search(r"<think>.*?</think>", content, re.DOTALL)
            or re.search(r"Step \d+:", content)
            or re.search(r"```python", content)
        )
        has_boxed = bool(re.search(r"\\boxed\{.+?\}", content))

        if has_reasoning and has_boxed:
            rewards.append(1.0)
        elif has_boxed:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards
