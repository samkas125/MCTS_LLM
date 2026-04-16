"""Accuracy reward function using symbolic equivalence checking."""

from src.rewards.answer_extraction import extract_answer


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    s = s.replace(" ", "")
    s = s.replace(",", "")
    # Remove trailing .0 for integers
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except ValueError:
        return s


def check_answer_equivalence(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer is mathematically equivalent to ground truth.

    Uses math_verify for symbolic comparison (\\frac{1}{2} == 0.5),
    falls back to string normalization.
    """
    # Try math_verify first (handles LaTeX, fractions, symbolic expressions)
    try:
        from math_verify import parse, verify
        from math_verify.extraction import ExprExtractionConfig, LatexExtractionConfig

        gold_parsed = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        pred_parsed = parse(
            predicted,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        return verify(gold_parsed, pred_parsed)
    except Exception:
        pass

    # Fallback: normalize and compare
    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)
    return norm_pred == norm_gt


def accuracy_reward_func(completions, solution, **kwargs) -> list[float]:
    """TRL-compatible accuracy reward function.

    Args:
        completions: list of completions (each is list[dict] in conversational format
                     or a string)
        solution: list[str] ground truth answers

    Returns:
        list[float]: 1.0 for correct, 0.0 for incorrect
    """
    rewards = []
    for completion, gt in zip(completions, solution):
        # Handle both conversational and string formats
        if isinstance(completion, list):
            content = completion[-1]["content"]
        else:
            content = completion

        predicted = extract_answer(content)
        if predicted is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if check_answer_equivalence(predicted, gt) else 0.0)
    return rewards
