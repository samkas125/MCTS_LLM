"""Prompt formatting for Qwen2.5-Math models."""

SYSTEM_PROMPT_COT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

SYSTEM_PROMPT_TIR = (
    "Please integrate natural language reasoning with programs to solve the problem above, "
    "and put your final answer within \\boxed{}."
)

STEP_INSTRUCTION = (
    "Write the next reasoning step. Include:\n"
    "1. A clear explanation in natural language\n"
    "2. Python code that verifies or computes the result\n"
    "Format:\n"
    "Step {step_num}: <explanation>\n"
    "```python\n<code>\n```\n"
    "Result: <describe what the code shows>\n"
)

FINAL_STEP_INSTRUCTION = (
    "If you have enough information to give the final answer, "
    "provide it now within \\boxed{{}}. "
    "Otherwise, write the next reasoning step with code.\n"
)


def build_expansion_prompt(
    problem: str,
    trajectory_so_far: str,
    step_num: int,
    mode: str = "tir",
) -> list[dict]:
    """Build prompt for MCTS expansion (generating a next reasoning step).

    Args:
        problem: The math problem text
        trajectory_so_far: Concatenated text of all steps so far
        step_num: Current step number (1-indexed)
        mode: "tir" for code-augmented CoT, "cot" for plain CoT

    Returns:
        Conversational format messages list
    """
    system = SYSTEM_PROMPT_TIR if mode == "tir" else SYSTEM_PROMPT_COT

    user_content = problem + "\n\n"
    if trajectory_so_far:
        user_content += f"Progress so far:\n{trajectory_so_far}\n\n"
    user_content += STEP_INSTRUCTION.format(step_num=step_num)

    if step_num >= 3:
        user_content += "\n" + FINAL_STEP_INSTRUCTION

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_grpo_prompt(problem: str) -> list[dict]:
    """Build prompt-only format for GRPO training.

    Args:
        problem: The math problem text

    Returns:
        Conversational format messages list
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": problem},
    ]


def build_eval_prompt(problem: str) -> list[dict]:
    """Build prompt for evaluation (greedy decoding).

    Same as GRPO prompt but used during eval with temperature=0.
    """
    return build_grpo_prompt(problem)
